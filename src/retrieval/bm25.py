"""BM25 (lexical / keyword) retriever.

Single responsibility: sparse keyword retrieval over the same corpus
that's stored in the Chroma vector store, used as one half of PHASE 2
hybrid retrieval (semantic + BM25).

BM25 is a plain in-memory index (via `rank_bm25`), rebuilt from
whatever documents currently exist in the vector store. That's cheap
enough for the corpus sizes this project targets (hundreds to low
thousands of chunks) and keeps this module free of its own persistence
concerns -- Chroma remains the single source of truth for chunk content
and metadata (source, page, section, filename, chunk_id, ...).

Note (not implemented): for a much larger corpus you'd want to persist
the BM25 index itself (e.g. pickle it alongside the manifest) instead
of rebuilding it from Chroma on every process start. Not worth the
complexity at this project's scale.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import config
from src.retrieval.base import BaseRetriever, RetrievedChunk
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    """Lowercase, alphanumeric-word tokenizer. Good enough for BM25."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever(BaseRetriever):
    """Retrieves chunks by lexical (keyword) overlap using Okapi BM25.

    The index is built once at construction time from every document
    currently in the vector store, preserving each chunk's full
    metadata (source, page, section, filename, chunk_id, document_id).
    """

    def __init__(self, vectorstore: Chroma, k: int = config.retrieval_k):
        self._k = k
        self._documents: List[Document] = self._load_corpus(vectorstore)

        if self._documents:
            tokenized_corpus = [_tokenize(doc.page_content) for doc in self._documents]
            self._bm25 = BM25Okapi(tokenized_corpus)
        else:
            self._bm25 = None
            logger.warning("BM25Retriever built over an empty vector store; retrieve() will return [].")

    @staticmethod
    def _load_corpus(vectorstore: Chroma) -> List[Document]:
        """Pull every chunk (content + metadata) currently in the vector store."""
        raw = vectorstore.get(include=["documents", "metadatas"])
        contents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        documents = [
            Document(page_content=content, metadata=metadata or {})
            for content, metadata in zip(contents, metadatas)
        ]
        logger.info("BM25Retriever indexed %d chunks from the vector store", len(documents))
        return documents

    def _ranked_indices_with_scores(self, query: str) -> List[Tuple[int, float]]:
        """Every corpus index with a positive BM25 score, best first."""
        if not self._bm25 or not self._documents:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in ranked_indices if scores[i] > 0]

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        k = k or self._k
        logger.debug("BM25 retrieval for query=%r (k=%d)", query, k)
        ranked = self._ranked_indices_with_scores(query)[:k]
        return [self._documents[i] for i, _ in ranked]

    def retrieve_with_diagnostics(self, query: str, k: Optional[int] = None) -> List[RetrievedChunk]:
        k = k or self._k
        ranked = self._ranked_indices_with_scores(query)[:k]
        return [
            RetrievedChunk(document=self._documents[i], final_rank=rank, bm25_score=score)
            for rank, (i, score) in enumerate(ranked)
        ]
