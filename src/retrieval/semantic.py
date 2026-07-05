"""Semantic (vector similarity) retriever.

The original retrieval strategy from PHASE 1, now one half of PHASE 2's
hybrid retrieval (semantic + BM25).
"""
from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from config import config
from src.retrieval.base import BaseRetriever, RetrievedChunk
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class SemanticRetriever(BaseRetriever):
    """Retrieves chunks by embedding similarity against a Chroma vector store."""

    def __init__(self, vectorstore: Chroma, k: int = config.retrieval_k):
        self._vectorstore = vectorstore
        self._k = k

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        k = k or self._k
        logger.debug("Semantic retrieval for query=%r (k=%d)", query, k)
        return self._vectorstore.similarity_search(query, k=k)

    def retrieve_with_diagnostics(self, query: str, k: Optional[int] = None) -> List[RetrievedChunk]:
        k = k or self._k
        try:
            # Normalized to roughly [0, 1], higher = more relevant, when
            # the vector store's embedding function supports it.
            pairs = self._vectorstore.similarity_search_with_relevance_scores(query, k=k)
        except Exception:
            # Fall back to raw distance-based scores if relevance scoring
            # isn't supported for this collection's distance metric.
            pairs = self._vectorstore.similarity_search_with_score(query, k=k)

        return [
            RetrievedChunk(document=doc, final_rank=i, semantic_score=float(score))
            for i, (doc, score) in enumerate(pairs)
        ]
