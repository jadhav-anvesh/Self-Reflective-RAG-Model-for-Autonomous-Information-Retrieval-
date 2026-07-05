"""Hybrid retriever: semantic search + BM25, merged via Reciprocal Rank Fusion.

Single responsibility: combine the results of two independent retrievers
into one ranked list. This is the "merge results intelligently" piece of
PHASE 2 -- rather than trying to normalize and average two incomparable
score scales (cosine similarity vs. BM25's unbounded scores), we fuse by
*rank position* using Reciprocal Rank Fusion (RRF):

    rrf_score(doc) = sum over retrievers r that returned doc of
                     1 / (rrf_k + rank_in_r)

RRF is the standard, score-scale-agnostic way to combine heterogeneous
rankers and is what most production hybrid-search systems use.

This also carries forward each candidate's `semantic_score`/`bm25_score`
into the fused `RetrievedChunk` (via `retrieve_with_diagnostics`), so a
debug dashboard can show exactly how a chunk scored at every stage.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from langchain_core.documents import Document

from config import config
from src.retrieval.base import BaseRetriever, RetrievedChunk
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _doc_key(doc: Document) -> str:
    """Stable identity for a chunk, used to de-duplicate across retrievers.

    Prefers (document_id, chunk_id) from ingestion metadata -- see
    `src/ingestion/chunking.py` -- and falls back to raw content for any
    document that lacks it (e.g. in unit tests).
    """
    document_id = doc.metadata.get("document_id")
    chunk_id = doc.metadata.get("chunk_id")
    if document_id is not None and chunk_id is not None:
        return f"{document_id}:{chunk_id}"
    return doc.page_content


class HybridRetriever(BaseRetriever):
    """Fuses a semantic retriever and a BM25 retriever via RRF.

    Both underlying retrievers implement `BaseRetriever`, so this class
    doesn't care whether "semantic" is Chroma-backed or something else
    entirely -- it only depends on the interface.
    """

    def __init__(
        self,
        semantic_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        k: int = config.retrieval_k,
        candidate_k: int = config.hybrid_candidate_k,
        rrf_k: int = config.rrf_k,
    ):
        self._semantic = semantic_retriever
        self._bm25 = bm25_retriever
        self._k = k
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        return [chunk.document for chunk in self.retrieve_with_diagnostics(query, k)]

    def retrieve_with_diagnostics(self, query: str, k: Optional[int] = None) -> List[RetrievedChunk]:
        k = k or self._k
        candidate_k = max(self._candidate_k, k)

        semantic_chunks = self._semantic.retrieve_with_diagnostics(query, k=candidate_k)
        bm25_chunks = self._bm25.retrieve_with_diagnostics(query, k=candidate_k)

        logger.debug(
            "Hybrid retrieval for query=%r: %d semantic candidates, %d BM25 candidates",
            query,
            len(semantic_chunks),
            len(bm25_chunks),
        )

        fused = self._reciprocal_rank_fusion([semantic_chunks, bm25_chunks])
        return fused[:k]

    def _reciprocal_rank_fusion(self, chunk_lists: List[List[RetrievedChunk]]) -> List[RetrievedChunk]:
        rrf_scores: Dict[str, float] = {}
        best_by_key: Dict[str, RetrievedChunk] = {}

        for chunk_list in chunk_lists:
            for rank, chunk in enumerate(chunk_list):
                key = _doc_key(chunk.document)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self._rrf_k + rank + 1)

                existing = best_by_key.get(key)
                if existing is None:
                    best_by_key[key] = RetrievedChunk(
                        document=chunk.document,
                        final_rank=0,  # placeholder, assigned below
                        semantic_score=chunk.semantic_score,
                        bm25_score=chunk.bm25_score,
                    )
                else:
                    # Merge in whichever score this list contributes that
                    # the other list didn't (e.g. semantic list sets
                    # semantic_score, BM25 list sets bm25_score).
                    if chunk.semantic_score is not None:
                        existing.semantic_score = chunk.semantic_score
                    if chunk.bm25_score is not None:
                        existing.bm25_score = chunk.bm25_score

        ordered_keys = sorted(rrf_scores.keys(), key=lambda key: rrf_scores[key], reverse=True)

        fused: List[RetrievedChunk] = []
        for rank, key in enumerate(ordered_keys):
            chunk = best_by_key[key]
            chunk.final_rank = rank
            chunk.rrf_score = rrf_scores[key]
            fused.append(chunk)
        return fused
