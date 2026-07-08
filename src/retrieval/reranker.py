"""Cross-encoder re-ranking retriever (PHASE 2, optional).

Single responsibility: take a candidate pool from any `BaseRetriever`
and re-order it with a cross-encoder, which scores (query, chunk) pairs
jointly and is far more accurate than bi-encoder/BM25 similarity alone
-- at the cost of being too slow to run over an entire corpus.

Pipeline:

    base retriever -> top `rerank_candidate_k` (e.g. 20)
                    -> cross-encoder scores each pair
                    -> top `rerank_k` (e.g. 5)
                    -> LLM

Enabled only via `config.use_reranking` (RAG_USE_RERANKING=true). The
model is loaded lazily and cached at module scope so importing this
module -- or constructing retrievers with reranking disabled -- never
pays the cost of loading the cross-encoder weights.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from langchain_core.documents import Document

from config import config
from src.retrieval.base import BaseRetriever, RetrievedChunk
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=None)
def _load_cross_encoder(model_name: str):
    """Load (and cache) a `sentence_transformers.CrossEncoder`.

    Imported lazily so environments that never enable reranking don't
    need the model weights downloaded/loaded at all.
    """
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder reranker model: %s", model_name)
    return CrossEncoder(
        model_name,
        device=config.embedding_device,
    )


class CrossEncoderReranker(BaseRetriever):
    """Wraps another `BaseRetriever` and re-ranks its output with a cross-encoder."""

    def __init__(
        self,
        base_retriever: BaseRetriever,
        model_name: str = config.reranker_model,
        candidate_k: int = config.rerank_candidate_k,
        top_k: int = config.rerank_k,
    ):
        self._base = base_retriever
        self._model_name = model_name
        self._candidate_k = candidate_k
        self._top_k = top_k

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        return [chunk.document for chunk in self.retrieve_with_diagnostics(query, k)]

    def retrieve_with_diagnostics(self, query: str, k: Optional[int] = None) -> List[RetrievedChunk]:
        top_k = k or self._top_k

        candidates = self._base.retrieve_with_diagnostics(query, k=self._candidate_k)
        if not candidates:
            return []

        logger.debug(
            "Cross-encoder re-ranking %d candidates for query=%r (top_k=%d)",
            len(candidates),
            query,
            top_k,
        )

        model = _load_cross_encoder(self._model_name)
        pairs = [[query, chunk.document.page_content] for chunk in candidates]
        scores = model.predict(pairs)

        for chunk, score in zip(candidates, scores):
            chunk.rerank_score = float(score)

        ranked = sorted(candidates, key=lambda chunk: chunk.rerank_score, reverse=True)[:top_k]
        for i, chunk in enumerate(ranked):
            chunk.final_rank = i
        return ranked
