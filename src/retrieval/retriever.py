"""Retriever factory.

Single place that decides which `BaseRetriever` implementation(s) to
construct and how to compose them. Callers (`app.py`, `main.py`,
`graph.py`) depend only on `get_retriever()` and the `BaseRetriever`
interface -- never on a concrete class -- so every PHASE 2 addition
(BM25, hybrid fusion, cross-encoder re-ranking) is purely additive here.

Composition, controlled by `config.py`:

    SemanticRetriever ────┐
                           ├─▶ HybridRetriever ─▶ [CrossEncoderReranker] ─▶ caller
    BM25Retriever ─────────┘        (RRF)           (optional, config.use_reranking)

`config.use_hybrid_retrieval=False` falls back to semantic-only, exactly
matching PHASE 1 behavior. `config.use_reranking=False` (the default)
skips the cross-encoder entirely, so the reranker model is never loaded
unless explicitly enabled.
"""
from __future__ import annotations

from langchain_community.vectorstores import Chroma

from config import config
from src.retrieval.base import BaseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.semantic import SemanticRetriever
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def get_retriever(vectorstore: Chroma, k: int = config.retrieval_k) -> BaseRetriever:
    """Return the fully-configured retriever over the given vector store.

    Args:
        vectorstore: A loaded Chroma vector store (see `src/indexing/vectorstore.py`).
        k: Number of documents the final retriever should return per query.

    Returns:
        A `BaseRetriever`. Depending on config this is a plain
        `SemanticRetriever`, a `HybridRetriever` (semantic + BM25 via
        RRF), or either of those wrapped in a `CrossEncoderReranker`.
    """
    semantic = SemanticRetriever(vectorstore, k=k)

    if config.use_hybrid_retrieval:
        logger.info("Hybrid retrieval enabled: semantic + BM25 (RRF fusion)")
        bm25 = BM25Retriever(vectorstore, k=k)
        base_retriever: BaseRetriever = HybridRetriever(semantic, bm25, k=k)
    else:
        logger.info("Hybrid retrieval disabled: using semantic-only retrieval")
        base_retriever = semantic

    if config.use_reranking:
        logger.info(
            "Cross-encoder re-ranking enabled: model=%s, candidate_k=%d, top_k=%d",
            config.reranker_model,
            config.rerank_candidate_k,
            config.rerank_k,
        )
        return CrossEncoderReranker(base_retriever, top_k=k)

    return base_retriever
