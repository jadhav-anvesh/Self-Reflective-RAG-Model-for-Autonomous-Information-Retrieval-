"""
Centralized configuration for the RAG application.

Every tunable parameter used anywhere in the codebase should live here
instead of being hardcoded in individual modules. Values can be overridden
at runtime via environment variables, which makes it easy to run
experiments (different chunk sizes, embedding models, etc.) without
touching code -- see PHASE 5 (experiments) in the project README.

Usage:
    from config import config
    print(config.chunk_size)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (single source of truth for the folder layout)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
MANIFEST_PATH = DATA_DIR / "manifest.json"


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


@dataclass(frozen=True)
class Config:
    """Immutable application configuration.

    Every field can be overridden with an environment variable of the
    same name in upper snake case (e.g. RAG_CHUNK_SIZE=400).
    """

    # --- Ingestion / chunking -------------------------------------------------
    chunk_size: int = _env_int("RAG_CHUNK_SIZE", 400)
    chunk_overlap: int = _env_int("RAG_CHUNK_OVERLAP", 50)

    # --- Embeddings -------------------------------------------------------
    embedding_model: str = _env_str("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedding_device: str = _env_str("RAG_EMBEDDING_DEVICE", "cuda")

    # --- LLM ----------------------------------------------------------------
    llm_model: str = _env_str("RAG_LLM_MODEL", "llama3.2")
    llm_temperature: float = _env_float("RAG_LLM_TEMPERATURE", 0.0)

    # --- Vector store -------------------------------------------------------
    collection_name: str = _env_str("RAG_COLLECTION_NAME", "rag-chroma")
    persist_directory: str = str(VECTORSTORE_DIR)

    # --- Retrieval ------------------------------------------------------------
    retrieval_k: int = _env_int("RAG_RETRIEVAL_K", 4)

    # --- PHASE 2: Hybrid retrieval (semantic + BM25) -------------------------
    use_hybrid_retrieval: bool = _env_str("RAG_USE_HYBRID_RETRIEVAL", "true").lower() == "true"
    # How many candidates each of semantic/BM25 fetch before fusion. Should be
    # >= retrieval_k so fusion has something to actually rank/merge.
    hybrid_candidate_k: int = _env_int("RAG_HYBRID_CANDIDATE_K", 10)
    # Constant in the Reciprocal Rank Fusion formula: 1 / (rrf_k + rank).
    # 60 is the standard value used in the original RRF paper.
    rrf_k: int = _env_int("RAG_RRF_K", 60)

    # --- PHASE 2: Cross-encoder re-ranking (optional) -------------------------
    use_reranking: bool = _env_str("RAG_USE_RERANKING", "true").lower() == "true"
    reranker_model: str = _env_str("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Pipeline: retriever -> top `rerank_candidate_k` -> cross-encoder -> top `rerank_k` -> LLM
    rerank_candidate_k: int = _env_int("RAG_RERANK_CANDIDATE_K", 20)
    rerank_k: int = _env_int("RAG_RERANK_K", 5)

    # --- Workflow -------------------------------------------------------------
    # recursion_limit is a hard safety net, not the primary loop terminator --
    # see max_hallucination_retries / max_query_rewrites below, which are what
    # actually bound the self-reflection loops. Raised from 8 -> 20 because 8
    # left almost no headroom: a single hallucination-retry + query-rewrite
    # cycle already costs ~6-7 graph steps, so harder questions could hit the
    # old limit and crash with GraphRecursionError before the bounded retry
    # logic even got a chance to terminate the loop gracefully.
    recursion_limit: int = _env_int("RAG_RECURSION_LIMIT", 20)
    # Max times `generate` is allowed to re-run because the hallucination
    # grader marked the previous generation as not grounded in the retrieved
    # documents. After this many attempts, the loop accepts the last
    # generation instead of retrying forever (see grade_generation_v_documents_and_question).
    max_hallucination_retries: int = _env_int("RAG_MAX_HALLUCINATION_RETRIES", 5)
    # Max times `transform_query` is allowed to re-run, whether triggered by
    # "no relevant documents" or "generation didn't address the question".
    max_query_rewrites: int = _env_int("RAG_MAX_QUERY_REWRITES", 5)

    # --- Ingestion bookkeeping ------------------------------------------------
    manifest_path: str = str(MANIFEST_PATH)
    raw_data_dir: str = str(RAW_DATA_DIR)


config = Config()
