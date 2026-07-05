"""Tests for src.retrieval.retriever.get_retriever (the composition factory).

`config` is a frozen dataclass singleton, so tests temporarily flip its
fields with `object.__setattr__` and restore the original values in a
fixture teardown -- this is the standard, safe way to mutate a frozen
dataclass instance for a test without a full module reload.
"""
import pytest
from langchain_core.documents import Document

import src.retrieval.reranker as reranker_module
from config import config
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.retriever import get_retriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.semantic import SemanticRetriever


class FakeVectorStore:
    """Minimal stand-in for Chroma satisfying both semantic and BM25 paths."""

    def __init__(self, documents):
        self._documents = documents

    def similarity_search(self, query, k=4):
        return self._documents[:k]

    def similarity_search_with_relevance_scores(self, query, k=4):
        return [(doc, 1.0 - 0.1 * i) for i, doc in enumerate(self._documents[:k])]

    def get(self, include=None):
        return {
            "documents": [d.page_content for d in self._documents],
            "metadatas": [d.metadata for d in self._documents],
        }


@pytest.fixture
def fake_vectorstore():
    docs = [
        Document(page_content="python programming basics", metadata={"document_id": "d1", "chunk_id": 0}),
        Document(page_content="deep learning with pytorch", metadata={"document_id": "d1", "chunk_id": 1}),
    ]
    return FakeVectorStore(docs)


@pytest.fixture
def restore_config():
    """Snapshot and restore the mutable-but-frozen config fields used here."""
    original = {
        "use_hybrid_retrieval": config.use_hybrid_retrieval,
        "use_reranking": config.use_reranking,
    }
    yield
    for key, value in original.items():
        object.__setattr__(config, key, value)


def test_get_retriever_returns_semantic_only_when_hybrid_disabled(fake_vectorstore, restore_config):
    object.__setattr__(config, "use_hybrid_retrieval", False)
    object.__setattr__(config, "use_reranking", False)

    retriever = get_retriever(fake_vectorstore, k=2)
    assert isinstance(retriever, SemanticRetriever)


def test_get_retriever_returns_hybrid_when_enabled(fake_vectorstore, restore_config):
    object.__setattr__(config, "use_hybrid_retrieval", True)
    object.__setattr__(config, "use_reranking", False)

    retriever = get_retriever(fake_vectorstore, k=2)
    assert isinstance(retriever, HybridRetriever)


def test_get_retriever_wraps_with_reranker_when_enabled(monkeypatch, fake_vectorstore, restore_config):
    class FakeCrossEncoder:
        def predict(self, pairs):
            return [1.0 for _ in pairs]

    monkeypatch.setattr(reranker_module, "_load_cross_encoder", lambda model_name: FakeCrossEncoder())

    object.__setattr__(config, "use_hybrid_retrieval", False)
    object.__setattr__(config, "use_reranking", True)

    retriever = get_retriever(fake_vectorstore, k=2)
    assert isinstance(retriever, CrossEncoderReranker)

    results = retriever.retrieve("python")
    assert len(results) <= 2


def test_get_retriever_end_to_end_returns_documents(fake_vectorstore, restore_config):
    object.__setattr__(config, "use_hybrid_retrieval", True)
    object.__setattr__(config, "use_reranking", False)

    retriever = get_retriever(fake_vectorstore, k=2)
    results = retriever.retrieve("python programming")
    assert len(results) > 0
    assert all(isinstance(d, Document) for d in results)
