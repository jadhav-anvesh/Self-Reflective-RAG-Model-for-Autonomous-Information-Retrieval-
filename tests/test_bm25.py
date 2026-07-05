"""Tests for src.retrieval.bm25.BM25Retriever."""
from langchain_core.documents import Document

from src.retrieval.bm25 import BM25Retriever


class FakeVectorStore:
    """Minimal stand-in for Chroma exposing only `.get(include=...)`."""

    def __init__(self, documents):
        self._documents = documents

    def get(self, include=None):
        return {
            "documents": [d.page_content for d in self._documents],
            "metadatas": [d.metadata for d in self._documents],
        }


def _corpus():
    return [
        Document(page_content="The cat sat on the mat", metadata={"document_id": "d1", "chunk_id": 0}),
        Document(page_content="Dogs are loyal pets", metadata={"document_id": "d1", "chunk_id": 1}),
        Document(
            page_content="Machine learning models require large datasets",
            metadata={"document_id": "d1", "chunk_id": 2},
        ),
    ]


def test_bm25_retrieves_lexically_relevant_document():
    retriever = BM25Retriever(FakeVectorStore(_corpus()), k=1)
    results = retriever.retrieve("cat mat")
    assert len(results) == 1
    assert "cat" in results[0].page_content.lower()


def test_bm25_respects_k_override():
    retriever = BM25Retriever(FakeVectorStore(_corpus()), k=1)
    results = retriever.retrieve("pets datasets models", k=2)
    assert len(results) == 2


def test_bm25_preserves_metadata():
    retriever = BM25Retriever(FakeVectorStore(_corpus()), k=1)
    results = retriever.retrieve("cat mat")
    assert results[0].metadata["document_id"] == "d1"
    assert results[0].metadata["chunk_id"] == 0


def test_bm25_returns_empty_list_for_empty_corpus():
    retriever = BM25Retriever(FakeVectorStore([]), k=3)
    assert retriever.retrieve("anything") == []


def test_bm25_excludes_zero_score_documents():
    retriever = BM25Retriever(FakeVectorStore(_corpus()), k=10)
    results = retriever.retrieve("xylophone quokka")
    assert results == []
