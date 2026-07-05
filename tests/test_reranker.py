"""Tests for src.retrieval.reranker.CrossEncoderReranker.

The real cross-encoder model is never loaded in tests -- `_load_cross_encoder`
is monkeypatched to return a tiny fake scorer, keeping these tests fast
and offline.
"""
from typing import List, Optional

from langchain_core.documents import Document

import src.retrieval.reranker as reranker_module
from src.retrieval.base import BaseRetriever
from src.retrieval.reranker import CrossEncoderReranker


class FakeRetriever(BaseRetriever):
    def __init__(self, docs: List[Document]):
        self._docs = docs

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        k = k or len(self._docs)
        return self._docs[:k]


class FakeCrossEncoder:
    """Scores pairs by how many words they share (higher = more relevant)."""

    def predict(self, pairs):
        scores = []
        for query, doc_text in pairs:
            query_words = set(query.lower().split())
            doc_words = set(doc_text.lower().split())
            scores.append(len(query_words & doc_words))
        return scores


def test_reranker_reorders_by_relevance(monkeypatch):
    monkeypatch.setattr(reranker_module, "_load_cross_encoder", lambda model_name: FakeCrossEncoder())

    docs = [
        Document(page_content="completely unrelated content about weather"),
        Document(page_content="a passing mention of python"),
        Document(page_content="python programming python programming tutorial"),
    ]
    base = FakeRetriever(docs)
    reranker = CrossEncoderReranker(base, candidate_k=3, top_k=2)

    results = reranker.retrieve("python programming")
    assert len(results) == 2
    assert results[0].page_content == "python programming python programming tutorial"


def test_reranker_respects_k_override(monkeypatch):
    monkeypatch.setattr(reranker_module, "_load_cross_encoder", lambda model_name: FakeCrossEncoder())

    docs = [Document(page_content=f"python doc {i}") for i in range(5)]
    base = FakeRetriever(docs)
    reranker = CrossEncoderReranker(base, candidate_k=5, top_k=2)

    results = reranker.retrieve("python", k=4)
    assert len(results) == 4


def test_reranker_returns_empty_list_when_base_has_no_candidates(monkeypatch):
    monkeypatch.setattr(reranker_module, "_load_cross_encoder", lambda model_name: FakeCrossEncoder())

    base = FakeRetriever([])
    reranker = CrossEncoderReranker(base, candidate_k=5, top_k=2)
    assert reranker.retrieve("anything") == []
