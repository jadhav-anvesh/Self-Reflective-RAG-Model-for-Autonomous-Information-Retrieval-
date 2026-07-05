"""Tests for src.retrieval.hybrid.HybridRetriever (Reciprocal Rank Fusion)."""
from typing import List, Optional

from langchain_core.documents import Document

from src.retrieval.base import BaseRetriever
from src.retrieval.hybrid import HybridRetriever


class FakeRetriever(BaseRetriever):
    """Returns a fixed, pre-ranked list of documents regardless of query."""

    def __init__(self, docs: List[Document]):
        self._docs = docs

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        k = k or len(self._docs)
        return self._docs[:k]


def _doc(text, document_id, chunk_id):
    return Document(page_content=text, metadata={"document_id": document_id, "chunk_id": chunk_id})


def test_hybrid_fuses_and_deduplicates_overlapping_results():
    shared = _doc("appears in both rankers", "d1", 0)
    semantic_only = _doc("only in semantic", "d1", 1)
    bm25_only = _doc("only in bm25", "d1", 2)

    semantic = FakeRetriever([shared, semantic_only])
    bm25 = FakeRetriever([shared, bm25_only])

    hybrid = HybridRetriever(semantic, bm25, k=3, candidate_k=2)
    results = hybrid.retrieve("query")

    # The document ranked #1 by both retrievers should get the highest
    # fused RRF score and come first; no duplicates should appear.
    keys = [(d.metadata["document_id"], d.metadata["chunk_id"]) for d in results]
    assert len(keys) == len(set(keys))
    assert results[0].page_content == "appears in both rankers"
    assert len(results) == 3


def test_hybrid_respects_final_k():
    docs = [_doc(f"doc {i}", "d1", i) for i in range(5)]
    semantic = FakeRetriever(docs)
    bm25 = FakeRetriever(list(reversed(docs)))

    hybrid = HybridRetriever(semantic, bm25, k=2, candidate_k=5)
    results = hybrid.retrieve("query")
    assert len(results) == 2


def test_hybrid_falls_back_to_single_ranker_when_other_is_empty():
    docs = [_doc(f"doc {i}", "d1", i) for i in range(3)]
    semantic = FakeRetriever(docs)
    bm25 = FakeRetriever([])

    hybrid = HybridRetriever(semantic, bm25, k=3, candidate_k=3)
    results = hybrid.retrieve("query")
    assert len(results) == 3
    assert results[0].page_content == "doc 0"
