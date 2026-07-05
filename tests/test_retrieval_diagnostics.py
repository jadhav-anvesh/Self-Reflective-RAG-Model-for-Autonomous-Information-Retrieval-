"""Tests for retrieve_with_diagnostics score propagation across the
retrieval pipeline: semantic score -> BM25 score -> RRF score -> rerank score.
"""
from langchain_core.documents import Document

import src.retrieval.reranker as reranker_module
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.semantic import SemanticRetriever


class FakeVectorStore:
    def __init__(self, documents):
        self._documents = documents

    def similarity_search(self, query, k=4):
        return self._documents[:k]

    def similarity_search_with_relevance_scores(self, query, k=4):
        return [(doc, round(1.0 - 0.1 * i, 2)) for i, doc in enumerate(self._documents[:k])]

    def get(self, include=None):
        return {
            "documents": [d.page_content for d in self._documents],
            "metadatas": [d.metadata for d in self._documents],
        }


def _corpus():
    return [
        Document(page_content="python programming tutorial", metadata={"document_id": "d1", "chunk_id": 0}),
        Document(page_content="deep learning with pytorch", metadata={"document_id": "d1", "chunk_id": 1}),
        Document(page_content="cooking recipes for beginners", metadata={"document_id": "d1", "chunk_id": 2}),
    ]


def test_semantic_retriever_diagnostics_populate_semantic_score():
    retriever = SemanticRetriever(FakeVectorStore(_corpus()), k=2)
    chunks = retriever.retrieve_with_diagnostics("python")
    assert len(chunks) == 2
    assert chunks[0].semantic_score is not None
    assert chunks[0].bm25_score is None
    assert chunks[0].final_rank == 0


def test_bm25_retriever_diagnostics_populate_bm25_score():
    class FakeVS:
        def get(self, include=None):
            docs = _corpus()
            return {"documents": [d.page_content for d in docs], "metadatas": [d.metadata for d in docs]}

    retriever = BM25Retriever(FakeVS(), k=2)
    chunks = retriever.retrieve_with_diagnostics("python programming")
    assert len(chunks) >= 1
    assert chunks[0].bm25_score is not None
    assert chunks[0].semantic_score is None


def test_hybrid_diagnostics_populate_semantic_bm25_and_rrf_scores():
    vs = FakeVectorStore(_corpus())
    semantic = SemanticRetriever(vs, k=2)
    bm25 = BM25Retriever(vs, k=2)
    hybrid = HybridRetriever(semantic, bm25, k=2, candidate_k=2)

    chunks = hybrid.retrieve_with_diagnostics("python programming")
    assert len(chunks) > 0
    top = chunks[0]
    assert top.rrf_score is not None
    # This corpus/query combination should surface signal from both stages
    # for at least one of the returned chunks.
    assert any(c.semantic_score is not None for c in chunks)
    assert any(c.bm25_score is not None for c in chunks)


def test_reranker_diagnostics_populate_rerank_score(monkeypatch):
    class FakeCrossEncoder:
        def predict(self, pairs):
            return [float(i) for i in range(len(pairs))]

    monkeypatch.setattr(reranker_module, "_load_cross_encoder", lambda model_name: FakeCrossEncoder())

    vs = FakeVectorStore(_corpus())
    semantic = SemanticRetriever(vs, k=2)
    reranker = CrossEncoderReranker(semantic, candidate_k=2, top_k=2)

    chunks = reranker.retrieve_with_diagnostics("python")
    assert len(chunks) == 2
    assert all(c.rerank_score is not None for c in chunks)
    # Highest predicted score (last pair, score=1.0) should be ranked first.
    assert chunks[0].rerank_score >= chunks[1].rerank_score
