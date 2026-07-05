"""Tests for src.evaluation.

Deterministic evaluation -- no LLM judge, no RAGAS, no external service --
so every test here runs with only the project's own lightweight modules
plus a fake compiled workflow/retriever, no network and no heavy deps.
"""
import json
import os
import tempfile

import pytest
from langchain_core.documents import Document

from src.evaluation.dataset_loader import load_eval_questions
from src.evaluation.evaluator import run_eval_set
from src.evaluation.metrics import (
    build_eval_record,
    compute_page_recall,
    records_to_dataframe,
    summarize_records,
)
from src.retrieval.base import RetrievedChunk


# ---------------------------------------------------------------------------
# dataset_loader
# ---------------------------------------------------------------------------
def test_load_eval_questions_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_eval_questions("/tmp/definitely_missing_eval_questions.json")


def test_load_eval_questions_empty_file_raises():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([], f)
        path = f.name
    try:
        with pytest.raises(ValueError):
            load_eval_questions(path)
    finally:
        os.remove(path)


def test_load_eval_questions_missing_question_field_raises():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([{"ground_truth": "no question here"}], f)
        path = f.name
    try:
        with pytest.raises(ValueError):
            load_eval_questions(path)
    finally:
        os.remove(path)


def test_load_eval_questions_roundtrip_with_expected_pages():
    questions = [{"question": "What is X?", "ground_truth": "X is Y.", "expected_pages": [8]}]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(questions, f)
        path = f.name
    try:
        loaded = load_eval_questions(path)
        assert loaded == questions
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# metrics.compute_page_recall
# ---------------------------------------------------------------------------
def test_page_recall_none_when_no_expected_pages():
    assert compute_page_recall([1, 2, 3], None) is None
    assert compute_page_recall([1, 2, 3], []) is None


def test_page_recall_full_hit():
    assert compute_page_recall([8, 9, 10], [8]) == 1.0
    assert compute_page_recall([8, 9], [8, 9]) == 1.0


def test_page_recall_partial_hit():
    assert compute_page_recall([8, 3], [8, 10]) == 0.5


def test_page_recall_zero_hit():
    assert compute_page_recall([1, 2, 3], [99]) == 0.0


def test_page_recall_ignores_none_in_retrieved_pages():
    assert compute_page_recall([None, 8, None], [8]) == 1.0


# ---------------------------------------------------------------------------
# metrics.build_eval_record / records_to_dataframe / summarize_records
# ---------------------------------------------------------------------------
def test_build_eval_record_computes_derived_fields():
    record = build_eval_record(
        question="What is X?",
        answer="X is Y.",
        retrieved_pages=[8, 9],
        expected_pages=[8],
        num_retrieved_chunks=2,
        num_reranked_chunks=2,
        retrieval_latency_ms=120.0,
        generation_latency_ms=800.0,
        workflow_latency_ms=950.0,
        retries=1,
        query_rewritten=False,
        low_confidence=False,
        sources=[{"filename": "input.pdf", "page": 8}],
        duplicate_citations_removed=1,
    )
    assert record["page_recall"] == 1.0
    assert record["answer_length"] == len("X is Y.")
    assert record["num_citations"] == 1
    assert record["citation_pages"] == [8]


def test_records_to_dataframe_has_required_columns():
    record = build_eval_record(
        question="Q?",
        answer="A.",
        retrieved_pages=[8],
        expected_pages=[8],
        num_retrieved_chunks=1,
        num_reranked_chunks=0,
        retrieval_latency_ms=50.0,
        generation_latency_ms=500.0,
        workflow_latency_ms=550.0,
        retries=0,
        query_rewritten=False,
        low_confidence=False,
        sources=[{"filename": "input.pdf", "page": 8}],
        duplicate_citations_removed=0,
    )
    df = records_to_dataframe([record])
    expected_columns = {
        "Question", "Answer", "Retrieved Pages", "Expected Pages", "Page Recall",
        "Retrieved Chunks", "Chunks After Reranking", "Retrieval Latency (ms)",
        "Generation Latency (ms)", "Workflow Latency (ms)", "Answer Length",
        "Retries", "Low Confidence", "Query Rewritten", "Sources",
    }
    assert expected_columns.issubset(set(df.columns))
    assert df.loc[0, "Question"] == "Q?"
    assert df.loc[0, "Sources"] == "input.pdf p.8"


def test_summarize_records_averages_and_rates():
    records = [
        build_eval_record(
            question="Q1?", answer="A1", retrieved_pages=[8], expected_pages=[8],
            num_retrieved_chunks=4, num_reranked_chunks=4, retrieval_latency_ms=100.0,
            generation_latency_ms=400.0, workflow_latency_ms=500.0, retries=1,
            query_rewritten=True, low_confidence=False, sources=[], duplicate_citations_removed=0,
        ),
        build_eval_record(
            question="Q2?", answer="A2", retrieved_pages=[3], expected_pages=[8],
            num_retrieved_chunks=4, num_reranked_chunks=0, retrieval_latency_ms=200.0,
            generation_latency_ms=600.0, workflow_latency_ms=800.0, retries=0,
            query_rewritten=False, low_confidence=True, sources=[], duplicate_citations_removed=0,
        ),
    ]
    summary = summarize_records(records)
    assert summary["num_questions"] == 2
    assert summary["avg_retrieval_latency_ms"] == 150.0
    assert summary["avg_workflow_latency_ms"] == 650.0
    assert summary["avg_page_recall"] == 0.5  # (1.0 + 0.0) / 2
    assert summary["page_recall_scored_questions"] == 2
    assert summary["retry_rate"] == 0.5
    assert summary["query_rewrite_rate"] == 0.5
    assert summary["low_confidence_rate"] == 0.5


def test_summarize_records_empty_list_returns_empty_dict():
    assert summarize_records([]) == {}


# ---------------------------------------------------------------------------
# evaluator.run_eval_set -- drives a fake *compiled workflow* AND a fake
# *retriever with diagnostics*, mirroring how the real create_workflow()/
# get_retriever() are used. No LLM judge anywhere in this path.
# ---------------------------------------------------------------------------
class FakeCompiledApp:
    """Stands in for `create_workflow(retriever)`'s return value."""

    def __init__(self, documents, generation, sources=None, low_confidence=False,
                 generation_attempts=1, query_rewrite_attempts=0, raise_recursion_error_for=None):
        self._documents = documents
        self._generation = generation
        self._sources = sources or []
        self._low_confidence = low_confidence
        self._generation_attempts = generation_attempts
        self._query_rewrite_attempts = query_rewrite_attempts
        self._raise_for = raise_recursion_error_for or set()

    def invoke(self, inputs, _config):
        from langgraph.errors import GraphRecursionError

        question = inputs["question"]
        if question in self._raise_for:
            raise GraphRecursionError(f"recursion limit hit for {question!r}")
        return {
            "question": question,
            "documents": self._documents,
            "generation": self._generation,
            "sources": self._sources,
            "low_confidence": self._low_confidence,
            "generation_attempts": self._generation_attempts,
            "query_rewrite_attempts": self._query_rewrite_attempts,
        }


class FakeRetriever:
    """Stands in for `get_retriever(vectorstore)`'s return value."""

    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve_with_diagnostics(self, query, k=None):
        return self._chunks


def test_run_eval_set_produces_records_with_retrieval_and_workflow_metrics():
    retrieved_chunks = [
        RetrievedChunk(document=Document(page_content="c1", metadata={"page": 8}), final_rank=0, rerank_score=0.9),
        RetrievedChunk(document=Document(page_content="c2", metadata={"page": 9}), final_rank=1, rerank_score=0.7),
    ]
    workflow_documents = [Document(page_content="c1", metadata={"page": 8})]  # post-grading filter
    app = FakeCompiledApp(documents=workflow_documents, generation="the answer")
    retriever = FakeRetriever(retrieved_chunks)

    records = run_eval_set(app, retriever, [{"question": "What is X?", "expected_pages": [8]}])

    assert len(records) == 1
    r = records[0]
    assert r["num_retrieved_chunks"] == 2
    assert r["num_reranked_chunks"] == 2  # both chunks have a rerank_score
    assert r["retrieved_pages"] == [8, 9]
    assert r["page_recall"] == 1.0
    assert r["answer"] == "the answer"
    assert r["retrieval_latency_ms"] >= 0
    assert r["workflow_latency_ms"] >= 0
    assert r["generation_latency_ms"] >= 0


def test_run_eval_set_counts_retries_and_query_rewrite_flag():
    chunks = [RetrievedChunk(document=Document(page_content="c1", metadata={"page": 1}), final_rank=0)]
    app = FakeCompiledApp(
        documents=[c.document for c in chunks],
        generation="answer",
        generation_attempts=3,  # 2 hallucination retries
        query_rewrite_attempts=1,  # 1 query rewrite
    )
    retriever = FakeRetriever(chunks)

    records = run_eval_set(app, retriever, [{"question": "Hard question?"}])

    assert records[0]["retries"] == 3  # (3 - 1) + 1
    assert records[0]["query_rewritten"] is True


def test_run_eval_set_carries_low_confidence_flag():
    chunks = [RetrievedChunk(document=Document(page_content="c", metadata={"page": 1}), final_rank=0)]
    app = FakeCompiledApp(documents=[c.document for c in chunks], generation="best-effort", low_confidence=True)
    retriever = FakeRetriever(chunks)

    records = run_eval_set(app, retriever, [{"question": "Hard question?"}])
    assert records[0]["low_confidence"] is True


def test_run_eval_set_computes_duplicate_citations_removed():
    doc = Document(page_content="c", metadata={"page": 8})
    chunks = [RetrievedChunk(document=doc, final_rank=0)]
    # Two documents from the same page reach generation, but citations
    # dedup down to one source -- 1 duplicate removed.
    app = FakeCompiledApp(
        documents=[doc, doc],
        generation="answer",
        sources=[{"filename": "input.pdf", "page": 8}],
    )
    retriever = FakeRetriever(chunks)

    records = run_eval_set(app, retriever, [{"question": "Q?"}])
    assert records[0]["duplicate_citations_removed"] == 1
    assert records[0]["num_citations"] == 1


def test_run_eval_set_skips_question_on_recursion_error_without_aborting_run():
    chunks = [RetrievedChunk(document=Document(page_content="c", metadata={"page": 1}), final_rank=0)]
    app = FakeCompiledApp(
        documents=[c.document for c in chunks], generation="answer",
        raise_recursion_error_for={"Impossible question?"},
    )
    retriever = FakeRetriever(chunks)
    questions = [{"question": "Impossible question?"}, {"question": "Fine question?"}]

    records = run_eval_set(app, retriever, questions)

    assert len(records) == 1
    assert records[0]["question"] == "Fine question?"


def test_zero_ragas_imports_in_evaluation_module():
    """Regression guard: the evaluation layer must not import ragas/datasets
    anywhere -- this is meant to be a fully offline, deterministic layer."""
    import src.evaluation.dataset_loader as dataset_loader
    import src.evaluation.evaluator as evaluator
    import src.evaluation.metrics as metrics

    for module in (dataset_loader, evaluator, metrics):
        source = open(module.__file__, encoding="utf-8").read()
        assert "ragas" not in source.lower(), f"{module.__name__} still references ragas"
