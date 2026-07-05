"""Deterministic evaluation metrics.

Single responsibility: turn `EvalRecord`s (one per question, produced by
`src.evaluation.evaluator.run_eval_set`) into a results DataFrame and an
averaged summary. No LLM judge, no external evaluation framework --
every metric here is computed directly from the real pipeline's own
output (timings, chunk counts, retrieved pages, citations), so nothing
here can itself throw a parser error, a timeout, or a NaN.

Metrics computed (see `evaluator.py` for exactly how each is captured):
    Retrieval:   retrieval latency, chunks retrieved, chunks reranked,
                 retrieved pages, expected pages, page recall
    Generation:  generation latency, workflow latency, answer length,
                 retries, query-rewritten flag, low-confidence flag
    Citations:   citation count, citation pages, duplicates removed

Page Recall (per question) = |retrieved_pages ∩ expected_pages| / |expected_pages|,
only computed for questions that set `expected_pages`; `None` otherwise.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from typing_extensions import TypedDict

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class EvalRecord(TypedDict, total=False):
    """One row: a question run through the real, compiled pipeline."""

    question: str
    answer: str
    ground_truth: Optional[str]

    # Retrieval
    retrieved_pages: List[Optional[int]]
    expected_pages: Optional[List[int]]
    page_recall: Optional[float]
    num_retrieved_chunks: int
    num_reranked_chunks: int
    retrieval_latency_ms: float

    # Generation / workflow
    generation_latency_ms: float
    workflow_latency_ms: float
    answer_length: int
    retries: int
    query_rewritten: bool
    low_confidence: bool

    # Citations
    sources: List[Dict[str, Any]]
    num_citations: int
    citation_pages: List[Optional[int]]
    duplicate_citations_removed: int


def compute_page_recall(retrieved_pages: List[Optional[int]], expected_pages: Optional[List[int]]) -> Optional[float]:
    """Page Recall = |retrieved ∩ expected| / |expected|, for one question.

    Args:
        retrieved_pages: Page numbers of the chunks retrieval actually
            returned for this question (duplicates/None allowed; both
            are harmless since this only checks set membership).
        expected_pages: The question's expected page numbers, if any.

    Returns:
        A float in [0, 1], or `None` if `expected_pages` wasn't set for
        this question (not "0.0" -- a question with no expected pages
        specified was never being checked in the first place).
    """
    if not expected_pages:
        return None
    expected = set(expected_pages)
    retrieved = {p for p in retrieved_pages if p is not None}
    hits = len(expected & retrieved)
    return hits / len(expected)


def build_eval_record(
    question: str,
    answer: str,
    retrieved_pages: List[Optional[int]],
    expected_pages: Optional[List[int]],
    num_retrieved_chunks: int,
    num_reranked_chunks: int,
    retrieval_latency_ms: float,
    generation_latency_ms: float,
    workflow_latency_ms: float,
    retries: int,
    query_rewritten: bool,
    low_confidence: bool,
    sources: List[Dict[str, Any]],
    duplicate_citations_removed: int,
    ground_truth: Optional[str] = None,
) -> EvalRecord:
    """Convenience constructor so callers don't hand-roll the dict shape
    or recompute `page_recall`/`answer_length`/`num_citations`/`citation_pages`
    from the raw pieces every time."""
    return EvalRecord(
        question=question,
        answer=answer,
        ground_truth=ground_truth,
        retrieved_pages=retrieved_pages,
        expected_pages=expected_pages,
        page_recall=compute_page_recall(retrieved_pages, expected_pages),
        num_retrieved_chunks=num_retrieved_chunks,
        num_reranked_chunks=num_reranked_chunks,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        workflow_latency_ms=workflow_latency_ms,
        answer_length=len(answer or ""),
        retries=retries,
        query_rewritten=query_rewritten,
        low_confidence=low_confidence,
        sources=sources,
        num_citations=len(sources),
        citation_pages=[s.get("page") for s in sources],
        duplicate_citations_removed=duplicate_citations_removed,
    )


def records_to_dataframe(records: List[EvalRecord]) -> pd.DataFrame:
    """Build the CSV-ready results table, one row per question.

    Column names/order match what `evaluate.py` writes to
    `results/evaluation_results.csv`.
    """
    rows = [
        {
            "Question": r["question"],
            "Answer": r["answer"],
            "Retrieved Pages": r.get("retrieved_pages", []),
            "Expected Pages": r.get("expected_pages"),
            "Page Recall": r.get("page_recall"),
            "Retrieved Chunks": r.get("num_retrieved_chunks"),
            "Chunks After Reranking": r.get("num_reranked_chunks"),
            "Retrieval Latency (ms)": r.get("retrieval_latency_ms"),
            "Generation Latency (ms)": r.get("generation_latency_ms"),
            "Workflow Latency (ms)": r.get("workflow_latency_ms"),
            "Answer Length": r.get("answer_length"),
            "Retries": r.get("retries"),
            "Low Confidence": r.get("low_confidence"),
            "Query Rewritten": r.get("query_rewritten"),
            "Sources": "; ".join(
                f"{s.get('filename', 'unknown')} p.{s.get('page')}" for s in r.get("sources", [])
            ),
        }
        for r in records
    ]
    return pd.DataFrame(rows)


def summarize_records(records: List[EvalRecord]) -> Dict[str, Any]:
    """Average every numeric metric across `records` for the JSON summary.

    Rates (retry_rate, low_confidence_rate, query_rewrite_rate) are the
    fraction of questions where that flag was true/nonzero.
    `avg_page_recall` only averages over questions that set
    `expected_pages` (others contributed no signal, not a 0).
    """
    if not records:
        return {}

    def _avg(key: str) -> float:
        values = [r[key] for r in records if r.get(key) is not None]
        return sum(values) / len(values) if values else 0.0

    page_recall_values = [r["page_recall"] for r in records if r.get("page_recall") is not None]

    return {
        "num_questions": len(records),
        "avg_retrieval_latency_ms": _avg("retrieval_latency_ms"),
        "avg_generation_latency_ms": _avg("generation_latency_ms"),
        "avg_workflow_latency_ms": _avg("workflow_latency_ms"),
        "avg_retrieved_chunks": _avg("num_retrieved_chunks"),
        "avg_reranked_chunks": _avg("num_reranked_chunks"),
        "avg_answer_length": _avg("answer_length"),
        "avg_page_recall": (sum(page_recall_values) / len(page_recall_values)) if page_recall_values else None,
        "page_recall_scored_questions": len(page_recall_values),
        "retry_rate": sum(1 for r in records if r.get("retries", 0) > 0) / len(records),
        "query_rewrite_rate": sum(1 for r in records if r.get("query_rewritten")) / len(records),
        "low_confidence_rate": sum(1 for r in records if r.get("low_confidence")) / len(records),
        "avg_num_citations": _avg("num_citations"),
        "avg_duplicate_citations_removed": _avg("duplicate_citations_removed"),
    }
