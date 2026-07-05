"""Evaluate the RAG pipeline's retrieval and generation quality.

Usage:
    python evaluate.py
    python evaluate.py --questions data/eval/questions.json --out results/evaluation_results.csv

Requires an existing vector store (`python ingest.py` first).

This is a deterministic, offline, INDEPENDENT evaluation layer -- it
does not modify or replace the retrieval pipeline, the LangGraph
workflow, or the Streamlit app. It builds the exact same retriever and
compiled workflow those entry points use (`get_retriever` +
`create_workflow`) and runs each question through them, measuring
retrieval/generation/workflow latency, chunk/citation counts, retries,
and page recall directly from the pipeline's own output.

No LLM is used as a judge -- every metric here is computed from timings
and counts, not from a model's opinion of the answer. That means this
never produces a NaN, a JSON-parsing failure, or a timeout of its own;
see the README's "Evaluation" section for why this fits an offline,
local-Ollama project better than an LLM-judged framework.

Runs entirely against the local Ollama LLM + local embedding model
already configured in `config.py` -- no external API key needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import config
from src.evaluation.dataset_loader import load_eval_questions
from src.evaluation.evaluator import run_eval_set
from src.evaluation.metrics import records_to_dataframe, summarize_records
from src.indexing.vectorstore import VectorStoreNotFoundError, load_vectorstore
from src.retrieval.retriever import get_retriever
from src.utils.logging_config import get_logger
from src.workflow.graph import create_workflow

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with deterministic engineering metrics.")
    parser.add_argument(
        "--questions",
        default="data/eval/questions.json",
        help="Path to a JSON file of {question, ground_truth, expected_pages} objects "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        default="results/evaluation_results.csv",
        help="Where to write the per-question results CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--summary-out",
        default="results/evaluation_summary.json",
        help="Where to write the aggregate summary JSON (default: %(default)s).",
    )
    return parser.parse_args()


def print_per_question_breakdown(records) -> None:
    """Print each question's metrics in a simple, readable block format."""
    print()
    for i, r in enumerate(records):
        print(f"Question {i + 1}: {r['question']}")
        print(f"  Retrieval Latency   {r['retrieval_latency_ms']:.1f} ms")
        print(f"  Generation Latency  {r['generation_latency_ms']:.1f} ms")
        print(f"  Workflow Latency    {r['workflow_latency_ms']:.1f} ms")
        print(f"  Retrieved Chunks    {r['num_retrieved_chunks']} ({r['num_reranked_chunks']} reranked)")
        if r.get("page_recall") is not None:
            print(f"  Page Recall         {r['page_recall']:.3f}")
        print(f"  Retries             {r['retries']} (query rewritten: {r['query_rewritten']})")
        print(f"  Low Confidence      {r['low_confidence']}")
        print(f"  Citations           {r['num_citations']}")
        print("-" * 40)


def main() -> None:
    args = parse_args()
    logger.info("=== Evaluation started (questions=%s) ===", args.questions)

    try:
        vectorstore = load_vectorstore()
    except VectorStoreNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    try:
        questions = load_eval_questions(args.questions)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    # Build the exact same retriever + compiled workflow the real app uses --
    # the real pipeline (hybrid retrieval -> [cross-encoder] -> LangGraph ->
    # LLM), not a second, simplified one.
    retriever = get_retriever(vectorstore)
    app = create_workflow(retriever)

    logger.info("Running %d question(s) through the real pipeline...", len(questions))
    records = run_eval_set(app, retriever, questions)
    if not records:
        logger.error("No questions completed successfully; nothing to report.")
        sys.exit(1)

    print_per_question_breakdown(records)

    results_df = records_to_dataframe(records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    logger.info("Wrote per-question results to %s", out_path)

    summary_metrics = summarize_records(records)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "questions_file": args.questions,
        "num_questions_requested": len(questions),
        "num_questions_scored": len(records),
        "metrics": summary_metrics,
        "config": {
            "llm_model": config.llm_model,
            "embedding_model": config.embedding_model,
            "use_hybrid_retrieval": config.use_hybrid_retrieval,
            "use_reranking": config.use_reranking,
        },
    }

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote aggregate summary to %s", summary_path)

    print("=== Average metrics (across all scored questions) ===")
    print(f"Avg Retrieval Latency   {summary_metrics['avg_retrieval_latency_ms']:.1f} ms")
    print(f"Avg Generation Latency  {summary_metrics['avg_generation_latency_ms']:.1f} ms")
    print(f"Avg Workflow Latency    {summary_metrics['avg_workflow_latency_ms']:.1f} ms")
    if summary_metrics.get("avg_page_recall") is not None:
        print(
            f"Avg Page Recall         {summary_metrics['avg_page_recall']:.3f} "
            f"(over {summary_metrics['page_recall_scored_questions']} question(s) with `expected_pages` set)"
        )
    print(f"Retry Rate              {summary_metrics['retry_rate']:.1%}")
    print(f"Low Confidence Rate     {summary_metrics['low_confidence_rate']:.1%}")
    print(f"\nPer-question results : {out_path}")
    print(f"Summary               : {summary_path}")


if __name__ == "__main__":
    main()
