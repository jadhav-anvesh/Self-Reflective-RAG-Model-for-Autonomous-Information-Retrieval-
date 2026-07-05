"""Evaluation Dashboard (offline, deterministic -- no LLM judge).

Two ways to view results:
  1. "Saved results" -- loads `results/evaluation_results.csv` /
     `evaluation_summary.json`, written by `python evaluate.py`. Instant.
  2. "Run live evaluation" -- runs `data/eval/questions.json` (or a
     custom path) through the real retriever + compiled workflow, right
     from this page, using the exact same call `evaluate.py` makes.

Every metric here is measured directly from the pipeline's own output
(latency, chunk counts, retrieved pages, citations, retries) -- there is
no LLM-as-judge step, so nothing on this page can produce a parsing
failure, a timeout, or a NaN score.

This page only reads/displays evaluation results -- it never changes
retrieval, the LangGraph workflow, or the vector store.
Run with: `streamlit run app.py` (this page appears in the sidebar nav).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config import config
from src.evaluation.dataset_loader import load_eval_questions
from src.evaluation.evaluator import run_eval_set
from src.evaluation.metrics import records_to_dataframe, summarize_records
from src.indexing.vectorstore import VectorStoreNotFoundError, load_vectorstore
from src.retrieval.retriever import get_retriever
from src.workflow.graph import create_workflow

st.set_page_config(page_title="Evaluation Dashboard", layout="wide")
st.title("📈 Evaluation Dashboard")
st.caption(
    "Deterministic engineering metrics -- latency, chunk counts, page recall, "
    "retries -- measured directly from the real retrieval + self-reflective "
    "generation pipeline. No LLM judge, so nothing here can time out or "
    "return NaN."
)

RESULTS_CSV = Path("results/evaluation_results.csv")
RESULTS_SUMMARY = Path("results/evaluation_summary.json")


def _render_results(results_df: pd.DataFrame, summary: dict) -> None:
    st.subheader("Average metrics")
    card_cols = st.columns(3)
    card_cols[0].metric("Avg Retrieval Latency", f"{summary.get('avg_retrieval_latency_ms', 0):.1f} ms")
    card_cols[1].metric("Avg Generation Latency", f"{summary.get('avg_generation_latency_ms', 0):.1f} ms")
    card_cols[2].metric("Avg Workflow Latency", f"{summary.get('avg_workflow_latency_ms', 0):.1f} ms")

    card_cols2 = st.columns(3)
    avg_page_recall = summary.get("avg_page_recall")
    card_cols2[0].metric("Avg Page Recall", f"{avg_page_recall:.3f}" if avg_page_recall is not None else "N/A")
    card_cols2[1].metric("Retry Rate", f"{summary.get('retry_rate', 0):.1%}")
    card_cols2[2].metric("Low Confidence Rate", f"{summary.get('low_confidence_rate', 0):.1%}")

    st.subheader("Charts")
    chart_cols = st.columns(2)
    if "Retrieval Latency (ms)" in results_df.columns:
        chart_cols[0].bar_chart(results_df["Retrieval Latency (ms)"])
        chart_cols[0].caption("Retrieval latency per question")
    if "Workflow Latency (ms)" in results_df.columns:
        chart_cols[1].bar_chart(results_df["Workflow Latency (ms)"])
        chart_cols[1].caption("Workflow latency per question")

    chart_cols2 = st.columns(2)
    if "Page Recall" in results_df.columns:
        chart_cols2[0].bar_chart(results_df["Page Recall"].dropna())
        chart_cols2[0].caption("Page recall per question (only questions with `expected_pages`)")
    if "Retrieved Chunks" in results_df.columns:
        chart_cols2[1].bar_chart(results_df["Retrieved Chunks"])
        chart_cols2[1].caption("Retrieved chunks per question")

    st.subheader("Per-question results")
    st.dataframe(results_df, use_container_width=True, hide_index=True)


with st.sidebar:
    st.header("Active pipeline")
    st.markdown(f"- **Hybrid retrieval:** `{config.use_hybrid_retrieval}`")
    st.markdown(f"- **Cross-encoder re-ranking:** `{config.use_reranking}`")
    st.markdown(f"- **LLM:** `{config.llm_model}`")

tab_saved, tab_live = st.tabs(["Saved results", "Run live evaluation"])

with tab_saved:
    if RESULTS_CSV.exists() and RESULTS_SUMMARY.exists():
        saved_df = pd.read_csv(RESULTS_CSV)
        saved_summary = json.loads(RESULTS_SUMMARY.read_text())
        st.caption(f"Loaded from {RESULTS_CSV} (run `python evaluate.py` to refresh).")
        _render_results(saved_df, saved_summary.get("metrics", saved_summary))
    else:
        st.info(
            "No saved results yet. Run `python evaluate.py` first, or use the "
            "'Run live evaluation' tab."
        )

with tab_live:
    questions_path = st.text_input("Questions file", value="data/eval/questions.json")

    if st.button("Run evaluation now"):
        try:
            vectorstore = load_vectorstore()
        except VectorStoreNotFoundError as e:
            st.error(str(e))
            st.stop()

        try:
            questions = load_eval_questions(questions_path)
        except (FileNotFoundError, ValueError) as e:
            st.error(str(e))
            st.stop()

        retriever = get_retriever(vectorstore)
        app = create_workflow(retriever)

        with st.spinner(f"Running {len(questions)} question(s) through the full pipeline..."):
            records = run_eval_set(app, retriever, questions)
            if not records:
                st.error("No questions completed successfully; nothing to report.")
                st.stop()
            live_df = records_to_dataframe(records)
            live_summary = summarize_records(records)

        _render_results(live_df, live_summary)
