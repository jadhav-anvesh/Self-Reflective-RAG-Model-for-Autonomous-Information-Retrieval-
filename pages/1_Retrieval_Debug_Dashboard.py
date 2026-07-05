"""Retrieval debug dashboard (PHASE 2 observability).

Shows exactly what the retrieval pipeline did for a given question,
independent of the LLM generation step:

    question -> semantic candidates -> BM25 candidates
             -> RRF fusion -> [cross-encoder re-ranking] -> final chunks

Every returned chunk carries whichever per-stage scores its pipeline
computed (see `src/retrieval/base.py::RetrievedChunk`), so this page is
just a thin rendering layer over `retriever.retrieve_with_diagnostics()`.
Run with: `streamlit run app.py` (this page appears in the sidebar nav).
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from config import config
from src.indexing.vectorstore import VectorStoreNotFoundError, load_vectorstore
from src.retrieval.retriever import get_retriever

st.set_page_config(page_title="Retrieval Debug Dashboard", layout="wide")
st.title("🔍 Retrieval Debug Dashboard")
st.caption(
    "Inspect what the retriever actually returns for a question -- per-stage "
    "scores, ranking, and metadata -- without running the full self-reflective "
    "generation loop."
)


@st.cache_resource(show_spinner="Loading vector index and retriever...")
def load_retriever():
    vectorstore = load_vectorstore()
    return get_retriever(vectorstore)


try:
    retriever = load_retriever()
except VectorStoreNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("Active pipeline")
    st.markdown(f"- **Hybrid retrieval (semantic + BM25):** `{config.use_hybrid_retrieval}`")
    st.markdown(f"- **Cross-encoder re-ranking:** `{config.use_reranking}`")
    if config.use_reranking:
        st.markdown(f"  - model: `{config.reranker_model}`")
        st.markdown(f"  - candidates considered: `{config.rerank_candidate_k}`")
    st.markdown(f"- **Final top-k:** `{config.rerank_k if config.use_reranking else config.retrieval_k}`")

question = st.text_input(
    "Question to trace through retrieval:",
    value="what is the requirement for Bachelor in Business Administration",
)
top_k = st.slider("Number of final chunks to show", min_value=1, max_value=20, value=config.retrieval_k)

if st.button("Run retrieval") and question.strip():
    start = time.perf_counter()
    chunks = retriever.retrieve_with_diagnostics(question, k=top_k)
    latency_ms = (time.perf_counter() - start) * 1000

    st.metric("Retrieval latency", f"{latency_ms:.1f} ms")

    if not chunks:
        st.warning("No chunks retrieved for this question.")
    else:
        rows = []
        for chunk in chunks:
            rows.append(
                {
                    "rank": chunk.final_rank + 1,
                    "snippet": (chunk.document.page_content[:160] + "...")
                    if len(chunk.document.page_content) > 160
                    else chunk.document.page_content,
                    "filename": chunk.document.metadata.get("filename"),
                    "page": chunk.document.metadata.get("page"),
                    "section": chunk.document.metadata.get("section"),
                    "semantic_score": round(chunk.semantic_score, 4) if chunk.semantic_score is not None else None,
                    "bm25_score": round(chunk.bm25_score, 4) if chunk.bm25_score is not None else None,
                    "rrf_score": round(chunk.rrf_score, 4) if chunk.rrf_score is not None else None,
                    "rerank_score": round(chunk.rerank_score, 4) if chunk.rerank_score is not None else None,
                }
            )

        df = pd.DataFrame(rows)
        # Drop score columns that are entirely unused for the current
        # pipeline config, so the table isn't cluttered with all-None columns.
        df = df.dropna(axis=1, how="all")

        st.subheader(f"Top {len(chunks)} retrieved chunks")
        st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("Full chunk text"):
            for chunk in chunks:
                st.markdown(
                    f"**#{chunk.final_rank + 1} -- {chunk.document.metadata.get('filename')}, "
                    f"Page {chunk.document.metadata.get('page')}**"
                )
                st.text(chunk.document.page_content)
                st.divider()
