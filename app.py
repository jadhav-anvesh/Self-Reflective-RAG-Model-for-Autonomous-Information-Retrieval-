"""Streamlit UI for the self-reflective RAG workflow.

Production fix vs. the original project: this file no longer builds
embeddings or the vector store on every run. It only *loads* the
persistent vector store built by `ingest.py`. If no index exists yet,
the user is shown a clear instruction to run `python ingest.py`.
"""
from __future__ import annotations

import streamlit as st
from langgraph.errors import GraphRecursionError

from config import config
from src.indexing.vectorstore import VectorStoreNotFoundError, load_vectorstore
from src.retrieval.retriever import get_retriever
from src.utils.logging_config import get_logger
from src.workflow.graph import create_workflow

logger = get_logger(__name__)

st.set_page_config(page_title="RAG Q&A", layout="centered")

st.sidebar.title("Instructions")
st.sidebar.info(
    """
    - Enter a question related to your document.
    - The app uses RAG (Retrieval-Augmented Generation) with self-reflection
      to retrieve and generate an answer.
    - New documents? Run `python ingest.py` first, then reload this page.
    - Want to see *why* a chunk was retrieved (semantic/BM25/RRF/rerank
      scores)? Open **Retrieval Debug Dashboard** in the sidebar (Streamlit
      multipage nav) to inspect retrieval internals for any question.
    """
)
st.sidebar.markdown(
    """
    ---
    ### **Jadhav Anvesh**
    *23110142*
    ---
    """
)


@st.cache_resource(show_spinner="Loading vector index...")
def load_workflow():
    """Load the vector store once per session and compile the workflow.

    Cached so repeated Streamlit re-runs (e.g. on every keystroke) don't
    reload the embedding model or vector store from disk.
    """
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore)
    return create_workflow(retriever)


st.title("RAG Workflow Q&A")

try:
    rag_app = load_workflow()
except VectorStoreNotFoundError as e:
    st.error(str(e))
    st.stop()

user_question = st.text_input(
    "Your Question:", value="What font should assignments use?"
)

if st.button("Get Answer") and user_question.strip():
    inputs = {"question": user_question}
    # Accumulate every node's partial update into the full state, rather than
    # keeping only the last node's delta. This matters now that grading can
    # terminate via "generate" -> END *or* via the "accept_best_effort" node
    # (retry_exhausted -> accept_best_effort -> END): accept_best_effort only
    # returns {"low_confidence": True}, so keeping just the last delta would
    # lose the actual "generation"/"sources" set earlier in the run.
    accumulated_state = dict(inputs)

    try:
        spinner_placeholder = st.empty()
        spinner_placeholder.text("Processing...")

        iteration = 1
        with st.spinner(""):
            for output in rag_app.stream(inputs, {"recursion_limit": config.recursion_limit}):
                for key, node_output in output.items():
                    accumulated_state.update(node_output)
                    if key == "transform_query":
                        iteration += 1
                    spinner_placeholder.text(f"Iteration = {iteration}")

        spinner_placeholder.text("")

        if accumulated_state.get("generation"):
            if accumulated_state.get("low_confidence"):
                st.warning(
                    "This answer couldn't be fully verified against the retrieved documents "
                    "within the configured retry limit -- treat it as a best-effort answer."
                )
            st.success("Answer:")
            st.write(accumulated_state["generation"])

            sources = accumulated_state.get("sources") or []
            if sources:
                st.markdown("**Sources:**")
                for source in sources:
                    label = f"- {source['filename']}, Page {source['page']}"
                    if source.get("section"):
                        label += f" ({source['section']})"
                    st.markdown(label)
        else:
            logger.warning(
                "Workflow finished with no generation in the final state for question=%r", user_question
            )
            st.error("Sorry, I didn't understand your question. Do you want to connect with a live agent?")

    except GraphRecursionError:
        spinner_placeholder.text("")
        logger.error(
            "Workflow hit recursion_limit=%d before terminating for question=%r",
            config.recursion_limit,
            user_question,
        )
        st.error(
            "This question needed more self-reflection steps than the configured limit "
            f"(recursion_limit={config.recursion_limit}). Try rephrasing the question, or "
            "raise RAG_MAX_HALLUCINATION_RETRIES / RAG_MAX_QUERY_REWRITES / RAG_RECURSION_LIMIT."
        )
    except Exception:
        spinner_placeholder.text("")
        logger.exception("Unexpected error while answering question=%r", user_question)
        st.error("Sorry, I didn't understand your question. Do you want to connect with a live agent?")
