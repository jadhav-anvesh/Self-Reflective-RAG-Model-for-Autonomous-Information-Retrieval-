"""LangGraph graph assembly for the self-reflective RAG workflow.

Wires the node functions from `nodes.py` into the retrieve -> grade ->
generate -> self-check control flow. This is the only place that knows
the graph's shape; node implementations stay agnostic of it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from src.retrieval.base import BaseRetriever

from src.workflow.nodes import (
    accept_best_effort,
    decide_to_generate,
    generate,
    grade_documents,
    grade_generation_v_documents_and_question,
    retrieve,
    transform_query,
)
from src.workflow.state import GraphState


def create_workflow(retriever: "BaseRetriever"):
    """Build and compile the self-reflective RAG LangGraph workflow.

    Args:
        retriever: A `BaseRetriever` (see `src/retrieval/base.py`) exposing
            `.retrieve(query)`. Swapping semantic for hybrid retrieval
            later requires no change here.

    Returns:
        A compiled LangGraph app with a `.stream(...)` / `.invoke(...)` API.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    # Terminal "give up gracefully" node: reached only when a retry budget
    # (config.max_hallucination_retries / config.max_query_rewrites) is
    # exhausted without grading actually passing. Keeps the self-reflection
    # loop's shape unchanged -- it only adds an honest way to end it.
    workflow.add_node("accept_best_effort", accept_best_effort)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
            "retry_exhausted": "accept_best_effort",
        },
    )
    workflow.add_edge("accept_best_effort", END)

    return workflow.compile()
