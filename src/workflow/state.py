"""Shared state definition for the LangGraph workflow."""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from typing_extensions import NotRequired, TypedDict

from src.workflow.citations import Citation


class GraphState(TypedDict):
    """State passed between nodes in the self-reflective RAG graph.

    Attributes:
        question: The current user question (may be rewritten mid-graph).
        generation: The LLM's generated answer.
        documents: Retrieved (and possibly filtered) documents.
        sources: De-duplicated (filename, page, section) citations for
            the documents actually used to produce `generation` -- see
            `src/workflow/citations.py`. Populated by the `generate` node.
        generation_attempts: Number of times `generate` has run for this
            question. Used to bound the "not supported" (hallucination)
            retry loop so it terminates instead of looping until the
            graph's global recursion_limit is hit. Not present on the
            first call, hence `NotRequired` -- nodes read it via
            `state.get("generation_attempts", 0)`.
        query_rewrite_attempts: Number of times `transform_query` has
            run for this question. Used to bound both the "no relevant
            documents" and "not useful" retry loops. Also `NotRequired`
            for the same reason.
        low_confidence: True only when a retry cap was hit and the
            workflow gave up gracefully rather than because grading
            actually passed. Set by `accept_best_effort` -- see its
            docstring in `nodes.py`. `NotRequired`/absent means grading
            passed normally.
    """

    question: str
    generation: str
    documents: List[Document]
    sources: List[Citation]
    generation_attempts: NotRequired[int]
    query_rewrite_attempts: NotRequired[int]
    low_confidence: NotRequired[bool]
