"""Node functions for the self-reflective RAG LangGraph workflow.

Each function implements exactly one step of the graph. Control-flow
(which node runs next) lives in `graph.py`, not here.

Root-cause context (see the bug investigation this file was patched
for): local/small LLMs served via Ollama don't always return the exact
'yes'/'no' the grading prompts ask for, even with structured output --
they occasionally emit synonyms, different casing, or trailing
punctuation. Combined with the fact that the self-reflection loops
("not supported" -> generate, "not useful"/"no relevant docs" ->
transform_query) previously had NO bounded retry counter of their own,
a single stubborn grading result on a harder question could loop
(deterministically, since temperature=0) until LangGraph's global
recursion_limit was hit, raising GraphRecursionError. `app.py` was
swallowing that in a bare `except Exception`, which is why the UI only
ever showed a generic fallback message instead of the real cause.

This file fixes both halves: `_normalize_binary_score` makes routing
robust to label variation, and `generation_attempts` /
`query_rewrite_attempts` bound each retry loop so it always terminates
on its own, with the global recursion_limit left as a safety net rather
than the thing actually stopping the loop.
"""
from __future__ import annotations

import time

from langchain_core.exceptions import OutputParserException

from config import config
from src.utils.logging_config import get_logger
from src.workflow.citations import format_sources
from src.workflow.prompts import (
    answer_grader,
    hallucination_grader,
    question_rewriter,
    rag_chain,
    retrieval_grader,
)
from src.workflow.state import GraphState

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Grader-output normalization
# ---------------------------------------------------------------------------
# Prompts in prompts.py explicitly ask for 'yes'/'no', but small/local models
# don't always comply. These sets cover the variations actually seen in
# practice; anything outside them is logged loudly and treated as a fail-safe
# negative rather than raising or silently mis-routing.
_POSITIVE_LABELS = {"yes", "y", "true", "relevant", "useful", "grounded", "supported", "1"}
_NEGATIVE_LABELS = {
    "no", "n", "false", "irrelevant", "not relevant", "not useful",
    "not supported", "not grounded", "ungrounded", "0",
}


def _normalize_binary_score(raw_score: str, *, context: str) -> bool:
    """Normalize a grader's binary_score field into True/False.

    Args:
        raw_score: The raw string returned by a grader (e.g. "Yes.", "YES",
            "relevant").
        context: Short label identifying which grader this came from, used
            only for logging when normalization falls through to the
            fail-safe default.

    Returns:
        True if the score means "positive" (relevant/grounded/useful),
        False otherwise -- including for unrecognized values, since
        treating an unparseable grade as a failure is safer than silently
        treating it as a pass.
    """
    cleaned = (raw_score or "").strip().lower().strip(" .!\"'")
    if cleaned in _POSITIVE_LABELS:
        return True
    if cleaned in _NEGATIVE_LABELS:
        return False
    logger.warning(
        "Unexpected grader output %r during %s (not in known yes/no synonyms); "
        "treating as a fail-safe negative so routing doesn't crash.",
        raw_score,
        context,
    )
    return False


def _invoke_grader_safely(chain, inputs: dict, *, context: str):
    """Invoke a structured-output grader chain, converting a parsing failure
    into a clearly-logged `None` instead of an uncaught crash.

    Root cause this guards against (see prompts.py for the full writeup):
    with `method="json_schema"` a grader chain raises `OutputParserException`
    -- rather than silently returning `None`, which is what the *previous*
    default (`method="function_calling"`, on the pinned
    `langchain-ollama<0.3`) did whenever llama3.2 responded with plain text
    instead of actually invoking the bound grading tool. That silent `None`
    was the direct cause of `AttributeError: 'NoneType' object has no
    attribute 'binary_score'`.

    `method="json_schema"` fixes the silent half of that bug, but a raised
    exception with nothing to catch it would just crash the graph instead --
    trading a silent data-corruption bug for an availability bug. This is
    the catch point: log the *actual* parsing failure (previously
    invisible), and return `None` so callers can route to their existing
    fail-safe-negative handling, the same place an unrecognized label
    already goes via `_normalize_binary_score`.

    Args:
        chain: A `prompt | llm.with_structured_output(...)` Runnable.
        inputs: The dict passed to `chain.invoke(...)`.
        context: Short label for logging (e.g. "hallucination grading").

    Returns:
        The parsed Pydantic object, or `None` if structured-output parsing
        failed for any reason.
    """
    try:
        return chain.invoke(inputs)
    except OutputParserException as e:
        logger.error(
            "Structured-output parsing failed during %s -- the model likely "
            "didn't produce schema-conforming output for this input. "
            "Treating as a fail-safe negative grade rather than crashing. "
            "Underlying error: %s",
            context,
            e,
        )
        return None


def retrieve(state: GraphState, retriever) -> dict:
    """Retrieve documents relevant to the current question.

    Args:
        state: Current graph state.
        retriever: A `BaseRetriever` (bound via closure in `graph.py`).
            Any implementation -- semantic, hybrid, reranked -- works
            here unchanged as long as it exposes `.retrieve(query)`.
    """
    logger.info("---RETRIEVE---")
    question = state["question"]
    documents = retriever.retrieve(question)
    logger.info("Retrieved %d documents for question=%r", len(documents), question)
    return {"documents": documents, "question": question}


def generate(state: GraphState) -> dict:
    """Generate an answer from the retrieved documents, with source citations."""
    logger.info("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    attempt = state.get("generation_attempts", 0) + 1

    formatted_docs = "\n\n".join(doc.page_content for doc in documents)
    logger.info(
        "Generation attempt %d/%d | question=%r | context_length=%d chars from %d document(s)",
        attempt,
        config.max_hallucination_retries + 1,
        question,
        len(formatted_docs),
        len(documents),
    )
    generation = rag_chain.invoke({"context": formatted_docs, "question": question})
    logger.info("Generated answer (%d chars): %.200r", len(generation), generation)
    sources = format_sources(documents)
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "sources": sources,
        "generation_attempts": attempt,
    }


def grade_documents(state: GraphState) -> dict:
    """Filter retrieved documents down to the ones relevant to the question."""
    logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    for d in documents:
        score = _invoke_grader_safely(
            retrieval_grader, {"question": question, "document": d.page_content}, context="document relevance grading"
        )
        if score is None:
            # Fail-safe: treat an unparseable grade as "not relevant" and drop
            # the chunk, same posture as an unrecognized label in
            # _normalize_binary_score -- a chunk we can't confirm is relevant
            # shouldn't be passed to generation.
            continue
        is_relevant = _normalize_binary_score(score.binary_score, context="document relevance grading")
        logger.info(
            "Document grade = %r (relevant=%s) | page=%s source=%s",
            score.binary_score,
            is_relevant,
            d.metadata.get("page"),
            d.metadata.get("source"),
        )
        if is_relevant:
            filtered_docs.append(d)
    logger.info("Kept %d/%d documents after relevance grading", len(filtered_docs), len(documents))
    return {"documents": filtered_docs, "question": question}


def transform_query(state: GraphState) -> dict:
    """Rewrite the question to be better suited for vector retrieval."""
    logger.info("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    attempt = state.get("query_rewrite_attempts", 0) + 1

    better_question = question_rewriter.invoke({"question": question})
    logger.info(
        "Query rewrite attempt %d/%d | %r -> %r",
        attempt,
        config.max_query_rewrites,
        question,
        better_question,
    )
    return {"documents": documents, "question": better_question, "query_rewrite_attempts": attempt}


def accept_best_effort(state: GraphState) -> dict:
    """Terminal node for when a retry cap was hit before grading actually passed.

    Reached only via the "retry_exhausted" routing label -- i.e. the
    hallucination or answer grader kept failing and the configured retry
    budget (`config.max_hallucination_retries` / `config.max_query_rewrites`)
    ran out. Rather than silently relabeling that outcome as "useful" (which
    would misrepresent a failed grading check as a pass), this node keeps
    the last generation and sources untouched and sets `low_confidence=True`
    so callers (e.g. `app.py`) can surface an honest caveat to the user
    instead of presenting an unverified answer with full confidence.
    """
    logger.warning(
        "Giving up gracefully for question=%r: retry budget exhausted "
        "(generation_attempts=%d, query_rewrite_attempts=%d) without grading passing. "
        "Returning the last generation with low_confidence=True.",
        state["question"],
        state.get("generation_attempts", 0),
        state.get("query_rewrite_attempts", 0),
    )
    return {"low_confidence": True}


def decide_to_generate(state: GraphState) -> str:
    """Route: generate an answer, or rewrite the query and retry retrieval."""
    logger.info("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    rewrite_attempts = state.get("query_rewrite_attempts", 0)

    if not filtered_documents:
        if rewrite_attempts >= config.max_query_rewrites:
            logger.warning(
                "No relevant documents after %d query rewrite(s) (limit=%d); "
                "generating a best-effort answer instead of rewriting again.",
                rewrite_attempts,
                config.max_query_rewrites,
            )
            decision = "generate"
        else:
            logger.info("Decision = transform_query (all retrieved documents graded not relevant)")
            decision = "transform_query"
    else:
        logger.info("Decision = generate (%d relevant document(s) found)", len(filtered_documents))
        decision = "generate"
    return decision


def grade_generation_v_documents_and_question(state: GraphState) -> str:
    """Route: check the generation is grounded and actually answers the question."""
    logger.info("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    generation_attempts = state.get("generation_attempts", 0)
    rewrite_attempts = state.get("query_rewrite_attempts", 0)

    formatted_docs = "\n\n".join(doc.page_content for doc in documents)
    logger.info(
        "Calling hallucination grader | context_length=%d chars | generation_length=%d chars",
        len(formatted_docs),
        len(generation),
    )
    _hallucination_call_start = time.perf_counter()
    score = _invoke_grader_safely(
        hallucination_grader,
        {"documents": formatted_docs, "generation": generation},
        context="hallucination grading",
    )
    logger.info(
        "Hallucination grader returned in %.1fs: %r",
        time.perf_counter() - _hallucination_call_start,
        score,
    )
    if score is None:
        logger.error(
            "Hallucination grader returned None.\n"
            "Question: %r\n"
            "Generation: %r\n"
            "Context length: %d chars",
            question,
            generation,
            len(formatted_docs),
        )
        return "retry_exhausted"
    is_grounded = _normalize_binary_score(score.binary_score, context="hallucination grading")
    logger.info(
        "Hallucination grade = %r (grounded=%s) | generation_attempts=%d/%d",
        score.binary_score,
        is_grounded,
        generation_attempts,
        config.max_hallucination_retries + 1,
    )

    if not is_grounded:
        if generation_attempts > config.max_hallucination_retries:
            logger.warning(
                "Hallucination-retry limit reached (%d/%d) for question=%r; "
                "routing to accept_best_effort instead of retrying forever.",
                generation_attempts,
                config.max_hallucination_retries,
                question,
            )
            decision = "retry_exhausted"
        else:
            logger.info("Decision = not supported (regenerate)")
            decision = "not supported"
        return decision

    logger.info("---GRADE GENERATION VS QUESTION---")
    answer_score = _invoke_grader_safely(
        answer_grader, {"question": question, "generation": generation}, context="answer-usefulness grading"
    )
    if answer_score is None:
        logger.warning(
            "Answer grader failed to parse for question=%r; treating as fail-safe "
            "'not useful' rather than crashing.",
            question,
        )
        is_useful = False
    else:
        is_useful = _normalize_binary_score(answer_score.binary_score, context="answer-usefulness grading")
        logger.info(
            "Answer-usefulness grade = %r (useful=%s) | query_rewrite_attempts=%d/%d",
            answer_score.binary_score,
            is_useful,
            rewrite_attempts,
            config.max_query_rewrites,
        )

    if is_useful:
        logger.info("Decision = useful (final answer)")
        return "useful"

    if rewrite_attempts >= config.max_query_rewrites:
        logger.warning(
            "Query-rewrite limit reached (%d/%d) for question=%r; "
            "routing to accept_best_effort instead of rewriting forever.",
            rewrite_attempts,
            config.max_query_rewrites,
            question,
        )
        return "retry_exhausted"

    logger.info("Decision = not useful (rewrite query and retry)")
    return "not useful"
