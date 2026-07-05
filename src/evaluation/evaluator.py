"""Evaluation orchestration.

Single responsibility: run a question set through the project's real
retriever and real, compiled self-reflective LangGraph workflow --
the same objects `app.py`/`main.py` use -- and package the results as
`EvalRecord`s. No LLM judge is used anywhere in this module; every
metric is derived directly from the pipeline's own output and timing.

Two calls per question, deliberately:
    1. `retriever.retrieve_with_diagnostics(question)` -- timed on its
       own so retrieval latency/chunk counts/reranked count/retrieved
       pages are measured independently of generation. This is the
       same diagnostics method the Retrieval Debug Dashboard already
       uses (see `src/retrieval/base.py`), so no retrieval code needed
       to change to support this.
    2. `app.invoke(...)` -- the full compiled workflow (hybrid retrieval
       -> [cross-encoder] -> self-reflective grade/rewrite/regenerate
       loop -> answer + citations), timed as "workflow latency".

`generation_latency_ms` is then estimated as
`workflow_latency_ms - retrieval_latency_ms` (floored at 0). This is an
approximation, not a re-instrumentation of the LangGraph nodes
themselves (which this project's constraints say not to touch): for a
question that triggers a query rewrite, the workflow re-runs retrieval
internally one or more additional times, so the single standalone
retrieval sample above isn't a perfect accounting of 100% of the
non-generation time in that run -- it's a representative single-pass
retrieval cost subtracted from the total, which is the right level of
precision for a benchmark/eval tool and avoids touching the workflow to
get an exact figure.
"""
from __future__ import annotations

import time
from typing import List

from langgraph.errors import GraphRecursionError

from config import config
from src.evaluation.dataset_loader import EvalQuestion
from src.evaluation.metrics import EvalRecord, build_eval_record
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def run_eval_set(app, retriever, questions: List[EvalQuestion]) -> List[EvalRecord]:
    """Run each question through the real retriever + compiled workflow.

    Args:
        app: A compiled graph from `src.workflow.graph.create_workflow` --
            the real production workflow, not a stand-in.
        retriever: The same `BaseRetriever` instance `app` was built
            with (i.e. `get_retriever(vectorstore)`) -- used separately,
            read-only, purely to measure retrieval-stage diagnostics.
        questions: Output of `dataset_loader.load_eval_questions`.

    Returns:
        One `EvalRecord` per question that completed successfully. A
        question that exhausts the workflow's `recursion_limit` is
        logged and skipped rather than aborting the whole evaluation run.
    """
    records: List[EvalRecord] = []

    for item in questions:
        question = item["question"]
        ground_truth = item.get("ground_truth")
        expected_pages = item.get("expected_pages")

        logger.info("evaluation: running question=%r", question)

        # --- Retrieval stage, timed and inspected independently ---------------
        retrieval_start = time.perf_counter()
        retrieved_chunks = retriever.retrieve_with_diagnostics(question)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        retrieved_pages = [chunk.document.metadata.get("page") for chunk in retrieved_chunks]
        num_retrieved_chunks = len(retrieved_chunks)
        num_reranked_chunks = sum(1 for chunk in retrieved_chunks if chunk.rerank_score is not None)

        logger.info(
            "evaluation: retrieval took %.1fms, %d chunk(s) (%d reranked), pages=%s",
            retrieval_latency_ms,
            num_retrieved_chunks,
            num_reranked_chunks,
            retrieved_pages,
        )

        # --- Full workflow, timed as a whole ------------------------------------
        workflow_start = time.perf_counter()
        try:
            final_state = app.invoke(
                {"question": question}, {"recursion_limit": config.recursion_limit}
            )
        except GraphRecursionError:
            logger.error(
                "Skipping question=%r: workflow hit recursion_limit=%d without terminating "
                "(this would also fail in the live app -- consider raising "
                "RAG_MAX_HALLUCINATION_RETRIES / RAG_MAX_QUERY_REWRITES / RAG_RECURSION_LIMIT).",
                question,
                config.recursion_limit,
            )
            continue
        workflow_latency_ms = (time.perf_counter() - workflow_start) * 1000
        generation_latency_ms = max(workflow_latency_ms - retrieval_latency_ms, 0.0)

        answer = final_state.get("generation", "")
        documents = final_state.get("documents", [])
        sources = final_state.get("sources", [])
        low_confidence = bool(final_state.get("low_confidence", False))

        generation_attempts = final_state.get("generation_attempts", 1) or 1
        query_rewrite_attempts = final_state.get("query_rewrite_attempts", 0) or 0
        retries = max(generation_attempts - 1, 0) + query_rewrite_attempts
        query_rewritten = query_rewrite_attempts > 0

        # Citations are already de-duplicated by the workflow's `generate`
        # node (src/workflow/citations.py); the raw pre-dedup count is just
        # how many final documents carried a citation-worthy page/filename.
        duplicate_citations_removed = max(len(documents) - len(sources), 0)

        logger.info(
            "evaluation: workflow took %.1fms (generation ~%.1fms), answer=%d chars, "
            "retries=%d, query_rewritten=%s, low_confidence=%s, citations=%d",
            workflow_latency_ms,
            generation_latency_ms,
            len(answer),
            retries,
            query_rewritten,
            low_confidence,
            len(sources),
        )

        records.append(
            build_eval_record(
                question=question,
                answer=answer,
                retrieved_pages=retrieved_pages,
                expected_pages=expected_pages,
                num_retrieved_chunks=num_retrieved_chunks,
                num_reranked_chunks=num_reranked_chunks,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                workflow_latency_ms=workflow_latency_ms,
                retries=retries,
                query_rewritten=query_rewritten,
                low_confidence=low_confidence,
                sources=sources,
                duplicate_citations_removed=duplicate_citations_removed,
                ground_truth=ground_truth,
            )
        )

    return records
