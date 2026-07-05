"""Evaluation dataset loading.

Single responsibility: read an evaluation question set (e.g.
`data/eval/questions.json`) into validated records. Contains no
retrieval/generation/metric logic -- see `evaluator.py` for that.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from typing_extensions import TypedDict

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class EvalQuestion(TypedDict, total=False):
    """One row of the evaluation dataset (`data/eval/questions.json`).

    Attributes:
        question: The question text. Required.
        ground_truth: Optional reference answer. Not consumed by any
            metric in this project's deterministic evaluation framework
            (there is no LLM judge) -- kept for schema backward
            compatibility and possible future use (e.g. exact-match
            scoring), never required.
        expected_pages: Optional list of 1-indexed PDF page numbers the
            answer is expected to come from. Powers the deterministic
            `Page Recall` metric in `metrics.compute_page_recall` --
            fraction of these pages actually present among the pages
            retrieval returned for this question.
    """

    question: str
    ground_truth: Optional[str]
    expected_pages: Optional[List[int]]


def load_eval_questions(path: str) -> List[EvalQuestion]:
    """Load and validate a JSON evaluation question set.

    Args:
        path: Path to a JSON file containing a list of question objects.

    Returns:
        The parsed list of question dicts (fields other than `question`
        are optional; callers read them defensively with `.get(...)`).

    Raises:
        FileNotFoundError: if `path` doesn't exist.
        ValueError: if the file isn't a non-empty JSON list, or any
            entry is missing a non-empty `question` field.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Eval question set not found at '{path}'. Create it -- see "
            f"README.md's 'Evaluation' section, or data/eval/questions.json "
            f"for the expected format -- or pass a different --questions path."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if not isinstance(questions, list):
        raise ValueError(f"Eval question set at '{path}' must be a JSON list of question objects.")
    if not questions:
        raise ValueError(f"Eval question set at '{path}' is empty.")
    for i, item in enumerate(questions):
        if not isinstance(item, dict) or not item.get("question"):
            raise ValueError(
                f"Eval question set at '{path}': entry {i} is missing a non-empty 'question' field."
            )

    logger.info("Loaded %d evaluation question(s) from %s", len(questions), path)
    return questions
