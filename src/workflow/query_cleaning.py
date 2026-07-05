"""Cleanup for LLM-rewritten retrieval queries.

Single responsibility: strip labels/reasoning the question-rewriter LLM
sometimes adds despite being asked not to (e.g. "Improved Question: ...
\\n\\n Reasoning: ..."), so the retriever always receives just the
rewritten question text. Kept in its own module (no Ollama/hub import)
so it's unit-testable without a network connection or a running model.
"""
from __future__ import annotations

import re

# Matches a "Reasoning:"/"Explanation:"/"Rationale:" section and
# everything after it, so it can be dropped -- these are the labels
# llama3.2 has been observed adding despite prompt instructions not to.
_REASONING_SECTION = re.compile(r"(?is)\n?\s*(reasoning|explanation|rationale)\s*:.*")

# Matches a leading label like "Improved Question:", "Rewritten Question:",
# "New Question:", or plain "Question:" so it can be stripped from the front.
_LEADING_LABEL = re.compile(r"(?i)^\s*(improved|rewritten|new|optimized)?\s*question\s*:\s*")

# Strips wrapping quote characters some models add around the whole answer.
_WRAPPING_QUOTES = re.compile(r'^[\'"](.*)[\'"]$')


def clean_rewritten_question(raw: str) -> str:
    """Return only the rewritten question text, with no label or reasoning.

    Defense-in-depth alongside the rewrite prompt itself: the prompt asks
    the model to output only the question, but small/local models don't
    always comply, so this strips the common failure patterns rather than
    trusting the model unconditionally.

    Args:
        raw: The rewriter LLM's raw string output.

    Returns:
        A single-line rewritten question with any "Reasoning:"/"Explanation:"
        section and any leading "Improved Question:"-style label removed.
    """
    text = (raw or "").strip()

    # Drop a trailing reasoning/explanation section, if present.
    text = _REASONING_SECTION.sub("", text).strip()

    # Drop a leading label like "Improved Question:".
    text = _LEADING_LABEL.sub("", text).strip()

    # If the model still wrapped output across multiple lines/paragraphs,
    # keep only the first non-empty line -- a rewritten question is a
    # single line, never a multi-paragraph block.
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), text)

    # Strip a single layer of wrapping quotes, if present.
    match = _WRAPPING_QUOTES.match(first_line)
    if match:
        first_line = match.group(1).strip()

    return first_line
