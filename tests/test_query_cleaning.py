"""Tests for src.workflow.query_cleaning.

Regression coverage for the bug where the query-rewriter LLM returned
"Improved Question: ...\\n\\nReasoning: ..." instead of just the
rewritten question, which the retriever then received verbatim.
"""
from src.workflow.query_cleaning import clean_rewritten_question


def test_strips_improved_question_label_and_reasoning_section():
    raw = (
        "Improved Question: What is the eligibility criteria for BBA admission?\n\n"
        "Reasoning: I rewrote the question to be more specific about the degree program."
    )
    assert clean_rewritten_question(raw) == "What is the eligibility criteria for BBA admission?"


def test_strips_rewritten_question_label():
    raw = "Rewritten Question: What courses are offered in semester one?"
    assert clean_rewritten_question(raw) == "What courses are offered in semester one?"


def test_strips_plain_question_label():
    raw = "Question: What is the minimum attendance requirement?"
    assert clean_rewritten_question(raw) == "What is the minimum attendance requirement?"


def test_strips_explanation_section():
    raw = "What is the grading policy?\nExplanation: this rephrases the original question."
    assert clean_rewritten_question(raw) == "What is the grading policy?"


def test_strips_rationale_section():
    raw = "What is the grading policy?\n\nRationale:\nMade it more specific.\nAnd concise."
    assert clean_rewritten_question(raw) == "What is the grading policy?"


def test_leaves_clean_output_unchanged():
    raw = "What is the eligibility criteria for BBA admission?"
    assert clean_rewritten_question(raw) == raw


def test_strips_surrounding_whitespace():
    raw = "   What is the eligibility criteria?   \n"
    assert clean_rewritten_question(raw) == "What is the eligibility criteria?"


def test_strips_wrapping_quotes():
    raw = '"What is the eligibility criteria for BBA admission?"'
    assert clean_rewritten_question(raw) == "What is the eligibility criteria for BBA admission?"


def test_takes_first_line_when_multiple_lines_remain():
    raw = "What is the eligibility criteria?\nSome trailing commentary the model added."
    assert clean_rewritten_question(raw) == "What is the eligibility criteria?"


def test_empty_input_returns_empty_string():
    assert clean_rewritten_question("") == ""
    assert clean_rewritten_question(None) == ""
