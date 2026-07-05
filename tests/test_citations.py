"""Tests for src.workflow.citations."""
from langchain_core.documents import Document

from src.workflow.citations import format_sources, format_sources_as_text


def _doc(filename, page, section=None):
    return Document(page_content="content", metadata={"filename": filename, "page": page, "section": section})


def test_format_sources_deduplicates_same_page():
    documents = [
        _doc("input.pdf", 7),
        _doc("input.pdf", 7),  # same page, different chunk -- should collapse
        _doc("input.pdf", 12),
    ]
    citations = format_sources(documents)
    assert len(citations) == 2
    assert citations[0]["page"] == 7
    assert citations[1]["page"] == 12


def test_format_sources_preserves_first_seen_order():
    documents = [_doc("a.pdf", 3), _doc("b.pdf", 1), _doc("a.pdf", 3)]
    citations = format_sources(documents)
    assert [(c["filename"], c["page"]) for c in citations] == [("a.pdf", 3), ("b.pdf", 1)]


def test_format_sources_includes_section_when_present():
    documents = [_doc("input.pdf", 12, section="Admissions Requirements")]
    citations = format_sources(documents)
    assert citations[0]["section"] == "Admissions Requirements"


def test_format_sources_empty_list_returns_empty():
    assert format_sources([]) == []


def test_format_sources_as_text_handles_empty_list():
    assert format_sources_as_text([]) == "No sources available."


def test_format_sources_as_text_formats_filename_and_page():
    documents = [_doc("input.pdf", 7)]
    text = format_sources_as_text(documents)
    assert "input.pdf" in text
    assert "Page 7" in text


def test_format_sources_as_text_includes_section_when_present():
    documents = [_doc("input.pdf", 12, section="Admissions")]
    text = format_sources_as_text(documents)
    assert "Admissions" in text


def test_format_sources_as_text_joins_multiple_sources():
    documents = [_doc("a.pdf", 1), _doc("b.pdf", 2)]
    text = format_sources_as_text(documents)
    assert "a.pdf" in text and "b.pdf" in text
    assert text.count("Page") == 2
