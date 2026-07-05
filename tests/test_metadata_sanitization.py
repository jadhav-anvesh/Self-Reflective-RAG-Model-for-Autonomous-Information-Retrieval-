"""Tests for src.utils.metadata (the ChromaDB metadata sanitization fix).

Covers the exact failure mode reported: `ValueError: Expected metadata
value to be a str, int, float or bool, got None` during
`vectorstore.add_documents()`, plus the other unsupported-type cases
called out in the bug report (nested dicts, lists, arbitrary objects).
"""
from langchain_core.documents import Document

from src.utils.metadata import sanitize_documents_metadata, sanitize_metadata


class _CustomObject:
    """Stand-in for "some arbitrary unsupported object" in metadata."""

    def __repr__(self):
        return "<CustomObject>"


# --- sanitize_metadata: None values ------------------------------------------


def test_none_values_are_removed():
    result = sanitize_metadata({"page": 12, "section": None, "source": "input.pdf"})
    assert result == {"page": 12, "source": "input.pdf"}
    assert "section" not in result


def test_all_none_metadata_returns_empty_dict():
    assert sanitize_metadata({"a": None, "b": None}) == {}


def test_empty_metadata_returns_empty_dict():
    assert sanitize_metadata({}) == {}


# --- sanitize_metadata: valid primitive types pass through unchanged --------


def test_valid_str_int_float_bool_are_preserved():
    metadata = {"filename": "input.pdf", "page": 7, "score": 0.87, "is_scanned": False}
    assert sanitize_metadata(metadata) == metadata


def test_bool_is_preserved_as_bool_not_cast_to_int():
    result = sanitize_metadata({"flag": True})
    assert result["flag"] is True
    assert isinstance(result["flag"], bool)


def test_zero_and_false_and_empty_string_are_preserved_not_treated_as_none():
    """Falsy-but-valid values must NOT be dropped -- only None is dropped."""
    result = sanitize_metadata({"count": 0, "flag": False, "note": ""})
    assert result == {"count": 0, "flag": False, "note": ""}


# --- sanitize_metadata: nested metadata (dicts) -----------------------------


def test_nested_dict_value_is_removed():
    result = sanitize_metadata({"page": 12, "info": {"nested": "value"}, "source": "input.pdf"})
    assert result == {"page": 12, "source": "input.pdf"}
    assert "info" not in result


def test_deeply_nested_structure_is_removed():
    result = sanitize_metadata({"data": {"a": {"b": [1, 2, {"c": 3}]}}})
    assert result == {}


# --- sanitize_metadata: lists / tuples / sets -------------------------------


def test_list_of_primitives_is_joined_into_string():
    result = sanitize_metadata({"tags": ["admissions", "requirements"]})
    assert result == {"tags": "admissions, requirements"}


def test_tuple_of_primitives_is_joined_into_string():
    result = sanitize_metadata({"tags": ("a", "b", "c")})
    assert result == {"tags": "a, b, c"}


def test_empty_list_is_dropped():
    result = sanitize_metadata({"tags": [], "page": 1})
    assert result == {"page": 1}
    assert "tags" not in result


def test_list_with_only_unsupported_items_is_dropped():
    result = sanitize_metadata({"items": [{"x": 1}, {"y": 2}], "page": 1})
    assert result == {"page": 1}


def test_list_with_mixed_supported_and_unsupported_items_keeps_only_supported():
    result = sanitize_metadata({"tags": ["a", {"bad": True}, "b", None]})
    assert result == {"tags": "a, b"}


# --- sanitize_metadata: unsupported objects ---------------------------------


def test_custom_object_is_removed():
    result = sanitize_metadata({"page": 1, "obj": _CustomObject()})
    assert result == {"page": 1}
    assert "obj" not in result


def test_function_value_is_removed():
    result = sanitize_metadata({"page": 1, "callback": lambda: None})
    assert result == {"page": 1}


# --- sanitize_metadata: never raises -----------------------------------------


def test_sanitize_metadata_never_raises_on_mixed_garbage():
    metadata = {
        "page": 12,
        "section": None,
        "source": "input.pdf",
        "tags": ["a", "b"],
        "nested": {"x": 1},
        "obj": _CustomObject(),
        "score": 0.5,
        "flag": True,
        "empty_list": [],
    }
    result = sanitize_metadata(metadata)
    assert result == {
        "page": 12,
        "source": "input.pdf",
        "tags": "a, b",
        "score": 0.5,
        "flag": True,
    }


# --- sanitize_documents_metadata: applied to Document objects ---------------


def test_sanitize_documents_metadata_mutates_documents_in_place():
    documents = [
        Document(page_content="chunk 1", metadata={"page": 12, "section": None, "source": "input.pdf"}),
        Document(page_content="chunk 2", metadata={"page": 13, "section": "Admissions", "source": "input.pdf"}),
    ]

    result = sanitize_documents_metadata(documents)

    assert result is documents  # same list, mutated in place
    assert "section" not in documents[0].metadata
    assert documents[0].metadata == {"page": 12, "source": "input.pdf"}
    assert documents[1].metadata == {"page": 13, "section": "Admissions", "source": "input.pdf"}


def test_sanitize_documents_metadata_handles_empty_list():
    assert sanitize_documents_metadata([]) == []


def test_sanitize_documents_metadata_removes_none_from_every_document():
    """The exact reported bug: 129 chunks, every one carrying `section: None`."""
    documents = [
        Document(page_content=f"chunk {i}", metadata={"page": i, "section": None, "source": "input.pdf"})
        for i in range(129)
    ]

    sanitize_documents_metadata(documents)

    for doc in documents:
        assert None not in doc.metadata.values()
        assert "section" not in doc.metadata
