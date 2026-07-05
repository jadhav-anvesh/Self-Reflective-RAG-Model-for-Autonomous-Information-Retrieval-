"""Tests for src.indexing.vectorstore.

These tests avoid loading a real embedding model or Chroma collection;
they exercise the pure filesystem/state logic (`persist_dir_has_data`,
`load_vectorstore`'s not-found error, `add_documents` no-op path) plus
the metadata-sanitization wiring (the actual `sanitize_metadata` logic
is tested independently in `test_metadata_sanitization.py`).
"""
import os
import tempfile

import pytest
from langchain_core.documents import Document

from src.indexing.vectorstore import (
    VectorStoreNotFoundError,
    add_documents,
    create_vectorstore,
    load_vectorstore,
    persist_dir_has_data,
)


def test_persist_dir_has_data_false_for_missing_dir():
    assert persist_dir_has_data("/tmp/definitely_does_not_exist_vectorstore_dir") is False


def test_persist_dir_has_data_false_for_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert persist_dir_has_data(tmpdir) is False


def test_persist_dir_has_data_true_when_dir_has_content():
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "chroma.sqlite3"), "w").close()
        assert persist_dir_has_data(tmpdir) is True


def test_load_vectorstore_raises_when_no_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(VectorStoreNotFoundError) as exc_info:
            load_vectorstore(collection_name="rag-chroma", persist_directory=tmpdir)
        # The error message should point the user at the fix.
        assert "python ingest.py" in str(exc_info.value)


def test_add_documents_no_op_on_empty_list():
    """Passing an empty list must not attempt to call the vector store at all."""

    class ExplodingVectorStore:
        def add_documents(self, documents):
            raise AssertionError("add_documents should not be called for an empty list")

    add_documents(ExplodingVectorStore(), [])


def test_add_documents_sanitizes_metadata_before_insertion():
    """Reproduces the reported bug: a None-valued field must never reach Chroma."""

    received = {}

    class SpyVectorStore:
        def add_documents(self, documents):
            received["documents"] = documents

    documents = [
        Document(page_content="chunk", metadata={"page": 12, "section": None, "source": "input.pdf"})
    ]

    add_documents(SpyVectorStore(), documents)

    inserted_metadata = received["documents"][0].metadata
    assert None not in inserted_metadata.values()
    assert inserted_metadata == {"page": 12, "source": "input.pdf"}


def test_create_vectorstore_sanitizes_metadata_before_insertion(monkeypatch):
    """`Chroma.from_documents` must only ever see already-sanitized metadata."""
    import src.indexing.vectorstore as vectorstore_module

    received = {}

    class FakeChroma:
        @staticmethod
        def from_documents(documents, collection_name, embedding, persist_directory):
            received["documents"] = documents
            return "fake-vectorstore"

    monkeypatch.setattr(vectorstore_module, "Chroma", FakeChroma)
    monkeypatch.setattr(vectorstore_module, "get_embedding_model", lambda: "fake-embeddings")

    documents = [
        Document(page_content="chunk", metadata={"page": 12, "section": None, "tags": ["a", "b"]})
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        result = create_vectorstore(documents, collection_name="test-collection", persist_directory=tmpdir)

    assert result == "fake-vectorstore"
    inserted_metadata = received["documents"][0].metadata
    assert None not in inserted_metadata.values()
    assert inserted_metadata == {"page": 12, "tags": "a, b"}
