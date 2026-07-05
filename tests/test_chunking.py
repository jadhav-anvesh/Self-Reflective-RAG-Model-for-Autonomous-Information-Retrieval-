"""Tests for src.ingestion.chunking."""
from src.ingestion.chunking import split_pages_into_documents, split_text_into_documents


def test_split_returns_documents():
    text = "This is a sentence. " * 200  # long enough to force multiple chunks
    docs = split_text_into_documents(text, chunk_size=50, chunk_overlap=10)
    assert len(docs) > 1
    assert all(doc.page_content for doc in docs)


def test_split_attaches_metadata_and_chunk_id():
    text = "Some content. " * 100
    docs = split_text_into_documents(
        text, chunk_size=50, chunk_overlap=10, metadata={"source": "input.pdf"}
    )
    assert len(docs) > 0
    for i, doc in enumerate(docs):
        assert doc.metadata["source"] == "input.pdf"
        assert doc.metadata["chunk_id"] == i


def test_split_without_metadata_has_empty_metadata():
    text = "Short text."
    docs = split_text_into_documents(text, chunk_size=50, chunk_overlap=10)
    assert len(docs) == 1
    assert docs[0].metadata == {}


def test_split_pages_preserves_page_numbers():
    pages = [
        (1, "Page one content. " * 30),
        (2, "Page two content. " * 30),
        (3, "Page three content. " * 30),
    ]
    docs = split_pages_into_documents(
        pages,
        document_id="abc123",
        source="input.pdf",
        filename="input.pdf",
        created_at="2026-07-03T00:00:00+00:00",
        chunk_size=50,
        chunk_overlap=10,
    )
    assert len(docs) > 3  # each page produced at least one chunk
    pages_seen = {doc.metadata["page"] for doc in docs}
    assert pages_seen == {1, 2, 3}
    # chunk_id is a running counter across the whole document, strictly increasing
    chunk_ids = [doc.metadata["chunk_id"] for doc in docs]
    assert chunk_ids == sorted(chunk_ids)
    assert chunk_ids == list(range(len(docs)))


def test_split_pages_attaches_full_metadata():
    pages = [(1, "Some content here. " * 20)]
    docs = split_pages_into_documents(
        pages,
        document_id="hash-xyz",
        source="report.pdf",
        filename="report.pdf",
        created_at="2026-07-03T00:00:00+00:00",
        chunk_size=50,
        chunk_overlap=10,
    )
    assert len(docs) >= 1
    for doc in docs:
        assert doc.metadata["document_id"] == "hash-xyz"
        assert doc.metadata["source"] == "report.pdf"
        assert doc.metadata["filename"] == "report.pdf"
        assert doc.metadata["page"] == 1
        assert doc.metadata["created_at"] == "2026-07-03T00:00:00+00:00"


def test_split_pages_skips_blank_pages():
    pages = [(1, "Real content here. " * 20), (2, "   \n  "), (3, "More real content. " * 20)]
    docs = split_pages_into_documents(
        pages,
        document_id="doc1",
        source="s.pdf",
        filename="s.pdf",
        created_at="2026-07-03T00:00:00+00:00",
        chunk_size=50,
        chunk_overlap=10,
    )
    pages_seen = {doc.metadata["page"] for doc in docs}
    assert 2 not in pages_seen
