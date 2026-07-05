"""Text chunking.

Single responsibility: split text into `Document` chunks ready for
embedding, with metadata attached. Chunk size / overlap come from
`config.py` by default so they are never hardcoded here.

`split_pages_into_documents` is the ingestion entry point: it chunks
page-by-page so every chunk keeps an accurate `page` number, which is
required for source citations (PHASE 3). Chunking whole-document text
in one shot (the old approach) throws that information away.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Heuristic heading detector for PHASE 2 "section" metadata: a short,
# standalone line that looks like a title/heading rather than prose
# (no trailing period, mostly uppercase or title-cased, not too long).
_HEADING_RE = re.compile(r"^[A-Z0-9][A-Za-z0-9 ,\-:&/']{2,80}$")


def _splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )


def _extract_headings(page_text: str) -> List[Tuple[int, str]]:
    """Find heading-like lines in a page, paired with their character offset.

    This is a lightweight heuristic (no layout/font information is
    available from plain-text PDF extraction), not a guarantee of a true
    document section. It's good enough to give chunks a human-readable
    `section` label for citations/debugging, and degrades gracefully to
    `None` when nothing heading-like is found.
    """
    headings: List[Tuple[int, str]] = []
    offset = 0
    for line in page_text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and len(stripped) <= 80 and _HEADING_RE.match(stripped):
            is_upper = stripped.isupper()
            is_title = stripped.istitle()
            if is_upper or is_title:
                headings.append((offset, stripped))
        offset += len(line)
    return headings


def _section_for_offset(headings: List[Tuple[int, str]], offset: int) -> Optional[str]:
    """Return the most recent heading at or before `offset`, if any."""
    section = None
    for heading_offset, heading_text in headings:
        if heading_offset <= offset:
            section = heading_text
        else:
            break
    return section


def split_text_into_documents(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    metadata: Optional[dict] = None,
) -> List[Document]:
    """Split a single block of text into overlapping chunks.

    Use this for text with no page structure. For PDFs, prefer
    `split_pages_into_documents` so page numbers are preserved.

    Args:
        text: The raw text to split.
        chunk_size: Override for `config.chunk_size` (in tokens).
        chunk_overlap: Override for `config.chunk_overlap` (in tokens).
        metadata: Optional metadata dict attached to every resulting chunk.

    Returns:
        A list of `Document` chunks.
    """
    chunk_size = chunk_size if chunk_size is not None else config.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap

    doc_splits = _splitter(chunk_size, chunk_overlap).create_documents([text])

    for doc in doc_splits:
        doc.metadata.pop("start_index", None)  # internal only; not part of this function's contract

    if metadata:
        for i, doc in enumerate(doc_splits):
            doc.metadata.update(metadata)
            doc.metadata["chunk_id"] = i

    logger.info(
        "Split text into %d chunks (chunk_size=%d, chunk_overlap=%d)",
        len(doc_splits),
        chunk_size,
        chunk_overlap,
    )
    return doc_splits


def split_pages_into_documents(
    pages: List[Tuple[int, str]],
    document_id: str,
    source: str,
    filename: str,
    created_at: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """Split page-tagged text into chunks, preserving page numbers.

    Splits each page independently (rather than concatenating the whole
    document first) so every chunk can be traced back to exactly one
    page. `chunk_id` is a running counter across the whole document, not
    per-page, so chunks have a stable, unique ordinal within the doc.

    Args:
        pages: Output of `pdf_loader.extract_pages_from_pdf`, i.e.
            (page_number, page_text) tuples.
        document_id: Stable identifier for the source document (we use
            its SHA-256 content hash -- see `hashing.py`).
        source: Human-readable source label (e.g. the filename).
        filename: The original filename.
        created_at: ISO timestamp of when this document was ingested.
        chunk_size: Override for `config.chunk_size` (in tokens).
        chunk_overlap: Override for `config.chunk_overlap` (in tokens).

    Returns:
        A list of `Document` chunks, each with metadata:
        {document_id, source, filename, page, section, chunk_id, created_at}.
        `section` is a best-effort heading heuristic (see
        `_extract_headings`) and may be `None` if no heading-like line
        was found on that page before the chunk.
    """
    chunk_size = chunk_size if chunk_size is not None else config.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap
    splitter = _splitter(chunk_size, chunk_overlap)

    all_chunks: List[Document] = []
    chunk_id = 0
    for page_number, page_text in pages:
        if not page_text.strip():
            continue  # skip blank pages (common with scanned/cover pages)
        headings = _extract_headings(page_text)
        page_chunks = splitter.create_documents([page_text])
        for chunk in page_chunks:
            start_index = chunk.metadata.pop("start_index", 0)
            chunk.metadata.update(
                {
                    "document_id": document_id,
                    "source": source,
                    "filename": filename,
                    "page": page_number,
                    "section": _section_for_offset(headings, start_index),
                    "chunk_id": chunk_id,
                    "created_at": created_at,
                }
            )
            all_chunks.append(chunk)
            chunk_id += 1

    logger.info(
        "Split %d pages into %d chunks (chunk_size=%d, chunk_overlap=%d) for document_id=%s",
        len(pages),
        len(all_chunks),
        chunk_size,
        chunk_overlap,
        document_id[:8],
    )
    return all_chunks
