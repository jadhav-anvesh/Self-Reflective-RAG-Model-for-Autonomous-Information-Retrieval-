"""PDF text extraction.

Single responsibility: turn a PDF file on disk into text. Does not know
about chunking, embeddings, or the vector store.

`extract_pages_from_pdf` is the primary function -- it keeps page
boundaries intact, which is what lets `chunking.py` attach an accurate
`page` number to every chunk. Without this, source citations
("Page 14") would require re-parsing the PDF later.
"""
from __future__ import annotations

from typing import List, Tuple

import fitz  # PyMuPDF

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def extract_pages_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    """Extract text from a PDF, one entry per page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of (page_number, page_text) tuples, 1-indexed to match
        how humans refer to pages (page 1, not page 0).
    """
    logger.info("Extracting text from PDF: %s", pdf_path)
    doc = fitz.open(pdf_path)
    try:
        pages = [(page_number, page.get_text()) for page_number, page in enumerate(doc, start=1)]
    finally:
        doc.close()
    logger.info("Extracted %d pages from %s", len(pages), pdf_path)
    return pages


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF as a single string (page info discarded).

    Kept for callers that only need raw text (e.g. quick scripts, tests).
    Ingestion should prefer `extract_pages_from_pdf` so page metadata
    isn't lost.
    """
    pages = extract_pages_from_pdf(pdf_path)
    return "".join(text for _, text in pages)
