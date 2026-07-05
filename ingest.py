"""Ingestion entry point.

Run this script whenever you add or change documents in `data/raw/`:

    python ingest.py

Pipeline:
    Read PDF(s) -> Chunk -> Generate Embeddings -> Persist ChromaDB

This is the ONLY place in the codebase allowed to generate embeddings or
mutate the vector store. `app.py` and `main.py` only ever *load* what
this script builds -- that's the fix for the original project always
recreating embeddings on every startup.

Duplicate detection: each source file's SHA-256 hash is checked against
`data/manifest.json`. Already-indexed files are skipped, so re-running
this script after adding one new PDF only embeds that new PDF
(incremental indexing), not the whole corpus.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from datetime import datetime, timezone

from langchain_core.documents import Document

from config import config
from src.indexing.vectorstore import add_documents, create_vectorstore, persist_dir_has_data
from src.ingestion.chunking import split_pages_into_documents
from src.ingestion.hashing import compute_file_hash
from src.ingestion.manifest import (
    is_already_indexed,
    load_manifest,
    record_indexed_document,
    save_manifest,
)
from src.ingestion.pdf_loader import extract_pages_from_pdf
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def find_source_pdfs(raw_data_dir: str) -> List[Path]:
    """Find all PDF files under the raw data directory."""
    raw_dir = Path(raw_data_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory '{raw_data_dir}' does not exist. "
            f"Create it and add your PDF(s) there."
        )
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDF files found in '{raw_data_dir}'. Add at least one PDF and re-run."
        )
    return pdfs


def build_chunks_for_file(pdf_path: Path, document_id: str) -> List[Document]:
    """Extract a PDF page-by-page and split it into embed-ready chunks.

    Each resulting chunk carries {document_id, source, filename, page,
    chunk_id, created_at} metadata -- everything PHASE 3 source
    citations need, with no re-parsing of the PDF required later.
    """
    pages = extract_pages_from_pdf(str(pdf_path))
    created_at = datetime.now(timezone.utc).isoformat()
    return split_pages_into_documents(
        pages,
        document_id=document_id,
        source=pdf_path.name,
        filename=pdf_path.name,
        created_at=created_at,
    )


def run_ingestion() -> None:
    """Ingest every new/changed PDF in `data/raw/` into the vector store."""
    logger.info("Starting ingestion. raw_data_dir=%s", config.raw_data_dir)

    pdf_paths = find_source_pdfs(config.raw_data_dir)
    manifest = load_manifest(config.manifest_path)

    new_chunks: List[Document] = []
    skipped, newly_indexed = 0, 0

    for pdf_path in pdf_paths:
        file_hash = compute_file_hash(str(pdf_path))

        if is_already_indexed(manifest, file_hash):
            logger.info("Skipping already-indexed file: %s (hash=%s...)", pdf_path.name, file_hash[:8])
            skipped += 1
            continue

        logger.info("Indexing new file: %s", pdf_path.name)
        chunks = build_chunks_for_file(pdf_path, document_id=file_hash)
        new_chunks.extend(chunks)

        manifest = record_indexed_document(
            manifest,
            file_hash=file_hash,
            filename=pdf_path.name,
            path=str(pdf_path),
            chunk_count=len(chunks),
        )
        newly_indexed += 1

    if not new_chunks:
        logger.info("Nothing new to index. %d file(s) already up to date.", skipped)
        return

    if persist_dir_has_data(config.persist_directory):
        # Incremental path: vector store already exists, only add new chunks.
        from src.indexing.vectorstore import load_vectorstore

        vectorstore = load_vectorstore()
        add_documents(vectorstore, new_chunks)
    else:
        # First-run path: no vector store yet, create it from scratch.
        vectorstore = create_vectorstore(new_chunks)

    save_manifest(config.manifest_path, manifest)

    logger.info(
        "Ingestion complete. %d file(s) newly indexed (%d chunks), %d file(s) skipped (already indexed).",
        newly_indexed,
        len(new_chunks),
        skipped,
    )


if __name__ == "__main__":
    try:
        run_ingestion()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
