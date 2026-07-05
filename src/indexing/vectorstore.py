"""Chroma vector store management.

Single responsibility: create, load, and incrementally update the
persistent Chroma vector store. This module intentionally knows nothing
about PDFs or chunking -- it only deals with `Document` objects and
embeddings.

Key production fix vs. the original project: embeddings are generated
ONCE by `ingest.py`. The application (`app.py` / `main.py`) only ever
calls `load_vectorstore()`, which is a fast, cheap operation.

Metadata sanitization: every document is passed through
`sanitize_documents_metadata()` (see `src/utils/metadata.py`) immediately
before insertion, because Chroma raises `ValueError` for any metadata
value that isn't a `str`/`int`/`float`/`bool` -- notably `None`, which
`section` in `src/ingestion/chunking.py` legitimately is when no heading
was detected. `langchain_community.vectorstores.utils.filter_complex_metadata`
does something similar, but only drops unsupported values outright
(including lists/tuples, with no attempt to preserve their content) and
operates on a whole `List[Document]` rather than being independently
testable at the single-dict level. `sanitize_metadata()` is a strict
superset of what `filter_complex_metadata` does -- notably: preserving
list/tuple values as a joined string instead of discarding them -- so
using both would be redundant; only `sanitize_documents_metadata()` is
used here.
"""
from __future__ import annotations

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import config
from src.utils.logging_config import get_logger
from src.utils.metadata import sanitize_documents_metadata

logger = get_logger(__name__)


class VectorStoreNotFoundError(RuntimeError):
    """Raised when the application tries to load a vector store that
    hasn't been built yet."""


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Construct the embedding model from `config.py` settings.

    Centralized here so ingestion and retrieval always use the exact
    same embedding model/config (mismatches would silently corrupt
    retrieval quality).
    """
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": config.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )


def persist_dir_has_data(persist_directory: str) -> bool:
    """Chroma creates the directory on first use even with zero vectors,
    so existence alone isn't a reliable signal -- check for content."""
    if not os.path.isdir(persist_directory):
        return False
    return any(os.scandir(persist_directory))


def create_vectorstore(
    documents: List[Document],
    collection_name: str = config.collection_name,
    persist_directory: str = config.persist_directory,
) -> Chroma:
    """Create a brand-new persistent vector store from documents.

    Used only by `ingest.py` on first run. Never call this from the
    application/UI layer.
    """
    logger.info(
        "Creating new vector store '%s' at %s with %d documents",
        collection_name,
        persist_directory,
        len(documents),
    )
    documents = sanitize_documents_metadata(documents)
    embedding_model = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=documents,
        collection_name=collection_name,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )
    return vectorstore


def load_vectorstore(
    collection_name: str = config.collection_name,
    persist_directory: str = config.persist_directory,
) -> Chroma:
    """Load an existing persistent vector store.

    Raises:
        VectorStoreNotFoundError: If no vector store has been built yet,
            with a message telling the user to run `python ingest.py`.
    """
    if not persist_dir_has_data(persist_directory):
        raise VectorStoreNotFoundError(
            f"No vector store found at '{persist_directory}'. "
            f"Run `python ingest.py` first to build the index from your "
            f"documents in '{config.raw_data_dir}'."
        )

    logger.info("Loading existing vector store '%s' from %s", collection_name, persist_directory)
    embedding_model = get_embedding_model()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )
    return vectorstore


def get_or_create_vectorstore(
    documents: Optional[List[Document]] = None,
    collection_name: str = config.collection_name,
    persist_directory: str = config.persist_directory,
) -> Chroma:
    """Load the vector store if it exists, otherwise create it from `documents`.

    Used by `ingest.py`, which is the only place allowed to create/mutate
    the vector store.
    """
    if persist_dir_has_data(persist_directory):
        vectorstore = load_vectorstore(collection_name, persist_directory)
        if documents:
            add_documents(vectorstore, documents)
        return vectorstore

    if not documents:
        raise ValueError("No existing vector store and no documents provided to create one.")
    return create_vectorstore(documents, collection_name, persist_directory)


def add_documents(vectorstore: Chroma, documents: List[Document]) -> None:
    """Incrementally add new documents to an existing vector store.

    This is what makes indexing incremental: only the chunks belonging
    to newly-added source files are embedded and inserted, the rest of
    the collection is untouched.
    """
    if not documents:
        logger.info("No new documents to add; vector store unchanged.")
        return
    documents = sanitize_documents_metadata(documents)
    logger.info("Adding %d new chunks to existing vector store", len(documents))
    vectorstore.add_documents(documents)
