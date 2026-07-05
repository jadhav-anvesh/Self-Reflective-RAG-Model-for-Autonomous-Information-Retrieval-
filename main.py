"""CLI entry point for the compiled RAG workflow.

Like `app.py`, this only *loads* the existing vector store -- it never
generates embeddings. Run `python ingest.py` first if you see a
VectorStoreNotFoundError.

Kept for terminal-based testing/debugging and for anything that wants
to import a ready-to-use `app` object, mirroring the original project's
`main.py` interface.
"""
from __future__ import annotations

import sys
import warnings

from src.indexing.vectorstore import VectorStoreNotFoundError, load_vectorstore
from src.retrieval.retriever import get_retriever
from src.utils.logging_config import get_logger
from src.workflow.graph import create_workflow

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

try:
    vectorstore = load_vectorstore()
except VectorStoreNotFoundError as e:
    logger.error(str(e))
    sys.exit(1)

retriever = get_retriever(vectorstore)
app = create_workflow(retriever)
logger.info("Workflow compiled successfully!")
