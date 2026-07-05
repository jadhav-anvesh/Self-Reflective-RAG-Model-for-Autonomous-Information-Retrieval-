"""Ingestion manifest.

Tracks which source documents have already been indexed, keyed by their
SHA-256 content hash. This is what makes incremental indexing and
duplicate detection possible: `ingest.py` consults the manifest before
doing any (expensive) embedding work.

manifest.json shape:
{
    "<sha256_hash>": {
        "filename": "input.pdf",
        "path": "data/raw/input.pdf",
        "chunk_count": 42,
        "indexed_at": "2026-07-03T12:00:00"
    },
    ...
}
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    """Load the manifest from disk, returning an empty dict if absent."""
    path = Path(manifest_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Manifest at %s is corrupt; starting fresh.", manifest_path)
            return {}


def save_manifest(manifest_path: str, manifest: Dict[str, Any]) -> None:
    """Persist the manifest to disk, creating parent directories if needed."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def is_already_indexed(manifest: Dict[str, Any], file_hash: str) -> bool:
    """Return True if a document with this content hash was already indexed."""
    return file_hash in manifest


def record_indexed_document(
    manifest: Dict[str, Any],
    file_hash: str,
    filename: str,
    path: str,
    chunk_count: int,
) -> Dict[str, Any]:
    """Add/update an entry in the manifest for a newly-indexed document."""
    manifest[file_hash] = {
        "filename": filename,
        "path": path,
        "chunk_count": chunk_count,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    return manifest
