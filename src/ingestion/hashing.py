"""File hashing for duplicate detection.

Single responsibility: compute a stable content hash for a file so the
ingestion pipeline can tell whether a document has already been indexed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def compute_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """Compute the SHA-256 hash of a file's contents.

    Reading in chunks keeps memory usage flat regardless of file size.

    Args:
        file_path: Path to the file to hash.
        chunk_size: Number of bytes read per iteration.

    Returns:
        Hex digest string, e.g. "a3f5...".
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def short_hash(file_path: str, length: int = 8) -> str:
    """Convenience wrapper returning a truncated hash for display/logging."""
    return compute_file_hash(file_path)[:length]
