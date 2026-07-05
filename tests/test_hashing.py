"""Tests for src.ingestion.hashing."""
import hashlib
import os
import tempfile

from src.ingestion.hashing import compute_file_hash, short_hash


def test_compute_file_hash_matches_stdlib():
    content = b"hello world, this is a test PDF payload"
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(content)
        path = f.name

    try:
        expected = hashlib.sha256(content).hexdigest()
        assert compute_file_hash(path) == expected
    finally:
        os.remove(path)


def test_compute_file_hash_is_deterministic():
    content = b"same content twice"
    with tempfile.NamedTemporaryFile(delete=False) as f1, tempfile.NamedTemporaryFile(delete=False) as f2:
        f1.write(content)
        f2.write(content)
        path1, path2 = f1.name, f2.name

    try:
        assert compute_file_hash(path1) == compute_file_hash(path2)
    finally:
        os.remove(path1)
        os.remove(path2)


def test_compute_file_hash_differs_for_different_content():
    with tempfile.NamedTemporaryFile(delete=False) as f1, tempfile.NamedTemporaryFile(delete=False) as f2:
        f1.write(b"content A")
        f2.write(b"content B")
        path1, path2 = f1.name, f2.name

    try:
        assert compute_file_hash(path1) != compute_file_hash(path2)
    finally:
        os.remove(path1)
        os.remove(path2)


def test_short_hash_is_prefix_of_full_hash():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"prefix test")
        path = f.name

    try:
        full = compute_file_hash(path)
        assert short_hash(path, length=8) == full[:8]
    finally:
        os.remove(path)
