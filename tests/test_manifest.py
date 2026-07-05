"""Tests for src.ingestion.manifest."""
import os
import tempfile

from src.ingestion.manifest import (
    is_already_indexed,
    load_manifest,
    record_indexed_document,
    save_manifest,
)


def test_load_manifest_missing_file_returns_empty_dict():
    manifest = load_manifest("/tmp/definitely_does_not_exist_manifest.json")
    assert manifest == {}


def test_record_and_check_indexed_document():
    manifest = {}
    manifest = record_indexed_document(
        manifest, file_hash="abc123", filename="input.pdf", path="data/raw/input.pdf", chunk_count=10
    )
    assert is_already_indexed(manifest, "abc123") is True
    assert is_already_indexed(manifest, "does-not-exist") is False
    assert manifest["abc123"]["chunk_count"] == 10


def test_save_and_load_manifest_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = os.path.join(tmpdir, "nested", "manifest.json")
        manifest = record_indexed_document(
            {}, file_hash="hash1", filename="a.pdf", path="a.pdf", chunk_count=3
        )
        save_manifest(manifest_path, manifest)

        loaded = load_manifest(manifest_path)
        assert loaded == manifest
        assert is_already_indexed(loaded, "hash1")
