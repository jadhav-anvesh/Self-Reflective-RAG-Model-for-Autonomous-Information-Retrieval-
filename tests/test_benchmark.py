"""Tests for benchmark.py's pure, model-free helper functions.

`run_config` (real ingestion/embedding/LLM calls) is intentionally not
unit tested here -- it's an integration-level function meant to run in
a real environment with Ollama + embedding models available.
"""
import csv
import os
import tempfile

import pytest

from benchmark import BenchmarkResult, _find_pdfs, plot_results, write_results_csv


def _sample_results():
    return [
        BenchmarkResult(
            chunk_size=200,
            embedding_model="fake-model",
            num_chunks=10,
            ingest_time_s=1.2,
            avg_retrieval_latency_ms=15.0,
            avg_generation_latency_ms=300.0,
            avg_workflow_latency_ms=315.0,
            avg_page_recall=0.8,
        ),
        BenchmarkResult(
            chunk_size=400,
            embedding_model="fake-model",
            num_chunks=6,
            ingest_time_s=1.0,
            avg_retrieval_latency_ms=12.0,
            avg_generation_latency_ms=280.0,
            avg_workflow_latency_ms=292.0,
            avg_page_recall=0.85,
        ),
    ]


def test_find_pdfs_raises_when_directory_has_no_pdfs():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            _find_pdfs(tmpdir)


def test_find_pdfs_finds_pdfs_sorted():
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "b.pdf"), "w").close()
        open(os.path.join(tmpdir, "a.pdf"), "w").close()
        open(os.path.join(tmpdir, "notes.txt"), "w").close()

        pdfs = _find_pdfs(tmpdir)
        assert [p.name for p in pdfs] == ["a.pdf", "b.pdf"]


def test_write_results_csv_writes_all_rows_and_columns():
    results = _sample_results()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "nested", "results.csv")
        write_results_csv(results, out_path)

        with open(out_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        assert rows[0]["chunk_size"] == "200"
        assert rows[1]["chunk_size"] == "400"
        assert "avg_page_recall" in rows[0]


def test_write_results_csv_handles_empty_results_without_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "empty.csv")
        write_results_csv([], out_path)
        assert os.path.exists(out_path)


def test_plot_results_returns_false_when_matplotlib_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot" or name.startswith("matplotlib"):
            raise ImportError("simulated missing matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with tempfile.TemporaryDirectory() as tmpdir:
        ok = plot_results(_sample_results(), "chunk_size", ["avg_retrieval_latency_ms"], f"{tmpdir}/plot.png")
        assert ok is False


def test_plot_results_creates_file_when_matplotlib_available():
    pytest.importorskip("matplotlib")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = f"{tmpdir}/plots/chunk_vs_latency.png"
        ok = plot_results(_sample_results(), "chunk_size", ["avg_retrieval_latency_ms"], out_path)
        assert ok is True
        assert os.path.exists(out_path)


def test_zero_ragas_references_in_benchmark_module():
    """Regression guard: benchmark.py must not call compute_ragas_metrics()
    or reference ragas in any way -- it measures engineering metrics only."""
    import benchmark

    source = open(benchmark.__file__, encoding="utf-8").read()
    assert "ragas" not in source.lower()
