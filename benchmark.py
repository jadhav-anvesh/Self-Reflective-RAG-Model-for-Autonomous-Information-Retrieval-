"""Benchmark different chunking/embedding configurations.

Answers the question "why chunk_size=250, why BAAI/bge-small-en-v1.5?"
with numbers instead of intuition. For each configuration in the grid,
this script:

    1. Re-ingests every PDF in data/raw/ into an isolated, throwaway
       Chroma collection (never touches your real data/vectorstore/).
    2. Builds a retriever over it and runs the eval question set
       (data/eval/questions.json by default) through retrieval + generation.
    3. Records ingestion time, retrieval latency, generation latency,
       workflow latency, and page recall -- all measured directly, no
       LLM judge involved.
    4. Writes a CSV per grid dimension to results/ and a comparison plot
       to results/plots/ (skipped gracefully if matplotlib isn't installed).

Usage:
    python benchmark.py --chunk-sizes 200 400 800
    python benchmark.py --embedding-models BAAI/bge-small-en-v1.5 intfloat/e5-base-v2

Runs completely offline: no judge LLM, no external evaluation
framework, no JSON-parsing step of any kind.

This intentionally does NOT run the full self-reflective LangGraph loop
(retrieval + grading + possible query rewrites) -- one retrieval + one
generation per question keeps benchmark runs fast and directly
comparable across configurations. "Workflow Latency" here is therefore
just retrieval + generation latency added together (single pass, no
retries), distinct from `evaluate.py`'s workflow latency, which times
the full self-reflective loop.
"""
from __future__ import annotations

import argparse
import gc
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from config import config
from src.evaluation.dataset_loader import load_eval_questions
from src.evaluation.metrics import compute_page_recall
from src.ingestion.chunking import split_pages_into_documents
from src.ingestion.pdf_loader import extract_pages_from_pdf
from src.retrieval.retriever import get_retriever
from src.utils.logging_config import get_logger
from src.utils.metadata import sanitize_documents_metadata

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    chunk_size: int
    embedding_model: str
    num_chunks: int
    ingest_time_s: float
    avg_retrieval_latency_ms: float
    avg_generation_latency_ms: float
    avg_workflow_latency_ms: float
    avg_page_recall: Optional[float] = None


def _find_pdfs(raw_data_dir: str) -> List[Path]:
    pdfs = sorted(Path(raw_data_dir).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in '{raw_data_dir}' to benchmark against.")
    return pdfs


def _release_chroma_resources(vectorstore) -> None:
    """Best-effort attempt to explicitly stop a Chroma client's underlying system.

    LangChain's `Chroma` wrapper doesn't expose an official `close()`
    across versions, so this tries the internal handle known to exist in
    the pinned `chromadb` version defensively. This alone is NOT enough
    to release the SQLite file handle -- the caller must also drop its
    own local references (e.g. `vectorstore`, `retriever`) and call
    `gc.collect()` afterward, since this function only holds a parameter
    reference and can't un-reference the caller's variables. See the
    `finally` block in `run_config()` for the full sequence.

    Never raises -- this is cleanup, not a step that should fail a
    benchmark run.
    """
    if vectorstore is None:
        return
    try:
        client = getattr(vectorstore, "_client", None)
        system = getattr(client, "_system", None) if client is not None else None
        if system is not None and hasattr(system, "stop"):
            system.stop()
    except Exception as e:  # defensive: this is best-effort cleanup, never fatal
        logger.debug("Could not explicitly stop the Chroma client (%s); relying on gc.collect() instead.", e)


def run_config(
    pdf_paths: List[Path],
    chunk_size: int,
    embedding_model: str,
    questions: List[dict],
) -> BenchmarkResult:
    """Run one (chunk_size, embedding_model) configuration end-to-end.

    Builds a fresh, isolated vector store in a temp directory so
    benchmarking never mutates the real `data/vectorstore/` index.
    """
    from datetime import datetime, timezone

    from langchain_huggingface import HuggingFaceEmbeddings
    from src.workflow.prompts import rag_chain

    logger.info("Benchmarking chunk_size=%d, embedding_model=%s", chunk_size, embedding_model)

    with tempfile.TemporaryDirectory(prefix="rag-benchmark-") as tmpdir:
        vectorstore = None
        retriever = None
        try:
            # --- Ingest ------------------------------------------------------
            ingest_start = time.perf_counter()
            all_chunks = []
            for pdf_path in pdf_paths:
                pages = extract_pages_from_pdf(str(pdf_path))
                chunks = split_pages_into_documents(
                    pages,
                    document_id=pdf_path.stem,
                    source=pdf_path.name,
                    filename=pdf_path.name,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    chunk_size=chunk_size,
                    chunk_overlap=config.chunk_overlap,
                )
                all_chunks.extend(chunks)

            # Reuse the exact same metadata sanitization the production
            # ingestion pipeline applies (src/indexing/vectorstore.py) --
            # chunks can carry None values (e.g. `section` when no heading
            # was detected), which Chroma rejects outright. Duplicating a
            # second copy of that cleaning logic here would be exactly the
            # kind of drift that lets one path silently diverge from the
            # other, so this calls the same shared helper instead.
            all_chunks = sanitize_documents_metadata(all_chunks)

            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": config.embedding_device},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
            )
            # Built directly (not via src.indexing.vectorstore.create_vectorstore)
            # so each grid point actually uses ITS OWN embedding model -- routing
            # through create_vectorstore() would silently re-embed with
            # config.embedding_model regardless of what this loop is testing.
            from langchain_community.vectorstores import Chroma

            vectorstore = Chroma.from_documents(
                documents=all_chunks,
                collection_name=f"bench-{chunk_size}-{abs(hash(embedding_model))}",
                embedding=embeddings,
                persist_directory=tmpdir,
            )
            ingest_time_s = time.perf_counter() - ingest_start

            # --- Retrieval + generation per question --------------------------
            retriever = get_retriever(vectorstore)
            retrieval_latencies_ms = []
            generation_latencies_ms = []
            page_recalls = []

            for item in questions:
                question = item["question"]
                expected_pages = item.get("expected_pages")

                t0 = time.perf_counter()
                retrieved_chunks = retriever.retrieve_with_diagnostics(question)
                retrieval_latencies_ms.append((time.perf_counter() - t0) * 1000)

                contexts = [chunk.document.page_content for chunk in retrieved_chunks]
                retrieved_pages = [chunk.document.metadata.get("page") for chunk in retrieved_chunks]
                page_recall = compute_page_recall(retrieved_pages, expected_pages)
                if page_recall is not None:
                    page_recalls.append(page_recall)

                t0 = time.perf_counter()
                rag_chain.invoke({"context": "\n\n".join(contexts), "question": question})
                generation_latencies_ms.append((time.perf_counter() - t0) * 1000)

            avg_retrieval_latency_ms = sum(retrieval_latencies_ms) / len(retrieval_latencies_ms)
            avg_generation_latency_ms = sum(generation_latencies_ms) / len(generation_latencies_ms)

            return BenchmarkResult(
                chunk_size=chunk_size,
                embedding_model=embedding_model,
                num_chunks=len(all_chunks),
                ingest_time_s=ingest_time_s,
                avg_retrieval_latency_ms=avg_retrieval_latency_ms,
                avg_generation_latency_ms=avg_generation_latency_ms,
                avg_workflow_latency_ms=avg_retrieval_latency_ms + avg_generation_latency_ms,
                avg_page_recall=(sum(page_recalls) / len(page_recalls)) if page_recalls else None,
            )
        finally:
            # The temp directory is deleted the instant this `with` block
            # exits. Chroma's SQLite backend holds an open connection to
            # `<tmpdir>/chroma.sqlite3` for as long as `vectorstore` (and
            # anything derived from it, e.g. `retriever`) is *referenced
            # from anywhere* -- which caused "PermissionError: chroma.sqlite3
            # is being used by another process" when cleanup ran before
            # those references were actually released. `_release_chroma_resources`
            # attempts an explicit stop; dropping this frame's own local
            # references and forcing gc.collect() here (not inside that
            # helper, which only holds a parameter binding it can't use to
            # un-reference *our* variables) is what actually lets the
            # connection's own cleanup run before the `with` block's rmtree.
            _release_chroma_resources(vectorstore)
            vectorstore = None
            retriever = None
            gc.collect()


def write_results_csv(results: List[BenchmarkResult], path: str) -> None:
    """Write benchmark results to CSV. Pure I/O, no model/embedding dependency."""
    import csv

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [asdict(r) for r in results]
    fieldnames = list(rows[0].keys()) if rows else []

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d benchmark result(s) to %s", len(results), out_path)


def plot_results(results: List[BenchmarkResult], x_field: str, y_fields: List[str], out_path: str) -> bool:
    """Plot `y_fields` against `x_field`. Returns False (no-op) if matplotlib is missing."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plot generation (pip install matplotlib)")
        return False

    x_values = [getattr(r, x_field) for r in results]

    fig, ax = plt.subplots()
    for y_field in y_fields:
        y_values = [getattr(r, y_field) for r in results]
        if all(v is None for v in y_values):
            continue
        ax.plot(x_values, y_values, marker="o", label=y_field)

    ax.set_xlabel(x_field)
    ax.legend()
    ax.set_title(f"{', '.join(y_fields)} vs {x_field}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved plot to %s", out)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark chunk sizes / embedding models for the RAG pipeline.")
    parser.add_argument("--chunk-sizes", type=int, nargs="*", default=[150, 200, 250, 300, 350, 400, 500, 600, 800])
    parser.add_argument("--embedding-models", type=str, nargs="*", default=[config.embedding_model])
    parser.add_argument("--questions", default="data/eval/questions.json")
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_paths = _find_pdfs(config.raw_data_dir)
    questions = load_eval_questions(args.questions)

    # --- Chunk size sweep (embedding model held fixed) -----------------------
    chunk_size_results = [
        run_config(pdf_paths, chunk_size=cs, embedding_model=args.embedding_models[0], questions=questions)
        for cs in args.chunk_sizes
    ]
    write_results_csv(chunk_size_results, f"{args.out_dir}/chunk_size_benchmark.csv")
    plot_results(
        chunk_size_results,
        x_field="chunk_size",
        y_fields=["avg_retrieval_latency_ms", "avg_generation_latency_ms", "avg_workflow_latency_ms"],
        out_path=f"{args.out_dir}/plots/chunk_size_vs_latency.png",
    )
    plot_results(
        chunk_size_results,
        x_field="chunk_size",
        y_fields=["avg_page_recall"],
        out_path=f"{args.out_dir}/plots/chunk_size_vs_page_recall.png",
    )
    plot_results(
        chunk_size_results,
        x_field="chunk_size",
        y_fields=["num_chunks"],
        out_path=f"{args.out_dir}/plots/chunk_size_vs_num_chunks.png",
    )
    plot_results(
        chunk_size_results,
        x_field="chunk_size",
        y_fields=["ingest_time_s"],
        out_path=f"{args.out_dir}/plots/chunk_size_vs_ingestion_time.png",
    )

    # --- Embedding model sweep (chunk size held fixed), only if >1 given ------
    if len(args.embedding_models) > 1:
        default_chunk_size = args.chunk_sizes[0]
        embedding_results = [
            run_config(pdf_paths, chunk_size=default_chunk_size, embedding_model=model, questions=questions)
            for model in args.embedding_models
        ]
        write_results_csv(embedding_results, f"{args.out_dir}/embedding_model_benchmark.csv")

    print(f"\nDone. Results written under {args.out_dir}/")


if __name__ == "__main__":
    main()
