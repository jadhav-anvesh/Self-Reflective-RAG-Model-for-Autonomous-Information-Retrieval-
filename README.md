# RAG Project

A self-reflective Retrieval-Augmented Generation pipeline (LangGraph +
Chroma + Ollama), built in phases with production-style engineering
practices: persistent vector indexing, incremental ingestion, an
interface-based retrieval layer, and centralized configuration.

## Architecture

```
                     Offline Ingestion Pipeline

           PDF Documents
                 │
                 ▼
         PDF Loader (PyMuPDF)
                 │
                 ▼
      Recursive Text Chunking
                 │
                 ▼
     HuggingFace Embeddings
                 │
                 ▼
      Persistent ChromaDB Index
       (incremental, SHA-256
        duplicate detection)


                  Online Query Pipeline

          User Question
                 │
                 ▼
      Hybrid Retrieval
   (Semantic + BM25 + RRF)
                 │
                 ▼
   Cross-Encoder Re-ranking
        (optional)
                 │
                 ▼
      LangGraph Workflow
  (Retrieve → Grade → Rewrite
     → Generate → Verify)
                 │
                 ▼
      Answer + Source Citations
                 │
                 ▼
   Evaluation / Debug Dashboard
```

## Features

- Local Llama 3.2 inference via Ollama -- fully offline, no API key
- HuggingFace embeddings (`BAAI/bge-small-en-v1.5`)
- Persistent ChromaDB indexing, separated from inference (`ingest.py` vs. `app.py`)
- Incremental ingestion -- only new/changed files get (re-)embedded
- Duplicate detection via SHA-256 content hashing (`data/manifest.json`)
- Hybrid Retrieval (Semantic + BM25)
- Reciprocal Rank Fusion (RRF)
- Optional Cross-Encoder Re-ranking
- Metadata-aware retrieval (page, section, filename)
- LangGraph self-reflective workflow (grade → rewrite → regenerate, with bounded retries)
- Source citations (page + filename, no PDF re-parsing needed)
- Retrieval Debug Dashboard (per-stage scores, latency, full chunk text)
- Deterministic Evaluation Dashboard (latency, page recall, retries -- no LLM judge)
- Benchmarking framework (chunk size / embedding model sweeps)

## Screenshots

**RAG Q&A** -- ask a question, get an answer with source citations:

![RAG Workflow Q&A](docs/screenshots/qna-page.png)

**Retrieval Debug Dashboard** -- trace exactly why a chunk was retrieved
(semantic / BM25 / RRF / re-rank scores), plus the full chunk text:

![Retrieval Debug Dashboard](docs/screenshots/retrieval-debug-dashboard.png)
![Retrieval Debug Dashboard -- full chunk text](docs/screenshots/retrieval-debug-chunk-detail.png)

**Evaluation Dashboard** -- deterministic metrics (latency, page recall,
retries, low-confidence rate) with no LLM judge involved:

![Evaluation Dashboard](docs/screenshots/evaluation-dashboard.png)

## Project layout

```
config.py                  Centralized, env-overridable configuration
ingest.py                  Offline ingestion entry point (PDF -> chunks -> embeddings -> Chroma)
main.py / app.py           Online entry points (CLI / Streamlit) -- load-only, never embed
evaluate.py                Deterministic evaluation CLI entry point (offline, independent of the app)
pages/                     Streamlit multipage nav (Retrieval Debug Dashboard, Evaluation Dashboard)
src/
  ingestion/                PDF loading, page-aware chunking, file hashing, manifest
  indexing/                 Chroma vector store creation/loading (embeddings live here only)
  retrieval/                Retriever interface + implementations (see below)
  workflow/                 LangGraph nodes/graph/prompts for the self-reflective RAG loop
  evaluation/                Dataset loading, real-workflow evaluation runner, deterministic metrics
  utils/                    Shared logging config
tests/                      Unit tests (pytest)
```

## Setup

```bash
pip install -r requirements.txt
python ingest.py        # builds the vector store from every PDF in data/raw/
streamlit run app.py    # or: python main.py
```

Add PDFs any time by dropping them into `data/raw/` and re-running
`python ingest.py` -- already-indexed files (tracked by SHA-256 content
hash in `data/manifest.json`) are skipped automatically, so only new or
changed files get (re-)embedded.

## Add Your Documents

This repository intentionally does not include any PDF documents --
to avoid distributing copyrighted or personal files. Bring your own
PDF(s) and place them here:

```
data/raw/
```

The expected layout looks like:

```
data/
├── raw/
│   └── your_document.pdf
├── eval/
└── vectorstore/   (generated automatically)
```

Once your PDF(s) are in place, build the vector database:

```bash
python ingest.py
```

`ingest.py` only embeds documents that are new or have changed since the
last run. Duplicate detection is based on a SHA-256 hash of each file's
content, tracked in `data/manifest.json`, so re-running it after adding
a single new PDF re-embeds only that file rather than the whole corpus.

Then start the application:

```bash
streamlit run app.py
```

`data/vectorstore/` (the Chroma index) and `data/manifest.json` are both
generated automatically by `ingest.py` and are intentionally excluded
from Git -- they're derived, machine-specific artifacts, not source.

## Retrieval architecture (PHASE 1 + PHASE 2)

Every retriever implements a single interface, `BaseRetriever.retrieve(query, k=None)`
(`src/retrieval/base.py`), so the LangGraph workflow (`src/workflow/`) never
needs to know which retrieval strategy is active.

```
SemanticRetriever  ──┐
 (Chroma / cosine)   ├──▶ HybridRetriever ──▶ [CrossEncoderReranker] ──▶ workflow
BM25Retriever ────────┘   (Reciprocal Rank      (optional)
 (lexical / keyword)         Fusion)
```

`src/retrieval/retriever.py` is the single factory (`get_retriever`) that
composes this pipeline based on `config.py`:

| Config field                 | Env var                     | Default | Effect |
|-------------------------------|------------------------------|---------|--------|
| `use_hybrid_retrieval`         | `RAG_USE_HYBRID_RETRIEVAL`   | `true`  | Combine semantic + BM25 via RRF instead of semantic-only |
| `hybrid_candidate_k`           | `RAG_HYBRID_CANDIDATE_K`     | `10`    | Candidates each of semantic/BM25 fetch before fusion |
| `rrf_k`                        | `RAG_RRF_K`                  | `60`    | RRF constant: `1 / (rrf_k + rank)` |
| `use_reranking`                | `RAG_USE_RERANKING`          | `false` | Add a cross-encoder re-ranking stage (optional, off by default) |
| `reranker_model`               | `RAG_RERANKER_MODEL`         | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `rerank_candidate_k`           | `RAG_RERANK_CANDIDATE_K`     | `20`    | Candidates pulled before re-ranking |
| `rerank_k`                     | `RAG_RERANK_K`               | `5`     | Final results returned after re-ranking |
| `retrieval_k`                  | `RAG_RETRIEVAL_K`            | `4`     | Final results returned when re-ranking is off |

**Why Reciprocal Rank Fusion (not score averaging):** BM25 scores and
cosine similarity live on incomparable scales, so naively averaging or
normalizing them is fragile. RRF instead fuses by *rank position*
(`sum of 1 / (rrf_k + rank)` across retrievers), which is scale-agnostic
and is what most production hybrid-search systems use.

**Why the cross-encoder is optional and lazy-loaded:** cross-encoders
score every `(query, candidate)` pair jointly, which is far more
accurate than bi-encoder/BM25 similarity alone but too slow to run over
a whole corpus. It's applied only as a final re-ranking pass over a
small candidate pool (`rerank_candidate_k`), and the model weights are
only loaded into memory the first time `config.use_reranking=True`
actually triggers a call -- enabling it costs nothing until it's used.

## Metadata-aware retrieval

Every chunk, from every retriever, carries the same metadata attached
during ingestion (`src/ingestion/chunking.py`):

```
document_id   SHA-256 hash of the source file (stable across re-ingests)
source        Human-readable source label (the filename)
filename      Original filename
page          1-indexed PDF page number
section       Best-effort heading detected on that page (heuristic; may be None)
chunk_id      Running counter, unique within the document
created_at    ISO timestamp of ingestion
```

This is what makes PHASE 3 (source citations, e.g. "Page 12, input.pdf")
a metadata lookup rather than a re-parse of the original PDF.

## Metadata sanitization (ChromaDB insertion fix)

Chroma raises `ValueError: Expected metadata value to be a str, int,
float or bool, got None` for any metadata value outside that set --
`section` in `src/ingestion/chunking.py` is legitimately `None` whenever
no heading was detected on a page, and that alone was enough to abort
the *entire* `add_documents()` batch (129 chunks built, zero stored).

`src/utils/metadata.py` provides `sanitize_metadata(dict) -> dict` and
`sanitize_documents_metadata(documents)`, applied automatically at both
Chroma insertion points in `src/indexing/vectorstore.py`
(`create_vectorstore` and `add_documents`) -- every document source
(ingestion, benchmarking, future loaders) is covered without needing to
remember to sanitize at the call site. Rules: `None` values are dropped,
`str`/`int`/`float`/`bool` pass through unchanged, lists/tuples/sets of
primitives are joined into a comma-separated string (so information like
`tags=["a","b"]` survives as `tags="a, b"` instead of being discarded),
and anything else (nested dicts, custom objects) is dropped.
LangChain's own `filter_complex_metadata` was evaluated but not used in
combination -- it only drops unsupported values outright with no attempt
to preserve list content, and `sanitize_metadata` is a strict superset of
what it does, so running both would be redundant.

## Source citations

Every answer comes with a de-duplicated list of `(filename, page[, section])`
citations (`src/workflow/citations.py`), built from the metadata already
attached at ingestion -- no re-parsing of the PDF needed. The `generate`
workflow node populates `state["sources"]`, and `app.py` renders it under
the answer as:

```
Answer: ...

Sources:
- input.pdf, Page 7
- input.pdf, Page 12 (Admissions Requirements)
```

## Retrieval debug dashboard

`pages/1_Retrieval_Debug_Dashboard.py` is a second Streamlit page
(auto-discovered by Streamlit's multipage nav when you run
`streamlit run app.py`) that traces a question through the retrieval
pipeline in isolation -- no LLM generation, just retrieval:

```
question -> semantic candidates -> BM25 candidates
         -> RRF fusion -> [cross-encoder re-ranking] -> final chunks
```

It shows a table of the final chunks with whichever per-stage scores
were computed (semantic similarity, BM25, RRF, cross-encoder), plus
retrieval latency and full chunk text. This is powered by
`BaseRetriever.retrieve_with_diagnostics()`, which every retriever
implements (`RetrievedChunk` in `src/retrieval/base.py`) -- the
LangGraph workflow still only ever calls the plain `.retrieve()`, so
none of this changes production behavior.

## Evaluation

A **deterministic, offline evaluation layer** -- it does not modify
retrieval, the LangGraph workflow, ChromaDB, or the Streamlit chat app.
It builds the exact same retriever and compiled workflow those use
(`get_retriever` + `create_workflow`) and runs each question through
them, measuring latency, chunk/citation counts, retries, and page
recall directly from the pipeline's own output.

**No LLM is used as a judge.** An earlier version of this project used
RAGAS (LLM-judged faithfulness/answer-relevancy/context-precision/recall),
but with a small local model (llama3.2 via Ollama) as the judge, RAGAS's
own judge calls regularly failed with `OutputParserException`s (the judge
not returning strict JSON), `TimeoutError`s, and NaN scores -- failures
inside the evaluation framework itself, not inside the RAG pipeline being
measured. This project now measures only things that can be counted or
timed, which a local model can't get "wrong": nothing here can produce a
parsing failure, a timeout, or a NaN. See "Why deterministic over
LLM-judged" below for the full reasoning.

```
src/evaluation/
    dataset_loader.py   loads + validates data/eval/questions.json
    evaluator.py         runs each question through the real retriever + compiled workflow
    metrics.py           computes page recall, builds the results table + summary
evaluate.py               CLI entry point
pages/2_Evaluation_Dashboard.py   optional Streamlit dashboard over the results
```

### 1. Create an evaluation dataset

Edit `data/eval/questions.json` -- a JSON list of question objects:

```json
[
  {
    "question": "What is the requirement for a Bachelor in Business Administration?",
    "ground_truth": null,
    "expected_pages": [8]
  }
]
```

- `question` -- required.
- `ground_truth` -- optional reference answer. Kept for schema
  compatibility and possible future use, but no metric in this
  framework currently consumes it (there's no LLM judge to compare
  against).
- `expected_pages` -- optional list of 1-indexed PDF page numbers the
  answer is expected to come from. Powers the deterministic **Page
  Recall** metric (see below).

### 2. Run it

```bash
python ingest.py                            # if you haven't already
python evaluate.py --questions data/eval/questions.json --out results/evaluation_results.csv
```

Runs entirely offline against the project's own local Ollama LLM and
HuggingFace embedding model -- no external API calls, no judge model,
no network dependency of any kind beyond what ingestion/generation
already need.

### 3. Sample output

```
Question 1: What font should assignments use?
  Retrieval Latency   42.3 ms
  Generation Latency  612.5 ms
  Workflow Latency    654.8 ms
  Retrieved Chunks    4 (4 reranked)
  Page Recall         1.000
  Retries             0 (query rewritten: False)
  Low Confidence      False
  Citations           1
----------------------------------------
Question 2: What is the requirement for a Bachelor in Business Administration?
  Retrieval Latency   47.1 ms
  Generation Latency  1188.4 ms
  Workflow Latency    1235.5 ms
  Retrieved Chunks    4 (4 reranked)
  Page Recall         1.000
  Retries             1 (query rewritten: False)
  Low Confidence      False
  Citations           2
----------------------------------------

=== Average metrics (across all scored questions) ===
Avg Retrieval Latency   44.7 ms
Avg Generation Latency  900.5 ms
Avg Workflow Latency    945.2 ms
Avg Page Recall         1.000 (over 2 question(s) with `expected_pages` set)
Retry Rate              50.0%
Low Confidence Rate     0.0%

Per-question results : results/evaluation_results.csv
Summary               : results/evaluation_summary.json
```

`results/evaluation_results.csv` has one row per question: Question,
Answer, Retrieved Pages, Expected Pages, Page Recall, Retrieved Chunks,
Chunks After Reranking, Retrieval/Generation/Workflow Latency, Answer
Length, Retries, Low Confidence, Query Rewritten, Sources.
`results/evaluation_summary.json` has the averaged metrics plus a run
manifest (timestamp, question count, LLM/embedding/retrieval config
used) -- useful for comparing runs after a retrieval or prompt change.

### 4. Metrics explained

| Metric | What it measures | How it's captured |
|---|---|---|
| **Retrieval Latency** | Time for the retriever alone (hybrid fusion + optional re-ranking) to return results. | Timed around a standalone `retriever.retrieve_with_diagnostics()` call. |
| **Generation Latency** | Approximate time spent generating (workflow latency minus the retrieval sample above). An estimate, not an exact instrumentation of the LangGraph nodes -- see `evaluator.py`'s docstring for why. | `workflow_latency - retrieval_latency`, floored at 0. |
| **Workflow Latency** | Total time for the full self-reflective loop (`app.invoke()`), including any retries/rewrites. | Timed around the compiled workflow's `.invoke()` call. |
| **Retrieved / Reranked Chunks** | How many chunks the retriever returned, and how many of those went through cross-encoder scoring. | From `RetrievedChunk.rerank_score is not None` on the retriever's diagnostics output. |
| **Page Recall** | `\|retrieved_pages ∩ expected_pages\| / \|expected_pages\|` for this question. | Only computed for questions with `expected_pages` set; `None` otherwise. |
| **Retries / Query Rewritten** | How many times the self-reflective loop retried generation or rewrote the query before accepting an answer. | From `generation_attempts`/`query_rewrite_attempts` in the workflow's final state. |
| **Low Confidence** | Whether the retry budget was exhausted before grading actually passed (see the workflow's `accept_best_effort` node). | From `low_confidence` in the workflow's final state. |
| **Citations / Duplicates Removed** | How many de-duplicated sources the answer cites, and how many raw document-level citations collapsed into those. | From `state["sources"]` vs. `state["documents"]`. |

### Why deterministic over LLM-judged

RAGAS-style metrics (faithfulness, answer relevancy, etc.) are valuable
*in principle*, but they require a second LLM call per metric per
question to actually judge the answer -- and that judge has to reliably
return structured (JSON) output. Hosted frontier models are decent at
this; a small local model run through Ollama, quantized for CPU/limited-
GPU inference, is not: it periodically ignores the requested output
format, runs slowly enough to time out under any real concurrency, and
when it *does* fail, RAGAS itself has no good way to distinguish "the
RAG pipeline gave a bad answer" from "the judge model didn't format its
opinion correctly." For a project whose whole premise is running fully
offline on local models, adding a *second* fragile local-LLM dependency
just to grade the first one is the wrong trade -- especially since
almost everything an LLM judge would tell you (is retrieval finding the
right pages? is the pipeline fast? is self-reflection kicking in often?)
is already directly measurable without asking a model's opinion at all.

### Optional: Streamlit dashboard

```bash
streamlit run app.py
```

Then open the **Evaluation Dashboard** page from the sidebar nav -- it
can either load the latest `results/evaluation_results.csv` /
`evaluation_summary.json`, or run evaluation live from the browser
(same underlying call as `python evaluate.py`). Shows metric cards
(average latencies, page recall, retry/low-confidence rates), latency/
page-recall/chunk-count charts, and the per-question table.

## Benchmarking (chunk size / embedding model sweeps)

```bash
python benchmark.py --chunk-sizes 200 400 800
python benchmark.py --embedding-models BAAI/bge-small-en-v1.5 intfloat/e5-base-v2
```

For each configuration, ingests into an isolated, throwaway Chroma
collection (your real `data/vectorstore/` is never touched), then
measures ingestion time, retrieval latency, generation latency, and
page recall (against `expected_pages` in the eval set) -- all
deterministic, no judge LLM involved. Results go to
`results/chunk_size_benchmark.csv` / `results/embedding_model_benchmark.csv`,
with comparison plots under `results/plots/` (skipped gracefully if
`matplotlib` isn't installed). Runs completely offline.

## Dependency notes (why these versions)

`requirements.txt` lists **direct dependencies only** -- transitive packages
(numpy, requests, huggingface_hub, torch's `nvidia-*` CUDA packages, etc.)
are intentionally left unpinned; pip resolves those automatically, and
pinning them adds maintenance burden without benefit.

| Package | Bound | Why |
|---|---|---|
| `langchain` / `langchain-community` | `==0.3.9` / `==0.3.8` exact | Released together; mismatched patch lines between them are a common source of subtle breakage. Kept exactly as the project was already built against. |
| `langchain-core`, `-text-splitters`, `-huggingface`, `-ollama` | floated within the `0.3.x` line compatible with the pins above | Each integration package tracks `langchain-core`'s interfaces; floating within the same minor line picks up patch fixes without risking the next breaking minor release. |
| `langgraph` | `>=0.2.53,<0.3` | Matches the `langchain-core 0.3.x` generation; `langgraph 0.3`+ tracks `langchain-core 0.4`/`1.x`, which isn't what's pinned here. |
| `chromadb` | `>=0.5.20,<0.6` | Contemporary with `langchain-community==0.3.8`'s `Chroma` wrapper; verified to resolve and import cleanly together. |
| `torch` / `transformers` / `sentence-transformers` | `2.5.x` / `4.46.x` / `3.3–3.x` | Mutually compatible generation that supports `CrossEncoder` (re-ranking) and `HuggingFaceEmbeddings` without requiring a CUDA toolkit newer than what's commonly available. |
| `pydantic` | `>=2.0,<3` | Used directly for the grading/routing schemas (`BaseModel`, `Field`) in `src/workflow/prompts.py`; `langchain-core 0.3.x` also requires Pydantic v2. |

**Removed:** `python-dotenv` was in the old (170+ line, `pip freeze`-style)
requirements file but isn't actually imported anywhere in the codebase --
`config.py` reads `os.environ` directly. Dropped as dead weight; add it
back (`pip install python-dotenv` + a `load_dotenv()` call at the top of
`config.py`) if you'd like `.env` file support. `ragas` and `datasets`
were removed entirely in a later revision -- see "Why deterministic over
LLM-judged" in the Evaluation section above.

**Also available:** `pyproject.toml` mirrors the same pinned dependencies
in the modern `[project.dependencies]` format, with `pytest` split into an
optional `dev` extra (`pip install -e ".[dev]"`). `requirements.txt` still
includes `pytest` directly so `pip install -r requirements.txt && pytest`
works standalone, per the deliverable checklist.

## Testing

```bash
pytest tests/ -q
```

Covers: file hashing, manifest/duplicate-detection, page-aware chunking,
config parsing, vector store load/create logic, all PHASE 2 retrieval
components (BM25, RRF fusion, cross-encoder re-ranking, retrieval
diagnostics, and the retriever factory's branching logic), source
citation formatting, and the deterministic evaluation/benchmark
modules (page recall, results-table building, summary averaging,
CSV writing, plotting, question-set loading) -- using lightweight
fakes throughout, so the suite needs no embedding model, live
Chroma/cross-encoder download, or Ollama instance to run.

## Roadmap

- ✅ PHASE 1 -- ingestion, persistent indexing, retriever abstraction, config
- ✅ PHASE 2 -- hybrid retrieval (BM25 + semantic via RRF), metadata-aware retrieval, optional cross-encoder re-ranking
- ✅ PHASE 3 -- source citations (page + filename), using the metadata already in place
- ✅ PHASE 4 -- retrieval debug dashboard (per-stage scores, latency, full chunk text)
- ✅ PHASE 5 -- deterministic evaluation (latency, chunk counts, page recall, retries -- no LLM judge)
- ✅ PHASE 6 -- benchmarking harness (chunk size / embedding model sweeps, CSV + plots)

What's intentionally *not* built (low ROI for this project's scope):
BM25 index persistence (fine at this corpus size), a `ChunkMetadata`
model in place of dicts, per-request latency breakdown beyond what the
debug dashboard already shows, agentic/graph RAG, fine-tuning, or a
custom vector DB / distributed deployment.