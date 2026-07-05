"""Retriever interface.

Defining this now -- before PHASE 2 -- means hybrid search, BM25, and
cross-encoder re-ranking can all be implemented as new classes that
satisfy this same interface, without touching `graph.py` or `nodes.py`.

PHASE 2 diagnostics: every retriever also supports `retrieve_with_diagnostics`,
returning a `RetrievedChunk` per document instead of a bare `Document`.
This is what powers the retrieval debug dashboard -- a `RetrievedChunk`
carries whichever per-stage scores the retriever pipeline computed
(semantic similarity, BM25, RRF fusion, cross-encoder), so the whole
pipeline

    retrieve -> semantic score -> BM25 score -> RRF score -> rerank score

is inspectable without changing what `retrieve()` returns to the
LangGraph workflow (which only ever needs plain `Document` objects).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document


@dataclass
class RetrievedChunk:
    """A retrieved document plus whatever per-stage scores were computed.

    Any score field left as `None` simply means that stage didn't run
    for this retriever pipeline (e.g. `bm25_score` is `None` when
    hybrid retrieval is disabled).
    """

    document: Document
    final_rank: int
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class BaseRetriever(ABC):
    """Common interface every retriever implementation must satisfy.

    `k` is optional on `retrieve()` so composite retrievers (hybrid fusion,
    cross-encoder re-ranking) can ask an inner retriever for a *candidate*
    pool larger than its default result size, without needing a second
    constructor argument or a new instance.
    """

    @abstractmethod
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Return the documents relevant to `query`.

        Args:
            query: The user query (or rewritten query).
            k: Optional override for the number of results to return.
                If omitted, the retriever's own configured default is used.
        """
        raise NotImplementedError

    def retrieve_with_diagnostics(self, query: str, k: Optional[int] = None) -> List[RetrievedChunk]:
        """Return documents wrapped with per-stage retrieval scores.

        Default implementation: no retriever-specific scores are known,
        so this just wraps `retrieve()`'s output with a final rank.
        Concrete retrievers override this to populate whichever score(s)
        they compute (see `SemanticRetriever`, `BM25Retriever`,
        `HybridRetriever`, `CrossEncoderReranker`).
        """
        documents = self.retrieve(query, k)
        return [RetrievedChunk(document=doc, final_rank=i) for i, doc in enumerate(documents)]
