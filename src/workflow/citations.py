"""Source citation formatting.

Single responsibility: turn the `Document` chunks used to produce an
answer into a de-duplicated, human-readable list of sources (filename +
page + section). This is possible with zero re-parsing of the original
PDFs because `src/ingestion/chunking.py` already attaches this metadata
to every chunk at ingestion time.
"""
from __future__ import annotations

from typing import List, TypedDict

from langchain_core.documents import Document


class Citation(TypedDict):
    filename: str
    page: int
    section: str | None


def format_sources(documents: List[Document]) -> List[Citation]:
    """De-duplicate `documents` into a list of citable (filename, page) sources.

    Order is preserved (first-seen), and duplicates -- multiple chunks
    from the same page -- collapse into a single citation, since "Page 7"
    is what a user wants to see, not "Page 7" repeated three times.
    """
    seen = set()
    citations: List[Citation] = []
    for doc in documents:
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page")
        key = (filename, page)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(filename=filename, page=page, section=doc.metadata.get("section"))
        )
    return citations


def format_sources_as_text(documents: List[Document]) -> str:
    """Render citations as a short human-readable string for CLI/plain-text output.

    Example: "input.pdf (Page 7), input.pdf (Page 12, Admissions)"
    """
    citations = format_sources(documents)
    if not citations:
        return "No sources available."

    parts = []
    for c in citations:
        label = f"{c['filename']} (Page {c['page']}" if c["page"] is not None else f"{c['filename']} (page unknown"
        if c["section"]:
            label += f", {c['section']}"
        label += ")"
        parts.append(label)
    return ", ".join(parts)
