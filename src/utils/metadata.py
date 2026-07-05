"""Metadata sanitization for vector store insertion.

Single responsibility: guarantee that a metadata dict only contains
values Chroma actually accepts (`str`, `int`, `float`, `bool`) before it
reaches `vectorstore.add_documents()` / `Chroma.from_documents()`.

Why this exists: Chroma raises

    ValueError: Expected metadata value to be a str, int, float or bool, got None

for *any* metadata value outside that set -- most commonly `None`
(e.g. `section` in `src/ingestion/chunking.py` is legitimately `None`
when no heading was detected on a page), but the same failure applies to
nested dicts, lists, and arbitrary objects. Since this raises during
`add_documents()`, a single bad chunk aborts the whole batch -- as seen
when 129 chunks were built but zero made it into the vector store.
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Chroma's own accepted metadata value types.
_ALLOWED_TYPES = (str, int, float, bool)


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `metadata` containing only Chroma-compatible values.

    Rules:
        - `None` values are dropped (the key is omitted entirely, not
          replaced with a placeholder -- an absent field is easier to
          reason about downstream than a stringified "None").
        - `str` / `int` / `float` / `bool` values are kept as-is.
        - `list` / `tuple` / `set` of otherwise-valid primitives are
          flattened into a single comma-separated string, so information
          like `tags=["a", "b"]` survives as `tags="a, b"` instead of
          being silently dropped. An empty or entirely-unsupported
          sequence is dropped like any other unsupported value.
        - Anything else (dicts, nested objects, custom classes, etc.) is
          dropped.

    This never raises -- worst case for a given key is that it's
    omitted from the sanitized result.
    """
    sanitized: Dict[str, Any] = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, _ALLOWED_TYPES):
            sanitized[key] = value
            continue

        if isinstance(value, (list, tuple, set)):
            primitives = [str(v) for v in value if isinstance(v, _ALLOWED_TYPES)]
            if primitives:
                sanitized[key] = ", ".join(primitives)
            continue

        # dicts, custom objects, etc. -- Chroma has no representation for
        # these, so they're dropped rather than causing an insertion failure.

    return sanitized


def sanitize_documents_metadata(documents: List) -> List:
    """Sanitize `.metadata` on every document in `documents`, in place.

    Applied immediately before every Chroma insertion point
    (`create_vectorstore` and `add_documents` in
    `src/indexing/vectorstore.py`) so no call site can forget it and no
    upstream document source (ingestion, benchmarking, future loaders)
    needs to independently guarantee clean metadata.

    Returns the same list (mutated in place) for convenient chaining.
    """
    dropped_key_count = 0
    affected_doc_count = 0

    for document in documents:
        original_keys = set(document.metadata.keys())
        document.metadata = sanitize_metadata(document.metadata)
        removed = original_keys - document.metadata.keys()
        if removed:
            dropped_key_count += len(removed)
            affected_doc_count += 1

    if dropped_key_count:
        logger.info(
            "Sanitized metadata on %d document(s): dropped %d unsupported/None key(s) total.",
            affected_doc_count,
            dropped_key_count,
        )

    return documents
