"""Build a provenance-validated historical reply corpus for retrieval."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

from historical_replies import HistoricalReply, validate_historical_replies
from retrieve import ReplyRetriever
from uber_history_adapter import build_historical_replies
from uber_history_derivation import derive_uber_history


def build_reply_retriever(
    embedder,
    records: Iterable[Mapping[str, Any]],
) -> tuple[ReplyRetriever, tuple[HistoricalReply, ...]]:
    """Validate explicitly annotated source records before retrieval."""
    historical_replies = build_historical_replies(records)
    retriever = ReplyRetriever(
        embedder,
        [reply.text for reply in historical_replies],
    )
    return retriever, historical_replies


def build_uber_reply_retriever(
    embedder,
    raw_csv_path: str | Path,
) -> tuple[ReplyRetriever, tuple[HistoricalReply, ...]]:
    """Build the Uber retrieval corpus directly from a TWCS CSV.

    The raw rows are passed through the documented narrow provenance heuristic
    before any FAISS index is constructed.  Threads that do not satisfy the
    terminal-agent rule are excluded rather than being treated as resolved.
    """
    path = Path(raw_csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    derived = derive_uber_history(rows)
    historical_replies = validate_historical_replies(
        reply.as_historical_reply() for reply in derived
    )
    retriever = ReplyRetriever(
        embedder,
        [reply.text for reply in historical_replies],
    )
    return retriever, historical_replies
