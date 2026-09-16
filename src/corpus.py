"""Build a provenance-validated historical reply corpus for retrieval."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from historical_replies import HistoricalReply
from retrieve import ReplyRetriever
from uber_history_adapter import build_historical_replies


def build_reply_retriever(
    embedder,
    records: Iterable[Mapping[str, Any]],
) -> tuple[ReplyRetriever, tuple[HistoricalReply, ...]]:
    """Validate source records before constructing the FAISS retrieval corpus."""
    historical_replies = build_historical_replies(records)
    retriever = ReplyRetriever(
        embedder,
        [reply.text for reply in historical_replies],
    )
    return retriever, historical_replies
