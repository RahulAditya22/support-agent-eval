"""Validated data contract and selection logic for historical resolved replies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Any


@dataclass(frozen=True)
class HistoricalReply:
    """Agent-authored outbound reply from a resolved/completed thread."""

    text: str
    thread_id: str
    author_type: str
    thread_status: str


def select_historical_replies(
    records: Iterable[Mapping[str, Any]],
) -> tuple[HistoricalReply, ...]:
    """Select only agent-authored replies attached to resolved/completed threads.

    The upstream data adapter must provide explicit ``author_type`` and
    ``thread_status`` fields. This function does not infer resolution from
    message text or reply position.
    """
    selected: list[HistoricalReply] = []

    for record in records:
        reply = HistoricalReply(
            text=str(record.get("text", "")),
            thread_id=str(record.get("thread_id", "")),
            author_type=str(record.get("author_type", "")),
            thread_status=str(record.get("thread_status", "")),
        )

        if reply.author_type != "agent":
            continue
        if reply.thread_status not in {"resolved", "completed"}:
            continue
        if not reply.text.strip() or not reply.thread_id.strip():
            continue

        selected.append(reply)

    if not selected:
        raise ValueError("no valid resolved agent replies were selected")

    return tuple(selected)


def validate_historical_replies(
    replies: Iterable[HistoricalReply],
) -> tuple[HistoricalReply, ...]:
    """Validate that only agent-authored replies from resolved threads are usable."""
    validated: list[HistoricalReply] = []

    for reply in replies:
        if not isinstance(reply, HistoricalReply):
            raise TypeError("all replies must be HistoricalReply instances")
        if not reply.text.strip():
            raise ValueError("historical reply text must be non-empty")
        if not reply.thread_id.strip():
            raise ValueError("thread_id must be non-empty")
        if reply.author_type != "agent":
            raise ValueError("historical reply must be agent-authored")
        if reply.thread_status not in {"resolved", "completed"}:
            raise ValueError("historical reply must belong to a resolved/completed thread")
        validated.append(reply)

    if not validated:
        raise ValueError("at least one valid historical reply is required")

    return tuple(validated)
