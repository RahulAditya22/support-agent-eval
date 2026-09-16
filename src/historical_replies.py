"""Validated data contract for historical resolved support replies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class HistoricalReply:
    """Agent-authored outbound reply from a resolved/completed thread."""

    text: str
    thread_id: str
    author_type: str
    thread_status: str


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
