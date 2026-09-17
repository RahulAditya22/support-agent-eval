"""Build validated historical replies from reconstructed Uber support rows.

The adapter deliberately requires explicit source metadata for author and thread
status. It does not guess whether a thread is resolved from message text or
reply position.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from historical_replies import HistoricalReply, validate_historical_replies


REQUIRED_FIELDS = (
    "text",
    "thread_id",
    "author_type",
    "thread_status",
)


def build_historical_replies(
    records: Iterable[Mapping[str, Any]],
) -> tuple[HistoricalReply, ...]:
    """Convert explicitly annotated reconstructed rows into validated replies.

    Rows without explicit source metadata are rejected rather than inferred.
    Only agent-authored records belonging to resolved/completed threads survive
    validation; customer messages and unresolved threads are excluded.
    """
    candidates: list[HistoricalReply] = []

    for record in records:
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError(
                "historical-reply records require explicit fields: "
                + ", ".join(missing)
            )

        candidate = HistoricalReply(
            text=str(record["text"]),
            thread_id=str(record["thread_id"]),
            author_type=str(record["author_type"]),
            thread_status=str(record["thread_status"]),
        )

        if candidate.author_type != "agent":
            continue
        if candidate.thread_status not in {"resolved", "completed"}:
            continue

        candidates.append(candidate)

    return validate_historical_replies(candidates)
