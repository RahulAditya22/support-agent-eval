"""Build validated historical-reply records from reconstructed support rows."""

from __future__ import annotations

from typing import Iterable, Mapping

from historical_replies import HistoricalReply, validate_historical_replies


def build_historical_replies(rows: Iterable[Mapping[str, object]]) -> tuple[HistoricalReply, ...]:
    """Select only agent-authored replies tied to explicitly resolved threads.

    Required row fields are ``text``, ``thread_id``, ``author_type``, and
    ``thread_status``. Rows that are not agent-authored or do not belong to a
    resolved/completed thread are excluded before validation.
    """
    candidates: list[HistoricalReply] = []

    for row in rows:
        try:
            text = str(row["text"])
            thread_id = str(row["thread_id"])
            author_type = str(row["author_type"])
            thread_status = str(row["thread_status"])
        except KeyError as exc:
            raise ValueError(f"missing required historical-reply field: {exc.args[0]}") from exc

        if author_type != "agent":
            continue
        if thread_status not in {"resolved", "completed"}:
            continue
        if not text.strip() or not thread_id.strip():
            continue

        candidates.append(
            HistoricalReply(
                text=text,
                thread_id=thread_id,
                author_type=author_type,
                thread_status=thread_status,
            )
        )

    return validate_historical_replies(candidates)
