"""Derive retrieval-corpus provenance from raw Customer Support on Twitter rows.

This module intentionally uses a narrow, deterministic heuristic because the
TWCS dataset exposes conversation structure and the ``inbound`` company/customer
indicator, but no explicit ``resolved`` field.  It is a corpus-construction
heuristic, not ground-truth resolution labeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from historical_replies import HistoricalReply


RAW_FIELDS = {
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
}


@dataclass(frozen=True)
class DerivedReply:
    """An agent-authored reply retained by the narrow resolution heuristic."""

    text: str
    thread_id: str
    author_type: str
    thread_status: str
    tweet_id: str

    def as_historical_reply(self) -> HistoricalReply:
        return HistoricalReply(
            text=self.text,
            thread_id=self.thread_id,
            author_type=self.author_type,
            thread_status=self.thread_status,
        )


def _is_missing(value: object) -> bool:
    return value is None or str(value).strip() in {"", "nan", "None"}


def _parse_ids(value: object) -> tuple[str, ...]:
    if _is_missing(value):
        return ()
    return tuple(
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    )


def derive_uber_history(
    records: Iterable[Mapping[str, object]],
    *,
    brand_handle: str = "Uber_Support",
) -> tuple[DerivedReply, ...]:
    """Derive candidate resolved Uber replies from raw TWCS rows.

    Author rule: ``inbound == False`` and ``author_id == brand_handle``.

    Resolution rule: reconstruct each thread from ``tweet_id`` /
    ``in_response_to_tweet_id`` / ``response_tweet_id``.  A thread is marked
    ``resolved`` only when every terminal message in the reconstructed thread
    is an Uber outbound message.  If any terminal message is customer-authored,
    or the graph is incomplete/ambiguous, the thread is treated as unresolved
    and contributes no retrieval replies.

    This is deliberately conservative.  It does not use text semantics or an
    LLM to decide whether a customer was actually satisfied.
    """
    rows = list(records)
    by_id: dict[str, Mapping[str, object]] = {}

    for row in rows:
        missing = RAW_FIELDS - set(row)
        if missing:
            raise ValueError(
                "raw TWCS rows require fields: " + ", ".join(sorted(missing))
            )
        tweet_id = str(row["tweet_id"]).strip()
        if not tweet_id:
            raise ValueError("tweet_id must be non-empty")
        by_id[tweet_id] = row

    children: dict[str, set[str]] = {tweet_id: set() for tweet_id in by_id}
    roots: list[str] = []

    for tweet_id, row in by_id.items():
        parent = row["in_response_to_tweet_id"]
        if _is_missing(parent):
            roots.append(tweet_id)
            continue
        parent_id = str(parent).strip()
        if parent_id not in by_id:
            # The reconstructed sample is incomplete around this row, so do
            # not make a resolution claim for its thread.
            continue
        children[parent_id].add(tweet_id)

    derived: list[DerivedReply] = []
    seen_threads: set[str] = set()

    for root_id in roots:
        root = by_id[root_id]
        text = str(root["text"] or "")
        if bool(root["inbound"]) is not True or "@uber_support" not in text.lower():
            continue

        stack = [root_id]
        thread_ids: set[str] = set()
        while stack:
            current = stack.pop()
            if current in thread_ids:
                continue
            thread_ids.add(current)
            stack.extend(children.get(current, ()))

        if root_id in seen_threads:
            continue
        seen_threads.add(root_id)

        terminals = [tweet_id for tweet_id in thread_ids if not children.get(tweet_id)]
        if not terminals:
            continue

        terminal_rows = [by_id[tweet_id] for tweet_id in terminals]
        complete = all(
            bool(row["inbound"]) is False
            and str(row["author_id"]).strip() == brand_handle
            for row in terminal_rows
        )
        if not complete:
            continue

        for tweet_id in sorted(thread_ids):
            row = by_id[tweet_id]
            if bool(row["inbound"]) is False and str(row["author_id"]).strip() == brand_handle:
                derived.append(
                    DerivedReply(
                        text=str(row["text"] or "").strip(),
                        thread_id=root_id,
                        author_type="agent",
                        thread_status="resolved",
                        tweet_id=tweet_id,
                    )
                )

    return tuple(reply for reply in derived if reply.text)
