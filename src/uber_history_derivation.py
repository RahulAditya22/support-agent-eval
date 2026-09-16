"""Derive retrieval-corpus provenance from raw Customer Support on Twitter rows."""

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


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"inbound must be boolean-like, got {value!r}")


def _parse_ids(value: object) -> tuple[str, ...]:
    if _is_missing(value):
        return ()
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def derive_uber_history(
    records: Iterable[Mapping[str, object]],
    *,
    brand_handle: str = "Uber_Support",
) -> tuple[DerivedReply, ...]:
    """Derive candidate resolved Uber replies from raw TWCS rows.

    Author rule: ``inbound == False`` and ``author_id == brand_handle``.

    Resolution rule: reconstruct each root thread through parent/child edges.
    A thread is heuristically resolved only when every terminal message is an
    Uber outbound message. Missing graph context or a customer terminal keeps
    the thread out of the retrieval corpus.
    """
    rows = list(records)
    by_id: dict[str, Mapping[str, object]] = {}

    for row in rows:
        missing = RAW_FIELDS - set(row)
        if missing:
            raise ValueError("raw TWCS rows require fields: " + ", ".join(sorted(missing)))
        tweet_id = str(row["tweet_id"]).strip()
        if not tweet_id:
            raise ValueError("tweet_id must be non-empty")
        by_id[tweet_id] = row

    children: dict[str, set[str]] = {tweet_id: set() for tweet_id in by_id}
    roots: list[str] = []
    missing_child_context: set[str] = set()

    for tweet_id, row in by_id.items():
        parent = row["in_response_to_tweet_id"]
        if _is_missing(parent):
            roots.append(tweet_id)
        else:
            parent_id = str(parent).strip()
            if parent_id in by_id:
                children[parent_id].add(tweet_id)

    # Validate response references only when they point to an ID that is
    # present in the sampled corpus's graph. A response ID may legitimately
    # belong to a tweet outside the selected root's connected component.
    for tweet_id, row in by_id.items():
        for child_id in _parse_ids(row["response_tweet_id"]):
            if child_id not in by_id:
                missing_child_context.add(tweet_id)

    derived: list[DerivedReply] = []

    for root_id in roots:
        root = by_id[root_id]
        root_text = str(root["text"] or "")
        if not _as_bool(root["inbound"]) or "@uber_support" not in root_text.lower():
            continue

        stack = [root_id]
        thread_ids: set[str] = set()
        while stack:
            current = stack.pop()
            if current in thread_ids:
                continue
            thread_ids.add(current)
            stack.extend(children.get(current, ()))

        if not thread_ids:
            continue

        # A dangling response reference in a member of this root thread is
        # still treated as incomplete, preserving the conservative synthetic
        # and real-corpus provenance rules.
        if thread_ids & missing_child_context:
            continue

        terminals = [tweet_id for tweet_id in thread_ids if not children.get(tweet_id)]
        if not terminals:
            continue

        complete = all(
            not _as_bool(by_id[tweet_id]["inbound"])
            and str(by_id[tweet_id]["author_id"]).strip() == brand_handle
            for tweet_id in terminals
        )
        if not complete:
            continue

        for tweet_id in sorted(thread_ids):
            row = by_id[tweet_id]
            if not _as_bool(row["inbound"]) and str(row["author_id"]).strip() == brand_handle:
                text = str(row["text"] or "").strip()
                if text:
                    derived.append(
                        DerivedReply(
                            text=text,
                            thread_id=root_id,
                            author_type="agent",
                            thread_status="resolved",
                            tweet_id=tweet_id,
                        )
                    )

    return tuple(derived)
