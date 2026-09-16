import csv
from pathlib import Path

import pytest

from uber_history_derivation import derive_uber_history


REQUIRED = {
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
}


def row(tweet_id, author_id, inbound, parent, responses="", text="message"):
    return {
        "tweet_id": str(tweet_id),
        "author_id": author_id,
        "inbound": inbound,
        "created_at": "Tue Oct 31 22:10:47 +0000 2017",
        "text": text,
        "response_tweet_id": responses,
        "in_response_to_tweet_id": parent,
    }


def test_marks_terminal_uber_agent_reply_as_resolved():
    records = [
        row("1", "101", True, "", "2", "@Uber_Support I need help"),
        row("2", "Uber_Support", False, "1", "", "We can help with that."),
    ]

    result = derive_uber_history(records)

    assert len(result) == 1
    assert result[0].tweet_id == "2"
    assert result[0].author_type == "agent"
    assert result[0].thread_status == "resolved"
    assert result[0].thread_id == "1"


def test_customer_terminal_message_keeps_thread_unresolved():
    records = [
        row("1", "101", True, "", "2", "@Uber_Support I need help"),
        row("2", "Uber_Support", False, "1", "3", "We can help with that."),
        row("3", "101", True, "2", "", "Thanks, but the issue remains."),
    ]

    assert derive_uber_history(records) == ()


def test_non_uber_outbound_is_not_agent_reply():
    records = [
        row("1", "101", True, "", "2", "@Uber_Support I need help"),
        row("2", "OtherSupport", False, "1", "", "Different company reply."),
    ]

    assert derive_uber_history(records) == ()


def test_missing_raw_field_is_rejected():
    records = [row("1", "101", True, "", "2")]
    records[0].pop("author_id")

    with pytest.raises(ValueError, match="raw TWCS rows require fields"):
        derive_uber_history(records)


def test_real_processed_uber_sample_runs_when_present():
    sample_path = Path("data/processed/uber_support_sample.csv")
    if not sample_path.exists():
        pytest.skip("local real Uber sample is not present")

    with sample_path.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))

    assert records
    assert REQUIRED <= set(records[0])

    for record in records:
        record["inbound"] = str(record["inbound"]).strip().lower() == "true"

    result = derive_uber_history(records)

    assert result, "real Uber sample should yield at least one heuristic-resolved reply"
    assert all(reply.author_type == "agent" for reply in result)
    assert all(reply.thread_status == "resolved" for reply in result)
    assert all(reply.text for reply in result)
