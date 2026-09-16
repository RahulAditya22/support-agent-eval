import pytest

from uber_history_adapter import build_historical_replies


def test_selects_only_agent_replies_from_resolved_threads():
    records = [
        {
            "text": "Please check your trip receipt.",
            "thread_id": "thread-1",
            "author_type": "agent",
            "thread_status": "resolved",
        },
        {
            "text": "Why did you charge me?",
            "thread_id": "thread-1",
            "author_type": "customer",
            "thread_status": "resolved",
        },
        {
            "text": "We are still investigating.",
            "thread_id": "thread-2",
            "author_type": "agent",
            "thread_status": "open",
        },
    ]

    result = build_historical_replies(records)

    assert len(result) == 1
    assert result[0].text == "Please check your trip receipt."
    assert result[0].thread_id == "thread-1"


def test_accepts_completed_agent_thread():
    records = [
        {
            "text": "We reviewed the issue.",
            "thread_id": "thread-3",
            "author_type": "agent",
            "thread_status": "completed",
        }
    ]

    assert build_historical_replies(records)[0].thread_status == "completed"


def test_rejects_records_without_explicit_source_metadata():
    records = [
        {
            "text": "Please check your trip receipt.",
            "thread_id": "thread-4",
        }
    ]

    with pytest.raises(ValueError, match="explicit fields"):
        build_historical_replies(records)


def test_rejects_when_no_valid_historical_reply_survives():
    records = [
        {
            "text": "Customer message",
            "thread_id": "thread-5",
            "author_type": "customer",
            "thread_status": "resolved",
        }
    ]

    with pytest.raises(ValueError, match="at least one"):
        build_historical_replies(records)
