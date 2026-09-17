import pytest

from historical_reply_builder import build_historical_replies


def test_selects_only_agent_replies_from_resolved_or_completed_threads():
    rows = [
        {
            "text": "Agent fare guidance.",
            "thread_id": "resolved-1",
            "author_type": "agent",
            "thread_status": "resolved",
        },
        {
            "text": "Customer complaint.",
            "thread_id": "resolved-2",
            "author_type": "customer",
            "thread_status": "resolved",
        },
        {
            "text": "Open agent reply.",
            "thread_id": "open-1",
            "author_type": "agent",
            "thread_status": "open",
        },
        {
            "text": "Completed agent guidance.",
            "thread_id": "completed-1",
            "author_type": "agent",
            "thread_status": "completed",
        },
    ]

    result = build_historical_replies(rows)

    assert [item.text for item in result] == [
        "Agent fare guidance.",
        "Completed agent guidance.",
    ]
    assert all(item.author_type == "agent" for item in result)
    assert all(item.thread_status in {"resolved", "completed"} for item in result)


def test_blank_text_and_thread_id_are_excluded():
    rows = [
        {
            "text": "   ",
            "thread_id": "thread-1",
            "author_type": "agent",
            "thread_status": "resolved",
        },
        {
            "text": "Agent reply",
            "thread_id": "   ",
            "author_type": "agent",
            "thread_status": "resolved",
        },
        {
            "text": "Valid reply",
            "thread_id": "thread-3",
            "author_type": "agent",
            "thread_status": "resolved",
        },
    ]

    result = build_historical_replies(rows)

    assert [item.text for item in result] == ["Valid reply"]


def test_all_filtered_rows_produce_validation_error():
    rows = [
        {
            "text": "Customer message",
            "thread_id": "thread-1",
            "author_type": "customer",
            "thread_status": "resolved",
        }
    ]

    with pytest.raises(ValueError, match="at least one"):
        build_historical_replies(rows)


def test_missing_required_field_is_rejected():
    rows = [
        {
            "text": "Agent reply",
            "thread_id": "thread-1",
            "author_type": "agent",
        }
    ]

    with pytest.raises(ValueError, match="thread_status"):
        build_historical_replies(rows)
