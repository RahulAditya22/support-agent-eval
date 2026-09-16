import pytest

from historical_replies import HistoricalReply, validate_historical_replies


def test_accepts_agent_reply_from_resolved_thread():
    reply = HistoricalReply(
        text="Please check your trip receipt.",
        thread_id="thread-1",
        author_type="agent",
        thread_status="resolved",
    )

    assert validate_historical_replies([reply]) == (reply,)


def test_accepts_agent_reply_from_completed_thread():
    reply = HistoricalReply(
        text="We can review the fare details.",
        thread_id="thread-2",
        author_type="agent",
        thread_status="completed",
    )

    assert validate_historical_replies([reply]) == (reply,)


def test_rejects_customer_authored_reply():
    reply = HistoricalReply(
        text="Why did you charge me?",
        thread_id="thread-3",
        author_type="customer",
        thread_status="resolved",
    )

    with pytest.raises(ValueError, match="agent-authored"):
        validate_historical_replies([reply])


def test_rejects_unresolved_thread():
    reply = HistoricalReply(
        text="Please investigate this fare.",
        thread_id="thread-4",
        author_type="agent",
        thread_status="open",
    )

    with pytest.raises(ValueError, match="resolved/completed"):
        validate_historical_replies([reply])


def test_rejects_empty_reply_text():
    reply = HistoricalReply(
        text="   ",
        thread_id="thread-5",
        author_type="agent",
        thread_status="resolved",
    )

    with pytest.raises(ValueError, match="non-empty"):
        validate_historical_replies([reply])


def test_rejects_empty_collection():
    with pytest.raises(ValueError, match="at least one"):
        validate_historical_replies([])


def test_rejects_wrong_object_type():
    with pytest.raises(TypeError, match="HistoricalReply"):
        validate_historical_replies(["not a historical reply"])
