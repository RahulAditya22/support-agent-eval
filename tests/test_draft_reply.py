from dataclasses import dataclass

import pytest

from draft_reply import build_draft_prompt, draft_reply
from llm_client import GroqLLMClient, LLMConfig


@dataclass(frozen=True)
class FakeRetrievedReply:
    text: str
    similarity: float


class FakeLLMClient(GroqLLMClient):
    """Test double that reuses the production client type without networking."""

    def __init__(self, response: str):
        self.response = response
        self.prompts = []

    def complete(self, prompt: str, *, max_output_tokens=None) -> str:
        self.prompts.append(prompt)
        return self.response


class FakeRetriever:
    """Test double demonstrating that retrieval can be mocked independently."""

    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, query: str, top_k: int = 5):
        self.queries.append((query, top_k))
        return self.results


def test_build_prompt_contains_customer_and_retrieved_evidence():
    replies = [
        FakeRetrievedReply("Please check your trip receipt.", 0.95),
        FakeRetrievedReply("We can review the fare details.", 0.88),
    ]

    prompt = build_draft_prompt("Why was I charged this fare?", replies)

    assert "Why was I charged this fare?" in prompt
    assert "Please check your trip receipt." in prompt
    assert "We can review the fare details." in prompt
    assert "only as grounding examples" in prompt


def test_draft_reply_uses_shared_llm_client_and_returns_evidence():
    replies = [
        FakeRetrievedReply("Please check your trip receipt.", 0.95),
        FakeRetrievedReply("We can review the fare details.", 0.88),
    ]
    llm = FakeLLMClient("I can help review the fare shown on your trip receipt.")

    result = draft_reply(
        "Why was I charged this fare?",
        replies,
        llm,
    )

    assert result.reply == (
        "I can help review the fare shown on your trip receipt."
    )
    assert result.evidence == (
        "Please check your trip receipt.",
        "We can review the fare details.",
    )
    assert len(llm.prompts) == 1


def test_draft_reply_does_not_make_a_second_llm_call_path():
    replies = [FakeRetrievedReply("Please send the trip receipt.", 0.9)]
    llm = FakeLLMClient("Please send the trip receipt so we can review it.")

    result = draft_reply("I need help with my fare.", replies, llm)

    assert result.reply.startswith("Please send")
    assert len(llm.prompts) == 1


def test_empty_customer_text_is_rejected():
    replies = [FakeRetrievedReply("Example reply", 0.9)]
    llm = FakeLLMClient("Example")

    with pytest.raises(ValueError, match="customer_text"):
        draft_reply("   ", replies, llm)


def test_missing_retrieval_context_is_rejected():
    llm = FakeLLMClient("Example")

    with pytest.raises(ValueError, match="retrieved_replies"):
        draft_reply("I need help.", [], llm)


def test_invalid_llm_client_is_rejected():
    replies = [FakeRetrievedReply("Example reply", 0.9)]

    with pytest.raises(TypeError, match="GroqLLMClient"):
        draft_reply("I need help.", replies, object())
