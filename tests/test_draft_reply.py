from dataclasses import dataclass

import pytest

from draft_reply import build_draft_prompt, draft_reply
from llm_client import GroqLLMClient


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
    """Test double for the retrieval dependency."""

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


def test_draft_reply_mocks_retriever_and_shared_llm_client():
    replies = [
        FakeRetrievedReply("Please check your trip receipt.", 0.95),
        FakeRetrievedReply("We can review the fare details.", 0.88),
    ]
    retriever = FakeRetriever(replies)
    llm = FakeLLMClient(
        "I can help review the fare shown on your trip receipt."
    )

    result = draft_reply(
        "Why was I charged this fare?",
        retriever,
        llm,
        top_k=2,
    )

    assert result.reply == (
        "I can help review the fare shown on your trip receipt."
    )
    assert result.evidence == (
        "Please check your trip receipt.",
        "We can review the fare details.",
    )
    assert retriever.queries == [("Why was I charged this fare?", 2)]
    assert len(llm.prompts) == 1


def test_draft_reply_uses_one_shared_llm_call():
    replies = [FakeRetrievedReply("Please send the trip receipt.", 0.9)]
    retriever = FakeRetriever(replies)
    llm = FakeLLMClient(
        "Please send the trip receipt so we can review it."
    )

    result = draft_reply("I need help with my fare.", retriever, llm)

    assert result.reply.startswith("Please send")
    assert len(retriever.queries) == 1
    assert len(llm.prompts) == 1


def test_empty_customer_text_is_rejected():
    retriever = FakeRetriever([])
    llm = FakeLLMClient("Example")

    with pytest.raises(ValueError, match="customer_text"):
        draft_reply("   ", retriever, llm)


def test_empty_retrieval_results_are_rejected():
    retriever = FakeRetriever([])
    llm = FakeLLMClient("Example")

    with pytest.raises(ValueError, match="retrieved_replies"):
        draft_reply("I need help.", retriever, llm)


def test_invalid_llm_client_is_rejected():
    replies = [FakeRetrievedReply("Example reply", 0.9)]
    retriever = FakeRetriever(replies)

    with pytest.raises(TypeError, match="GroqLLMClient"):
        draft_reply("I need help.", retriever, object())
