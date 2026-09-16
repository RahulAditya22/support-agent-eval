from dataclasses import dataclass

from classify import ClassificationResult
from draft_reply import DraftReplyResult
from llm_client import GroqLLMClient
from pipeline import run_pipeline


@dataclass(frozen=True)
class FakeRetrievedReply:
    text: str
    similarity: float


class FakeLLMClient(GroqLLMClient):
    """Test double that never performs a network request."""

    def __init__(self):
        self.calls = []

    def complete(self, prompt: str, *, max_output_tokens=None) -> str:
        self.calls.append(prompt)
        return "Please share the trip receipt so we can review the fare."


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query: str, top_k: int = 5):
        self.queries.append((query, top_k))
        return [FakeRetrievedReply("Please share the trip receipt so we can review the fare.", 0.91)]


def test_pipeline_wires_classifier_retriever_and_llm():
    llm = FakeLLMClient()
    retriever = FakeRetriever()
    classifier_calls = []

    expected = ClassificationResult(
        intent="fare_price_dispute",
        confidence=0.94,
        reason="Customer disputes the fare charged for a trip.",
    )

    def fake_classifier(customer_text, llm_client):
        classifier_calls.append((customer_text, llm_client))
        return expected

    result = run_pipeline(
        "Why was I charged this fare?",
        retriever,
        llm,
        top_k=1,
        classifier=fake_classifier,
    )

    assert result.classification == expected
    assert result.draft == DraftReplyResult(
        reply="Please share the trip receipt so we can review the fare.",
        evidence=("Please share the trip receipt so we can review the fare.",),
    )
    assert classifier_calls == [("Why was I charged this fare?", llm)]
    assert retriever.queries == [("Why was I charged this fare?", 1)]
    assert len(llm.calls) == 1


def test_pipeline_rejects_blank_customer_text():
    llm = FakeLLMClient()
    retriever = FakeRetriever()

    try:
        run_pipeline("   ", retriever, llm, classifier=lambda *_: None)
    except ValueError as exc:
        assert "customer_text" in str(exc)
    else:
        raise AssertionError("blank customer text should be rejected")
