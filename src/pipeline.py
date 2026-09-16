"""End-to-end support-agent orchestration for one customer message."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from classify import ClassificationResult, classify
from draft_reply import DraftReplyResult, RetrieverLike, draft_reply
from llm_client import GroqLLMClient


class Classifier(Protocol):
    """Callable contract for customer-message classification."""

    def __call__(self, customer_text: str, llm_call: Callable[[str], str]) -> ClassificationResult:
        ...


@dataclass(frozen=True)
class PipelineResult:
    """Combined classification and grounded draft for one customer message."""

    classification: ClassificationResult
    draft: DraftReplyResult


def run_pipeline(
    customer_text: str,
    retriever: RetrieverLike,
    llm_client: GroqLLMClient,
    *,
    top_k: int = 5,
    classifier: Classifier = classify,
) -> PipelineResult:
    """Classify a customer message, retrieve evidence, and draft a reply."""
    if not isinstance(customer_text, str) or not customer_text.strip():
        raise ValueError("customer_text must be a non-empty string")

    if not isinstance(llm_client, GroqLLMClient):
        raise TypeError("llm_client must be a GroqLLMClient")

    classification = classifier(customer_text.strip(), llm_client.complete)
    draft = draft_reply(
        customer_text.strip(),
        retriever,
        llm_client,
        top_k=top_k,
    )

    return PipelineResult(
        classification=classification,
        draft=draft,
    )
