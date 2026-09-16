"""Grounded support-reply drafting using the shared Groq LLM client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from llm_client import GroqLLMClient


class ReplyLike(Protocol):
    """Minimal retrieval-result interface accepted by the drafter."""

    text: str
    similarity: float


@dataclass(frozen=True)
class DraftReplyResult:
    """A generated support reply and the evidence used to ground it."""

    reply: str
    evidence: tuple[str, ...]


def build_draft_prompt(
    customer_text: str,
    retrieved_replies: Sequence[ReplyLike],
) -> str:
    """Build a prompt that constrains the model to retrieved evidence."""
    if not isinstance(customer_text, str) or not customer_text.strip():
        raise ValueError("customer_text must be a non-empty string")

    if not retrieved_replies:
        raise ValueError("retrieved_replies must contain at least one item")

    evidence_lines = "\n".join(
        f"{i}. {item.text}"
        for i, item in enumerate(retrieved_replies, start=1)
    )

    return f"""You draft one concise Uber Support reply.

Use the historical replies below only as grounding examples. Do not invent
policies, refunds, credits, timelines, or facts that are not supported by the
customer message or the historical evidence.

Rules:
1. Address the customer's actual request directly.
2. Be concise, professional, and empathetic.
3. Do not mention that you are using historical replies.
4. Do not claim an action has already been taken unless the evidence supports it.
5. If the evidence is insufficient to safely answer, ask for the minimum useful
   information rather than inventing an answer.
6. Return only the reply text, with no labels or commentary.

Customer message:
{customer_text.strip()}

Historical resolved-reply examples:
{evidence_lines}
"""


def draft_reply(
    customer_text: str,
    retrieved_replies: Sequence[ReplyLike],
    llm_client: GroqLLMClient,
) -> DraftReplyResult:
    """Generate a grounded reply through the shared GroqLLMClient."""
    if not isinstance(llm_client, GroqLLMClient):
        raise TypeError("llm_client must be a GroqLLMClient")

    prompt = build_draft_prompt(customer_text, retrieved_replies)
    reply = llm_client.complete(prompt)

    if not reply.strip():
        raise ValueError("LLM returned an empty draft reply")

    evidence = tuple(item.text for item in retrieved_replies)
    return DraftReplyResult(reply=reply.strip(), evidence=evidence)
