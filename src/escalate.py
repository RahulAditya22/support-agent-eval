"""Deterministic escalation policy for support-agent outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from classify import ClassificationResult


@dataclass(frozen=True)
class EscalationResult:
    """Policy decision explaining whether a case needs human handling."""

    escalate: bool
    reason: str


# Safety-sensitive or otherwise high-risk intents should not be auto-handled.
DEFAULT_ESCALATION_INTENTS: frozenset[str] = frozenset(
    {
        "unauthorized_transaction",
        "driver_partner_safety_issue",
        "other_out_of_scope",
    }
)


def decide_escalation(
    classification: ClassificationResult,
    *,
    min_confidence: float = 0.70,
    escalation_intents: frozenset[str] = DEFAULT_ESCALATION_INTENTS,
) -> EscalationResult:
    """Return a deterministic human-escalation decision from validated output."""
    if not isinstance(classification, ClassificationResult):
        raise TypeError("classification must be a ClassificationResult")

    if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
        raise TypeError("min_confidence must be numeric")

    min_confidence = float(min_confidence)
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")

    if classification.intent in escalation_intents:
        return EscalationResult(
            escalate=True,
            reason=f"intent_requires_human:{classification.intent}",
        )

    if classification.confidence < min_confidence:
        return EscalationResult(
            escalate=True,
            reason=f"low_confidence:{classification.confidence:.2f}<{min_confidence:.2f}",
        )

    return EscalationResult(
        escalate=False,
        reason="within_auto_handle_policy",
    )
