"""Intent classification for the Uber Support agent.

Phase 3 contract:
- Classify the initiating customer tweet, not the reconstructed thread.
- Use exactly one taxonomy label.
- Keep LLM interaction behind a small callable so tests can mock it.
- Parse and validate structured JSON; never silently accept an unknown label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping


INTENTS: tuple[str, ...] = (
    "cancellation_no_show_charge",
    "unauthorized_transaction",
    "fare_price_dispute",
    "missing_incomplete_ride",
    "uber_eats_order_delivery_issue",
    "ride_pass_promotion_coupon",
    "account_login_app_access",
    "payment_refund_billing",
    "driver_partner_safety_issue",
    "other_out_of_scope",
)


INTENT_DEFINITIONS: Mapping[str, str] = {
    "cancellation_no_show_charge":
        "Disputed charge or fee caused by cancellation, driver cancellation, or a no-show.",
    "unauthorized_transaction":
        "Unrecognized or unauthorized Uber transaction or account activity.",
    "fare_price_dispute":
        "Dispute about the ride fare, quoted price, pricing calculation, or unexpected price.",
    "missing_incomplete_ride":
        "Ride/trip is missing, disappeared, failed to appear correctly, or was not completed as expected.",
    "uber_eats_order_delivery_issue":
        "Uber Eats order or delivery problem, including missing/wrong items or delivery condition.",
    "ride_pass_promotion_coupon":
        "Ride Pass, promotion, coupon, discount, or promotional eligibility/application issue.",
    "account_login_app_access":
        "Account, login, verification, app-access, or app-level technical problem preventing use.",
    "payment_refund_billing":
        "Payment, refund, billing, card, wallet, or reimbursement issue not primarily unauthorized or fare-related.",
    "driver_partner_safety_issue":
        "Driver/partner problem, driver conduct issue, safety concern, or driver-related incident.",
    "other_out_of_scope":
        "Insufficient information or an issue that cannot be defensibly mapped to a named intent.",
}


@dataclass(frozen=True)
class ClassificationResult:
    """Validated single-label classification returned by the agent."""

    intent: str
    confidence: float
    reason: str


def build_classification_prompt(text: str) -> str:
    """Build a deterministic prompt for the real or mocked LLM."""

    intent_lines = "\n".join(
        f"- {name}: {INTENT_DEFINITIONS[name]}"
        for name in INTENTS
    )

    return f"""You classify one initiating customer tweet for Uber Support.

Choose exactly ONE intent from this list:
{intent_lines}

Rules:
1. Classify the initiating customer message only.
2. If multiple problems appear, prioritize the explicitly requested resolution,
   then the blocking issue, then the issue receiving the greatest emphasis/detail.
3. Do not invent missing facts.
4. Use other_out_of_scope when no named intent can be defended.
5. Return ONLY valid JSON with exactly these keys:
   {{"intent":"<one listed intent>","confidence":<0.0-1.0>,"reason":"<brief reason>"}}

Customer tweet:
{text}
"""


def parse_classification_response(raw: str) -> ClassificationResult:
    """Parse and validate the LLM's JSON response."""

    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response was not valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")

    intent = payload.get("intent")
    confidence = payload.get("confidence")
    reason = payload.get("reason")

    if intent not in INTENTS:
        raise ValueError(f"Unknown intent: {intent!r}")

    if isinstance(confidence, bool) or not isinstance(
        confidence,
        (int, float),
    ):
        raise ValueError("confidence must be numeric")

    confidence = float(confidence)

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")

    return ClassificationResult(
        intent=intent,
        confidence=confidence,
        reason=reason.strip(),
    )


def classify(
    text: str,
    llm_call: Callable[[str], str],
) -> ClassificationResult:
    """Classify text using an injected LLM callable.

    The callable receives the complete prompt and returns the model's raw text.
    This keeps provider/API details outside the classifier and makes CI fully
    deterministic with a mock.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    prompt = build_classification_prompt(text.strip())

    raw = llm_call(prompt)

    if not isinstance(raw, str):
        raise TypeError("llm_call must return a string")

    return parse_classification_response(raw)