import pytest

from classify import ClassificationResult
from escalate import decide_escalation


def classification(intent: str, confidence: float) -> ClassificationResult:
    return ClassificationResult(
        intent=intent,
        confidence=confidence,
        reason="test result",
    )


def test_high_risk_intent_escalates_even_with_high_confidence():
    result = decide_escalation(
        classification("driver_partner_safety_issue", 0.99)
    )

    assert result.escalate is True
    assert result.reason == "intent_requires_human:driver_partner_safety_issue"


def test_unauthorized_transaction_escalates():
    result = decide_escalation(
        classification("unauthorized_transaction", 0.95)
    )

    assert result.escalate is True
    assert result.reason == "intent_requires_human:unauthorized_transaction"


def test_other_out_of_scope_escalates():
    result = decide_escalation(
        classification("other_out_of_scope", 0.90)
    )

    assert result.escalate is True
    assert result.reason == "intent_requires_human:other_out_of_scope"


def test_low_confidence_escalates_for_normal_intent():
    result = decide_escalation(
        classification("fare_price_dispute", 0.69)
    )

    assert result.escalate is True
    assert result.reason == "low_confidence:0.69<0.70"


def test_confidence_at_threshold_is_auto_handled():
    result = decide_escalation(
        classification("fare_price_dispute", 0.70)
    )

    assert result == type(result)(
        escalate=False,
        reason="within_auto_handle_policy",
    )


def test_high_confidence_normal_intent_is_auto_handled():
    result = decide_escalation(
        classification("payment_refund_billing", 0.93)
    )

    assert result.escalate is False
    assert result.reason == "within_auto_handle_policy"


def test_invalid_classification_type_is_rejected():
    with pytest.raises(TypeError, match="ClassificationResult"):
        decide_escalation(object())


def test_invalid_threshold_is_rejected():
    result = classification("fare_price_dispute", 0.80)

    with pytest.raises(ValueError, match="between 0 and 1"):
        decide_escalation(result, min_confidence=1.1)
