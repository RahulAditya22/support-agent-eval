import pytest

from classify import INTENTS, classify, parse_classification_response


def test_valid_response_is_parsed():
    result = parse_classification_response(
        '{"intent":"fare_price_dispute","confidence":0.91,'
        '"reason":"Customer disputes the fare."}'
    )

    assert result.intent == "fare_price_dispute"
    assert result.confidence == pytest.approx(0.91)


def test_unknown_intent_is_rejected():
    with pytest.raises(ValueError, match="Unknown intent"):
        parse_classification_response(
            '{"intent":"banking77","confidence":0.9,'
            '"reason":"bad taxonomy"}'
        )


def test_invalid_confidence_is_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_classification_response(
            '{"intent":"fare_price_dispute","confidence":2,'
            '"reason":"too high"}'
        )


def test_classifier_injects_mock_llm():
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt

        return (
            '{"intent":"cancellation_no_show_charge",'
            '"confidence":0.88,'
            '"reason":"Customer disputes a cancellation fee."}'
        )

    result = classify(
        "@Uber_Support why was I charged after my driver cancelled?",
        fake_llm,
    )

    assert result.intent == "cancellation_no_show_charge"
    assert 0 <= result.confidence <= 1
    assert "Customer tweet:" in seen["prompt"]
    assert all(intent in seen["prompt"] for intent in INTENTS)


def test_empty_text_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        classify("   ", lambda _: "{}")