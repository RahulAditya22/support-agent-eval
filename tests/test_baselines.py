import pytest

from baselines import (
    KeywordRetriever,
    TfidfLogisticBaseline,
    baseline_keyword_retrieval,
    baseline_zero_shot,
    majority_baseline,
    naive_keyword_retrieval_llm,
    zero_shot_direct_llm,
)


def test_majority_baseline_still_works():
    result = majority_baseline(["fare", "fare", "refund"], 2)
    assert result.predictions == ("fare", "fare")


def test_tfidf_classifier_still_works():
    model = TfidfLogisticBaseline().fit(
        ["fare too high", "need refund", "fare price"],
        ["fare_price_dispute", "payment_refund_billing", "fare_price_dispute"],
    )
    assert model.predict(["fare price is wrong"])[0] == "fare_price_dispute"


def test_zero_shot_baseline_uses_injected_llm_without_retrieval():
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        if "Classify this" in prompt:
            return '{"intent":"fare_price_dispute","confidence":0.9,"reason":"fare"}'
        return "Please share the trip receipt."

    result = zero_shot_direct_llm("Why was my fare too high?", fake_llm)
    assert result.intent == "fare_price_dispute"
    assert result.reply == "Please share the trip receipt."
    assert result.evidence == ()
    assert len(calls) == 2
    assert all("Retrieved" not in prompt for prompt in calls)


def test_zero_shot_alias_works():
    result = baseline_zero_shot(
        "Fare issue",
        lambda prompt: (
            '{"intent":"fare_price_dispute","confidence":0.8,"reason":"fare"}'
            if "Classify this" in prompt
            else "We can review the fare."
        ),
    )
    assert result.intent == "fare_price_dispute"


def test_keyword_retriever_returns_lexical_matches():
    retriever = KeywordRetriever(
        ["refund for card", "fare receipt", "driver safety"]
    )
    assert retriever.search("I need a fare receipt", top_k=1) == ("fare receipt",)


def test_keyword_baseline_executes_with_mock_llm():
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        if "Classify this" in prompt:
            return '{"intent":"fare_price_dispute","confidence":0.9,"reason":"fare"}'
        return "Please send the fare receipt."

    result = naive_keyword_retrieval_llm(
        "I was charged the wrong fare",
        ["refund for card", "fare receipt", "driver safety"],
        fake_llm,
        top_k=2,
    )
    assert result.intent == "fare_price_dispute"
    assert result.reply == "Please send the fare receipt."
    assert len(result.evidence) == 2
    assert "Retrieved examples:" in calls[1]


def test_keyword_baseline_alias_works():
    result = baseline_keyword_retrieval(
        "refund please",
        ["refund example", "fare example"],
        lambda prompt: (
            '{"intent":"payment_refund_billing","confidence":0.8,"reason":"refund"}'
            if "Classify this" in prompt
            else "We can review the refund."
        ),
        top_k=1,
    )
    assert result.intent == "payment_refund_billing"


def test_baselines_reject_blank_text():
    with pytest.raises(ValueError, match="customer_text"):
        zero_shot_direct_llm(" ", lambda _: "x")
    with pytest.raises(ValueError, match="customer_text"):
        naive_keyword_retrieval_llm(" ", ["x"], lambda _: "x")


def test_keyword_top_k_validation():
    retriever = KeywordRetriever(["one document"])
    with pytest.raises(ValueError, match="positive integer"):
        retriever.search("one", top_k=0)
