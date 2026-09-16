import pytest

from eval.harness import (
    evaluate_classification,
    evaluate_escalation,
    evaluate_retrieval,
    similarity_distribution,
)


def rows():
    return [
        {
            "human_intent_label": "fare_price_dispute",
            "predicted_intent": "fare_price_dispute",
            "human_auto_handle_label": "true",
            "predicted_escalate": False,
            "gold_retrieval_reply": "fare example",
            "retrieved_replies": ["fare example", "other"],
            "retrieval_similarities": [0.91, 0.72],
        },
        {
            "human_intent_label": "payment_refund_billing",
            "predicted_intent": "fare_price_dispute",
            "human_auto_handle_label": "false",
            "predicted_escalate": True,
            "gold_retrieval_reply": "missing",
            "retrieved_replies": ["other"],
            "retrieval_similarities": [0.63, 0.42],
        },
        {
            "human_intent_label": "fare_price_dispute",
            "predicted_intent": "fare_price_dispute",
            "human_auto_handle_label": "false",
            "predicted_escalate": False,
            "retrieval_similarities": [0.55],
        },
    ]


def test_classification_scores_accuracy_macro_f1_and_per_intent():
    result = evaluate_classification(rows())
    assert result.n == 3
    assert result.accuracy == pytest.approx(2 / 3)
    assert result.macro_f1 == pytest.approx(0.4)
    assert result.per_intent["fare_price_dispute"].precision == pytest.approx(2 / 3)
    assert result.per_intent["fare_price_dispute"].recall == 1.0
    assert result.per_intent["payment_refund_billing"].precision == 0.0
    assert result.per_intent["payment_refund_billing"].recall == 0.0


def test_classification_requires_human_labels():
    data = rows()
    data[0]["human_intent_label"] = ""
    with pytest.raises(ValueError, match="human_intent_label"):
        evaluate_classification(data)


def test_escalation_scores_precision_recall_and_false_auto_handle_rate():
    result = evaluate_escalation(rows())
    # Human auto=true -> no escalation; false -> escalation.
    # Predictions: no, yes, no => TP=1, FP=0, FN=1; false auto=1/3.
    assert result.n == 3
    assert result.precision == 1.0
    assert result.recall == 0.5
    assert result.false_auto_handle_rate == pytest.approx(1 / 3)


def test_escalation_requires_human_labels():
    data = rows()
    data[1]["human_auto_handle_label"] = ""
    with pytest.raises(ValueError, match="human_auto_handle_label"):
        evaluate_escalation(data)


def test_retrieval_top_k_recall_skips_rows_without_gold_target():
    result = evaluate_retrieval(rows(), top_k=2)
    assert result.n == 2
    assert result.recall == pytest.approx(0.5)
    assert result.similarity.n == 5
    assert result.similarity.mean == pytest.approx((0.91 + 0.72 + 0.63 + 0.42 + 0.55) / 5)


def test_similarity_distribution_has_summary_statistics():
    result = similarity_distribution([0.1, 0.2, 0.3, 0.4, 0.5])
    assert result.n == 5
    assert result.mean == pytest.approx(0.3)
    assert result.median == pytest.approx(0.3)
    assert result.minimum == pytest.approx(0.1)
    assert result.maximum == pytest.approx(0.5)
    assert result.p25 == pytest.approx(0.2)
    assert result.p75 == pytest.approx(0.4)


def test_retrieval_without_targets_is_cleanly_reported():
    result = evaluate_retrieval(
        [{"retrieved_replies": ["reply"], "retrieval_similarities": [0.8]}]
    )
    assert result.n == 0
    assert result.recall is None
    assert result.similarity is not None
