import pytest

from eval.metrics import classification_metrics, cohen_kappa, pearson_correlation, top_k_recall


def test_classification_metrics():
    result = classification_metrics(
        ["fare", "fare", "refund", "refund"],
        ["fare", "refund", "refund", "refund"],
    )
    assert result.n == 4
    assert result.accuracy == pytest.approx(0.75)
    assert result.macro_f1 == pytest.approx(11 / 15)


def test_top_k_recall_exact_match():
    assert top_k_recall("reply two", ["reply one", "reply two"]) == 1.0
    assert top_k_recall("reply three", ["reply one", "reply two"]) == 0.0


def test_agreement_metrics():
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "a", "b", "b"]) == pytest.approx(1.0)
    assert pearson_correlation([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_metric_validation():
    with pytest.raises(ValueError):
        classification_metrics([], [])
    with pytest.raises(ValueError):
        classification_metrics(["a"], ["b", "a"])
    with pytest.raises(ValueError):
        top_k_recall("", ["reply"])
