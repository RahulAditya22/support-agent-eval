"""Deterministic evaluation metrics for the support-agent assignment.

This module deliberately contains no fabricated labels or benchmark results.
The caller supplies human-owned golden labels and system predictions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    n: int


def _validate_pairs(y_true: Sequence[str], y_pred: Sequence[str]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("at least one labelled example is required")


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> ClassificationMetrics:
    """Compute exact accuracy and macro F1 without external state."""
    _validate_pairs(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    accuracy = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    f1_values: list[float] = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return ClassificationMetrics(accuracy=accuracy, macro_f1=sum(f1_values) / len(f1_values), n=len(y_true))


def top_k_recall(expected_reply: str, retrieved_replies: Iterable[str]) -> float:
    """Return binary recall@k for an exact expected reply match."""
    if not isinstance(expected_reply, str) or not expected_reply.strip():
        raise ValueError("expected_reply must be non-empty")
    replies = list(retrieved_replies)
    if not replies:
        return 0.0
    target = expected_reply.strip()
    return 1.0 if any(reply.strip() == target for reply in replies) else 0.0


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Compute unweighted Cohen's kappa for two human annotators."""
    _validate_pairs(labels_a, labels_b)
    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    classes = sorted(set(labels_a) | set(labels_b))
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum((counts_a[label] / n) * (counts_b[label] / n) for label in classes)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Compute Pearson correlation for judge/human numeric scores."""
    if len(xs) != len(ys) or not xs:
        raise ValueError("score sequences must be non-empty and equal length")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]
    numerator = sum(x * y for x, y in zip(centered_x, centered_y))
    denom = sqrt(sum(x * x for x in centered_x) * sum(y * y for y in centered_y))
    return numerator / denom if denom else 0.0
