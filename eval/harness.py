"""Evaluation harness for the human-labelled support-agent golden set.

The harness never invents labels. Human-evaluation metrics require explicit,
non-empty human labels. Retrieval metrics are only computed for rows that
contain an explicit retrieval target.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from eval.metrics import classification_metrics, top_k_recall


@dataclass(frozen=True)
class PerIntentMetrics:
    precision: float
    recall: float


@dataclass(frozen=True)
class ClassificationEvaluation:
    accuracy: float
    macro_f1: float
    n: int
    per_intent: dict[str, PerIntentMetrics]


@dataclass(frozen=True)
class EscalationEvaluation:
    precision: float
    recall: float
    false_auto_handle_rate: float
    n: int


@dataclass(frozen=True)
class SimilarityDistribution:
    n: int
    mean: float
    median: float
    minimum: float
    maximum: float
    p25: float
    p75: float


@dataclass(frozen=True)
class RetrievalEvaluation:
    top_k: int
    recall: float | None
    n: int
    similarity: SimilarityDistribution | None


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_human_labels(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> list[str]:
    if not rows:
        raise ValueError("rows must be non-empty")
    labels: list[str] = []
    for index, row in enumerate(rows):
        value = row.get(field)
        if not _nonempty(value):
            raise ValueError(
                f"human evaluation requires non-empty {field!r}; "
                f"missing at row {index}"
            )
        labels.append(value.strip())
    return labels


def _prediction(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    raise ValueError(f"missing prediction field; expected one of {keys}")


def evaluate_classification(
    rows: Sequence[Mapping[str, Any]],
    *,
    prediction_field: str = "predicted_intent",
) -> ClassificationEvaluation:
    """Score accuracy, macro F1, and per-intent precision/recall."""
    y_true = _require_human_labels(rows, "human_intent_label")
    y_pred: list[str] = []
    for index, row in enumerate(rows):
        value = row.get(prediction_field)
        if not _nonempty(value):
            raise ValueError(
                f"missing non-empty {prediction_field!r} at row {index}"
            )
        y_pred.append(value.strip())

    metrics = classification_metrics(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    per_intent: dict[str, PerIntentMetrics] = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_intent[label] = PerIntentMetrics(precision, recall)

    return ClassificationEvaluation(
        accuracy=metrics.accuracy,
        macro_f1=metrics.macro_f1,
        n=metrics.n,
        per_intent=per_intent,
    )


def _as_escalate(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"escalate", "escalated", "true", "1", "yes"}:
            return True
        if normalized in {"auto_handle", "auto-handled", "auto handled", "false", "0", "no"}:
            return False
    raise ValueError(
        "escalation labels must be booleans or explicit "
        "'escalate'/'auto_handle' values"
    )


def evaluate_escalation(
    rows: Sequence[Mapping[str, Any]],
    *,
    prediction_field: str = "predicted_escalate",
) -> EscalationEvaluation:
    """Score escalation precision/recall and false auto-handle rate.

    The human label is interpreted as whether the case should be auto-handled.
    A false auto-handle is a human-escalate case that the system auto-handles.
    """
    raw_true = _require_human_labels(rows, "human_auto_handle_label")
    y_true_escalate = [not _as_escalate(v) for v in raw_true]
    y_pred_escalate = [
        _as_escalate(row.get(prediction_field))
        for row in rows
    ]

    tp = sum(t and p for t, p in zip(y_true_escalate, y_pred_escalate))
    fp = sum((not t) and p for t, p in zip(y_true_escalate, y_pred_escalate))
    fn = sum(t and (not p) for t, p in zip(y_true_escalate, y_pred_escalate))
    false_auto = sum(t and (not p) for t, p in zip(y_true_escalate, y_pred_escalate))

    return EscalationEvaluation(
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        false_auto_handle_rate=false_auto / len(rows),
        n=len(rows),
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def similarity_distribution(similarities: Iterable[float]) -> SimilarityDistribution:
    values = [float(value) for value in similarities]
    if not values:
        raise ValueError("at least one similarity score is required")
    return SimilarityDistribution(
        n=len(values),
        mean=mean(values),
        median=median(values),
        minimum=min(values),
        maximum=max(values),
        p25=_percentile(values, 0.25),
        p75=_percentile(values, 0.75),
    )


def evaluate_retrieval(
    rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 5,
    target_field: str = "gold_retrieval_reply",
    retrieved_field: str = "retrieved_replies",
    similarity_field: str = "retrieval_similarities",
) -> RetrievalEvaluation:
    """Compute retrieval recall@k only where an explicit gold target exists.

    Rows without a target are skipped rather than treated as misses.
    Similarity statistics are computed from explicit numeric similarity lists.
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    recall_values: list[float] = []
    similarities: list[float] = []
    for row in rows:
        target = row.get(target_field)
        retrieved = row.get(retrieved_field)
        if _nonempty(target) and isinstance(retrieved, (list, tuple)):
            recall_values.append(top_k_recall(target, retrieved[:top_k]))
        raw_scores = row.get(similarity_field)
        if isinstance(raw_scores, (list, tuple)):
            for score in raw_scores[:top_k]:
                similarities.append(float(score))

    return RetrievalEvaluation(
        top_k=top_k,
        recall=mean(recall_values) if recall_values else None,
        n=len(recall_values),
        similarity=similarity_distribution(similarities) if similarities else None,
    )


def evaluate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    classification_prediction_field: str = "predicted_intent",
    escalation_prediction_field: str = "predicted_escalate",
    retrieval_top_k: int = 5,
) -> dict[str, object]:
    """Run all human-labelled metrics plus optional retrieval analysis.

    Classification and escalation are required human-labelled evaluations.
    Retrieval gracefully reports ``n=0``/``recall=None`` when no explicit
    retrieval target is present in the supplied rows.
    """
    return {
        "classification": evaluate_classification(
            rows, prediction_field=classification_prediction_field
        ),
        "escalation": evaluate_escalation(
            rows, prediction_field=escalation_prediction_field
        ),
        "retrieval": evaluate_retrieval(rows, top_k=retrieval_top_k),
    }
