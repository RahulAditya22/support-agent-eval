"""Simple, reproducible baselines for the Uber intent task."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class BaselineResult:
    name: str
    predictions: tuple[str, ...]


def majority_baseline(y_train: Sequence[str], n_predictions: int) -> BaselineResult:
    if not y_train:
        raise ValueError("y_train must be non-empty")
    if n_predictions < 1:
        raise ValueError("n_predictions must be positive")
    majority = Counter(y_train).most_common(1)[0][0]
    return BaselineResult("majority", tuple([majority] * n_predictions))


class TfidfLogisticBaseline:
    """TF-IDF word n-gram logistic model using human-labelled training data."""

    def __init__(self) -> None:
        self.model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        )

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "TfidfLogisticBaseline":
        if len(texts) != len(labels) or not texts:
            raise ValueError("texts and labels must be non-empty and equal length")
        self.model.fit(texts, labels)
        return self

    def predict(self, texts: Sequence[str]) -> tuple[str, ...]:
        if not texts:
            raise ValueError("texts must be non-empty")
        return tuple(str(value) for value in self.model.predict(texts))
