"""Reproducible baseline systems for the Uber support evaluation.

The original majority and TF-IDF classifier baselines are preserved. Phase 4
adds:
1. zero-shot direct LLM (classification + draft, no retrieval);
2. naive TF-IDF lexical retrieval followed by an LLM draft.

All LLM calls are injectable for deterministic tests. A real caller can be
``GroqLLMClient.complete`` and therefore inherits its call/retry/timeout budget.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from classify import INTENTS, parse_classification_response


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


@dataclass(frozen=True)
class LLMRunResult:
    """Output from an LLM-backed baseline for one customer tweet."""

    intent: str
    confidence: float
    reason: str
    reply: str
    evidence: tuple[str, ...] = ()


def _direct_classification_prompt(customer_text: str) -> str:
    labels = ", ".join(INTENTS)
    return f"""Classify this Uber Support customer tweet using exactly one label:
{labels}

Return ONLY JSON with exactly these keys:
{{"intent":"<label>","confidence":0.0,"reason":"brief reason"}}

Customer tweet:
{customer_text.strip()}
"""


def _parse_direct_classification(raw: str):
    return parse_classification_response(raw)


def _direct_draft_prompt(customer_text: str) -> str:
    return f"""Draft a concise Uber Support reply to this customer tweet.

This is a zero-shot baseline: there is NO historical retrieval context.
Do not invent account-specific facts, refunds, actions, or guarantees.
If details are missing, ask for the minimum information needed.

Return ONLY the reply text.

Customer tweet:
{customer_text.strip()}
"""


def _baseline2_draft_prompt(customer_text: str, evidence: Sequence[str]) -> str:
    evidence_text = "\n".join(
        f"{i + 1}. {text}" for i, text in enumerate(evidence)
    )
    return f"""Draft a concise Uber Support reply using the lexical retrieval
examples below only as grounding examples. Do not claim that an example
action has already happened to this customer and do not invent facts.

Retrieved examples:
{evidence_text}

Customer tweet:
{customer_text.strip()}

Return ONLY the reply text.
"""


def _complete(llm_call: Callable[[str], str], prompt: str) -> str:
    if not callable(llm_call):
        raise TypeError("llm_call must be callable")
    raw = llm_call(prompt)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("llm_call must return non-empty text")
    return raw.strip()


class KeywordRetriever:
    """Naive TF-IDF lexical retriever used only as a baseline."""

    def __init__(self, documents: Sequence[str]) -> None:
        if not documents:
            raise ValueError("documents must contain at least one item")
        cleaned = [text.strip() for text in documents]
        if any(not text for text in cleaned):
            raise ValueError("documents must contain non-empty strings")
        self.documents = tuple(cleaned)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.matrix = self.vectorizer.fit_transform(self.documents)

    def search(self, query: str, top_k: int = 5) -> tuple[str, ...]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        query_vector = self.vectorizer.transform([query.strip()])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        count = min(top_k, len(self.documents))
        # Stable deterministic tie-breaking: score descending, source index ascending.
        indices = sorted(
            range(len(self.documents)),
            key=lambda i: (-float(scores[i]), i),
        )[:count]
        return tuple(self.documents[i] for i in indices)


def zero_shot_direct_llm(
    customer_text: str,
    llm_call: Callable[[str], str],
) -> LLMRunResult:
    """Baseline 1: direct LLM classification and drafting, no retrieval."""
    if not isinstance(customer_text, str) or not customer_text.strip():
        raise ValueError("customer_text must be non-empty")

    classification = _parse_direct_classification(
        _complete(llm_call, _direct_classification_prompt(customer_text))
    )
    reply = _complete(llm_call, _direct_draft_prompt(customer_text))
    return LLMRunResult(
        intent=classification.intent,
        confidence=classification.confidence,
        reason=classification.reason,
        reply=reply,
    )


def naive_keyword_retrieval_llm(
    customer_text: str,
    documents: Sequence[str] | KeywordRetriever,
    llm_call: Callable[[str], str],
    *,
    top_k: int = 5,
) -> LLMRunResult:
    """Baseline 2: direct LLM classification + naive lexical retrieval + LLM draft."""
    if not isinstance(customer_text, str) or not customer_text.strip():
        raise ValueError("customer_text must be non-empty")
    retriever = (
        documents if isinstance(documents, KeywordRetriever)
        else KeywordRetriever(documents)
    )
    classification = _parse_direct_classification(
        _complete(llm_call, _direct_classification_prompt(customer_text))
    )
    evidence = retriever.search(customer_text, top_k=top_k)
    reply = _complete(
        llm_call,
        _baseline2_draft_prompt(customer_text, evidence),
    )
    return LLMRunResult(
        intent=classification.intent,
        confidence=classification.confidence,
        reason=classification.reason,
        reply=reply,
        evidence=evidence,
    )


# Friendly aliases for callers/tests.
baseline_zero_shot = zero_shot_direct_llm
baseline_keyword_retrieval = naive_keyword_retrieval_llm
