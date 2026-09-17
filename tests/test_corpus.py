from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from corpus import build_reply_retriever, build_uber_reply_retriever


@dataclass
class FakeEmbedder:
    mapping: dict[str, list[float]]

    def encode(self, texts, **kwargs):
        return np.asarray([self.mapping[text] for text in texts], dtype="float32")


class DeterministicEmbedder:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            value = sum(ord(char) for char in text) % 997
            vectors.append([float(value + 1), 1.0])
        return np.asarray(vectors, dtype="float32")


def test_build_reply_retriever_filters_before_faiss_index():
    records = [
        {"text": "Please check your trip receipt.", "thread_id": "t1", "author_type": "agent", "thread_status": "resolved"},
        {"text": "Why did you charge me?", "thread_id": "t2", "author_type": "customer", "thread_status": "resolved"},
        {"text": "We are still investigating this.", "thread_id": "t3", "author_type": "agent", "thread_status": "open"},
        {"text": "We can review the fare details.", "thread_id": "t4", "author_type": "agent", "thread_status": "completed"},
    ]
    embedder = FakeEmbedder({
        "Please check your trip receipt.": [1.0, 0.0],
        "We can review the fare details.": [0.9, 0.1],
    })

    retriever, history = build_reply_retriever(embedder, records)

    assert [item.text for item in history] == [
        "Please check your trip receipt.",
        "We can review the fare details.",
    ]
    assert retriever.replies == tuple(item.text for item in history)


def test_missing_provenance_fields_are_rejected():
    with pytest.raises(ValueError, match="explicit fields"):
        build_reply_retriever(FakeEmbedder({"reply": [1.0, 0.0]}), [{"text": "reply", "thread_id": "t1"}])


def test_real_uber_sample_builds_retrieval_corpus_when_present():
    sample_path = Path("data/processed/uber_support_sample.csv")
    if not sample_path.exists():
        pytest.skip("local real Uber sample is not present")

    retriever, history = build_uber_reply_retriever(DeterministicEmbedder(), sample_path)

    assert history
    assert retriever.replies
    assert retriever.replies == tuple(item.text for item in history)
