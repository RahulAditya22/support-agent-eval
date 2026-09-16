from types import SimpleNamespace

import numpy as np
import pytest

from retrieve import ReplyRetriever


class FakeEmbedder:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.asarray([self.mapping[text] for text in texts], dtype="float32")


def test_retriever_builds_index_and_returns_top_k():
    embedder = FakeEmbedder(
        {
            "fare problem": [1.0, 0.0],
            "refund problem": [0.0, 1.0],
            "fare help": [0.9, 0.1],
        }
    )

    retriever = ReplyRetriever(
        embedder,
        ["fare problem", "refund problem", "fare help"],
    )

    results = retriever.search("fare problem", top_k=2)

    assert len(results) == 2
    assert results[0].text == "fare problem"
    assert results[0].similarity == pytest.approx(1.0)
    assert results[1].text == "fare help"
    assert results[1].similarity > results[0].similarity - 1.0


def test_retriever_requests_normalized_embeddings():
    embedder = FakeEmbedder(
        {
            "reply one": [1.0, 0.0],
            "reply two": [0.0, 1.0],
        }
    )

    ReplyRetriever(embedder, ["reply one", "reply two"])

    assert embedder.calls[0][1]["normalize_embeddings"] is True
    assert embedder.calls[0][1]["convert_to_numpy"] is True


def test_search_caps_top_k_to_available_replies():
    embedder = FakeEmbedder(
        {
            "one": [1.0, 0.0],
            "two": [0.0, 1.0],
        }
    )

    retriever = ReplyRetriever(embedder, ["one", "two"])

    results = retriever.search("one", top_k=10)

    assert len(results) == 2


def test_empty_replies_are_rejected():
    embedder = FakeEmbedder({})

    with pytest.raises(ValueError, match="at least one"):
        ReplyRetriever(embedder, [])


def test_empty_query_and_invalid_top_k_are_rejected():
    embedder = FakeEmbedder({"reply": [1.0, 0.0]})
    retriever = ReplyRetriever(embedder, ["reply"])

    with pytest.raises(ValueError, match="non-empty"):
        retriever.search("   ")

    with pytest.raises(ValueError, match="positive integer"):
        retriever.search("reply", top_k=0)


def test_embedder_shape_is_validated():
    class BadEmbedder:
        def encode(self, texts, **kwargs):
            return np.asarray([1.0, 2.0], dtype="float32")

    with pytest.raises(ValueError, match="2D array"):
        ReplyRetriever(BadEmbedder(), ["reply"])
