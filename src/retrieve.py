"""FAISS retrieval over resolved historical support replies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import faiss
import numpy as np


class Embedder(Protocol):
    """Minimal embedding interface used by the retriever."""

    def encode(self, texts: Sequence[str], **kwargs) -> np.ndarray:
        ...


@dataclass(frozen=True)
class RetrievedReply:
    """A historical reply and its cosine similarity to the query."""

    text: str
    similarity: float
    index: int


class ReplyRetriever:
    """Nearest-neighbor retrieval using normalized MiniLM embeddings and FAISS."""

    def __init__(
        self,
        embedder: Embedder,
        replies: Sequence[str],
    ) -> None:
        if not replies:
            raise ValueError("replies must contain at least one item")

        cleaned = [reply.strip() for reply in replies]
        if any(not reply for reply in cleaned):
            raise ValueError("replies must contain non-empty strings")

        self.embedder = embedder
        self.replies = tuple(cleaned)
        self.index = self._build_index(self.replies)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.embedder.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors = np.asarray(vectors, dtype="float32")

        if vectors.ndim != 2:
            raise ValueError("embedder must return a 2D array")

        if vectors.shape[0] != len(texts):
            raise ValueError("embedder returned the wrong number of vectors")

        return vectors

    def _build_index(self, replies: Sequence[str]) -> faiss.Index:
        vectors = self._encode(replies)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    def search(self, query: str, top_k: int = 5) -> list[RetrievedReply]:
        """Return the top-k historical replies by cosine similarity."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        query_vector = self._encode([query.strip()])
        scores, indices = self.index.search(
            query_vector,
            min(top_k, len(self.replies)),
        )

        results: list[RetrievedReply] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            results.append(
                RetrievedReply(
                    text=self.replies[int(index)],
                    similarity=float(score),
                    index=int(index),
                )
            )

        return results
