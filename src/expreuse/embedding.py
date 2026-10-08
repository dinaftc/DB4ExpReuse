"""Goal and state embeddings."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one unit-norm row per text."""


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norm == 0, 1, norm)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(normalize(a) @ normalize(b))


class SentenceTransformerEmbedder:
    """all-mpnet-base-v2 by default, as in the paper."""

    def __init__(self, model: str = "sentence-transformers/all-mpnet-base-v2", device=None):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model, device=device)
        self.dim = self._model.get_sentence_embedding_dimension()
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vectors = self._model.encode(missing, normalize_embeddings=True,
                                         show_progress_bar=False)
            self._cache.update(zip(missing, np.asarray(vectors, dtype=np.float32)))
        return np.stack([self._cache[t] for t in texts])


class HashEmbedder:
    """Deterministic bag-of-words embedding. For tests and benchmarks that
    need embeddings without loading a model; not a semantic encoder."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                rng = np.random.default_rng(int.from_bytes(digest, "little"))
                out[i] += rng.standard_normal(self.dim).astype(np.float32)
        return normalize(out)
