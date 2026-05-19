"""Embedding helpers for the Milvus memory provider."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

from .config import MilvusConfig


class EmbeddingClient:
    """Small embedding facade with a deterministic offline fallback."""

    def __init__(self, config: MilvusConfig):
        self.config = config

    def embed(self, text: str) -> list[float]:
        provider = (self.config.embedding_provider or "deterministic").lower()
        if provider in {"deterministic", "local", "fake"}:
            return deterministic_embedding(text, self.config.embedding_dimension)
        if provider == "openai":
            return self._openai_embed(text)
        raise RuntimeError(f"Unsupported embedding provider: {provider}")

    def _openai_embed(self, text: str) -> list[float]:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package is required for OpenAI embeddings") from exc
        client = OpenAI()
        response = client.embeddings.create(
            model=self.config.embedding_model,
            input=text,
        )
        return list(response.data[0].embedding)


def deterministic_embedding(text: str, dimension: int = 384) -> list[float]:
    """Return a stable token-hashing vector for tests and smoke checks.

    This is not a production semantic embedding model. It is intentionally a
    lightweight lexical feature-hash vector so local smoke tests can retrieve
    text that shares important words without requiring an external embedding
    API.
    """
    dimension = max(8, int(dimension or 384))
    vector = [0.0] * dimension
    tokens = re.findall(r"[\w.-]+", (text or "").lower())
    if not tokens:
        tokens = ["hermes"]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).digest()
        idx = int.from_bytes(digest[:8], "big") % dimension
        sign = 1.0 if digest[8] % 2 == 0 else -1.0
        vector[idx] += sign
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def ensure_dimension(vector: Iterable[float], dimension: int) -> list[float]:
    values = [float(v) for v in vector]
    if len(values) != dimension:
        raise ValueError(f"Embedding dimension mismatch: {len(values)} != {dimension}")
    return values
