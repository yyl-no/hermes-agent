"""Milvus collection schema constants."""

from __future__ import annotations

DEFAULT_COLLECTION = "hermes_memory"
VECTOR_FIELD = "vector"
TEXT_FIELD = "text"

METADATA_FIELDS = [
    "id",
    "text",
    "memory_type",
    "source",
    "session_id",
    "parent_session_id",
    "platform",
    "profile",
    "workspace",
    "user_id",
    "chat_id",
    "created_at",
    "updated_at",
    "tags",
    "provenance",
]


def build_index_params() -> dict:
    return {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
    }

