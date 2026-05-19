"""Storage and retrieval for Milvus memory records."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .client import MilvusClientWrapper
from .config import MilvusConfig
from .embeddings import EmbeddingClient, ensure_dimension
from .schema import TEXT_FIELD, VECTOR_FIELD


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict[str, Any]


class MilvusMemoryStore:
    def __init__(
        self,
        config: MilvusConfig,
        *,
        client: MilvusClientWrapper | None = None,
        embedder: EmbeddingClient | None = None,
    ):
        self.config = config
        self.client_wrapper = client or MilvusClientWrapper(config)
        self.embedder = embedder or EmbeddingClient(config)
        self._initialized = False

    def initialize(self) -> None:
        client = self.client_wrapper.connect()
        if not client.has_collection(self.config.collection):
            client.create_collection(
                collection_name=self.config.collection,
                dimension=self.config.embedding_dimension,
                metric_type="COSINE",
                auto_id=False,
            )
        _safe_load_collection(client, self.config.collection)
        self._initialized = True

    def close(self) -> None:
        self.client_wrapper.close()
        self._initialized = False

    def insert_record(self, text: str, metadata: dict[str, Any]) -> str:
        if not text or not text.strip():
            return ""
        if not self._initialized:
            self.initialize()
        now = int(time.time())
        record_id = _coerce_int64_id(metadata.get("id"))
        clean_meta = dict(metadata)
        clean_meta.setdefault("id", record_id)
        clean_meta.setdefault("created_at", now)
        clean_meta["updated_at"] = now
        clean_meta["text"] = text
        clean_meta["provenance"] = _json_string(clean_meta.get("provenance", {}))
        clean_meta["tags"] = _json_string(clean_meta.get("tags", []))
        vector = ensure_dimension(
            self.embedder.embed(text),
            self.config.embedding_dimension,
        )
        entity = {**clean_meta, VECTOR_FIELD: vector}
        client = self.client_wrapper.connect()
        client.insert(collection_name=self.config.collection, data=[entity])
        _safe_flush(client, self.config.collection)
        _safe_load_collection(client, self.config.collection)
        return str(record_id)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        memory_type: str | None = None,
        session_id: str | None = None,
        include_types: list[str] | None = None,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            return []
        if not self._initialized:
            self.initialize()
        top_k = max(1, min(int(top_k or self.config.top_k), 50))
        vector = ensure_dimension(
            self.embedder.embed(query),
            self.config.embedding_dimension,
        )
        filters = []
        if memory_type:
            filters.append(f'memory_type == "{_escape_filter(memory_type)}"')
        elif include_types:
            quoted = ", ".join(f'"{_escape_filter(t)}"' for t in include_types)
            filters.append(f"memory_type in [{quoted}]")
        if session_id:
            filters.append(f'session_id == "{_escape_filter(session_id)}"')
        filter_expr = " and ".join(filters) if filters else None
        client = self.client_wrapper.connect()
        kwargs: dict[str, Any] = {
            "collection_name": self.config.collection,
            "data": [vector],
            "limit": top_k,
            "output_fields": ["*"],
        }
        if filter_expr:
            kwargs["filter"] = filter_expr
        raw = client.search(**kwargs)
        hits = raw[0] if raw else []
        results: list[SearchResult] = []
        for hit in hits:
            entity = _hit_entity(hit)
            score = float(_hit_score(hit))
            if score < self.config.min_score:
                continue
            text = str(entity.get(TEXT_FIELD) or "")
            results.append(SearchResult(text=text, score=score, metadata=entity))
        return results


def _json_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _coerce_int64_id(value: Any) -> int:
    """Return an int64-compatible Milvus primary key.

    ``MilvusClient.create_collection(..., auto_id=False)`` creates an int64
    primary key named ``id``.  Keep that default schema and generate stable
    integer ids here instead of inserting string UUIDs.
    """
    if isinstance(value, int):
        return value & ((1 << 63) - 1)
    if isinstance(value, str) and value.strip():
        try:
            return int(value) & ((1 << 63) - 1)
        except ValueError:
            return uuid.uuid5(uuid.NAMESPACE_URL, value).int & ((1 << 63) - 1)
    return uuid.uuid4().int & ((1 << 63) - 1)


def _escape_filter(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _hit_entity(hit: Any) -> dict[str, Any]:
    if isinstance(hit, dict):
        entity = hit.get("entity") or hit
        return dict(entity)
    entity = getattr(hit, "entity", None)
    if entity is None:
        return {}
    if isinstance(entity, dict):
        return dict(entity)
    try:
        return dict(entity)
    except Exception:
        return {}


def _hit_score(hit: Any) -> float:
    if isinstance(hit, dict):
        return float(hit.get("distance", hit.get("score", 0.0)) or 0.0)
    return float(getattr(hit, "distance", getattr(hit, "score", 0.0)) or 0.0)


def _safe_flush(client: Any, collection_name: str) -> None:
    flush = getattr(client, "flush", None)
    if not callable(flush):
        return
    try:
        flush(collection_name=collection_name)
    except TypeError:
        try:
            flush(collection_name)
        except Exception:
            return
    except Exception:
        return


def _safe_load_collection(client: Any, collection_name: str) -> None:
    load = getattr(client, "load_collection", None)
    if not callable(load):
        return
    try:
        load(collection_name=collection_name)
    except TypeError:
        try:
            load(collection_name)
        except Exception:
            return
    except Exception:
        return
