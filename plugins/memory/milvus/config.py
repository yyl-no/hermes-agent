"""Configuration helpers for the Milvus memory provider."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MilvusConfig:
    uri: str = ""
    token: str = ""
    database: str = "default"
    collection: str = "hermes_memory"
    embedding_provider: str = "deterministic"
    embedding_model: str = "deterministic-384"
    embedding_dimension: int = 384
    top_k: int = 8
    max_chars: int = 3000
    min_score: float = 0.35
    mode: str = "mirror"
    include_types: list[str] = field(
        default_factory=lambda: ["curated_memory", "turn", "summary"]
    )


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_json_config(hermes_home: str | Path | None = None) -> dict[str, Any]:
    if hermes_home is None:
        try:
            from hermes_constants import get_hermes_home

            hermes_home = get_hermes_home()
        except Exception:
            hermes_home = Path.home() / ".hermes"
    path = Path(hermes_home) / "milvus.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_config(hermes_home: str | Path | None = None) -> MilvusConfig:
    """Load Milvus config from env vars, then $HERMES_HOME/milvus.json."""
    data: dict[str, Any] = {
        "uri": os.environ.get("MILVUS_URI", ""),
        "token": os.environ.get("MILVUS_TOKEN", ""),
        "database": os.environ.get("MILVUS_DATABASE", "default"),
        "collection": os.environ.get("MILVUS_COLLECTION", "hermes_memory"),
        "embedding_provider": os.environ.get(
            "MILVUS_EMBEDDING_PROVIDER", "deterministic"
        ),
        "embedding_model": os.environ.get(
            "MILVUS_EMBEDDING_MODEL", "deterministic-384"
        ),
        "embedding_dimension": os.environ.get("MILVUS_EMBEDDING_DIMENSION", 384),
        "top_k": os.environ.get("MILVUS_TOP_K", 8),
        "max_chars": os.environ.get("MILVUS_MAX_CHARS", 3000),
        "min_score": os.environ.get("MILVUS_MIN_SCORE", 0.35),
        "mode": os.environ.get("MILVUS_MODE", "mirror"),
    }
    file_cfg = _load_json_config(hermes_home)
    data.update({k: v for k, v in file_cfg.items() if v not in (None, "")})
    include_types = data.get("include_types") or ["curated_memory", "turn", "summary"]
    if isinstance(include_types, str):
        include_types = [p.strip() for p in include_types.split(",") if p.strip()]
    return MilvusConfig(
        uri=str(data.get("uri", "")),
        token=str(data.get("token", "")),
        database=str(data.get("database", "default") or "default"),
        collection=str(data.get("collection", "hermes_memory") or "hermes_memory"),
        embedding_provider=str(data.get("embedding_provider", "deterministic")),
        embedding_model=str(data.get("embedding_model", "deterministic-384")),
        embedding_dimension=_as_int(data.get("embedding_dimension"), 384),
        top_k=max(1, min(_as_int(data.get("top_k"), 8), 50)),
        max_chars=max(500, _as_int(data.get("max_chars"), 3000)),
        min_score=_as_float(data.get("min_score"), 0.35),
        mode=str(data.get("mode", "mirror") or "mirror"),
        include_types=list(include_types),
    )


def write_config(values: dict[str, Any], hermes_home: str | Path) -> None:
    path = Path(hermes_home) / "milvus.json"
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing.update(values)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

