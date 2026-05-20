"""Stage 3 Milvus memory mode switching tests."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes_cli.config import DEFAULT_CONFIG
from plugins.memory.milvus import MilvusMemoryProvider
from plugins.memory.milvus.config import (
    MilvusConfig,
    load_config,
    normalize_memory_mode,
)
from plugins.memory.milvus.store import SearchResult
from run_agent import AIAgent


class FakeStore:
    def __init__(self):
        self.initialized = False
        self.closed = False
        self.search_calls = []
        self.results = [
            SearchResult(
                text="Stage3 mode switching memory.",
                score=0.9,
                metadata={"memory_type": "curated_memory", "source": "test"},
            )
        ]

    def initialize(self):
        self.initialized = True

    def close(self):
        self.closed = True

    def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return list(self.results)


def _provider(mode="mirror", *, top_k=20, max_chars=3000):
    cfg = MilvusConfig(
        uri="mock://milvus",
        mode=mode,
        top_k=top_k,
        max_chars=max_chars,
        min_score=0.0,
        embedding_dimension=8,
    )
    store = FakeStore()
    provider = MilvusMemoryProvider(config=cfg, store=store)
    return provider, store


def test_default_config_exposes_stage3_memory_mode_controls():
    memory_cfg = DEFAULT_CONFIG["memory"]

    assert memory_cfg["mode"] == "mirror"
    assert memory_cfg["markdown_mirror"] is True


def test_normalize_memory_mode_rejects_unknown_values():
    assert normalize_memory_mode("mirror") == "mirror"
    assert normalize_memory_mode("HYBRID") == "hybrid"
    assert normalize_memory_mode("primary") == "primary"
    assert normalize_memory_mode("exclusive") == "mirror"
    assert normalize_memory_mode("") == "mirror"


def test_milvus_json_mode_is_loaded_and_normalized(tmp_path):
    (tmp_path / "milvus.json").write_text(
        json.dumps({"uri": "mock://milvus", "mode": "primary"}),
        encoding="utf-8",
    )

    cfg = load_config(tmp_path)

    assert cfg.uri == "mock://milvus"
    assert cfg.mode == "primary"


def test_milvus_invalid_json_mode_falls_back_to_mirror(tmp_path):
    (tmp_path / "milvus.json").write_text(
        json.dumps({"uri": "mock://milvus", "mode": "exclusive"}),
        encoding="utf-8",
    )

    cfg = load_config(tmp_path)

    assert cfg.mode == "mirror"


def test_provider_initialize_accepts_memory_mode_and_markdown_mirror():
    provider, store = _provider("mirror")

    provider.initialize(
        "stage3-session",
        platform="cli",
        memory_mode="hybrid",
        markdown_mirror=False,
    )

    assert provider.mode == "hybrid"
    assert provider.markdown_mirror is False
    assert store.initialized is True
    provider.shutdown()


def test_provider_invalid_memory_mode_falls_back_to_mirror():
    provider, _store = _provider("primary")

    provider.initialize("stage3-session", memory_mode="exclusive")

    assert provider.mode == "mirror"
    provider.shutdown()


def test_prefetch_mode_controls_recall_budget():
    mirror, mirror_store = _provider("mirror", top_k=20, max_chars=4000)
    hybrid, hybrid_store = _provider("hybrid", top_k=20, max_chars=4000)
    primary, primary_store = _provider("primary", top_k=20, max_chars=4000)

    mirror.initialize("stage3")
    hybrid.initialize("stage3")
    primary.initialize("stage3")

    assert mirror.prefetch("stage3")
    assert hybrid.prefetch("stage3")
    assert primary.prefetch("stage3")

    assert mirror_store.search_calls[0][1]["top_k"] == 4
    assert hybrid_store.search_calls[0][1]["top_k"] == 8
    assert primary_store.search_calls[0][1]["top_k"] == 12

    mirror.shutdown()
    hybrid.shutdown()
    primary.shutdown()


def test_run_agent_init_passes_memory_mode_controls_to_provider():
    source = inspect.getsource(AIAgent.__init__)

    assert '"memory_mode": mem_config.get("mode", "mirror")' in source
    assert '"markdown_mirror": mem_config.get("markdown_mirror", True)' in source
