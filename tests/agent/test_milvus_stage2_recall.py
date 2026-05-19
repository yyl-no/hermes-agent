"""Stage 2 Milvus recall integration tests."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plugins.memory.milvus import MilvusMemoryProvider
from plugins.memory.milvus.config import MilvusConfig
from plugins.memory.milvus.store import SearchResult
from run_agent import AIAgent


class FakeStore:
    def __init__(self, results=None):
        self.initialized = False
        self.closed = False
        self.search_calls = []
        self.results = list(results or [])

    def initialize(self):
        self.initialized = True

    def close(self):
        self.closed = True

    def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return list(self.results)


class FailingStore(FakeStore):
    def search(self, query, **kwargs):
        raise RuntimeError("milvus unavailable")


def _provider(store, *, max_chars=1200):
    cfg = MilvusConfig(
        uri="mock://milvus",
        embedding_dimension=8,
        top_k=4,
        max_chars=max_chars,
        min_score=0.0,
    )
    provider = MilvusMemoryProvider(config=cfg, store=store)
    provider.initialize("session-1", platform="cli")
    return provider


def test_prefetch_filters_types_and_formats_without_context_fence():
    store = FakeStore(
        [
            SearchResult(
                text="User prefers concise answers.",
                score=0.82,
                metadata={
                    "memory_type": "curated_memory",
                    "source": "memory_tool",
                    "session_id": "session-1",
                },
            )
        ]
    )
    provider = _provider(store)

    rendered = provider.prefetch("how should I answer?", session_id="session-1")

    assert rendered.startswith("Milvus recalled memory:")
    assert "User prefers concise answers." in rendered
    assert "<memory-context>" not in rendered
    assert "</memory-context>" not in rendered
    assert store.search_calls[0][1]["include_types"] == [
        "curated_memory",
        "turn",
        "summary",
    ]
    provider.shutdown()


def test_prefetch_sorts_deduplicates_and_truncates_results():
    long_text = "Long memory " + ("x" * 900)
    store = FakeStore(
        [
            SearchResult(
                text="lower score",
                score=0.2,
                metadata={"memory_type": "turn", "source": "conversation"},
            ),
            SearchResult(
                text=long_text,
                score=0.9,
                metadata={"memory_type": "curated_memory", "source": "memory_tool"},
            ),
            SearchResult(
                text=long_text,
                score=0.8,
                metadata={"memory_type": "curated_memory", "source": "memory_tool"},
            ),
        ]
    )
    provider = _provider(store, max_chars=500)

    rendered = provider.prefetch("long memory")

    assert rendered.index("score=0.90") < rendered.index("score=0.20")
    assert rendered.count("Long memory") == 1
    assert "..." in rendered
    assert len(rendered) <= 500
    provider.shutdown()


def test_prefetch_failure_returns_empty_context():
    provider = _provider(FailingStore())

    assert provider.prefetch("anything") == ""

    provider.shutdown()


class FakeMemoryManager:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def prefetch_all(self, query, *, session_id=""):
        self.calls.append((query, session_id))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_agent_builds_fenced_external_memory_context_for_turn():
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "session-abc"
    agent._memory_manager = FakeMemoryManager("Milvus recalled memory:\n\n1. fact")

    block = agent._build_external_memory_context_for_turn("remember this?")

    assert block.startswith("<memory-context>")
    assert "NOT new user input" in block
    assert "Milvus recalled memory" in block
    assert agent._memory_manager.calls == [("remember this?", "session-abc")]


def test_agent_external_memory_context_is_best_effort():
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "session-abc"
    agent._memory_manager = FakeMemoryManager(RuntimeError("offline"))

    assert agent._build_external_memory_context_for_turn("remember this?") == ""


def test_agent_external_memory_context_skips_empty_query():
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "session-abc"
    agent._memory_manager = FakeMemoryManager("memory")

    assert agent._build_external_memory_context_for_turn("   ") == ""
    assert agent._memory_manager.calls == []
