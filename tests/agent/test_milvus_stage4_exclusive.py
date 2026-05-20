"""Stage 4 Milvus exclusive memory mode tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.memory_manager import MemoryManager
from hermes_cli.config import DEFAULT_CONFIG
from plugins.memory.milvus import MilvusMemoryProvider
from plugins.memory.milvus.config import MilvusConfig, normalize_memory_mode
from plugins.memory.milvus.store import SearchResult
from run_agent import AIAgent


class FakeStore:
    def __init__(self):
        self.initialized = False
        self.closed = False
        self.inserted = []
        self.records = []

    def initialize(self):
        self.initialized = True

    def close(self):
        self.closed = True

    def insert_record(self, text, metadata):
        record = (text, dict(metadata))
        self.inserted.append(record)
        self.records.append(
            SearchResult(text=text, score=1.0, metadata=dict(metadata))
        )
        return f"id-{len(self.inserted)}"

    def list_records(self, **kwargs):
        include_types = set(kwargs.get("include_types") or [])
        if not include_types:
            return list(self.records)
        return [
            r for r in self.records
            if r.metadata.get("memory_type") in include_types
        ]

    def delete_by_text(self, old_text, **kwargs):
        before = len(self.records)
        self.records = [r for r in self.records if old_text not in r.text]
        return before - len(self.records)


def _provider():
    cfg = MilvusConfig(
        uri="mock://milvus",
        mode="exclusive",
        embedding_dimension=8,
        max_chars=2000,
        min_score=0.0,
    )
    store = FakeStore()
    provider = MilvusMemoryProvider(config=cfg, store=store)
    provider.initialize(
        "stage4-session",
        platform="cli",
        memory_mode="exclusive",
        markdown_mirror=False,
    )
    return provider, store


def test_stage4_config_defaults_keep_safe_fallbacks():
    memory_cfg = DEFAULT_CONFIG["memory"]

    assert memory_cfg["mode"] == "mirror"
    assert memory_cfg["markdown_mirror"] is True
    assert memory_cfg["fallback_to_markdown"] is True


def test_exclusive_mode_is_valid():
    assert normalize_memory_mode("exclusive") == "exclusive"


def test_provider_write_memory_adds_curated_memory_to_milvus():
    provider, store = _provider()

    raw = provider.write_memory(
        "add",
        "memory",
        "Stage4 exclusive memory writes to Milvus.",
        metadata={"session_id": "stage4-session", "tool_name": "memory"},
    )
    payload = json.loads(raw)

    assert payload["success"] is True
    assert payload["mode"] == "exclusive"
    assert payload["markdown_mirror"] is False
    text, metadata = store.inserted[0]
    assert text == "Stage4 exclusive memory writes to Milvus."
    assert metadata["memory_type"] == "curated_memory"
    assert metadata["source"] == "memory_tool"
    provider.shutdown()


def test_provider_write_memory_user_target_becomes_user_profile():
    provider, store = _provider()

    raw = provider.write_memory("add", "user", "User prefers direct answers.")
    payload = json.loads(raw)

    assert payload["success"] is True
    assert store.inserted[0][1]["memory_type"] == "user_profile"
    provider.shutdown()


def test_provider_write_memory_remove_deletes_matching_memory():
    provider, _store = _provider()
    provider.write_memory("add", "memory", "Remove this Milvus memory.")

    raw = provider.write_memory("remove", "memory", old_text="Remove this")
    payload = json.loads(raw)

    assert payload["success"] is True
    assert payload["removed"] == 1
    provider.shutdown()


def test_stable_memory_block_prioritizes_user_profile_then_memory():
    provider, _store = _provider()
    provider.write_memory("add", "memory", "Project convention lives in Milvus.")
    provider.write_memory("add", "user", "User likes concise summaries.")

    block = provider.build_stable_memory_block()

    assert "Milvus persistent memory:" in block
    assert block.index("User profile:") < block.index("Long-term memory:")
    assert "User likes concise summaries." in block
    assert "Project convention lives in Milvus." in block
    provider.shutdown()


def test_memory_manager_routes_exclusive_memory_write():
    provider, _store = _provider()
    manager = MemoryManager()
    manager.add_provider(provider)

    raw = manager.write_memory("add", "memory", "Manager routed memory.")
    payload = json.loads(raw)

    assert payload["success"] is True
    provider.shutdown()


def test_provider_import_and_export_markdown_memory():
    provider, _store = _provider()

    report = provider.import_markdown_memory(
        memory_text="- Project uses Milvus\n- Project uses Milvus",
        user_text="- User prefers concise replies",
    )
    exported = provider.export_markdown_memory()

    assert report["success"] is True
    assert report["memory"] == 1
    assert report["user"] == 1
    assert report["skipped_duplicate"] == 1
    assert "Project uses Milvus" in exported["memory"]
    assert "User prefers concise replies" in exported["user"]
    provider.shutdown()


class FakeExternalMemoryManager:
    def __init__(self, *, write_result=None):
        self.writes = []
        self.write_result = write_result

    def write_memory(self, action, target, content, *, old_text="", metadata=None):
        self.writes.append((action, target, content, old_text, metadata or {}))
        if self.write_result is not None:
            return self.write_result
        return json.dumps({"success": True, "target": target, "entries": [content]})


def test_agent_memory_tool_routes_to_external_provider_in_exclusive_mode():
    agent = AIAgent.__new__(AIAgent)
    agent._memory_provider_mode = "exclusive"
    agent._memory_manager = FakeExternalMemoryManager()
    agent._memory_store = object()
    agent.session_id = "stage4-agent"
    agent._parent_session_id = ""
    agent.platform = "cli"
    agent._memory_write_origin = "assistant_tool"
    agent._memory_write_context = "foreground"

    raw = agent._handle_memory_tool_call(
        {
            "action": "add",
            "target": "memory",
            "content": "Exclusive routed memory.",
        },
        effective_task_id="task-1",
        tool_call_id="call-1",
    )
    payload = json.loads(raw)

    assert payload["success"] is True
    assert agent._memory_manager.writes[0][0:3] == (
        "add",
        "memory",
        "Exclusive routed memory.",
    )


class FakeMarkdownMemoryStore:
    def __init__(self):
        self.calls = []

    def add(self, target, content):
        self.calls.append(("add", target, content))
        return {"success": True, "target": target, "entries": [content]}

    def replace(self, target, old_text, new_content):
        self.calls.append(("replace", target, old_text, new_content))
        return {"success": True, "target": target, "entries": [new_content]}

    def remove(self, target, old_text):
        self.calls.append(("remove", target, old_text))
        return {"success": True, "target": target, "entries": []}


def test_agent_exclusive_memory_tool_falls_back_to_markdown_on_provider_error():
    agent = AIAgent.__new__(AIAgent)
    agent._memory_provider_mode = "exclusive"
    agent._memory_fallback_to_markdown = True
    agent._memory_manager = FakeExternalMemoryManager(
        write_result=json.dumps({"success": False, "error": "Milvus down"})
    )
    agent._memory_store = FakeMarkdownMemoryStore()
    agent.session_id = "stage4-agent"
    agent._parent_session_id = ""
    agent.platform = "cli"
    agent._memory_write_origin = "assistant_tool"
    agent._memory_write_context = "foreground"

    raw = agent._handle_memory_tool_call(
        {
            "action": "add",
            "target": "memory",
            "content": "Fallback Markdown memory.",
        }
    )
    payload = json.loads(raw)

    assert payload["success"] is True
    assert agent._memory_store.calls == [
        ("add", "memory", "Fallback Markdown memory.")
    ]


class FakeNonExclusiveMemoryManager:
    def __init__(self):
        self.events = []

    def on_memory_write(self, action, target, content, metadata=None):
        self.events.append((action, target, content, dict(metadata or {})))


def test_agent_nonexclusive_memory_tool_mirrors_remove_with_old_text():
    agent = AIAgent.__new__(AIAgent)
    agent._memory_provider_mode = "hybrid"
    agent._memory_manager = FakeNonExclusiveMemoryManager()
    agent._memory_store = FakeMarkdownMemoryStore()
    agent.session_id = "stage4-agent"
    agent._parent_session_id = ""
    agent.platform = "cli"
    agent._memory_write_origin = "assistant_tool"
    agent._memory_write_context = "foreground"

    raw = agent._handle_memory_tool_call(
        {
            "action": "remove",
            "target": "memory",
            "old_text": "Stale Milvus memory",
        }
    )
    payload = json.loads(raw)

    assert payload["success"] is True
    assert agent._memory_manager.events[0][0:3] == ("remove", "memory", "")
    assert agent._memory_manager.events[0][3]["old_text"] == "Stale Milvus memory"


def test_agent_exclusive_system_prompt_skips_markdown_block():
    agent = AIAgent.__new__(AIAgent)
    agent._memory_provider_mode = "exclusive"
    agent._memory_manager = MemoryManager()
    provider, _store = _provider()
    provider.write_memory("add", "memory", "Stable Milvus prompt memory.")
    agent._memory_manager.add_provider(provider)

    assert agent._external_memory_exclusive_active() is True
    block = agent._memory_manager.build_stable_memory_block()

    assert "Stable Milvus prompt memory." in block
    provider.shutdown()
