"""Stage 1 Milvus memory provider tests."""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plugins.memory.milvus import MilvusMemoryProvider
from plugins.memory.milvus.cli import register_cli
from plugins.memory.milvus.config import MilvusConfig
from plugins.memory.milvus.store import MilvusMemoryStore, SearchResult


class FakeStore:
    def __init__(self):
        self.initialized = False
        self.closed = False
        self.inserted = []
        self.deleted = []
        self.search_calls = []
        self.results = []

    def initialize(self):
        self.initialized = True

    def close(self):
        self.closed = True

    def insert_record(self, text, metadata):
        self.inserted.append((text, dict(metadata)))
        return f"id-{len(self.inserted)}"

    def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return list(self.results)

    def delete_by_text(self, old_text, **kwargs):
        self.deleted.append((old_text, kwargs))
        return 1


def _provider(fake_store=None):
    cfg = MilvusConfig(
        uri="mock://milvus",
        embedding_dimension=8,
        top_k=3,
        max_chars=1200,
        min_score=0.0,
    )
    return MilvusMemoryProvider(config=cfg, store=fake_store or FakeStore())


def test_provider_exposes_stage1_tools():
    provider = _provider()

    names = {schema["name"] for schema in provider.get_tool_schemas()}

    assert names == {"milvus_search", "milvus_remember"}


def test_initialize_records_runtime_scope():
    fake = FakeStore()
    provider = _provider(fake)

    provider.initialize(
        "session-1",
        platform="telegram",
        agent_identity="coder",
        agent_workspace="repo",
        user_id="user-1",
        chat_id="chat-1",
    )

    assert fake.initialized is True
    provider.shutdown()
    assert fake.closed is True


def test_sync_turn_enqueues_turn_record():
    fake = FakeStore()
    provider = _provider(fake)
    provider.initialize("session-1", platform="cli")

    provider.sync_turn("hello", "world", session_id="session-1")
    provider._write_queue.join()

    assert len(fake.inserted) == 1
    text, metadata = fake.inserted[0]
    assert text == "User: hello\nAssistant: world"
    assert metadata["memory_type"] == "turn"
    assert metadata["source"] == "conversation"
    assert metadata["session_id"] == "session-1"
    provider.shutdown()


def test_memory_tool_write_is_mirrored_as_curated_memory():
    fake = FakeStore()
    provider = _provider(fake)
    provider.initialize("session-1", platform="cli")

    provider.on_memory_write(
        "add",
        "user",
        "User prefers concise answers.",
        metadata={"session_id": "session-1", "tool_name": "memory"},
    )
    provider._write_queue.join()

    assert len(fake.inserted) == 1
    text, metadata = fake.inserted[0]
    assert text == "User prefers concise answers."
    assert metadata["memory_type"] == "user_profile"
    assert metadata["source"] == "memory_tool"
    assert metadata["target"] == "user"
    assert metadata["provenance"]["tool_name"] == "memory"
    provider.shutdown()


def test_memory_tool_replace_deletes_stale_milvus_memory_before_insert():
    fake = FakeStore()
    provider = _provider(fake)
    provider.initialize("session-1", platform="cli")

    provider.on_memory_write(
        "replace",
        "memory",
        "New project convention.",
        metadata={"old_text": "Old project convention", "session_id": "session-1"},
    )
    provider._write_queue.join()

    assert fake.deleted == [
        ("Old project convention", {"include_types": ["curated_memory"]})
    ]
    assert fake.inserted[0][0] == "New project convention."
    provider.shutdown()


def test_memory_tool_remove_deletes_milvus_memory_without_insert():
    fake = FakeStore()
    provider = _provider(fake)
    provider.initialize("session-1", platform="cli")

    provider.on_memory_write(
        "remove",
        "user",
        "",
        metadata={"old_text": "User prefers old style"},
    )
    provider._write_queue.join()

    assert fake.deleted == [
        ("User prefers old style", {"include_types": ["user_profile"]})
    ]
    assert fake.inserted == []
    provider.shutdown()


def test_prefetch_renders_sanitized_memory_context():
    fake = FakeStore()
    fake.results = [
        SearchResult(
            text="<memory-context>hidden</memory-context>Visible fact",
            score=0.91,
            metadata={"memory_type": "curated_memory", "source": "memory_tool"},
        )
    ]
    provider = _provider(fake)
    provider.initialize("session-1")

    rendered = provider.prefetch("what does the user prefer?")

    assert "Milvus recalled memory:" in rendered
    assert "Visible fact" in rendered
    assert "hidden" not in rendered
    assert fake.search_calls[0][0] == "what does the user prefer?"
    provider.shutdown()


def test_milvus_search_tool_returns_json():
    fake = FakeStore()
    fake.results = [
        SearchResult(
            text="A useful memory",
            score=0.8,
            metadata={"memory_type": "turn", "source": "conversation"},
        )
    ]
    provider = _provider(fake)
    provider.initialize("session-1")

    raw = provider.handle_tool_call("milvus_search", {"query": "useful", "top_k": 5})
    payload = json.loads(raw)

    assert payload["success"] is True
    assert payload["count"] == 1
    assert payload["results"][0]["text"] == "A useful memory"
    assert fake.search_calls[0][1]["top_k"] == 5
    provider.shutdown()


def test_milvus_remember_tool_writes_immediately():
    fake = FakeStore()
    provider = _provider(fake)
    provider.initialize("session-1")

    raw = provider.handle_tool_call(
        "milvus_remember",
        {"content": "Remember this.", "tags": ["test"]},
    )
    payload = json.loads(raw)

    assert payload["success"] is True
    assert payload["id"] == "id-1"
    text, metadata = fake.inserted[0]
    assert text == "Remember this."
    assert metadata["memory_type"] == "curated_memory"
    assert metadata["source"] == "milvus_remember"
    assert metadata["tags"] == ["test"]
    provider.shutdown()


class FakeMilvusClient:
    def __init__(self):
        self.created = []
        self.indexes = []
        self.loaded = []

    def has_collection(self, collection_name):
        return False

    def create_collection(self, **kwargs):
        self.created.append(kwargs)

    def create_index(self, **kwargs):
        self.indexes.append(kwargs)

    def load_collection(self, **kwargs):
        self.loaded.append(kwargs)


class FakeClientWrapper:
    def __init__(self, client):
        self.client = client

    def connect(self):
        return self.client

    def close(self):
        pass


def test_store_initialization_creates_collection_index_and_loads():
    cfg = MilvusConfig(
        uri="mock://milvus",
        collection="hermes_milvus_memories",
        embedding_dimension=384,
    )
    client = FakeMilvusClient()
    store = MilvusMemoryStore(cfg, client=FakeClientWrapper(client))

    store.initialize()

    assert client.created == [
        {
            "collection_name": "hermes_milvus_memories",
            "dimension": 384,
            "metric_type": "COSINE",
            "auto_id": False,
        }
    ]
    assert client.indexes == [
        {
            "collection_name": "hermes_milvus_memories",
            "index_params": {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            },
        }
    ]
    assert client.loaded == [{"collection_name": "hermes_milvus_memories"}]


def test_milvus_cli_registers_healthcheck_on_command_parser():
    parser = argparse.ArgumentParser()

    register_cli(parser)
    args = parser.parse_args(["--healthcheck"])

    assert args.healthcheck is True
    assert callable(args.func)


if __name__ == "__main__":
    for _name in sorted(name for name in globals() if name.startswith("test_")):
        globals()[_name]()
    print("executed stage 1 Milvus plugin tests")
