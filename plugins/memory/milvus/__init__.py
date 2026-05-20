"""Milvus memory provider for Hermes.

Enable with:

    memory:
      provider: milvus
"""

from __future__ import annotations

import json
import logging
import threading
import time
from queue import Queue
from typing import Any, Dict, List

from agent.memory_manager import sanitize_context
from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .client import MilvusClientWrapper
from .config import MilvusConfig, load_config, normalize_memory_mode, write_config
from .store import MilvusMemoryStore, SearchResult

logger = logging.getLogger(__name__)


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _split_markdown_entries(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    has_delimiter = "\n§\n" in text
    chunks = text.split("\n§\n") if has_delimiter else text.splitlines()
    entries: list[str] = []
    for chunk in chunks:
        item = chunk.strip()
        if item.startswith(("-", "*")):
            item = item[1:].strip()
        if item and item != "§" and not item.startswith("#"):
            entries.append(item)
    return entries


SEARCH_SCHEMA = {
    "name": "milvus_search",
    "description": "Search Milvus long-term memory by semantic similarity.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "top_k": {
                "type": "integer",
                "description": "Maximum results (default 5, max 20).",
            },
            "memory_type": {
                "type": "string",
                "description": "Optional memory type filter.",
            },
            "session_id": {
                "type": "string",
                "description": "Optional session id filter.",
            },
        },
        "required": ["query"],
    },
}


REMEMBER_SCHEMA = {
    "name": "milvus_remember",
    "description": "Store a durable memory in Milvus.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Memory content."},
            "memory_type": {
                "type": "string",
                "description": "Memory type (default curated_memory).",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags.",
            },
        },
        "required": ["content"],
    },
}


class MilvusMemoryProvider(MemoryProvider):
    """Milvus-backed mirror memory provider."""

    def __init__(
        self,
        config: MilvusConfig | None = None,
        store: MilvusMemoryStore | None = None,
    ):
        self._config = config
        self._store = store
        self._session_id = ""
        self._platform = "cli"
        self._profile = ""
        self._workspace = ""
        self._user_id = ""
        self._chat_id = ""
        self._mode = normalize_memory_mode(config.mode if config else "mirror")
        self._markdown_mirror = True
        self._write_queue: Queue[tuple[str, dict[str, Any]]] = Queue()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_prefetch = ""
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "milvus"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def markdown_mirror(self) -> bool:
        return self._markdown_mirror

    def is_available(self) -> bool:
        cfg = self._config or load_config()
        return bool(cfg.uri) and MilvusClientWrapper.dependency_available()

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home")
        self._config = self._config or load_config(hermes_home)
        self._store = self._store or MilvusMemoryStore(self._config)
        self._session_id = session_id or ""
        self._platform = kwargs.get("platform") or "cli"
        self._profile = kwargs.get("agent_identity") or kwargs.get("profile") or ""
        self._workspace = kwargs.get("agent_workspace") or kwargs.get("workspace") or ""
        self._user_id = kwargs.get("user_id") or ""
        self._chat_id = kwargs.get("chat_id") or ""
        self._mode = normalize_memory_mode(
            kwargs.get("memory_mode") or self._config.mode
        )
        self._markdown_mirror = _as_bool(kwargs.get("markdown_mirror"), True)
        self._store.initialize()
        self._start_worker()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query or not query.strip() or not self._store:
            return ""
        try:
            include_types = self._config.include_types if self._config else None
            if not include_types:
                include_types = ["curated_memory", "turn", "summary"]
            top_k = self._prefetch_top_k()
            results = self._store.search(
                query,
                top_k=top_k,
                include_types=include_types,
            )
            rendered = self._render_results(results)
            with self._lock:
                self._last_prefetch = rendered
            logger.debug(
                "Milvus prefetch mode=%s query chars=%d results=%d context chars=%d",
                self._mode,
                len(query),
                len(results),
                len(rendered),
            )
            return rendered
        except Exception as exc:
            logger.debug("Milvus prefetch failed: %s", exc, exc_info=True)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        # The provider does synchronous bounded prefetch. This hook is kept as a
        # no-op so turn completion stays cheap in mirror mode.
        return None

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        text = f"User: {user_content}\nAssistant: {assistant_content}".strip()
        metadata = self._base_metadata(session_id=session_id or self._session_id)
        metadata.update({"memory_type": "turn", "source": "conversation", "role": "turn"})
        self._enqueue(text, metadata)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, REMEMBER_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if tool_name == "milvus_search":
                return self._tool_search(args)
            if tool_name == "milvus_remember":
                return self._tool_remember(args)
            return tool_error(f"Unknown Milvus memory tool: {tool_name}")
        except Exception as exc:
            logger.warning("Milvus tool %s failed: %s", tool_name, exc, exc_info=True)
            return tool_error(f"Milvus tool failed: {exc}")

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        if action not in {"add", "replace"} or not content:
            return
        record_meta = self._base_metadata(
            session_id=(metadata or {}).get("session_id") or self._session_id
        )
        record_meta.update(
            {
                "memory_type": "curated_memory",
                "source": "memory_tool",
                "target": target or "memory",
                "action": action,
                "provenance": dict(metadata or {}),
            }
        )
        self._enqueue(content, record_meta)

    def write_memory(
        self,
        action: str,
        target: str,
        content: str = "",
        *,
        old_text: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        if target not in {"memory", "user"}:
            return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False)
        if action not in {"add", "replace", "remove"}:
            return tool_error(f"Unknown action '{action}'. Use: add, replace, remove", success=False)
        content = str(content or "").strip()
        old_text = str(old_text or "").strip()
        if action in {"add", "replace"} and not content:
            return tool_error("Content is required for this memory action.", success=False)
        if action in {"replace", "remove"} and not old_text:
            return tool_error("old_text is required for this memory action.", success=False)
        if not self._store:
            return tool_error("Milvus provider is not initialized", success=False)
        include_types = ["user_profile"] if target == "user" else ["curated_memory"]
        if action == "remove":
            removed = self._store.delete_by_text(old_text, include_types=include_types)
            if not removed:
                return tool_error(f"No Milvus memory matched '{old_text}'.", success=False)
            return json.dumps(
                {
                    "success": True,
                    "target": target,
                    "message": "Entry removed from Milvus memory.",
                    "removed": removed,
                    "mode": self._mode,
                    "markdown_mirror": self._markdown_mirror,
                },
                ensure_ascii=False,
            )
        removed = 0
        if action == "replace":
            removed = self._store.delete_by_text(old_text, include_types=include_types)
            if not removed:
                return tool_error(f"No Milvus memory matched '{old_text}'.", success=False)

        record_meta = self._base_metadata(
            session_id=(metadata or {}).get("session_id") or self._session_id
        )
        record_meta.update(
            {
                "memory_type": "user_profile" if target == "user" else "curated_memory",
                "source": "memory_tool",
                "target": target,
                "action": action,
                "old_text": old_text or "",
                "provenance": dict(metadata or {}),
            }
        )
        record_id = self._store.insert_record(content, record_meta)
        return json.dumps(
            {
                "success": True,
                "target": target,
                "message": "Entry stored in Milvus memory.",
                "id": record_id,
                "entry_count": None,
                "entries": [content],
                "mode": self._mode,
                "markdown_mirror": self._markdown_mirror,
                "removed": removed,
            },
            ensure_ascii=False,
        )

    def build_stable_memory_block(self, max_chars: int = 3000) -> str:
        if self._mode != "exclusive" or not self._store:
            return ""
        try:
            results = self._store.list_records(
                include_types=["user_profile", "curated_memory"],
                limit=50,
            )
        except Exception as exc:
            logger.debug("Milvus stable memory block failed: %s", exc, exc_info=True)
            return ""
        if not results:
            return ""

        user_items: list[SearchResult] = []
        memory_items: list[SearchResult] = []
        for result in results:
            if result.metadata.get("memory_type") == "user_profile":
                user_items.append(result)
            else:
                memory_items.append(result)

        parts = ["Milvus persistent memory:"]
        used = len(parts[0])

        def add_section(title: str, items: list[SearchResult]) -> None:
            nonlocal used
            if not items:
                return
            section_lines = [title]
            for idx, item in enumerate(items, 1):
                text = sanitize_context(item.text).strip()
                if not text:
                    continue
                line = f"{idx}. {text[:700].rstrip()}"
                block_size = len(line) + 1
                if used + len(title) + block_size > max_chars:
                    break
                section_lines.append(line)
                used += block_size
            if len(section_lines) > 1:
                parts.append("\n".join(section_lines))
                used += len(title)

        add_section("User profile:", user_items)
        add_section("Long-term memory:", memory_items)
        return "\n\n".join(parts) if len(parts) > 1 else ""

    def import_markdown_memory(self, memory_text: str = "", user_text: str = "") -> dict[str, Any]:
        if not self._store:
            return {"success": False, "error": "Milvus provider is not initialized"}
        report = {"memory": 0, "user": 0, "skipped_duplicate": 0, "failed": 0}
        seen: set[tuple[str, str]] = set()
        for target, raw in (("memory", memory_text), ("user", user_text)):
            for entry in _split_markdown_entries(raw):
                key = (target, " ".join(entry.lower().split()))
                if key in seen:
                    report["skipped_duplicate"] += 1
                    continue
                seen.add(key)
                payload = json.loads(
                    self.write_memory(
                        "add",
                        target,
                        entry,
                        metadata={
                            "source": "migration",
                            "tool_name": "milvus_import",
                        },
                    )
                )
                if payload.get("success"):
                    report[target] += 1
                else:
                    report["failed"] += 1
        report["success"] = report["failed"] == 0
        return report

    def export_markdown_memory(self) -> dict[str, Any]:
        if not self._store:
            return {"success": False, "error": "Milvus provider is not initialized"}
        records = self._store.list_records(
            include_types=["user_profile", "curated_memory"],
            limit=200,
        )
        memory_entries: list[str] = []
        user_entries: list[str] = []
        seen: set[tuple[str, str]] = set()
        for record in records:
            target = "user" if record.metadata.get("memory_type") == "user_profile" else "memory"
            text = sanitize_context(record.text).strip()
            if not text:
                continue
            key = (target, " ".join(text.lower().split()))
            if key in seen:
                continue
            seen.add(key)
            if target == "user":
                user_entries.append(text)
            else:
                memory_entries.append(text)
        return {
            "success": True,
            "memory": "\n§\n".join(memory_entries),
            "user": "\n§\n".join(user_entries),
            "memory_count": len(memory_entries),
            "user_count": len(user_entries),
        }

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        self._session_id = new_session_id or self._session_id

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not messages:
            return ""
        text = "\n".join(
            str(m.get("content", ""))
            for m in messages[-12:]
            if isinstance(m, dict) and m.get("content")
        )
        if not text:
            return ""
        metadata = self._base_metadata(session_id=self._session_id)
        metadata.update({"memory_type": "summary", "source": "compression"})
        self._enqueue(text[:4000], metadata)
        return ""

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs,
    ) -> None:
        text = f"Delegated task: {task}\nResult: {result}".strip()
        if not text:
            return
        metadata = self._base_metadata(session_id=self._session_id)
        metadata.update(
            {
                "memory_type": "summary",
                "source": "delegation",
                "child_session_id": child_session_id,
            }
        )
        self._enqueue(text, metadata)

    def shutdown(self) -> None:
        self._stop.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=5)
        if self._store:
            try:
                self._store.close()
            except Exception:
                pass

    def get_config_schema(self):
        return [
            {
                "key": "uri",
                "description": "Milvus URI, for example http://localhost:19530",
                "required": True,
            },
            {
                "key": "token",
                "description": "Milvus/Zilliz token",
                "secret": True,
                "required": False,
                "env_var": "MILVUS_TOKEN",
            },
            {"key": "database", "description": "Milvus database", "default": "default"},
            {
                "key": "collection",
                "description": "Milvus collection",
                "default": "hermes_memory",
            },
            {
                "key": "embedding_provider",
                "description": "Embedding provider",
                "default": "deterministic",
                "choices": ["deterministic", "openai"],
            },
            {
                "key": "embedding_dimension",
                "description": "Embedding dimension",
                "default": "384",
            },
        ]

    def save_config(self, values, hermes_home):
        write_config(values, hermes_home)

    def _tool_search(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query") or "")
        top_k = max(1, min(int(args.get("top_k") or 5), 20))
        if not self._store:
            return tool_error("Milvus provider is not initialized", success=False)
        results = self._store.search(
            query,
            top_k=top_k,
            memory_type=args.get("memory_type") or None,
            session_id=args.get("session_id") or None,
        )
        return json.dumps(
            {
                "success": True,
                "query": query,
                "results": [self._result_to_dict(r) for r in results],
                "count": len(results),
            },
            ensure_ascii=False,
        )

    def _tool_remember(self, args: Dict[str, Any]) -> str:
        content = str(args.get("content") or "").strip()
        if not content:
            return tool_error("content is required", success=False)
        memory_type = str(args.get("memory_type") or "curated_memory")
        metadata = self._base_metadata(session_id=self._session_id)
        metadata.update(
            {
                "memory_type": memory_type,
                "source": "milvus_remember",
                "tags": args.get("tags") or [],
            }
        )
        if not self._store:
            return tool_error("Milvus provider is not initialized", success=False)
        record_id = self._store.insert_record(content, metadata)
        return json.dumps({"success": True, "id": record_id}, ensure_ascii=False)

    def _enqueue(self, text: str, metadata: Dict[str, Any]) -> None:
        if not text or not text.strip():
            return
        self._start_worker()
        self._write_queue.put((text, metadata))

    def _start_worker(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="milvus-memory-writer",
            daemon=True,
        )
        self._worker.start()

    def _worker_loop(self) -> None:
        while not self._stop.is_set() or not self._write_queue.empty():
            try:
                text, metadata = self._write_queue.get(timeout=0.2)
            except Exception:
                continue
            try:
                if self._store:
                    self._store.insert_record(text, metadata)
            except Exception as exc:
                logger.debug("Milvus memory write failed: %s", exc, exc_info=True)
            finally:
                self._write_queue.task_done()

    def _base_metadata(self, *, session_id: str = "") -> Dict[str, Any]:
        now = int(time.time())
        return {
            "session_id": session_id or self._session_id,
            "parent_session_id": "",
            "platform": self._platform,
            "profile": self._profile,
            "workspace": self._workspace,
            "user_id": self._user_id,
            "chat_id": self._chat_id,
            "created_at": now,
            "updated_at": now,
        }

    def _render_results(self, results: List[SearchResult]) -> str:
        if not results:
            return ""
        max_chars = self._prefetch_max_chars()
        per_result_chars = min(700, max(120, max_chars // 2))
        parts = ["Milvus recalled memory:"]
        used = len(parts[0])
        seen: set[str] = set()
        rendered_count = 0
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        for result in sorted_results:
            text = sanitize_context(result.text).strip()
            if not text:
                continue
            normalized = " ".join(text.lower().split())
            if normalized in seen:
                continue
            seen.add(normalized)
            if len(text) > per_result_chars:
                text = text[: per_result_chars - 3].rstrip() + "..."
            meta = result.metadata
            rendered_count += 1
            header = (
                f"{rendered_count}. [{meta.get('memory_type', 'memory')}, "
                f"score={result.score:.2f}, source={meta.get('source', 'unknown')}]"
            )
            session_id = meta.get("session_id")
            if session_id:
                header = header[:-1] + f", session={session_id}]"
            block = f"{header}\n   {text}"
            if used + len(block) > max_chars:
                rendered_count -= 1
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts) if len(parts) > 1 else ""

    def _prefetch_top_k(self) -> int:
        configured = self._config.top_k if self._config else 8
        if self._mode == "mirror":
            return max(1, min(configured, 4))
        if self._mode == "exclusive":
            return max(1, min(configured, 16))
        if self._mode == "primary":
            return max(1, min(configured, 12))
        return max(1, min(configured, 8))

    def _prefetch_max_chars(self) -> int:
        configured = self._config.max_chars if self._config else 3000
        if self._mode == "mirror":
            return max(500, min(configured, 1500))
        if self._mode == "exclusive":
            return max(2500, configured)
        if self._mode == "primary":
            return max(2000, configured)
        return configured

    @staticmethod
    def _result_to_dict(result: SearchResult) -> Dict[str, Any]:
        return {
            "text": result.text,
            "score": result.score,
            "metadata": result.metadata,
        }


def register(ctx):
    ctx.register_memory_provider(MilvusMemoryProvider())
