# Milvus Memory Provider

This directory contains the Hermes Milvus memory provider. In `exclusive`
mode, Hermes uses Milvus as the primary long-term semantic memory backend while
leaving the official SQLite session store and FTS5 session search intact.

This fork ships Milvus as an in-tree provider for convenience. Upstream Hermes
policy no longer accepts new providers under `plugins/memory/`; if you want to
submit this integration upstream, publish it as a standalone memory-provider
plugin or pip package instead.

Milvus replaces the long-term semantic memory path, not the whole Hermes
storage system:

```text
Sessions, messages, metadata -> SQLite state.db
Session full-text search     -> SQLite FTS5 / session_search
Long-term semantic memory    -> Milvus
Model reasoning and answers  -> configured LLM provider
```

## Install

From the repository root:

```bash
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"
```

The `[all]` extra includes `pymilvus`, so no separate Milvus Python client
install is required.

## Configure Hermes

Create or update `~/.hermes/config.yaml`:

```yaml
memory:
  provider: milvus
  mode: exclusive
  markdown_mirror: false
  fallback_to_markdown: true
```

The supported memory modes are:

- `mirror`: keep Markdown memory as the main path and mirror writes to Milvus.
- `hybrid`: use Milvus recall together with the built-in memory path.
- `primary`: prefer Milvus recall, with Markdown still available.
- `exclusive`: route long-term memory reads and writes through Milvus.

## Configure Milvus

Copy the example config:

```bash
mkdir -p ~/.hermes
cp plugins/memory/milvus/milvus.example.json ~/.hermes/milvus.json
```

Edit `~/.hermes/milvus.json` if your Milvus endpoint differs:

```json
{
  "uri": "http://localhost:19530",
  "database": "default",
  "collection": "hermes_milvus_memories",
  "embedding_provider": "deterministic",
  "embedding_model": "deterministic-384",
  "embedding_dimension": 384,
  "top_k": 8,
  "max_chars": 3000,
  "min_score": 0.0
}
```

`collection` is the Milvus collection Hermes will read and write. The default
example uses `hermes_milvus_memories` as a production-friendly name.

Environment variables can override the JSON file:

```bash
export MILVUS_URI=http://localhost:19530
export MILVUS_COLLECTION=hermes_milvus_memories
export MILVUS_DATABASE=default
```

## Run Hermes

```bash
hermes
```

or, from a development checkout:

```bash
python -m cli
```

Ask Hermes to remember a fact, then restart Hermes and ask for that fact again.
In `exclusive` mode the answer should come from persistent Milvus memory.

## Verify

Run the focused tests:

```bash
python -m pytest \
  tests/agent/test_milvus_stage1_plugin.py \
  tests/agent/test_milvus_stage2_recall.py \
  tests/agent/test_milvus_stage3_modes.py \
  tests/agent/test_milvus_stage4_exclusive.py \
  tests/agent/test_memory_provider.py \
  tests/agent/test_memory_session_switch.py \
  -q
```

Run a syntax check:

```bash
python -m py_compile \
  run_agent.py \
  agent/memory_manager.py \
  agent/memory_provider.py \
  hermes_cli/config.py \
  plugins/memory/milvus/*.py
```

## Notes

Milvus is a vector database, not an LLM. Hermes still calls the configured model
provider to understand user requests and write final answers. Milvus stores and
retrieves the long-term memory context that Hermes passes to the model.

Do not commit local user files such as `~/.hermes/config.yaml`,
`~/.hermes/milvus.json`, `~/.hermes/.env`, or your virtual environment.
