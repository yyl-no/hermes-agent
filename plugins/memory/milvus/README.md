# Milvus Memory Provider

This directory contains the Hermes Milvus memory provider for stage 1.

Enable it in `config.yaml`:

```yaml
memory:
  provider: milvus
```

Stage 1 runs in mirror mode. It does not replace `MEMORY.md`, `USER.md`,
SQLite `state.db`, or `session_search`.
