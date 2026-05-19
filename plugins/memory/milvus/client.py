"""Milvus client wrapper."""

from __future__ import annotations

from .config import MilvusConfig


class MilvusClientWrapper:
    """Thin wrapper around pymilvus MilvusClient.

    Importing pymilvus is delayed so Hermes can start without Milvus
    dependencies when the provider is not enabled.
    """

    def __init__(self, config: MilvusConfig):
        self.config = config
        self.client = None

    @staticmethod
    def dependency_available() -> bool:
        try:
            import pymilvus  # noqa: F401

            return True
        except Exception:
            return False

    def connect(self):
        if self.client is not None:
            return self.client
        try:
            from pymilvus import MilvusClient
        except Exception as exc:
            raise RuntimeError("pymilvus package is required for Milvus memory") from exc
        kwargs = {"uri": self.config.uri}
        if self.config.token:
            kwargs["token"] = self.config.token
        if self.config.database:
            kwargs["db_name"] = self.config.database
        self.client = MilvusClient(**kwargs)
        return self.client

    def close(self) -> None:
        client = self.client
        self.client = None
        close = getattr(client, "close", None)
        if callable(close):
            close()

