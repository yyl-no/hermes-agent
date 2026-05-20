"""CLI hooks for the Milvus memory provider."""

from __future__ import annotations


def register_cli(subparser):
    subparser.add_argument("--healthcheck", action="store_true", help="Check config/deps")
    subparser.set_defaults(func=_handle)


def _handle(args):
    if args.healthcheck:
        from .client import MilvusClientWrapper
        from .config import load_config

        cfg = load_config()
        print(f"Milvus URI configured: {bool(cfg.uri)}")
        print(f"pymilvus installed: {MilvusClientWrapper.dependency_available()}")
        print(f"collection: {cfg.collection}")
        return
    print("Use --healthcheck to verify Milvus memory configuration.")
