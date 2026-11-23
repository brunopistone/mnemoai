"""Cancelled document summaries must release pools and never cache late replies."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import FastMCP

from mnemoai.client.local_mcp import LocalMCPClient
from mnemoai.models.controllers import llm_controller
from mnemoai.server import factory
from mnemoai.server.tools.readers import chunking_helper as chunks
from mnemoai.server.tools.thread_offload import ThreadedToolServer


def test_cancel_releases_chunk_workers_before_provider_replies(monkeypatch, tmp_path):
    started = threading.Barrier(3)
    release, finished, cancelled = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    writes = []

    class Model:
        def invoke(self, messages):
            started.wait(timeout=5)
            release.wait(5)
            return SimpleNamespace(content="summary")

    class Controller:
        def initialize_model(self):
            pass

        def get_model(self):
            return Model()

    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path))
    monkeypatch.setattr(llm_controller, "LangChainLLMController", Controller)
    monkeypatch.setattr(chunks, "config", {"CHUNKING_CONCURRENCY": 2})
    monkeypatch.setattr(chunks, "__count_tokens", len)
    monkeypatch.setattr(
        chunks, "__split_into_chunks", lambda *args: ["first", "second"]
    )
    monkeypatch.setattr(chunks, "_get_cached_summary", lambda key: None)
    monkeypatch.setattr(
        chunks, "_set_cached_summary", lambda *args: writes.append(args)
    )
    server = FastMCP("summaries")
    threaded = ThreadedToolServer(server)

    @threaded.tool()
    def summarize() -> str:
        try:
            result, _ = chunks.process_large_content("long input", chunk_size=2)
            return result
        finally:
            finished.set()

    monkeypatch.setattr(factory, "create_server", lambda: server)
    client = LocalMCPClient()
    client._cancel_probe = cancelled.is_set
    try:
        client.__enter__()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(client.call_tool_sync, "summarize", {})
            try:
                started.wait(timeout=5)
                cancelled.set()
                with pytest.raises(KeyboardInterrupt):
                    pending.result(timeout=3)
                assert finished.wait(1), "Chunk pool still waits on the provider"
                client.shutdown()
                assert not release.is_set()
                assert writes == []
            finally:
                release.set()
    finally:
        release.set()
        client.shutdown()
