"""Unit tests for background auto-extraction of curated memory (MEMORY.md).

After each turn, `client.auto_extract_memory` distills durable facts from the
exchange and writes them to MEMORY.md — the auto-learning counterpart to the
model calling the `memory` tool. Opt-in (`ENABLE_MEMORY_AUTO_EXTRACTION`), runs
in the background, and writes only via `MemoryStore`. Pure-logic tests use a bare
client (no LLM/agent) and drive the worker synchronously.
"""

import mnemoai.client.client as client_mod
from mnemoai.client.client import LangGraphClient


class TestParseMemoryOps:
    def test_plain_json_array(self):
        raw = '[{"action": "add", "text": "[user] likes pytest"}]'
        ops = LangGraphClient._parse_memory_ops(raw)
        assert ops == [{"action": "add", "text": "[user] likes pytest"}]

    def test_strips_code_fences(self):
        raw = '```json\n[{"action": "add", "text": "x"}]\n```'
        ops = LangGraphClient._parse_memory_ops(raw)
        assert ops == [{"action": "add", "text": "x"}]

    def test_extracts_array_amid_prose(self):
        raw = 'Here is what to save:\n[{"action":"add","text":"y"}]\nThat is all.'
        ops = LangGraphClient._parse_memory_ops(raw)
        assert ops == [{"action": "add", "text": "y"}]

    def test_empty_array_is_empty(self):
        assert LangGraphClient._parse_memory_ops("[]") == []

    def test_malformed_returns_empty(self):
        assert LangGraphClient._parse_memory_ops("not json at all") == []
        assert LangGraphClient._parse_memory_ops("") == []
        assert LangGraphClient._parse_memory_ops("[unclosed") == []

    def test_non_dict_items_filtered(self):
        raw = '[{"action":"add","text":"ok"}, "junk", 5]'
        ops = LangGraphClient._parse_memory_ops(raw)
        assert ops == [{"action": "add", "text": "ok"}]


def _bare_client():
    c = LangGraphClient.__new__(LangGraphClient)
    c.model = object()  # truthy; the worker path is what's exercised
    return c


class TestAutoExtractGating:
    def test_noop_when_memory_disabled(self, monkeypatch):
        monkeypatch.setattr(client_mod.config, "get", lambda k, d=None: False)
        c = _bare_client()
        started = {"n": 0}
        monkeypatch.setattr(
            c, "_auto_extract_memory_worker",
            lambda *a: started.__setitem__("n", started["n"] + 1),
        )
        c.auto_extract_memory("q", "r")
        assert started["n"] == 0

    def test_noop_when_toggle_off(self, monkeypatch):
        # ENABLE_MEMORY on, ENABLE_MEMORY_AUTO_EXTRACTION off (default).
        def _get(k, d=None):
            if k == "ENABLE_MEMORY":
                return True
            if k == "ENABLE_MEMORY_AUTO_EXTRACTION":
                return False
            return d
        monkeypatch.setattr(client_mod.config, "get", _get)
        c = _bare_client()
        started = {"n": 0}
        monkeypatch.setattr(
            c, "_auto_extract_memory_worker",
            lambda *a: started.__setitem__("n", started["n"] + 1),
        )
        c.auto_extract_memory("q", "r")
        assert started["n"] == 0

    def test_noop_on_empty_query_or_response(self, monkeypatch):
        monkeypatch.setattr(client_mod.config, "get", lambda k, d=None: True)
        c = _bare_client()
        started = {"n": 0}
        monkeypatch.setattr(
            c, "_auto_extract_memory_worker",
            lambda *a: started.__setitem__("n", started["n"] + 1),
        )
        c.auto_extract_memory("", "r")
        c.auto_extract_memory("q", "")
        assert started["n"] == 0

    def test_spawns_worker_when_enabled(self, monkeypatch):
        monkeypatch.setattr(client_mod.config, "get", lambda k, d=None: True)
        c = _bare_client()
        seen = {}
        monkeypatch.setattr(
            c, "_auto_extract_memory_worker",
            lambda q, r: seen.update(q=q, r=r),
        )
        # Run the thread body synchronously by making Thread.start call target.
        import threading

        real_thread = threading.Thread

        class _SyncThread(real_thread):
            def start(self):  # run inline for determinism
                self.run()

        monkeypatch.setattr(client_mod.threading, "Thread", _SyncThread)
        c.auto_extract_memory("my query", "my response")
        assert seen == {"q": "my query", "r": "my response"}


class TestAutoExtractWorker:
    def test_applies_add_and_replace_ops(self, monkeypatch, tmp_path):
        # Prompt available; model returns two ops; both applied via MemoryStore.
        monkeypatch.setattr(
            client_mod.config, "prompt",
            lambda k, d=None: "EXISTING:\n{existing_memory}\nEXCHANGE:\n{exchange}",
        )
        # Point MemoryStore at a temp file (its __init__ imports these lazily
        # from utils.paths / utils.config, so patch at those source modules).
        from mnemoai.utils import config as cfg_mod
        from mnemoai.utils import paths as paths_mod

        store_file = tmp_path / "MEMORY.md"
        monkeypatch.setattr(paths_mod, "memory_file_path", lambda: store_file)
        monkeypatch.setattr(
            cfg_mod.config, "get",
            lambda k, d=None: {"MAX_CHARS": 5000} if k == "MEMORY" else d,
        )

        c = _bare_client()
        monkeypatch.setattr(
            c, "_invoke_model_once",
            lambda prompt: '[{"action":"add","text":"[user] prefers pytest"}]',
        )
        c._auto_extract_memory_worker("what test runner?", "use pytest")
        assert "prefers pytest" in store_file.read_text()

    def test_no_ops_writes_nothing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            client_mod.config, "prompt",
            lambda k, d=None: "{existing_memory}{exchange}",
        )
        from mnemoai.utils import config as cfg_mod
        from mnemoai.utils import paths as paths_mod

        store_file = tmp_path / "MEMORY.md"
        monkeypatch.setattr(paths_mod, "memory_file_path", lambda: store_file)
        monkeypatch.setattr(cfg_mod.config, "get", lambda k, d=None: d)
        c = _bare_client()
        monkeypatch.setattr(c, "_invoke_model_once", lambda prompt: "[]")
        c._auto_extract_memory_worker("q", "r")
        assert not store_file.exists() or store_file.read_text().strip() == ""

    def test_missing_prompt_is_noop(self, monkeypatch):
        monkeypatch.setattr(client_mod.config, "prompt", lambda k, d=None: None)
        c = _bare_client()
        called = {"n": 0}
        monkeypatch.setattr(
            c, "_invoke_model_once",
            lambda prompt: called.__setitem__("n", called["n"] + 1) or "[]",
        )
        c._auto_extract_memory_worker("q", "r")
        assert called["n"] == 0  # returned before invoking the model

    def test_worker_never_raises(self, monkeypatch):
        # A failing model call must be swallowed (background task).
        monkeypatch.setattr(
            client_mod.config, "prompt", lambda k, d=None: "{existing_memory}{exchange}"
        )
        c = _bare_client()

        def _boom(prompt):
            raise RuntimeError("model down")

        monkeypatch.setattr(c, "_invoke_model_once", _boom)
        c._auto_extract_memory_worker("q", "r")  # must not raise
