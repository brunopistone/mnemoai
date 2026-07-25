"""Unit tests for STEERING.md — user-authored always-on instructions.

Covers the pure-logic pieces (no LLM): discovery/precedence (utils.paths),
concatenation (SteeringStore), and the ephemeral-strip that keeps the injected
block out of stored history (so compaction never summarizes it).
"""


from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.client.memory.steering_store import SteeringStore
from mnemoai.utils import paths


class TestSteeringDiscovery:
    def test_global_and_project_walk_up_order(self, tmp_path, monkeypatch):
        # Global (app home) + project files walked up to a .git root, applied
        # broadest -> most specific.
        home = tmp_path / "home"
        home.mkdir()
        (home / "STEERING.md").write_text("GLOBAL")
        monkeypatch.setattr(paths, "app_home", lambda: home)

        repo = tmp_path / "proj"
        (repo / ".git").mkdir(parents=True)
        (repo / "STEERING.md").write_text("PROJECT ROOT")
        sub = repo / "src" / "pkg"
        sub.mkdir(parents=True)
        (sub / "STEERING.md").write_text("DEEP")

        files = paths.steering_files(cwd=sub)
        names = [f.read_text() for f in files]
        # global first, then repo root, then the deepest (cwd) last
        assert names == ["GLOBAL", "PROJECT ROOT", "DEEP"]

    def test_no_files_returns_empty(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(paths, "app_home", lambda: home)
        empty = tmp_path / "nowhere"
        empty.mkdir()
        assert paths.steering_files(cwd=empty) == []

    def test_walk_stops_at_git_root(self, tmp_path, monkeypatch):
        # A STEERING.md ABOVE the repo root must not be picked up.
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(paths, "app_home", lambda: home)
        (tmp_path / "STEERING.md").write_text("ABOVE REPO")  # outside the repo
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        (repo / "STEERING.md").write_text("REPO")
        files = paths.steering_files(cwd=repo)
        assert [f.read_text() for f in files] == ["REPO"]


class TestSteeringStore:
    def test_concatenates_with_headers_in_order(self, tmp_path):
        a = tmp_path / "a" / "STEERING.md"
        a.parent.mkdir()
        a.write_text("first rule")
        b = tmp_path / "b" / "STEERING.md"
        b.parent.mkdir()
        b.write_text("second rule")
        out = SteeringStore(files=[a, b]).read()
        assert out.index("first rule") < out.index("second rule")
        assert "Contents of" in out
        assert str(a) in out and str(b) in out

    def test_missing_file_skipped_not_fatal(self, tmp_path):
        good = tmp_path / "STEERING.md"
        good.write_text("kept")
        gone = tmp_path / "gone" / "STEERING.md"  # never created
        out = SteeringStore(files=[gone, good]).read()
        assert out.strip().endswith("kept")

    def test_empty_when_no_files(self):
        assert SteeringStore(files=[]).read() == ""


class TestSteeringEphemeralStrip:
    def test_steering_block_stripped_from_stored_prompt(self):
        prompt = "<steering>always use tabs</steering>\n\nrefactor this"
        assert LangGraphAgent._strip_ephemeral(prompt).strip() == "refactor this"

    def test_plan_and_steering_both_stripped(self):
        prompt = (
            "<steering>rule</steering>\n\n"
            "<plan-mode-active>read only</plan-mode-active>\n\n"
            "do the thing"
        )
        assert LangGraphAgent._strip_ephemeral(prompt).strip() == "do the thing"

    def test_non_block_text_untouched(self):
        assert LangGraphAgent._strip_ephemeral("just a prompt") == "just a prompt"


class TestSteeringReminderNoToggle:
    """`_steering_reminder` is gated ONLY by the file's presence — there is no
    ENABLE_STEERING config key. No file → empty; file present → injected block."""

    def _client(self):
        from mnemoai.client.client import LangGraphClient

        return LangGraphClient.__new__(LangGraphClient)

    def test_empty_when_no_steering_file(self, tmp_path, monkeypatch):
        # No STEERING.md anywhere → empty (its absence is the off switch).
        monkeypatch.setattr(paths, "steering_files", lambda cwd=None: [])
        assert self._client()._steering_reminder() == ""

    def test_injected_when_file_present(self, tmp_path, monkeypatch):
        f = tmp_path / "STEERING.md"
        f.write_text("Always use British spelling.")
        monkeypatch.setattr(paths, "steering_files", lambda cwd=None: [f])
        out = self._client()._steering_reminder()
        assert "Always use British spelling." in out
        assert "OVERRIDE" in out  # authoritative framing
        assert out.strip().startswith("<steering>")
        assert out.strip().endswith("</steering>")
