"""Unit tests for the model-scoped memory key and the unforking of directories
written before it was normalized.

Pure path logic plus real (tiny) Chroma and playbook stores on a tmp dir — no
config.yaml, no LLM, no embedding model: episodes are copied with the vectors
already stored beside them.
"""

import json

import pytest

from mnemoai.client.memory import model_dir_merge
from mnemoai.client.memory.model_dir_merge import (
    Unfork,
    forked_dirs,
    merge_playbook_entries,
    unfork_model_dirs,
)
from mnemoai.utils.paths import model_dir, normalize_model_key, sanitize_model_name

_DIM = 4


class TestNormalizeModelKey:
    @pytest.mark.parametrize(
        "model_id",
        [
            "global.anthropic.claude-opus-5",
            "us.anthropic.claude-opus-5",
            "eu.anthropic.claude-opus-5",
            "apac.anthropic.claude-opus-5",
            "us-gov.anthropic.claude-opus-5",
        ],
    )
    def test_a_routing_prefix_is_not_a_different_model(self, model_id):
        assert normalize_model_key(model_id) == "anthropic.claude-opus-5"

    def test_an_unprefixed_id_is_untouched(self):
        assert normalize_model_key("anthropic.claude-opus-5") == (
            "anthropic.claude-opus-5"
        )

    def test_only_one_prefix_comes_off(self):
        # Two routing scopes is not a shape AWS produces; stripping both would
        # guess at what the second one means.
        assert normalize_model_key("global.us.anthropic.claude-opus-5") == (
            "us.anthropic.claude-opus-5"
        )

    def test_a_local_model_whose_name_starts_that_way_is_kept(self):
        # Ollama tags are user-chosen: "us." here is part of the name, and the
        # remainder is not provider-qualified.
        assert normalize_model_key("us-tuned-qwen") == "us-tuned-qwen"
        assert normalize_model_key("global.mymodel") == "global.mymodel"

    def test_a_bare_prefix_keeps_itself(self):
        # Nothing follows the scope, so there is no base id to key memory by.
        assert normalize_model_key("global.") == "global."
        assert normalize_model_key("us") == "us"

    def test_it_still_sanitizes(self):
        assert normalize_model_key("us.meta.llama3-70b-instruct-v1:0") == (
            "meta.llama3-70b-instruct-v1_0"
        )
        assert normalize_model_key("") == "default"
        assert normalize_model_key(None) == "default"

    def test_it_is_idempotent(self):
        # It runs over directory NAMES, which were written by an earlier version
        # of the same function.
        for model_id in ("global.anthropic.claude-opus-5", "qwen3:32b", ""):
            once = normalize_model_key(model_id)
            assert normalize_model_key(once) == once

    def test_sanitize_model_name_is_left_alone(self):
        # The prefix is a MEMORY-key concern; the path sanitizer has other callers.
        assert sanitize_model_name("global.anthropic.claude-opus-5") == (
            "global.anthropic.claude-opus-5"
        )

    def test_model_dir_uses_the_normalized_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path))
        prefixed = model_dir("global.anthropic.claude-opus-5", profile="p")
        plain = model_dir("anthropic.claude-opus-5", profile="p")
        assert prefixed == plain
        assert prefixed.name == "anthropic.claude-opus-5"


# --- Fixtures for the on-disk half -----------------------------------------


def _episodic_store(models_root, model, episodes, fingerprint="fp-1024|score=2"):
    """A real episodic collection with pre-computed vectors."""
    import chromadb

    path = models_root / model / "episodic_memory"
    path.mkdir(parents=True)
    client = chromadb.PersistentClient(path=str(path))
    collection = client.create_collection(
        name="episodic_memory",
        metadata={"description": "test", "embed_fingerprint": fingerprint},
    )
    for i, (episode_id, tools) in enumerate(episodes):
        collection.add(
            ids=[episode_id],
            embeddings=[[float(i + 1)] * _DIM],
            metadatas=[
                {
                    "task": f"task {episode_id}",
                    "tools": tools,
                    "outcome": "success",
                    "timestamp": "2026-01-01T00:00:00",
                }
            ],
        )
    return path


def _read_episodes(path):
    """{id: metadata} straight from the store on disk."""
    import chromadb
    from chromadb.api.client import SharedSystemClient

    SharedSystemClient.clear_system_cache()
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_collection(name="episodic_memory")
    got = collection.get(include=["metadatas"])
    return dict(zip(got["ids"], got["metadatas"]))


def _playbook(models_root, model, strategies):
    path = models_root / model / "playbook"
    path.mkdir(parents=True)
    (path / "playbook.json").write_text(
        json.dumps([{"strategy": s, "confidence": 0.5} for s in strategies])
    )
    return path


def _entries(path):
    return json.loads((path / "playbook.json").read_text())


@pytest.fixture
def models_root(tmp_path, monkeypatch):
    """``{profile}/models/`` under an isolated app home."""
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path))
    root = tmp_path / "tester" / "models"
    root.mkdir(parents=True)
    monkeypatch.setattr(
        model_dir_merge, "profile_dir", lambda profile=None: root.parent
    )
    return root


class TestForkedDirs:
    def test_finds_the_prefixed_sibling(self, models_root):
        (models_root / "global.anthropic.claude-opus-5").mkdir()
        (models_root / "anthropic.claude-opus-5").mkdir()
        (models_root / "qwen3_32b").mkdir()
        found = forked_dirs(models_root, "anthropic.claude-opus-5")
        assert [p.name for p in found] == ["global.anthropic.claude-opus-5"]

    def test_the_normalized_dir_is_never_its_own_donor(self, models_root):
        (models_root / "anthropic.claude-opus-5").mkdir()
        assert forked_dirs(models_root, "anthropic.claude-opus-5") == []

    def test_an_aside_copy_is_not_rediscovered(self, models_root):
        # Otherwise a completed merge would run again on every startup.
        (models_root / "global.anthropic.claude-opus-5.merged-20260914").mkdir()
        assert forked_dirs(models_root, "anthropic.claude-opus-5") == []

    def test_missing_root(self, tmp_path):
        assert forked_dirs(tmp_path / "nope", "anything") == []


class TestReadEntries:
    def test_an_empty_playbook_is_not_an_unreadable_one(self, tmp_path):
        # [] has nothing to carry; None means don't touch the donor.
        (tmp_path / "playbook.json").write_text("[]")
        assert model_dir_merge._read_entries(tmp_path / "playbook.json") == []

    @pytest.mark.parametrize("body", ["not json", '{"a": 1}', ""])
    def test_unreadable_shapes(self, tmp_path, body):
        (tmp_path / "playbook.json").write_text(body)
        assert model_dir_merge._read_entries(tmp_path / "playbook.json") is None

    def test_missing_file(self, tmp_path):
        assert model_dir_merge._read_entries(tmp_path / "nope.json") is None


class TestMergePlaybookEntries:
    def test_unseen_strategies_are_appended_after_the_target(self):
        merged = merge_playbook_entries(
            [{"strategy": "b"}, {"strategy": "c"}], [{"strategy": "a"}]
        )
        assert [e["strategy"] for e in merged] == ["a", "b", "c"]

    def test_the_target_keeps_its_own_copy_of_a_shared_strategy(self):
        target = [{"strategy": "a", "confidence": 0.9}]
        merged = merge_playbook_entries([{"strategy": "a", "confidence": 0.1}], target)
        assert merged == target

    def test_donor_duplicates_collapse(self):
        merged = merge_playbook_entries([{"strategy": "b"}, {"strategy": "b"}], [])
        assert len(merged) == 1

    def test_junk_is_dropped_not_raised(self):
        merged = merge_playbook_entries(
            [None, 7, {"strategy": ""}, {"no_strategy": 1}, {"strategy": "b"}],
            "not a list",
        )
        assert [e["strategy"] for e in merged] == ["b"]


class TestAdoption:
    """Only one side exists: the directory is moved, nothing is opened."""

    def test_a_lone_prefixed_store_is_moved_into_place(self, models_root):
        _episodic_store(models_root, "global.anthropic.claude-opus-5", [("e1", "fs_read")])
        _playbook(models_root, "global.anthropic.claude-opus-5", ["always read first"])

        records = unfork_model_dirs("global.anthropic.claude-opus-5")

        assert len(records) == 1
        assert records[0].moved == ["episodic_memory", "playbook"]
        assert records[0].episodes == 0 and records[0].strategies == 0
        target = models_root / "anthropic.claude-opus-5"
        assert (target / "episodic_memory" / "chroma.sqlite3").is_file()
        assert _read_episodes(target / "episodic_memory").keys() == {"e1"}
        assert [e["strategy"] for e in _entries(target / "playbook")] == [
            "always read first"
        ]

    def test_the_emptied_donor_is_gone(self, models_root):
        _episodic_store(models_root, "us.anthropic.claude-opus-5", [("e1", "fs_read")])
        unfork_model_dirs("us.anthropic.claude-opus-5")
        assert not (models_root / "us.anthropic.claude-opus-5").exists()

    def test_half_and_half(self, models_root):
        # The playbook has to merge, the episodic store only has one side.
        _episodic_store(models_root, "global.anthropic.claude-opus-5", [("e1", "x")])
        _playbook(models_root, "global.anthropic.claude-opus-5", ["donor rule"])
        _playbook(models_root, "anthropic.claude-opus-5", ["target rule"])

        records = unfork_model_dirs("anthropic.claude-opus-5")

        assert records[0].moved == ["episodic_memory"]
        assert records[0].strategies == 1
        strategies = [
            e["strategy"] for e in _entries(models_root / "anthropic.claude-opus-5" / "playbook")
        ]
        assert strategies == ["target rule", "donor rule"]

    def test_nothing_to_do(self, models_root):
        _episodic_store(models_root, "anthropic.claude-opus-5", [("e1", "fs_read")])
        assert unfork_model_dirs("global.anthropic.claude-opus-5") == []


class TestMerge:
    """Both sides exist: episodes are copied and the donor is kept aside."""

    def test_episodes_are_carried_over_with_their_vectors(self, models_root):
        _episodic_store(
            models_root, "global.anthropic.claude-opus-5", [("e1", "fs_read"), ("e2", "x")]
        )
        _episodic_store(models_root, "anthropic.claude-opus-5", [("e3", "git_safe")])

        records = unfork_model_dirs("anthropic.claude-opus-5")

        assert records[0].episodes == 2
        episodes = _read_episodes(
            models_root / "anthropic.claude-opus-5" / "episodic_memory"
        )
        assert set(episodes) == {"e1", "e2", "e3"}
        assert episodes["e1"]["task"] == "task e1"

    def test_the_donor_is_kept_aside_and_the_merge_does_not_re_run(self, models_root):
        _episodic_store(models_root, "global.anthropic.claude-opus-5", [("e1", "a")])
        _episodic_store(models_root, "anthropic.claude-opus-5", [("e2", "b")])

        first = unfork_model_dirs("anthropic.claude-opus-5")
        assert first[0].kept_as.startswith("global.anthropic.claude-opus-5.merged-")
        aside = models_root / first[0].kept_as
        assert (aside / "episodic_memory" / "chroma.sqlite3").is_file()

        # Nothing left that normalizes to the key, so no second pass.
        assert unfork_model_dirs("anthropic.claude-opus-5") == []

    def test_a_legacy_tools_payload_is_not_copied_forward(self, models_root):
        legacy = str([{"name": "fs_read", "args": {"p": 1}, "result": "r" * 5000}])
        _episodic_store(models_root, "global.anthropic.claude-opus-5", [("e1", legacy)])
        _episodic_store(models_root, "anthropic.claude-opus-5", [("e2", "git_safe")])

        unfork_model_dirs("anthropic.claude-opus-5")

        episodes = _read_episodes(
            models_root / "anthropic.claude-opus-5" / "episodic_memory"
        )
        assert episodes["e1"]["tools"] == "fs_read"

    def test_an_already_present_episode_is_not_duplicated(self, models_root):
        _episodic_store(models_root, "global.anthropic.claude-opus-5", [("e1", "a")])
        _episodic_store(models_root, "anthropic.claude-opus-5", [("e1", "a")])

        records = unfork_model_dirs("anthropic.claude-opus-5")

        # Nothing to carry, so nothing is reported — but the donor is redundant,
        # so it still moves aside and the merge doesn't come back.
        assert records == []
        assert not (models_root / "global.anthropic.claude-opus-5").exists()
        episodes = _read_episodes(
            models_root / "anthropic.claude-opus-5" / "episodic_memory"
        )
        assert list(episodes) == ["e1"]

    def test_a_different_embedding_model_is_left_alone(self, models_root):
        _episodic_store(
            models_root,
            "global.anthropic.claude-opus-5",
            [("e1", "a")],
            fingerprint="other-model|score=2",
        )
        _episodic_store(models_root, "anthropic.claude-opus-5", [("e2", "b")])

        records = unfork_model_dirs("anthropic.claude-opus-5")

        # Incomparable vectors: nothing is carried, and nothing is destroyed.
        assert records == []
        assert (
            models_root / "global.anthropic.claude-opus-5" / "episodic_memory"
        ).is_dir()
        assert list(
            _read_episodes(models_root / "anthropic.claude-opus-5" / "episodic_memory")
        ) == ["e2"]

    def test_a_component_that_could_not_be_carried_keeps_the_donor_named(
        self, models_root
    ):
        # The playbook merges, the episodic store can't: renaming the donor aside
        # would leave those episodes under a key nothing resolves to.
        _episodic_store(
            models_root,
            "global.anthropic.claude-opus-5",
            [("e1", "a")],
            fingerprint="other-model|score=2",
        )
        _playbook(models_root, "global.anthropic.claude-opus-5", ["donor rule"])
        _episodic_store(models_root, "anthropic.claude-opus-5", [("e2", "b")])
        _playbook(models_root, "anthropic.claude-opus-5", ["target rule"])

        records = unfork_model_dirs("anthropic.claude-opus-5")

        assert records[0].strategies == 1
        assert records[0].blocked == ["episodic_memory"]
        assert records[0].kept_as is None
        assert (models_root / "global.anthropic.claude-opus-5").is_dir()

        # A second pass re-reads it and copies nothing, rather than duplicating.
        assert unfork_model_dirs("anthropic.claude-opus-5") == []
        entries = _entries(models_root / "anthropic.claude-opus-5" / "playbook")
        assert [e["strategy"] for e in entries] == ["target rule", "donor rule"]


class TestItNeverRaises:
    def test_an_unreadable_donor_store_leaves_both_in_place(self, models_root):
        _episodic_store(models_root, "anthropic.claude-opus-5", [("e1", "a")])
        broken = models_root / "global.anthropic.claude-opus-5" / "episodic_memory"
        broken.mkdir(parents=True)
        (broken / "chroma.sqlite3").write_bytes(b"not a database")

        assert unfork_model_dirs("anthropic.claude-opus-5") == []
        assert (broken / "chroma.sqlite3").is_file()

    def test_no_profile_dir_at_all(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            model_dir_merge, "profile_dir", lambda profile=None: tmp_path / "nope"
        )
        assert unfork_model_dirs("global.anthropic.claude-opus-5") == []

    def test_a_failing_profile_lookup_is_swallowed(self, monkeypatch):
        def boom(profile=None):
            raise RuntimeError("no home")

        monkeypatch.setattr(model_dir_merge, "profile_dir", boom)
        assert unfork_model_dirs("global.anthropic.claude-opus-5") == []


class TestReporting:
    def test_a_no_op_is_not_reported(self):
        assert Unfork(donor="a", target="b").carried is False

    def test_the_line_names_both_sides_and_the_copy(self):
        line = model_dir_merge._describe(
            Unfork(
                donor="global.anthropic.claude-opus-5",
                target="anthropic.claude-opus-5",
                episodes=618,
                strategies=3,
                kept_as="global.anthropic.claude-opus-5.merged-20260914",
            )
        )
        assert "global.anthropic.claude-opus-5" in line
        assert "618 episodes" in line
        assert "3 strategies" in line
        assert "merged-20260914" in line
