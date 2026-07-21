"""Unit tests for instance-unique session ids + restart-orphan cleanup.

A ``/model``/``/params`` change re-execs the process (os.execv preserves the
env, so MNEMOAI_INSTANCE_ID and thus instance_id() survive) and mints a fresh
session_id — which named a new chunk_cache/rag_store while the previous run's was
orphaned, accumulating stale .db files. The session_id now embeds the instance id
so (a) two same-second tabs on one profile get DISTINCT ids (no shared/clobbered
files) and (b) an instance can safely delete its OWN prior-session artifacts on
restart without ever touching a concurrent tab's. No LLM/network needed.
"""

import os

import pytest

from mnemoai.client.client import LangGraphClient
from mnemoai.utils import paths


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOAI_HOME", str(tmp_path))
    return tmp_path


def _client(iid):
    """A bare client (no start/LLM) with a pinned instance id."""
    os.environ["MNEMOAI_INSTANCE_ID"] = iid
    c = LangGraphClient.__new__(LangGraphClient)
    return c


class TestSessionIdUniqueness:
    def test_session_id_embeds_instance_id(self, tmp_home):
        c = _client("iid_9")
        sid = c._new_session_id()
        assert sid.endswith("_iid_9")  # instance id is the trailing component

    def test_same_second_different_instances_differ(self, tmp_home):
        # Two tabs on the same profile starting in the same second must NOT share
        # a session_id (they'd otherwise share on-disk artifacts and clobber).
        a = _client("tabA")._new_session_id()
        b = _client("tabB")._new_session_id()
        assert a != b
        assert a.endswith("_tabA") and b.endswith("_tabB")


class TestRestartOrphanCleanup:
    def test_restart_cleans_own_prior_chunk_cache(self, tmp_home):
        c1 = _client("tabA")
        c1.session_id = "p_20260101_100000_tabA"
        c1._initialize_chunk_cache()
        prof = paths.profile_dir()
        old_db = prof / "chunk_cache_p_20260101_100000_tabA.db"
        assert old_db.exists()

        # Restart: same instance id (execv-preserved), new session_id.
        c2 = _client("tabA")
        c2.session_id = "p_20260101_100500_tabA"
        c2._initialize_chunk_cache()
        new_db = prof / "chunk_cache_p_20260101_100500_tabA.db"

        assert new_db.exists()
        assert not old_db.exists()  # this instance's own orphan was cleaned
        assert sorted(p.name for p in prof.glob("chunk_cache_*.db")) == [new_db.name]

    def test_restart_does_not_delete_concurrent_tab_db(self, tmp_home):
        # A live sibling tab (different instance id) must keep its db when THIS
        # instance restarts — the counterexample the fix must defeat.
        a1 = _client("tabA")
        a1.session_id = "p_20260101_100000_tabA"
        a1._initialize_chunk_cache()

        b = _client("tabB")
        b.session_id = "p_20260101_100000_tabB"  # same second, different instance
        b._initialize_chunk_cache()
        prof = paths.profile_dir()
        b_db = prof / "chunk_cache_p_20260101_100000_tabB.db"
        assert b_db.exists()

        a2 = _client("tabA")
        a2.session_id = "p_20260101_100500_tabA"
        a2._initialize_chunk_cache()

        assert b_db.exists()  # sibling B untouched by A's restart cleanup
        assert not (prof / "chunk_cache_p_20260101_100000_tabA.db").exists()

    def test_same_second_restart_reuses_one_db(self, tmp_home):
        # A restart within the same second → identical session_id → the db is
        # reused, not orphaned; no wrongful delete, no accumulation.
        c1 = _client("tabC")
        c1.session_id = "p_20260101_120000_tabC"
        c1._initialize_chunk_cache()
        c2 = _client("tabC")
        c2.session_id = "p_20260101_120000_tabC"  # same id
        c2._initialize_chunk_cache()
        prof = paths.profile_dir()
        assert len(list(prof.glob("chunk_cache_*.db"))) == 1

    def test_restart_cleans_own_prior_rag_store(self, tmp_home):
        c1 = _client("tabA")
        c1.session_id = "p_20260101_100000_tabA"
        c1._initialize_rag_session()
        prof = paths.profile_dir()
        # Simulate the FAISS store the prior session would have created.
        old_store = prof / "rag_store_p_20260101_100000_tabA.faiss"
        old_store.write_text("x")

        c2 = _client("tabA")
        c2.session_id = "p_20260101_100500_tabA"
        c2._initialize_rag_session()

        assert not old_store.exists()  # this instance's own prior store cleaned
