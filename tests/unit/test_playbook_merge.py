"""Regression tests for PlaybookStore dedup/merge.

Focus: the embedding-merge fallback must NOT reference a non-existent
``__wrapped__`` (which raised AttributeError), and must fall back to the
keyword dedup instead — a real crash path with no prior coverage.
"""

from mnemoai.client.memory.playbook_store import PlaybookStore


def _store():
    # __new__ so we don't touch disk/config; set only what the merge methods use.
    s = PlaybookStore.__new__(PlaybookStore)
    s.max_entries = 100
    s.similarity_threshold = 0.85
    s.embeddings = None
    return s


def _entries():
    return [
        {"strategy": "use glob before grep", "confidence": 0.9, "timestamp": "2"},
        {"strategy": "use glob before grep", "confidence": 0.5, "timestamp": "1"},
        {"strategy": "read file before editing", "confidence": 0.8, "timestamp": "3"},
    ]


class TestMergeByStrategyKey:
    def test_keyword_dedup_keeps_highest_confidence(self):
        s = _store()
        out = s._merge_by_strategy_key(_entries())
        # Two distinct strategies; the higher-confidence duplicate is kept.
        strategies = sorted(e["strategy"] for e in out)
        assert strategies == ["read file before editing", "use glob before grep"]
        glob = next(e for e in out if e["strategy"] == "use glob before grep")
        assert glob["confidence"] == 0.9

    def test_merge_similar_dispatches_to_keyword_when_no_embeddings(self):
        s = _store()  # embeddings=None
        out = s._merge_similar(_entries())
        assert len(out) == 2  # deduped by strategy key

    def test_embedding_merge_fallback_does_not_crash(self):
        # The bug: fallback called self._merge_similar.__wrapped__ (no such attr).
        # Now it must fall back to _merge_by_strategy_key without raising.
        s = _store()

        class _BrokenEmbeddings:
            def embed(self, texts):
                raise RuntimeError("embedding backend down")

        s.embeddings = _BrokenEmbeddings()
        out = s._merge_with_embeddings(_entries())  # must NOT raise
        assert len(out) == 2  # fell back to keyword dedup

    def test_single_entry_is_noop(self):
        s = _store()
        one = [{"strategy": "x", "confidence": 1.0, "timestamp": "1"}]
        assert s._merge_similar(one) == one


class TestTheInjectedBlockClaimsOnlyWhatItCanSupport:
    """The block is paid on EVERY turn and never reclaimable by compaction, so
    what it asserts and how big it gets both matter.

    Nothing in the store can support a claim of learning or effectiveness: the
    strategies come from Reflector's static tables (no model call exists in the
    reflection path) and no entry records whether one ever helped — there is no
    usefulness count, and ``confidence`` can only ever rise. A block headed
    "Learned Strategies" listing "Effective strategies" asserted both.
    """

    def _entries(self, n_fail=6, n_ok=6):
        fail = [
            {"outcome": "failure", "context": f"ctx{i}", "strategy": f"avoid {i}"}
            for i in range(n_fail)
        ]
        ok = [
            {"outcome": "success", "context": f"ctx{i}", "strategy": f"do {i}"}
            for i in range(n_ok)
        ]
        return fail + ok

    def test_the_header_does_not_claim_the_notes_were_learned(self):
        block = _store().format_for_prompt(self._entries())
        first = block.splitlines()[0]
        assert "learned" not in first.lower()
        assert "strategies" not in first.lower()

    def test_no_group_label_claims_effectiveness(self):
        block = _store().format_for_prompt(self._entries()).lower()
        assert "effective" not in block
        assert "avoid these patterns" not in block

    def test_each_group_is_capped_regardless_of_how_many_are_passed(self):
        # MAX_INJECT is set explicitly to 10 in every config.yaml written so far,
        # so lowering its default would reach no existing install — the cap has to
        # live here, at the one chokepoint that produces the injected text.
        block = _store().format_for_prompt(self._entries(n_fail=50, n_ok=50))
        assert len([ln for ln in block.splitlines() if ln.startswith("  ")]) == 4

    def test_the_whole_block_stays_small(self):
        # The live 10-entry block measured 1050 chars (~262 tokens per turn).
        block = _store().format_for_prompt(self._entries(n_fail=50, n_ok=50))
        assert len(block) < 400

    def test_no_entries_still_injects_nothing(self):
        assert _store().format_for_prompt([]) == ""

    def test_one_group_alone_does_not_emit_the_other_label(self):
        block = _store().format_for_prompt(self._entries(n_fail=0, n_ok=3))
        assert "past errors" not in block
        assert "past successes" in block

    def test_the_marker_is_the_one_context_report_segments_on(self):
        # /context splits the LIVE prompt by marker, so a drifted header silently
        # stops attributing this block.
        from mnemoai.client import context_report
        from mnemoai.client.memory.playbook_store import PLAYBOOK_BLOCK_MARKER

        block = _store().format_for_prompt(self._entries())
        assert block.startswith(PLAYBOOK_BLOCK_MARKER)
        assert PLAYBOOK_BLOCK_MARKER in [m for m, _ in context_report._SYSTEM_SEGMENTS]
