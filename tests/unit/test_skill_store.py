"""Unit tests for the agent-skills store (client/memory/skill_store.py).

Skills are authored SKILL.md instruction packs under a skills root. The store
scans them tolerantly (a bad/incomplete skill is skipped, never fatal), exposes
tier-1 metadata for the system prompt, and loads a full body by name for the
use_skill tool. Pure file logic — no LLM/MCP — so these run in the unit tier.
"""

from mnemoai.client.memory.skill_store import (
    SkillStore,
    _parse_frontmatter,
    format_available_skills,
    render_skill_body,
)


def _write_skill(root, name, frontmatter: str, body: str = "Body here.") -> None:
    """Create root/<name>/SKILL.md with the given raw frontmatter + body."""
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n{body}\n")


def _valid(name="alpha", desc="Use when the user asks for alpha."):
    return f"name: {name}\ndescription: {desc}"


class TestParseFrontmatter:
    def test_parses_frontmatter_and_body(self):
        front, body = _parse_frontmatter("---\nname: x\ndescription: y\n---\nHello.\n")
        assert front == {"name": "x", "description": "y"}
        assert body.strip() == "Hello."

    def test_no_frontmatter_returns_empty_dict(self):
        front, body = _parse_frontmatter("Just a body, no fence.")
        assert front == {}
        assert "Just a body" in body


class TestListSkills:
    def test_parses_a_valid_skill(self, tmp_path):
        _write_skill(tmp_path, "alpha", _valid(), body="# Alpha\nDo the thing.")
        skills = SkillStore(tmp_path).list_skills()
        assert len(skills) == 1
        s = skills[0]
        assert s.name == "alpha"  # directory name is canonical id
        assert s.description == "Use when the user asks for alpha."
        assert "Do the thing." in s.body
        assert s.path == tmp_path / "alpha"

    def test_absent_root_returns_empty(self, tmp_path):
        assert SkillStore(tmp_path / "does-not-exist").list_skills() == []

    def test_skips_missing_skill_md_but_keeps_others(self, tmp_path):
        (tmp_path / "no_md").mkdir()  # dir without SKILL.md
        _write_skill(tmp_path, "good", _valid())
        names = [s.name for s in SkillStore(tmp_path).list_skills()]
        assert names == ["good"]

    def test_skips_missing_description(self, tmp_path):
        _write_skill(tmp_path, "bad", "name: bad")  # no description
        _write_skill(tmp_path, "good", _valid())
        names = [s.name for s in SkillStore(tmp_path).list_skills()]
        assert names == ["good"]

    def test_tolerates_extra_optional_keys(self, tmp_path):
        front = (
            "name: extras\ndescription: Uses extra frontmatter keys.\n"
            "license: MIT\nallowed-tools: Read Grep\ncompatibility: anything\n"
            "metadata:\n  author: someone"
        )
        _write_skill(tmp_path, "extras", front)
        skills = SkillStore(tmp_path).list_skills()
        assert [s.name for s in skills] == ["extras"]

    def test_extra_unknown_key_is_tolerated(self, tmp_path):
        _write_skill(tmp_path, "x", _valid() + "\nsomething_custom: 1")
        assert [s.name for s in SkillStore(tmp_path).list_skills()] == ["x"]

    def test_skips_malformed_yaml(self, tmp_path):
        # Unparseable YAML in the frontmatter -> skipped, not fatal.
        _write_skill(tmp_path, "bad", "name: [unclosed\ndescription: x")
        _write_skill(tmp_path, "good", _valid())
        names = [s.name for s in SkillStore(tmp_path).list_skills()]
        assert names == ["good"]

    def test_ignores_non_directory_entries(self, tmp_path):
        (tmp_path / "stray.md").write_text("not a skill dir")
        _write_skill(tmp_path, "good", _valid())
        names = [s.name for s in SkillStore(tmp_path).list_skills()]
        assert names == ["good"]

    def test_sorted_by_name(self, tmp_path):
        _write_skill(tmp_path, "zeta", _valid("zeta"))
        _write_skill(tmp_path, "alpha", _valid("alpha"))
        names = [s.name for s in SkillStore(tmp_path).list_skills()]
        assert names == ["alpha", "zeta"]


class TestListIssues:
    """Rejected skills are reported (not silently dropped) so /skills can show them."""

    def test_missing_description_reported(self, tmp_path):
        _write_skill(tmp_path, "bad", "name: bad")
        issues = SkillStore(tmp_path).list_issues()
        assert len(issues) == 1
        assert issues[0].name == "bad"
        assert "description" in issues[0].reason

    def test_malformed_yaml_reported(self, tmp_path):
        _write_skill(tmp_path, "bad", "name: [unclosed\ndescription: x")
        issues = SkillStore(tmp_path).list_issues()
        assert issues[0].name == "bad"
        assert "YAML" in issues[0].reason or "frontmatter" in issues[0].reason

    def test_overlong_description_reported(self, tmp_path):
        _write_skill(tmp_path, "bad", f"name: bad\ndescription: {'x' * 1100}")
        issues = SkillStore(tmp_path).list_issues()
        assert issues[0].name == "bad"
        assert "too long" in issues[0].reason

    def test_valid_skill_produces_no_issue(self, tmp_path):
        _write_skill(tmp_path, "good", _valid())
        assert SkillStore(tmp_path).list_issues() == []

    def test_dir_without_skill_md_is_not_an_issue(self, tmp_path):
        (tmp_path / "not_a_skill").mkdir()  # no SKILL.md -> not an attempt
        assert SkillStore(tmp_path).list_issues() == []


class TestMetadataAndLoad:
    def test_list_metadata_shape(self, tmp_path):
        _write_skill(tmp_path, "alpha", _valid("alpha", "desc a"))
        assert SkillStore(tmp_path).list_metadata() == [("alpha", "desc a")]

    def test_load_body_known(self, tmp_path):
        _write_skill(tmp_path, "alpha", _valid(), body="# Alpha\nSteps.")
        skill = SkillStore(tmp_path).load_body("alpha")
        assert skill is not None
        assert "Steps." in skill.body

    def test_load_body_unknown_returns_none(self, tmp_path):
        _write_skill(tmp_path, "alpha", _valid())
        assert SkillStore(tmp_path).load_body("nope") is None

    def test_load_body_blank_returns_none(self, tmp_path):
        assert SkillStore(tmp_path).load_body("") is None


class TestFormatAvailableSkills:
    def test_empty_returns_empty_string(self):
        assert format_available_skills([]) == ""

    def test_block_lists_names_and_descriptions(self):
        block = format_available_skills([("alpha", "do alpha"), ("beta", "do beta")])
        assert "<available_skills>" in block
        assert "</available_skills>" in block
        assert "use_skill" in block  # instructs the model to call the tool
        assert "alpha: do alpha" in block
        assert "beta: do beta" in block

    def test_long_description_truncated(self):
        long = "x" * 500
        block = format_available_skills([("alpha", long)])
        assert "…" in block
        # The full 500-char description must not appear verbatim; it's capped.
        assert long not in block
        assert block.count("x") < 250  # truncated near the ~200-char cap


class TestArgumentHint:
    def test_argument_hint_parsed(self, tmp_path):
        _write_skill(
            tmp_path,
            "alpha",
            _valid() + "\nargument_hint: a PR number and repo",
        )
        skill = SkillStore(tmp_path).list_skills()[0]
        assert skill.argument_hint == "a PR number and repo"

    def test_argument_hint_defaults_empty(self, tmp_path):
        _write_skill(tmp_path, "alpha", _valid())
        assert SkillStore(tmp_path).list_skills()[0].argument_hint == ""

    def test_hint_rendered_in_listing_for_skill_objects(self, tmp_path):
        _write_skill(tmp_path, "alpha", _valid() + "\nargument_hint: a filename")
        block = format_available_skills(SkillStore(tmp_path).list_skills())
        assert "expects: a filename" in block

    def test_tuple_input_still_supported(self):
        # Legacy (name, description) tuples must keep working (no hint shown).
        block = format_available_skills([("alpha", "do alpha")])
        assert "alpha: do alpha" in block
        assert "expects:" not in block


class TestListingBudget:
    def test_large_library_collapses_to_plus_more(self):
        # Many skills with long descriptions must not blow the aggregate budget;
        # the overflow is summarized as a "+N more" line.
        from mnemoai.client.memory.skill_store import _MAX_LISTING_CHARS

        meta = [(f"skill{i}", "d" * 180) for i in range(100)]
        block = format_available_skills(meta)
        assert "more — see /skills" in block
        assert len(block) < _MAX_LISTING_CHARS + 500  # bounded

    def test_small_library_no_more_line(self):
        block = format_available_skills([("alpha", "a"), ("beta", "b")])
        assert "more — see /skills" not in block

    def test_first_skill_always_listed_even_if_huge(self):
        # A single oversized description still yields a line (never an empty list).
        block = format_available_skills([("alpha", "x" * 5000)])
        assert "alpha:" in block

    def test_listing_bounded_by_token_budget(self, monkeypatch):
        # Make each line "expensive" in tokens while staying tiny in chars, so
        # the TOKEN budget (not the char cap) is what forces the "+N more" line.
        import mnemoai.client.memory.skill_store as ss

        monkeypatch.setattr(ss, "_estimate_tokens", lambda text: 300)
        # 3 lines * 300 tokens = 900; _MAX_LISTING_TOKENS default is 1000, so the
        # 4th line trips the token budget well before the 4000-char cap.
        meta = [(f"s{i}", "short") for i in range(10)]
        block = format_available_skills(meta)
        assert "more — see /skills" in block


class TestRenderSkillBody:
    def _skill(self, tmp_path, body):
        from mnemoai.client.memory.skill_store import Skill

        return Skill(name="alpha", description="d", body=body, path=tmp_path)

    def test_arguments_substituted(self, tmp_path):
        s = self._skill(tmp_path, "Handle $ARGUMENTS now.")
        out = render_skill_body(s, "PR 42")
        assert "PR 42" in out and "$ARGUMENTS" not in out

    def test_skill_dir_substituted_braced_and_unbraced(self, tmp_path):
        s = self._skill(tmp_path, "run ${SKILL_DIR}/go.sh and $CLAUDE_SKILL_DIR/x")
        out = render_skill_body(s, "")
        assert str(tmp_path) in out
        assert "${SKILL_DIR}" not in out and "$CLAUDE_SKILL_DIR" not in out

    def test_no_placeholders_returned_unchanged(self, tmp_path):
        s = self._skill(tmp_path, "Just plain instructions.")
        assert render_skill_body(s, "ignored") == "Just plain instructions."

    def test_blank_arguments_becomes_empty(self, tmp_path):
        s = self._skill(tmp_path, "x=[$ARGUMENTS]")
        assert render_skill_body(s, "") == "x=[]"

    def test_unbraced_token_does_not_eat_longer_identifier(self, tmp_path):
        # $SKILL_DIR must NOT match inside $SKILL_DIRECTORY (word-bounded).
        s = self._skill(tmp_path, "$SKILL_DIRECTORY vs ${SKILL_DIR}")
        out = render_skill_body(s, "")
        assert "$SKILL_DIRECTORY" in out  # untouched
        assert str(tmp_path) in out  # the braced one was substituted

    def test_arguments_value_is_not_re_substituted(self, tmp_path):
        # A ${SKILL_DIR} literal inside the args value must survive (single pass).
        s = self._skill(tmp_path, "args=$ARGUMENTS")
        out = render_skill_body(s, "keep ${SKILL_DIR} literal")
        assert out == "args=keep ${SKILL_DIR} literal"


class TestScanMemoization:
    def _clear(self):
        import mnemoai.client.memory.skill_store as ss

        ss._SCAN_CACHE.clear()

    def test_unchanged_dir_served_from_cache(self, tmp_path, monkeypatch):
        self._clear()
        _write_skill(tmp_path, "alpha", _valid())
        calls = {"n": 0}
        import pathlib

        orig = pathlib.Path.read_text

        def counting(self, *a, **k):
            if self.name == "SKILL.md":
                calls["n"] += 1
            return orig(self, *a, **k)

        monkeypatch.setattr(pathlib.Path, "read_text", counting)
        SkillStore(tmp_path).list_skills()
        SkillStore(tmp_path).list_skills()  # second scan, unchanged dir
        assert calls["n"] == 1  # parsed once, then cache-served

    def test_edit_invalidates_cache(self, tmp_path, monkeypatch):
        import time

        self._clear()
        _write_skill(tmp_path, "alpha", _valid(desc="original"))
        assert SkillStore(tmp_path).load_body("alpha").description == "original"
        time.sleep(0.01)
        (tmp_path / "alpha" / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: edited\n---\nBody\n"
        )
        # mtime bumped → signature changes → re-parsed.
        assert SkillStore(tmp_path).load_body("alpha").description == "edited"

    def test_added_skill_invalidates_cache(self, tmp_path):
        self._clear()
        _write_skill(tmp_path, "alpha", _valid())
        assert len(SkillStore(tmp_path).list_skills()) == 1
        _write_skill(tmp_path, "beta", _valid(name="beta"))
        assert len(SkillStore(tmp_path).list_skills()) == 2  # new dir seen
