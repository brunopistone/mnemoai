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

    def test_tolerates_cc_optional_keys(self, tmp_path):
        # Claude-Code frontmatter keys must not cause rejection.
        front = (
            "name: cc\ndescription: Use for CC stuff.\n"
            "license: MIT\nallowed-tools: Read Grep\ncompatibility: anything\n"
            "metadata:\n  author: someone"
        )
        _write_skill(tmp_path, "cc", front)
        skills = SkillStore(tmp_path).list_skills()
        assert [s.name for s in skills] == ["cc"]

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
