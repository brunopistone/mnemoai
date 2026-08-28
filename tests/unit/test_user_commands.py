"""Unit tests for user-defined slash commands (client/user_commands.py).

Pure file logic: a markdown file per command, argument substitution, tolerant
scanning with a visible reason per rejected file. No LLM, no terminal.
"""

import os

import pytest

from mnemoai.client import user_commands
from mnemoai.client.user_commands import (
    BUILTIN_COMMANDS,
    UserCommandStore,
    substitute,
)


@pytest.fixture(autouse=True)
def _clear_scan_cache():
    """The scan is memoized process-wide; keep tests independent of each other."""
    user_commands._SCAN_CACHE.clear()
    yield
    user_commands._SCAN_CACHE.clear()


def _write(root, name, text):
    root.mkdir(parents=True, exist_ok=True)
    p = root / name
    p.write_text(text)
    return p


class TestSubstitute:
    def test_arguments_placeholder(self):
        assert substitute("Review $ARGUMENTS please", "the router") == (
            "Review the router please"
        )

    def test_positional_placeholders(self):
        assert substitute("$1 then $2", "alpha beta") == "alpha then beta"

    def test_missing_positional_becomes_empty(self):
        # A command called with fewer args than it references must still run.
        assert substitute("[$1][$2]", "alpha") == "[alpha][]"

    def test_no_placeholder_appends_arguments(self):
        # Silently dropping what the user typed is the one outcome that reads as
        # a bug, so the args are appended instead.
        assert substitute("Summarize this file:", "src/x.py") == (
            "Summarize this file:\n\nsrc/x.py"
        )

    def test_no_placeholder_no_arguments_is_unchanged(self):
        assert substitute("Run the suite.") == "Run the suite."

    def test_two_digit_positional_left_alone(self):
        # $12 is not "$1 followed by 2" — rewriting half of it would be worse
        # than leaving it in place.
        assert substitute("cost is $12", "alpha") == "cost is $12\n\nalpha"

    def test_substituted_value_is_not_rescanned(self):
        # One pass: an argument that itself contains a placeholder stays literal.
        assert substitute("say $ARGUMENTS", "$1 dollars") == "say $1 dollars"

    def test_arguments_are_stripped(self):
        assert substitute("x=$ARGUMENTS", "   spaced   ") == "x=spaced"


class TestScan:
    def test_missing_root_is_empty(self, tmp_path):
        store = UserCommandStore(root=tmp_path / "nope")
        assert store.list_commands() == []
        assert store.list_issues() == []

    def test_frontmatter_command(self, tmp_path):
        _write(
            tmp_path,
            "review.md",
            "---\ndescription: Review a diff\nargument_hint: <path>\n---\n\nReview $ARGUMENTS.\n",
        )
        (cmd,) = UserCommandStore(root=tmp_path).list_commands()
        assert cmd.name == "review"
        assert cmd.description == "Review a diff"
        assert cmd.argument_hint == "<path>"
        assert cmd.body == "Review $ARGUMENTS."
        assert cmd.label == "/review <path>"

    def test_bare_markdown_needs_no_frontmatter(self, tmp_path):
        # Unlike a skill (whose description is what the model routes on), a plain
        # markdown file is a valid command; the description falls back to the body.
        _write(tmp_path, "ship.md", "# Ship it\n\nRun the release checklist.\n")
        (cmd,) = UserCommandStore(root=tmp_path).list_commands()
        assert cmd.name == "ship"
        assert cmd.description == "Ship it"
        assert cmd.argument_hint == ""

    def test_underscore_and_dot_files_ignored(self, tmp_path):
        _write(tmp_path, "_README.md", "authoring notes\n")
        _write(tmp_path, ".draft.md", "work in progress\n")
        _write(tmp_path, "real.md", "do the thing\n")
        store = UserCommandStore(root=tmp_path)
        assert [c.name for c in store.list_commands()] == ["real"]
        assert store.list_issues() == []  # ignored, not rejected

    def test_builtin_name_is_rejected_with_a_reason(self, tmp_path):
        # The dispatcher matches built-ins first, so this file could never fire —
        # saying so beats looking broken.
        _write(tmp_path, "compact.md", "squeeze it\n")
        store = UserCommandStore(root=tmp_path)
        assert store.list_commands() == []
        (issue,) = store.list_issues()
        assert issue.name == "compact.md"
        assert "built-in" in issue.reason

    def test_builtin_check_is_case_insensitive(self, tmp_path):
        _write(tmp_path, "Compact.md", "squeeze it\n")
        assert UserCommandStore(root=tmp_path).list_commands() == []

    def test_unusable_name_is_rejected(self, tmp_path):
        _write(tmp_path, "my command.md", "body\n")
        _write(tmp_path, "-lead.md", "body\n")
        store = UserCommandStore(root=tmp_path)
        assert store.list_commands() == []
        assert {i.name for i in store.list_issues()} == {"my command.md", "-lead.md"}
        assert all("single /command token" in i.reason for i in store.list_issues())

    def test_empty_body_is_rejected(self, tmp_path):
        _write(tmp_path, "hollow.md", "---\ndescription: nothing\n---\n\n   \n")
        store = UserCommandStore(root=tmp_path)
        assert store.list_commands() == []
        assert "no prompt text" in store.list_issues()[0].reason

    def test_oversize_file_is_rejected(self, tmp_path):
        _write(tmp_path, "huge.md", "x" * (user_commands._MAX_BODY_CHARS + 1))
        store = UserCommandStore(root=tmp_path)
        assert store.list_commands() == []
        assert "too large" in store.list_issues()[0].reason

    def test_invalid_frontmatter_is_rejected(self, tmp_path):
        _write(tmp_path, "broken.md", "---\ndescription: [unclosed\n---\n\nbody\n")
        store = UserCommandStore(root=tmp_path)
        assert store.list_commands() == []
        assert "YAML" in store.list_issues()[0].reason

    def test_one_bad_file_does_not_hide_the_good_ones(self, tmp_path):
        _write(tmp_path, "broken.md", "---\ndescription: [unclosed\n---\n\nbody\n")
        _write(tmp_path, "fine.md", "do the thing\n")
        store = UserCommandStore(root=tmp_path)
        assert [c.name for c in store.list_commands()] == ["fine"]

    def test_description_and_hint_are_clipped(self, tmp_path):
        # Both are rendered in fixed-width UI, so they're bounded where they're read.
        _write(
            tmp_path,
            "long.md",
            "---\ndescription: " + "d" * 200 + "\nargument_hint: " + "h" * 200 + "\n---\n\nbody\n",
        )
        (cmd,) = UserCommandStore(root=tmp_path).list_commands()
        assert len(cmd.description) <= user_commands._MAX_DESC_CHARS
        assert len(cmd.argument_hint) <= user_commands._MAX_HINT_CHARS
        assert cmd.description.endswith("…")

    def test_multiline_description_is_collapsed(self, tmp_path):
        _write(tmp_path, "multi.md", "---\ndescription: |\n  one\n  two\n---\n\nbody\n")
        (cmd,) = UserCommandStore(root=tmp_path).list_commands()
        assert cmd.description == "one two"


class TestMemoization:
    def test_edit_applies_to_the_next_read(self, tmp_path):
        p = _write(tmp_path, "x.md", "first\n")
        store = UserCommandStore(root=tmp_path)
        assert store.list_commands()[0].body == "first"
        p.write_text("second\n")
        os.utime(p, (0, 0))  # force a distinct mtime regardless of clock resolution
        assert UserCommandStore(root=tmp_path).list_commands()[0].body == "second"

    def test_new_file_applies_to_the_next_read(self, tmp_path):
        _write(tmp_path, "x.md", "first\n")
        store = UserCommandStore(root=tmp_path)
        assert len(store.list_commands()) == 1
        _write(tmp_path, "y.md", "second\n")
        assert len(UserCommandStore(root=tmp_path).list_commands()) == 2

    def test_unchanged_dir_is_not_reparsed(self, tmp_path):
        _write(tmp_path, "x.md", "body\n")
        first = UserCommandStore(root=tmp_path).list_commands()
        second = UserCommandStore(root=tmp_path).list_commands()
        assert first is second  # same cached list object


class TestExpand:
    def test_expands_with_arguments(self, tmp_path):
        _write(tmp_path, "explain.md", "Explain $ARGUMENTS in this repo.\n")
        got = UserCommandStore(root=tmp_path).expand("/explain the router")
        assert got is not None
        cmd, prompt = got
        assert cmd.name == "explain"
        assert prompt == "Explain the router in this repo."

    def test_expands_with_no_arguments(self, tmp_path):
        _write(tmp_path, "explain.md", "Explain $ARGUMENTS in this repo.\n")
        _, prompt = UserCommandStore(root=tmp_path).expand("/explain")
        assert prompt == "Explain  in this repo."

    def test_case_insensitive_lookup(self, tmp_path):
        _write(tmp_path, "Explain.md", "body\n")
        assert UserCommandStore(root=tmp_path).expand("/explain") is not None

    def test_prose_is_not_a_command(self, tmp_path):
        _write(tmp_path, "explain.md", "body\n")
        assert UserCommandStore(root=tmp_path).expand("explain the router") is None

    def test_unknown_slash_line_falls_through(self, tmp_path):
        # An unknown /thing keeps its current meaning (prose), not an error.
        assert UserCommandStore(root=tmp_path).expand("/nope now") is None

    def test_path_prefixed_message_is_not_a_command(self, tmp_path):
        _write(tmp_path, "usr.md", "body\n")
        assert UserCommandStore(root=tmp_path).expand("/usr/local/bin is on PATH") is None

    def test_get_accepts_slash_or_bare(self, tmp_path):
        _write(tmp_path, "x.md", "body\n")
        store = UserCommandStore(root=tmp_path)
        assert store.get("x") is not None
        assert store.get("/x") is not None
        assert store.get("") is None


class TestCompletions:
    def test_pairs_carry_the_hint_in_the_meta_text(self, tmp_path):
        _write(
            tmp_path,
            "review.md",
            "---\ndescription: Review a diff\nargument_hint: <path>\n---\n\nbody\n",
        )
        assert UserCommandStore(root=tmp_path).completions() == [
            ("/review", "Review a diff · <path>")
        ]

    def test_no_hint_is_just_the_description(self, tmp_path):
        _write(tmp_path, "ship.md", "---\ndescription: Ship it\n---\n\nbody\n")
        assert UserCommandStore(root=tmp_path).completions() == [("/ship", "Ship it")]


class TestReservedNames:
    def test_every_builtin_is_reserved(self):
        """BUILTIN_COMMANDS must cover the terminal UI's own command table.

        The two lists live apart on purpose — this module stays pure file logic so
        ``/doctor`` can consult it without importing the UI — so a new built-in
        added without reserving it would let a same-named file load and never fire.
        """
        from mnemoai.client.ui.chat_interface import ChatInterface

        builtin = {name.lstrip("/").lower() for name, _ in ChatInterface._COMMANDS}
        assert builtin <= set(BUILTIN_COMMANDS), (
            "add these to user_commands.BUILTIN_COMMANDS: "
            f"{sorted(builtin - set(BUILTIN_COMMANDS))}"
        )

    def test_default_store_reserves_them(self, tmp_path):
        store = UserCommandStore(root=tmp_path)
        assert "help" in store.reserved
        assert "doctor" in store.reserved
