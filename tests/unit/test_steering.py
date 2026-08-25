"""Unit tests for always-on user instructions (STEERING.md / CLAUDE.md).

Covers the pure-logic pieces (no LLM): discovery/precedence (utils.paths),
per-directory filename shadowing, concatenation (SteeringStore), the
ephemeral-strip that keeps the injected block out of stored history (so
compaction never summarizes it), and that the block never reaches the router.
"""


import re
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from mnemoai.client.agent.agent import LangGraphAgent
from mnemoai.client.agent.router import is_trivial_query
from mnemoai.client.memory import steering_store
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


class TestClaudeMdAlsoAccepted:
    """``CLAUDE.md`` is honored wherever ``STEERING.md`` is, and loses to it
    within a single directory — see ``paths.STEERING_FILENAMES``.

    Shadowing is per-DIRECTORY, not global: the point of these tests is that a
    project ``STEERING.md`` must not suppress a global ``CLAUDE.md``, and a
    parent directory's choice must not constrain a child's.
    """

    @staticmethod
    def _home(tmp_path, monkeypatch, *, files=()):
        """An app home containing the given ``(name, text)`` files."""
        home = tmp_path / "home"
        home.mkdir()
        for name, text in files:
            (home / name).write_text(text)
        monkeypatch.setattr(paths, "app_home", lambda: home)
        return home

    def test_claude_md_alone_is_discovered(self, tmp_path, monkeypatch):
        # A repo carrying only CLAUDE.md needs no second file to be picked up.
        self._home(tmp_path, monkeypatch)
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        (repo / "CLAUDE.md").write_text("FROM CLAUDE")
        files = paths.steering_files(cwd=repo)
        assert [f.read_text() for f in files] == ["FROM CLAUDE"]

    def test_steering_wins_over_a_sibling_claude_md(self, tmp_path, monkeypatch):
        # Both in ONE directory: STEERING.md is read, its sibling is skipped —
        # so the two can coexist and only one costs context.
        self._home(tmp_path, monkeypatch)
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        (repo / "STEERING.md").write_text("WINNER")
        (repo / "CLAUDE.md").write_text("SHADOWED")
        files = paths.steering_files(cwd=repo)
        assert [f.read_text() for f in files] == ["WINNER"]
        assert all(f.name == "STEERING.md" for f in files)

    def test_shadowing_is_per_directory_not_global(self, tmp_path, monkeypatch):
        # The regression this pins: a project STEERING.md must NOT suppress a
        # global CLAUDE.md, and a deeper CLAUDE.md with no sibling still applies.
        self._home(tmp_path, monkeypatch, files=[("CLAUDE.md", "GLOBAL")])
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        (repo / "STEERING.md").write_text("PROJECT")
        (repo / "CLAUDE.md").write_text("PROJECT SHADOWED")
        deep = repo / "src" / "pkg"
        deep.mkdir(parents=True)
        (deep / "CLAUDE.md").write_text("DEEP")
        files = paths.steering_files(cwd=deep)
        assert [f.read_text() for f in files] == ["GLOBAL", "PROJECT", "DEEP"]

    def test_a_global_steering_md_does_not_suppress_a_project_claude_md(
        self, tmp_path, monkeypatch
    ):
        # The other direction of the same invariant: finding a STEERING.md in one
        # tier must not switch the OTHER name off for the remaining tiers. A
        # global-shadowing implementation passes the test above but fails here.
        self._home(tmp_path, monkeypatch, files=[("STEERING.md", "GLOBAL STEERING")])
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        (repo / "CLAUDE.md").write_text("PROJECT CLAUDE")
        files = paths.steering_files(cwd=repo)
        assert [f.read_text() for f in files] == ["GLOBAL STEERING", "PROJECT CLAUDE"]

    def test_global_tier_falls_back_to_claude_md(self, tmp_path, monkeypatch):
        # The global tier reads the app home only, under either name.
        self._home(tmp_path, monkeypatch, files=[("CLAUDE.md", "GLOBAL CLAUDE")])
        empty = tmp_path / "elsewhere"
        empty.mkdir()
        files = paths.steering_files(cwd=empty)
        assert [f.read_text() for f in files] == ["GLOBAL CLAUDE"]

    def test_global_steering_md_wins_in_the_app_home_too(self, tmp_path, monkeypatch):
        self._home(
            tmp_path,
            monkeypatch,
            files=[("STEERING.md", "GLOBAL STEERING"), ("CLAUDE.md", "GLOBAL SHADOW")],
        )
        empty = tmp_path / "elsewhere"
        empty.mkdir()
        files = paths.steering_files(cwd=empty)
        assert [f.read_text() for f in files] == ["GLOBAL STEERING"]

    def test_a_directory_named_claude_md_is_not_read(self, tmp_path, monkeypatch):
        # is_file() must reject a DIRECTORY with the accepted name, rather than
        # letting a read raise later.
        self._home(tmp_path, monkeypatch)
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        (repo / "CLAUDE.md").mkdir()  # a directory, not a file
        assert paths.steering_files(cwd=repo) == []

    def test_authoring_path_stays_steering_md(self):
        # Reading accepts both names; WRITING has one canonical target, so the
        # creator skill can't be talked into authoring the fallback name.
        assert paths.global_steering_path().name == "STEERING.md"

    def test_precedence_order_is_declared_not_incidental(self):
        # The whole "STEERING.md wins" rule is this tuple's order.
        assert paths.STEERING_FILENAMES[0] == "STEERING.md"
        assert "CLAUDE.md" in paths.STEERING_FILENAMES

    def test_same_file_reached_twice_is_injected_once(self, tmp_path, monkeypatch):
        # When the app home lies inside the walked project chain, the global tier
        # and the walk reach the SAME file; it must not be injected twice.
        home = tmp_path / "repo" / "home"
        home.mkdir(parents=True)
        (home / ".git").mkdir()  # terminate the walk at the app home itself
        (home / "CLAUDE.md").write_text("ONCE")
        monkeypatch.setattr(paths, "app_home", lambda: home)
        files = paths.steering_files(cwd=home)
        assert [f.read_text() for f in files] == ["ONCE"]

    def test_de_dup_survives_a_symlinked_app_home(self, tmp_path, monkeypatch):
        # The de-dup must key on the REAL path, not the spelling: app_home() is
        # returned unresolved while the project walk resolves, so one file
        # reached under two spellings would otherwise be injected twice — paying
        # double context and stating every rule twice to the model.
        real = tmp_path / "real"
        real.mkdir()
        (real / ".git").mkdir()  # terminate the walk here
        (real / "CLAUDE.md").write_text("ONCE")
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)
        monkeypatch.setattr(paths, "app_home", lambda: link)
        files = paths.steering_files(cwd=link)
        assert [f.read_text() for f in files] == ["ONCE"]


class TestWalkIsBoundedAtHome:
    """With no ``.git`` anywhere, the walk must stop at the home dir.

    Unbounded it reaches the filesystem root, so one ``~/CLAUDE.md`` would become
    always-on instructions for EVERY non-git directory under it. That is a real
    setup — other tooling puts an instructions file in the home dir — and it
    would quietly break the rule that only the app home is global.
    """

    def test_a_home_level_file_is_not_picked_up(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        (home / "work" / "scratch").mkdir(parents=True)
        (home / "CLAUDE.md").write_text("STRAY HOME FILE")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(paths, "app_home", lambda: tmp_path / "apphome")

        assert paths.steering_files(cwd=home / "work" / "scratch") == []

    def test_a_file_above_home_is_not_picked_up(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        (home / "work").mkdir(parents=True)
        (tmp_path / "CLAUDE.md").write_text("ABOVE HOME")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(paths, "app_home", lambda: tmp_path / "apphome")

        assert paths.steering_files(cwd=home / "work") == []

    def test_a_non_git_dir_below_home_still_contributes(self, tmp_path, monkeypatch):
        # The boundary excludes home and up — NOT the ordinary dirs beneath it.
        home = tmp_path / "home"
        work = home / "work"
        work.mkdir(parents=True)
        (work / "STEERING.md").write_text("WORK RULES")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(paths, "app_home", lambda: tmp_path / "apphome")

        assert [f.read_text() for f in paths.steering_files(cwd=work)] == ["WORK RULES"]

    def test_an_explicit_git_root_at_home_still_wins(self, tmp_path, monkeypatch):
        # A dotfiles repo checked out AT $HOME is a deliberate choice, so a real
        # .git there must still bound the walk inclusively — the boundary only
        # applies when there is no project root at all.
        home = tmp_path / "home"
        (home / ".git").mkdir(parents=True)
        (home / "CLAUDE.md").write_text("DOTFILES ROOT")
        sub = home / "proj"
        sub.mkdir()
        (sub / "STEERING.md").write_text("PROJ")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(paths, "app_home", lambda: tmp_path / "apphome")

        names = [f.read_text() for f in paths.steering_files(cwd=sub)]
        assert names == ["DOTFILES ROOT", "PROJ"]


class TestSteeringRobustness:
    """A bad instruction file must not break the conversation.

    This content is read on EVERY turn, so anything that raises in here fails
    the whole turn, not one feature — and these files are often authored by other
    tooling, so odd encodings and sizes are expected input, not corner cases.
    """

    def test_non_utf8_file_does_not_raise(self, tmp_path):
        # UnicodeDecodeError is a ValueError, NOT an OSError: an OSError-only
        # guard let it escape SteeringStore -> the client -> every turn.
        f = tmp_path / "CLAUDE.md"
        f.write_bytes("Always café rules".encode("latin-1"))
        out = SteeringStore(files=[f]).read()
        assert "Always caf" in out  # the stray byte degrades, the rules survive

    def test_unreadable_file_is_skipped_not_fatal(self, tmp_path):
        f = tmp_path / "STEERING.md"
        f.write_text("secret")
        f.chmod(0o000)
        try:
            assert SteeringStore(files=[f]).read() == ""
        finally:
            f.chmod(0o644)  # so tmp_path cleanup can remove it

    def test_oversized_file_is_capped_and_says_so(self, tmp_path):
        # Never silent: the model is told it saw a prefix, so it reads the file
        # itself instead of assuming the rest was empty.
        #
        # Sized RELATIVE to the real default, not a hardcoded number: a literal
        # that happens to sit under a raised cap stops testing truncation while
        # still passing.
        oversized = steering_store._DEFAULT_MAX_CHARS * 2
        f = tmp_path / "CLAUDE.md"
        f.write_text("y" * oversized)
        out = SteeringStore(files=[f]).read()
        assert len(out) < oversized
        assert "truncated" in out
        assert str(f) in out  # names the file to read

    def test_the_documented_default_cap_matches_the_shipped_config(self):
        # The examples and the guide quote a number; the code default is what
        # actually applies when nobody edits config. A drift between them is a
        # docs bug that no other test would notice.
        examples = list(
            (Path(steering_store.__file__).parents[2] / "utils").glob(
                "config.yaml*.example"
            )
        )
        assert examples, "no config examples found"
        for ex in examples:
            quoted = re.search(
                r"^STEERING:\s*\n\s*MAX_CHARS:\s*(\d+)", ex.read_text(), re.MULTILINE
            )
            assert quoted, f"{ex.name} does not document STEERING.MAX_CHARS"
            assert int(quoted.group(1)) == steering_store._DEFAULT_MAX_CHARS, ex.name

    def test_a_normal_sized_file_is_untouched(self, tmp_path):
        f = tmp_path / "STEERING.md"
        f.write_text("Use tabs.\n" * 50)
        out = SteeringStore(files=[f]).read()
        assert "truncated" not in out
        assert out.count("Use tabs.") == 50

    def test_unreadable_candidate_falls_through_to_the_other_name(
        self, tmp_path, monkeypatch
    ):
        # Readability is part of the per-directory CHOICE: otherwise an
        # unreadable STEERING.md shadows a good CLAUDE.md beside it and the
        # directory contributes nothing at all.
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(paths, "app_home", lambda: home)
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        blocked = repo / "STEERING.md"
        blocked.write_text("UNREADABLE")
        blocked.chmod(0o000)
        (repo / "CLAUDE.md").write_text("READABLE FALLBACK")
        try:
            files = paths.steering_files(cwd=repo)
            assert [f.read_text() for f in files] == ["READABLE FALLBACK"]
        finally:
            blocked.chmod(0o644)

    def test_one_unreadable_ancestor_does_not_abandon_the_walk(
        self, tmp_path, monkeypatch
    ):
        # A directory whose own instruction file can't be probed must be SKIPPED,
        # with the walk continuing — not abort the whole resolution. Before the
        # per-directory guard, one PermissionError anywhere in the chain returned
        # early and dropped every file collected after it.
        #
        # Note what is and isn't recoverable: an unreadable dir also blocks
        # access to its CHILDREN (the OS denies the stat), so a file below it is
        # genuinely unreachable. What must survive is everything reachable — here
        # the repo-root file, which is collected after the bad directory is
        # skipped.
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(paths, "app_home", lambda: home)
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        (repo / "STEERING.md").write_text("ROOT")
        mid = repo / "mid"
        mid.mkdir()
        deep = mid / "deep"
        deep.mkdir()
        mid.chmod(0o000)  # unreadable directory between cwd and the repo root
        try:
            got = [f.read_text() for f in paths.steering_files(cwd=deep)]
        finally:
            mid.chmod(0o755)
        assert got == ["ROOT"], got  # reachable instructions still applied

    def test_walk_stops_at_a_git_FILE_not_just_a_dir(self, tmp_path, monkeypatch):
        # A git worktree/submodule writes .git as a FILE; the walk must still
        # treat it as the project root and not climb above it.
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(paths, "app_home", lambda: home)
        (tmp_path / "CLAUDE.md").write_text("ABOVE THE WORKTREE")
        wt = tmp_path / "wt"
        wt.mkdir()
        (wt / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt")
        (wt / "CLAUDE.md").write_text("WORKTREE")
        files = paths.steering_files(cwd=wt)
        assert [f.read_text() for f in files] == ["WORKTREE"]


class TestSteeringDoesNotCorruptRouting:
    """Routing must run on what the USER asked, not on the injected block.

    The graph carries the full prompt (steering included, by design), but every
    routing decision reads message text — so an always-on file of any size buries
    a short query, and its paths/extensions trip the deterministic signal
    patterns. Accepting ``CLAUDE.md`` made this reachable in practice: repos
    already carry large ones (this project's own is ~56k chars).
    """

    @staticmethod
    def _steered(query, rules="See src/main.py and docs/notes.pdf for conventions."):
        # A realistic block: prose plus the file paths/extensions that are exactly
        # what the router's deterministic signals look for.
        return HumanMessage(content=f"<steering>{rules}</steering>\n\n{query}")

    def test_a_greeting_stays_trivial_behind_a_steering_block(self):
        # Without the strip, is_trivial_query() sees the whole block: over the
        # word gate AND carrying path signals -> "Hello" gets decomposed.
        recovered = LangGraphAgent._last_human_query([self._steered("Hello")])
        assert recovered == "Hello"
        assert is_trivial_query(recovered)

    def test_the_last_human_message_wins_over_earlier_turns(self):
        msgs = [
            self._steered("first question"),
            AIMessage(content="an answer"),
            self._steered("second question"),
        ]
        assert LangGraphAgent._last_human_query(msgs) == "second question"

    def test_no_human_message_yields_empty_not_an_error(self):
        assert LangGraphAgent._last_human_query([AIMessage(content="x")]) == ""
        assert LangGraphAgent._last_human_query([]) == ""

    def test_classify_sees_the_stripped_query(self):
        # _classify feeds the router directly; it must strip too, or the LLM
        # classifier is handed the instruction file instead of the question.
        agent = LangGraphAgent.__new__(LangGraphAgent)
        seen = []

        class _Router:
            def classify(self, query, context=""):
                seen.append(query)
                return "simple_qa"

        agent.router = _Router()
        state = {"messages": [self._steered("What time is it?")]}
        assert agent._classify(state) == {"route": "simple_qa"}
        assert seen == ["What time is it?"]


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


class TestSteeringSizes:
    """``sizes()`` backs the per-file rows of ``/context``, so what it reports has
    to be what is actually injected — the capped text, not the file on disk."""

    def test_one_entry_per_file_in_apply_order(self, tmp_path):
        a = tmp_path / "a" / "STEERING.md"
        a.parent.mkdir()
        a.write_text("short")
        b = tmp_path / "b" / "STEERING.md"
        b.parent.mkdir()
        b.write_text("a much longer rule set")
        sizes = SteeringStore(files=[a, b]).sizes()
        assert [path for path, _ in sizes] == [a, b]
        assert len(sizes[1][1]) > len(sizes[0][1])

    def test_empty_and_missing_files_get_no_row(self, tmp_path):
        blank = tmp_path / "blank" / "STEERING.md"
        blank.parent.mkdir()
        blank.write_text("   \n")
        gone = tmp_path / "gone" / "STEERING.md"
        real = tmp_path / "STEERING.md"
        real.write_text("kept")
        assert [p for p, _ in SteeringStore(files=[blank, gone, real]).sizes()] == [real]

    def test_reports_the_capped_size_not_the_file_size(self, tmp_path, monkeypatch):
        monkeypatch.setattr(SteeringStore, "_max_chars", staticmethod(lambda: 100))
        big = tmp_path / "STEERING.md"
        big.write_text("x" * 5000)
        _, text = SteeringStore(files=[big]).sizes()[0]
        assert len(text) < 5000

    def test_no_files_no_rows(self):
        assert SteeringStore(files=[]).sizes() == []


class TestSteeringReferences:
    """``@path`` lets a long ruleset be split into focused files. The reference is
    resolved against the file that MENTIONS it and only inlined when it names a
    real file, which is what keeps a false positive (a decorator, a handle) from
    turning into noise."""

    def _steering(self, tmp_path, text):
        f = tmp_path / "STEERING.md"
        f.write_text(text)
        return f

    def test_a_referenced_file_is_injected_with_its_own_header(self, tmp_path):
        (tmp_path / "rules").mkdir()
        (tmp_path / "rules" / "style.md").write_text("two-space indents")
        f = self._steering(tmp_path, "Follow @rules/style.md at all times.")
        out = SteeringStore(files=[f]).read()
        assert "two-space indents" in out
        assert "referenced by @rules/style.md" in out
        assert str(tmp_path / "rules" / "style.md") in out

    def test_the_reference_stays_in_the_prose(self, tmp_path):
        (tmp_path / "x.md").write_text("body")
        f = self._steering(tmp_path, "See @x.md for details.")
        out = SteeringStore(files=[f]).read()
        assert "See @x.md for details." in out  # the sentence still reads

    def test_relative_to_the_referencing_file_not_the_process_cwd(self, tmp_path):
        # The project's steering file means ITS neighbours, wherever the app ran.
        deep = tmp_path / "proj" / "sub"
        deep.mkdir(parents=True)
        (deep / "extra.md").write_text("nested rule")
        f = deep / "STEERING.md"
        f.write_text("also @extra.md")
        assert "nested rule" in SteeringStore(files=[f]).read()

    def test_absolute_and_home_paths_resolve(self, tmp_path, monkeypatch):
        target = tmp_path / "abs.md"
        target.write_text("absolute rule")
        f = self._steering(tmp_path, f"see @{target}")
        assert "absolute rule" in SteeringStore(files=[f]).read()

        home_file = tmp_path / "home.md"
        home_file.write_text("home rule")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        g = tmp_path / "other" / "STEERING.md"
        g.parent.mkdir()
        g.write_text("see @~/home.md")
        assert "home rule" in SteeringStore(files=[g]).read()

    def test_a_nonexistent_reference_is_left_alone(self, tmp_path):
        f = self._steering(tmp_path, "see @nope/missing.md for details")
        out = SteeringStore(files=[f]).read()
        assert "see @nope/missing.md for details" in out
        assert "referenced by" not in out

    def test_a_decorator_or_handle_is_not_a_reference(self, tmp_path):
        # Nothing resolves, so nothing is injected — the false positive is inert.
        f = self._steering(
            tmp_path, "Use @staticmethod here. Ask @someone. Mail a@b.com."
        )
        out = SteeringStore(files=[f]).read()
        assert "referenced by" not in out
        assert "@staticmethod" in out and "a@b.com" in out

    def test_an_email_address_never_matches(self, tmp_path):
        # Even when a file with the domain's name exists: the @ is mid-word.
        (tmp_path / "b.com").write_text("SHOULD NOT APPEAR")
        f = self._steering(tmp_path, "contact a@b.com")
        assert "SHOULD NOT APPEAR" not in SteeringStore(files=[f]).read()

    def test_a_directory_reference_is_not_inlined(self, tmp_path):
        (tmp_path / "rules").mkdir()
        f = self._steering(tmp_path, "see @rules")
        assert "referenced by" not in SteeringStore(files=[f]).read()

    def test_references_are_followed_transitively(self, tmp_path):
        (tmp_path / "a.md").write_text("level one, then @b.md")
        (tmp_path / "b.md").write_text("level two")
        f = self._steering(tmp_path, "start at @a.md")
        out = SteeringStore(files=[f]).read()
        assert "level one" in out and "level two" in out

    def test_a_reference_cycle_terminates(self, tmp_path):
        (tmp_path / "a.md").write_text("A refers to @b.md")
        (tmp_path / "b.md").write_text("B refers back to @a.md")
        f = self._steering(tmp_path, "see @a.md")
        out = SteeringStore(files=[f]).read()
        assert out.count("A refers to") == 1
        assert out.count("B refers back") == 1

    def test_a_file_is_injected_once_however_many_times_it_is_named(self, tmp_path):
        (tmp_path / "one.md").write_text("single copy")
        f = self._steering(tmp_path, "@one.md and again @one.md and ./@one.md")
        assert SteeringStore(files=[f]).read().count("single copy") == 1

    def test_a_reference_to_another_discovered_file_is_not_duplicated(self, tmp_path):
        glob = tmp_path / "global" / "STEERING.md"
        glob.parent.mkdir()
        glob.write_text("global rule")
        proj = tmp_path / "proj" / "STEERING.md"
        proj.parent.mkdir()
        proj.write_text(f"inherit @{glob}")
        out = SteeringStore(files=[glob, proj]).read()
        assert out.count("global rule") == 1

    def test_deep_chains_stop_at_the_depth_limit(self, tmp_path):
        # Each level points at the next; beyond the limit nothing more is pulled in.
        depth = steering_store._MAX_INCLUDE_DEPTH
        for i in range(depth + 3):
            (tmp_path / f"l{i}.md").write_text(f"level {i} then @l{i + 1}.md")
        f = self._steering(tmp_path, "start @l0.md")
        out = SteeringStore(files=[f]).read()
        assert "level 0" in out
        assert f"level {depth + 2}" not in out

    def test_includes_share_the_files_own_size_budget(self, tmp_path, monkeypatch):
        # Splitting a ruleset across references must not sidestep STEERING.MAX_CHARS.
        # Patch the effective cap, not the code default — a runtime config.yaml
        # that sets MAX_CHARS would otherwise win and the test would pass blindly.
        monkeypatch.setattr(SteeringStore, "_max_chars", staticmethod(lambda: 200))
        (tmp_path / "big.md").write_text("z" * 500)
        f = self._steering(tmp_path, "see @big.md")
        out = SteeringStore(files=[f]).read()
        assert "z" * 500 not in out
        assert "not included" in out  # and it says so

    def test_an_unreadable_reference_costs_only_itself(self, tmp_path):
        bad = tmp_path / "bad.md"
        bad.write_text("secret")
        bad.chmod(0o000)
        (tmp_path / "good.md").write_text("good rule")
        f = self._steering(tmp_path, "see @bad.md and @good.md")
        try:
            out = SteeringStore(files=[f]).read()
        finally:
            bad.chmod(0o644)
        assert "good rule" in out


class TestSteeringReferenceScan:
    """The pure text scan, independent of the filesystem."""

    def test_finds_references_in_order_without_duplicates(self):
        found = steering_store.references("@a.md then @b/c.md then @a.md")
        assert found == ["a.md", "b/c.md"]

    def test_a_trailing_period_is_not_part_of_the_path(self):
        assert steering_store.references("read @docs/style.md.") == ["docs/style.md"]

    def test_start_of_line_and_bracketed_forms_match(self):
        assert steering_store.references("@top.md") == ["top.md"]
        assert steering_store.references("(@paren.md)") == ["paren.md"]
        assert steering_store.references("`@code.md`") == ["code.md"]

    def test_mid_word_at_signs_never_match(self):
        assert steering_store.references("user@host.com") == []
        assert steering_store.references("x/y@v1.md") == []

    def test_empty_input_is_safe(self):
        assert steering_store.references("") == []
        assert steering_store.references(None) == []


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
