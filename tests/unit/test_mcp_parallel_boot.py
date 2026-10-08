"""Unit tests: the MCP servers are connected — and asked for their tools — all
at once, without changing anything the ORDER used to decide.

Every external server in ``mcp.json`` is a subprocess that has to be spawned and
then answer ``initialize()``, so a boot that waited for each in turn paid the SUM
of those waits before the first prompt appeared. Overlapping them is only safe if
nothing order-sensitive moved onto the worker threads: the member list, the
collision namespacing, the failure reports (which PRINT) and the built-in
server's veto over startup all have to read exactly as they did.

Concurrency is proven with a ``Barrier``, not with timings: a barrier can only be
passed if every member is inside its call at the same moment, so a sequential
implementation fails the test instead of merely being slower.
"""

import threading

import pytest

from mnemoai.client.mcp_tool_wrapper import MultiMCPClient, _in_parallel

# Long enough that a machine under load doesn't fail the test, short enough that
# a genuinely sequential boot doesn't hang the suite for a noticeable time.
_BARRIER_TIMEOUT = 5.0


class _Tool:
    def __init__(self, name):
        self.name = name
        self.mcp_tool = type("T", (), {"name": name})()


class _Member:
    """A stand-in server that records how, and on which thread, it was driven."""

    def __init__(self, tools=(), barrier=None, fail=None, list_fail=None, gate=None):
        self._tools = [_Tool(t) for t in tools]
        self._barrier = barrier
        self._fail = fail
        self._list_fail = list_fail
        self._gate = gate
        self.connected = False
        self.shut_down = False
        self.connect_thread = None
        self.list_thread = None

    def _sync(self):
        if self._barrier is not None:
            self._barrier.wait(timeout=_BARRIER_TIMEOUT)
        if self._gate is not None:
            self._gate.wait(timeout=_BARRIER_TIMEOUT)

    def __enter__(self):
        self.connect_thread = threading.current_thread()
        self._sync()
        if self._fail is not None:
            raise self._fail
        self.connected = True
        return self

    def __exit__(self, *a):
        return False

    def list_tools_sync(self):
        self.list_thread = threading.current_thread()
        self._sync()
        if self._list_fail is not None:
            raise self._list_fail
        return list(self._tools)

    def shutdown(self):
        self.shut_down = True


def _multi(members):
    """A MultiMCPClient over stand-in members, with no real subprocess."""
    m = MultiMCPClient.__new__(MultiMCPClient)
    m._members = members
    m._tools = []
    m._cancel_probe = None
    m._startup = None
    m._startup_merged = False
    return m


@pytest.fixture
def reports(monkeypatch):
    """Capture the user-facing failure reports and the thread they came from."""
    seen = []
    monkeypatch.setattr(
        MultiMCPClient,
        "_report_member_failure",
        staticmethod(
            lambda what, exc: seen.append((what, exc, threading.current_thread()))
        ),
    )
    return seen


class TestConnectingHappensAtOnce:
    def test_every_member_connects_at_the_same_time(self):
        # The barrier can only be passed if all three are inside __enter__
        # together — a sequential boot deadlocks on the first one.
        barrier = threading.Barrier(3)
        members = [
            ("builtin", _Member(barrier=barrier)),
            ("aws", _Member(barrier=barrier)),
            ("playwright", _Member(barrier=barrier)),
        ]
        m = _multi(members)
        assert m.__enter__() is m
        assert all(w.connected for _, w in members)

    def test_each_member_is_connected_off_the_calling_thread(self):
        barrier = threading.Barrier(2)
        members = [
            ("builtin", _Member(barrier=barrier)),
            ("aws", _Member(barrier=barrier)),
        ]
        _multi(members).__enter__()
        here = threading.current_thread()
        assert all(w.connect_thread is not here for _, w in members)

    def test_a_lone_member_stays_on_the_calling_thread(self):
        # The common case is the built-in server by itself: nothing to overlap,
        # so it must not cost a thread hop.
        member = _Member()
        _multi([("builtin", member)]).__enter__()
        assert member.connect_thread is threading.current_thread()

    def test_the_live_list_keeps_declaration_order(self):
        # Whichever server answers first, 'builtin' stays first — the merge below
        # depends on it to keep the core tool names.
        late = threading.Event()
        m = _multi([("builtin", _Member(gate=late)), ("aws", _Member())])
        threading.Timer(0.05, late.set).start()
        m.__enter__()
        assert [name for name, _w in m._members] == ["builtin", "aws"]


class TestAFailingServerIsStillJustSkipped:
    def test_an_external_failure_leaves_the_rest_live(self, reports):
        good = _Member()
        bad = _Member(fail=RuntimeError("no such binary"))
        m = _multi([("builtin", good), ("broken", bad)])
        m.__enter__()
        assert [name for name, _w in m._members] == ["builtin"]
        assert good.connected is True

    def test_the_failure_is_reported_once_and_names_the_server(self, reports):
        m = _multi([
            ("builtin", _Member()),
            ("broken", _Member(fail=RuntimeError("no such binary"))),
        ])
        m.__enter__()
        assert len(reports) == 1
        what, exc, _thread = reports[0]
        assert "broken" in what and "failed to start" in what
        assert isinstance(exc, RuntimeError)

    def test_reports_come_out_in_member_order(self, reports):
        # Two servers failing in the opposite order to their declaration: the
        # report order must follow mcp.json, not the race.
        late = threading.Event()
        m = _multi([
            ("builtin", _Member()),
            ("a", _Member(fail=RuntimeError("a died"), gate=late)),
            ("b", _Member(fail=RuntimeError("b died"))),
        ])
        threading.Timer(0.05, late.set).start()
        m.__enter__()
        assert [str(exc) for _what, exc, _t in reports] == ["a died", "b died"]

    def test_reports_are_printed_from_the_calling_thread(self, reports):
        # _report_member_failure prints; from a pool thread the lines could
        # interleave with each other and with the startup spinner.
        m = _multi([
            ("builtin", _Member()),
            ("broken", _Member(fail=RuntimeError("x"))),
        ])
        m.__enter__()
        assert reports[0][2] is threading.current_thread()


class TestTheBuiltinServerStillVetoesStartup:
    def test_a_builtin_failure_is_re_raised(self, reports):
        m = _multi([
            ("builtin", _Member(fail=RuntimeError("builtin died"))),
            ("aws", _Member()),
        ])
        with pytest.raises(RuntimeError, match="builtin died"):
            m.__enter__()

    def test_the_externals_that_came_up_are_handed_back(self, reports):
        # New with a parallel boot: connecting in turn, no external had started
        # yet when the built-in failed. Now they have subprocesses to release.
        external = _Member()
        m = _multi([
            ("builtin", _Member(fail=RuntimeError("builtin died"))),
            ("aws", external),
        ])
        with pytest.raises(RuntimeError):
            m.__enter__()
        assert external.connected is True
        assert external.shut_down is True

    def test_an_aborted_boot_reports_nothing(self, reports):
        # Startup is over, so a skipped-server notice would be noise above the
        # failure that actually ended it.
        m = _multi([
            ("builtin", _Member(fail=RuntimeError("builtin died"))),
            ("aws", _Member(fail=RuntimeError("aws died"))),
        ])
        with pytest.raises(RuntimeError, match="builtin died"):
            m.__enter__()
        assert reports == []


class TestListingToolsHappensAtOnce:
    def test_every_member_is_asked_at_the_same_time(self):
        barrier = threading.Barrier(3)
        m = _multi([
            ("builtin", _Member(["read_file"], barrier=barrier)),
            ("aws", _Member(["aws_docs"], barrier=barrier)),
            ("playwright", _Member(["click"], barrier=barrier)),
        ])
        names = [t.name for t in m.list_tools_sync()]
        assert names == ["read_file", "aws_docs", "click"]

    def test_the_merge_follows_member_order_not_the_race(self):
        # The namespacing rule is "built-in names win", which only holds if the
        # built-in server's tools are merged first however late they arrive.
        late = threading.Event()
        m = _multi([
            ("builtin", _Member(["read_file"], gate=late)),
            ("ext", _Member(["read_file"])),
        ])
        threading.Timer(0.05, late.set).start()
        names = [t.name for t in m.list_tools_sync()]
        assert names == ["read_file", "ext__read_file"]

    def test_a_member_that_cannot_list_is_skipped(self, reports):
        m = _multi([
            ("builtin", _Member(["read_file"])),
            ("broken", _Member(list_fail=RuntimeError("server died"))),
        ])
        assert [t.name for t in m.list_tools_sync()] == ["read_file"]
        assert "could not list tools" in reports[0][0]

    def test_listing_reports_also_come_out_in_member_order(self, reports):
        late = threading.Event()
        m = _multi([
            ("builtin", _Member(["ok"])),
            ("a", _Member(list_fail=RuntimeError("a died"), gate=late)),
            ("b", _Member(list_fail=RuntimeError("b died"))),
        ])
        threading.Timer(0.05, late.set).start()
        m.list_tools_sync()
        assert [str(exc) for _what, exc, _t in reports] == ["a died", "b died"]


class TestTheParallelHelper:
    def test_results_are_returned_in_member_order(self):
        members = [("a", 1), ("b", 2), ("c", 3)]
        out = _in_parallel(members, lambda w: w * 10)
        assert [(m[0], v, exc) for m, v, exc in out] == [
            ("a", 10, None),
            ("b", 20, None),
            ("c", 30, None),
        ]

    def test_a_raising_member_hands_its_exception_back(self):
        boom = ValueError("nope")

        def _work(w):
            if w == "bad":
                raise boom
            return w

        out = _in_parallel([("ok", "ok"), ("bad", "bad")], _work)
        assert out[0][1:] == ("ok", None)
        assert out[1][1] is None and out[1][2] is boom

    def test_a_sibling_still_runs_after_a_failure(self):
        ran = []

        def _work(w):
            ran.append(w)
            if w == "first":
                raise RuntimeError("x")
            return w

        _in_parallel([("a", "first"), ("b", "second")], _work)
        assert sorted(ran) == ["first", "second"]

    def test_an_interrupt_is_not_one_members_failure(self):
        # Esc must end the boot, not be filed as "that server didn't start".
        def _work(w):
            if w == "first":
                raise KeyboardInterrupt
            return w

        with pytest.raises(KeyboardInterrupt):
            _in_parallel([("a", "first"), ("b", "second")], _work)

    def test_no_members_is_an_empty_result(self):
        assert _in_parallel([], lambda w: w) == []


class TestBackgroundStartup:
    def test_status_notices_a_server_that_dies_after_discovery(self):
        external = _Member(["browser"])
        m = _multi([("builtin", _Member(["read"])), ("browser", external)])
        try:
            m.start_background()
            m.finish_startup()
            external._connected = False
            assert m.startup_status()[1].state == "disconnected"
        finally:
            m.shutdown()

    def test_external_named_builtin_does_not_overwrite_the_core_result(self):
        m = _multi([
            ("builtin", _Member(["read"])), ("builtin", _Member(["read", "external"])),
        ])
        try:
            assert [t.name for t in m.start_background()] == ["read"]
            assert [t.name for t in m.finish_startup()] == ["read", "builtin__read", "external"]
        finally:
            m.shutdown()

    def test_worker_interrupt_is_not_reported_as_an_optional_server_failure(self):
        m = _multi([
            ("builtin", _Member(["read"])), ("external", _Member(fail=KeyboardInterrupt())),
        ])
        try:
            m.start_background()
            with pytest.raises(KeyboardInterrupt):
                m.finish_startup()
        finally:
            m.shutdown()

    def test_prompt_does_not_wait_for_external_schemas(self):
        gate = threading.Event()
        external = _Member(["browser"], gate=gate)
        m = _multi([("builtin", _Member(["read"])), ("browser", external)])
        try:
            assert [t.name for t in m.start_background()] == ["read"]
            assert not gate.is_set()
            assert [(s.name, s.state) for s in m.startup_status()] == [
                ("builtin", "ready"), ("browser", "connecting"),
            ]
            # Re-entering the context (RAG setup/query) must not start it twice.
            assert m.__enter__() is m
            assert [t.name for t in m.finish_startup(wait=False)] == ["read"]
            gate.set()
            assert [t.name for t in m.list_tools_sync()] == ["read", "browser"]
            assert m.finish_startup() is m.list_tools_sync()
        finally:
            gate.set()
            m.shutdown()

    def test_cancelled_wait_leaves_discovery_for_the_next_turn(self):
        gate = threading.Event()
        external = _Member(["browser"], gate=gate)
        m = _multi([("builtin", _Member(["read"])), ("browser", external)])
        try:
            m.start_background()
            m.set_cancel_probe(lambda: True)
            assert external._cancel_probe is None
            with pytest.raises(KeyboardInterrupt):
                m.finish_startup()
            gate.set()
            m.set_cancel_probe(lambda: False)
            assert [t.name for t in m.finish_startup()] == ["read", "browser"]
            assert external._cancel_probe is m._cancel_probe
        finally:
            gate.set()
            m.shutdown()

    def test_failure_report_waits_for_foreground_and_only_prints_once(self, reports):
        m = _multi([
            ("builtin", _Member(["read"])),
            ("broken", _Member(list_fail=ValueError("bad schema"))),
        ])
        try:
            m.start_background()
            m._startup.wait()
            assert reports == []
            assert m.startup_status()[1].state == "failed"
            assert [t.name for t in m.finish_startup()] == ["read"]
            m.finish_startup()
            assert len(reports) == 1
            assert reports[0][2] is threading.current_thread()
        finally:
            m.shutdown()

    def test_builtin_schema_failure_still_aborts_and_closes_every_member(self):
        members = [
            ("builtin", _Member(list_fail=RuntimeError("no core schema"))),
            ("external", _Member(["tool"])),
        ]
        m = _multi(members)
        with pytest.raises(RuntimeError, match="no core schema"):
            m.start_background()
        assert all(w.shut_down for _, w in members)
        assert all(s.state == "closed" for s in m.startup_status())

    def test_shutdown_rejects_late_results(self):
        gate = threading.Event()
        m = _multi([
            ("builtin", _Member(["read"])),
            ("external", _Member(["tool"], gate=gate)),
        ])
        m.start_background()
        m.shutdown()
        gate.set()
        for thread in m._startup.threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
        with pytest.raises(RuntimeError, match="closed"):
            m.finish_startup()
        assert [t.name for t in m._tools] == ["read"]

    def test_collisions_stay_stable_without_mutating_provider_wrappers(self):
        builtin = _Member(["read", "ext__read"])
        external = _Member(["read"])
        m = _multi([("builtin", builtin), ("ext", external)])
        assert [t.name for t in m.list_tools_sync()] == ["read", "ext__read", "ext__read_2"]
        assert external._tools[0].name == "read"
        assert [t.name for t in m.list_tools_sync()] == ["read", "ext__read", "ext__read_2"]
