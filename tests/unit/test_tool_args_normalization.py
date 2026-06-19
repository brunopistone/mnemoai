"""Unit tests for LangGraphAgent._normalize_tool_args.

Smaller models sometimes emit a malformed tool-args dict where a
``field="value"`` expression is packed into a single KEY with an empty value
(e.g. ``{'query="USPTO fees"': ''}``) instead of ``{'query': 'USPTO fees'}``.
The normalizer repairs exactly that shape and leaves everything else untouched.
"""

from mnemoai.client.agent.agent import LangGraphAgent

n = LangGraphAgent._normalize_tool_args


def test_repairs_double_quoted_value():
    assert n({'query="USPTO trademark registration fee 2026"': ""}) == {
        "query": "USPTO trademark registration fee 2026"
    }


def test_repairs_single_quoted_value():
    assert n({"query='hello world'": ""}) == {"query": "hello world"}


def test_repairs_bare_value():
    assert n({"timeout=30": ""}) == {"timeout": "30"}


def test_repairs_when_value_is_none():
    # Some models put None rather than "" for the empty value.
    assert n({'query="x"': None}) == {"query": "x"}


def test_leaves_wellformed_single_arg():
    assert n({"query": "normal query"}) == {"query": "normal query"}


def test_leaves_multi_arg_dict():
    assert n({"path": "/tmp/x", "command": "create"}) == {
        "path": "/tmp/x",
        "command": "create",
    }


def test_leaves_when_value_populated():
    # A real single-arg call whose key happens to contain '=' is NOT repaired,
    # because the value is present (the malformation always has an empty value).
    assert n({"query=x": "real value"}) == {"query=x": "real value"}


def test_leaves_non_field_expression_key():
    assert n({"just a sentence": ""}) == {"just a sentence": ""}


def test_leaves_empty_dict():
    assert n({}) == {}


def test_leaves_non_dict():
    assert n("raw") == "raw"
    assert n(None) is None
