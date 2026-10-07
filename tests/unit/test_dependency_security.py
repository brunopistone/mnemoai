"""Security floors must protect both locked checkouts and PyPI installations."""

import tomllib
from pathlib import Path

import pytest
from langgraph_sdk import Auth
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
# PyJWT 2.15 also fixes CVE-2026-101918, beyond the 2.14 algorithm-confusion fixes.
# urllib3 2.8 fixes CVE-2026-97687/97688/97689.
SECURITY_FLOORS = [
    # GHSA-fvww-7h3r-vfhp: resource decorators previously ignored actions=.
    ("langgraph-sdk", "0.4.4", ("0.4.2", "0.4.3")),
    # 6.19.0 also includes the 6.17/6.18 resource-exhaustion fixes.
    ("pypdf", "6.19.0", ("6.16.2", "6.18.0", "6.18.1")),
    ("pyjwt", "2.15.0", ("2.13.0", "2.14.0")),
    ("urllib3", "2.8.0", ("2.6.3", "2.7.0")),
]


def _load(name):
    return tomllib.loads((ROOT / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name,fixed,vulnerable", SECURITY_FLOORS)
def test_pypi_dependencies_reject_known_vulnerable_versions(name, fixed, vulnerable):
    # pip does not read uv.lock, and can otherwise keep a vulnerable transitive
    # dependency already present when the user upgrades mnemoai.
    requirements = {
        canonicalize_name(req.name): req
        for req in map(Requirement, _load("pyproject.toml")["project"]["dependencies"])
    }
    assert name in requirements, f"{name} needs a published security minimum"
    requirement = requirements[name]
    assert requirement.marker is None, "the floor must apply on every platform"
    assert all(version not in requirement.specifier for version in vulnerable)
    assert fixed in requirement.specifier


@pytest.mark.parametrize("name,fixed,vulnerable", SECURITY_FLOORS)
def test_locked_dependencies_satisfy_security_floor(name, fixed, vulnerable):
    locked = [
        package for package in _load("uv.lock")["package"]
        if canonicalize_name(package["name"]) == name
    ]
    assert locked
    assert all(Version(package["version"]) >= Version(fixed) for package in locked)


def test_lockfile_project_version_matches_package_metadata():
    project = _load("pyproject.toml")["project"]
    locked = next(
        package for package in _load("uv.lock")["package"]
        if package["name"] == project["name"]
    )
    assert locked["version"] == project["version"]


@pytest.mark.parametrize("resource", ["threads", "assistants", "crons"])
def test_langgraph_resource_auth_actions_do_not_bypass_default_deny(resource):
    auth = Auth()

    @auth.on
    async def deny_other_actions(ctx, value):
        return False

    @getattr(auth.on, resource)(actions=["create"])
    async def allow_create(ctx, value):
        return None

    # Pin the upstream fix against the real SDK registry, not a version mock.
    assert auth._handlers == {(resource, "create"): [allow_create]}
    assert (resource, "*") not in auth._handlers
    assert auth._global_handlers == [deny_other_actions]


@pytest.mark.parametrize("actions", [[], ["unknown_action"], ["create", "create"]])
def test_langgraph_rejects_invalid_action_scopes(actions):
    auth = Auth()

    async def handler(ctx, value):
        return None

    with pytest.raises(ValueError):
        auth.on.threads(actions=actions)(handler)
    assert auth._handlers == {}


def test_langgraph_unscoped_resource_handler_remains_supported():
    auth = Auth()

    @auth.on.threads
    async def handler(ctx, value):
        return None

    assert auth._handlers == {("threads", "*"): [handler]}
