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
    # Preserve action-scope enforcement and encode streaming resource identifiers.
    # GHSA-fvww-7h3r-vfhp and GHSA-3fx2-cqw9-xr3c.
    ("langgraph-sdk", "0.4.6", ("0.4.2", "0.4.3", "0.4.4", "0.4.5")),
    ("langgraph", "1.2.14", ("1.2.3", "1.2.9", "1.2.13")),
    # Includes the malformed-reference and parser resource-exhaustion fixes.
    ("pypdf", "6.21.2", ("6.18.1", "6.19.0", "6.20.1", "6.21.1")),
    ("pyjwt", "2.15.0", ("2.13.0", "2.14.0")),
    ("urllib3", "2.8.0", ("2.6.3", "2.7.0")),
    ("certifi", "2026.10.19", ("2026.7.22", "2026.10.17")),
    ("aiohttp", "3.14.4", ("3.14.2", "3.14.3")),
    ("brotli", "1.2.1", ("1.1.0", "1.2.0")),
    ("filelock", "4.2.1", ("3.32.0", "4.2.0")),
    # Proxy credential-file access and salt-key reuse: GHSA-g5ff-637f-6q2m /
    # GHSA-7hp6-4w63-5g45.
    ("litellm", "1.100.4", ("1.93.0", "1.93.1", "1.93.2", "1.94.0", "1.100.3")),
    # Session-scope delimiter collision: GHSA-5324-3w58-qpp2.
    ("mem0ai", "2.0.18", ("2.0.14", "2.0.17")),
    ("mcp", "1.30.0", ("1.28.1", "1.29.0")),
    ("multidict", "6.9.1", ("6.7.1", "6.9.0")),
    ("nltk", "3.10.4", ("3.9.4", "3.10.3")),
    ("oauthlib", "4.0.0", ("3.3.1",)),
    ("opentelemetry-api", "1.45.1", ("1.44.0", "1.45.0")),
    ("pillow", "13.0.0", ("12.3.0",)),
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


def test_certifi_excludes_root_with_compromised_intermediate_keys():
    import ssl

    import certifi

    context = ssl.create_default_context(cafile=certifi.where())
    common_names = {
        value
        for cert in context.get_ca_certs()
        for rdn in cert["subject"]
        for key, value in rdn
        if key == "commonName"
    }
    assert common_names
    assert "emSign ECC Root CA - G3" not in common_names


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
