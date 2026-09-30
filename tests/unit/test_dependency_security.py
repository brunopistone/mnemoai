"""Security floors must protect both locked checkouts and PyPI installations."""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
# PyJWT 2.15 also fixes CVE-2026-101918, beyond the 2.14 algorithm-confusion fixes.
# urllib3 2.8 fixes CVE-2026-97687/97688/97689.
SECURITY_FLOORS = [
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
