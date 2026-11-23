"""Documentation links in the root README must resolve to real pages.

``mkdocs build --strict`` only validates files inside ``docs/``, and neither
README.md nor CLAUDE.md is in the nav -- so the four broken doc links that
shipped in the README were invisible to CI. This test closes that gap by
resolving every published docs URL in the README back to a file on disk.

The mapping mirrors how mkdocs serves pages with ``use_directory_urls`` (the
default): ``docs/guides/usage.md`` is served at ``/guides/usage/``, and an
``index.md`` is served at its directory root.
"""

import re
from pathlib import Path
from urllib.parse import urlsplit

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
SITE_PREFIX = "https://brunopistone.github.io/mnemoai/"


def _published_paths() -> set[str]:
    """Every URL path mkdocs will serve, derived from the files in docs/."""
    paths = set()
    for md in DOCS_DIR.rglob("*.md"):
        rel = md.relative_to(DOCS_DIR).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "index":
            parts = parts[:-1]
        paths.add("/".join(parts))
    return paths


def _readme_doc_links() -> list[str]:
    """The site-relative paths of every docs link in the README.

    ``*`` is excluded from the path charset so the markdown bold around a bare
    URL (``**https://…/**``) isn't captured as part of the path.
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(rf"{re.escape(SITE_PREFIX)}([^)\s#*]*)", readme)
    return [link.strip("/") for link in links]


def test_readme_has_doc_links():
    """Guard the guard: if the README stops linking docs, this test is moot."""
    assert _readme_doc_links(), "no docs links found in README - has the URL changed?"


def test_every_readme_doc_link_resolves():
    published = _published_paths()
    broken = [link for link in _readme_doc_links() if link and link not in published]
    assert not broken, (
        f"README links to non-existent doc pages: {broken}. "
        f"Available: {sorted(published)}"
    )


def test_site_root_link_is_valid():
    """The bare site URL must have a home page behind it."""
    assert (DOCS_DIR / "index.md").exists()


def test_mkdocs_nav_entries_exist():
    """Every nav entry in mkdocs.yml points at a file that exists.

    mkdocs --strict already catches this on the docs job, but that job only runs
    on pushes to main that touch docs/ -- so a nav typo on a feature branch is
    otherwise unnoticed until release.
    """
    config = yaml.safe_load((REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8"))

    def entries(node):
        if isinstance(node, str):
            yield node
            return
        assert isinstance(node, (dict, list)), f"Unexpected nav entry: {node!r}"
        for child in node.values() if isinstance(node, dict) else node:
            yield from entries(child)

    referenced = list(entries(config["nav"]))
    assert referenced, "mkdocs.yml has no navigation entries"
    urls = (urlsplit(ref) for ref in referenced)
    local_paths = [url.path for url in urls if not url.scheme and not url.netloc]
    missing = [ref for ref in local_paths if not (DOCS_DIR / ref).is_file()]
    assert not missing, f"mkdocs.yml nav references missing files: {missing}"
