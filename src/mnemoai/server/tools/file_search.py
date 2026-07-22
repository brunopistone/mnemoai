"""Fast search tools using glob and ripgrep."""

import base64
import glob
import json
import os
import subprocess
from typing import Optional

from mcp.server.fastmcp import FastMCP

from mnemoai.utils.logger import logger
from mnemoai.utils.path_utils import normalize_path

# ripgrep's --max-columns is a no-op in --json mode (it emits the full line
# regardless), so we cap long lines here to stop minified/base64 lines from
# blowing up the result.
MAX_LINE_CHARS = 500

# Noise dirs glob_search skips by default (opt out with include_ignored=True).
# ripgrep already honors .gitignore/.git/hidden; stdlib glob does not, so glob
# needs its own set — kept on stdlib so glob still works with no rg installed.
_DEFAULT_IGNORED_DIRS = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}

# Hard cap on matches buffered when sorting by mtime, so "collect all before
# sorting" on an enormous tree can't exhaust memory.
_GLOB_SCAN_CEILING = 100000


def register_search_tools(mcp: FastMCP) -> None:
    """Register fast search tools.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    async def glob_search(
        pattern: str,
        path: str = None,
        max_results: int = 1000,
        sort_by_mtime: bool = True,
        include_ignored: bool = False,
    ) -> str:
        """Fast file pattern matching.

        Use this to find files by name patterns, NOT for content search.
        For large searches (entire home dir or system-wide), use execute_bash with 'find' instead.

        Args:
            pattern: Glob pattern (e.g., "**/*.py", "src/**/*.ts", "*.yaml")
            path: Directory to search (default: current directory)
            max_results: Maximum number of results to return (default: 1000, use 0 for unlimited)
            sort_by_mtime: Sort by modification time (default: True, disable for faster large searches)
            include_ignored: Include files inside common noise dirs (.git, node_modules, .venv, __pycache__, build, dist, ...); default False

        Returns:
            JSON string with matching file paths

        Examples:
            glob_search(pattern="**/*.py", max_results=100)  # First 100 Python files
            glob_search(pattern="*.yaml", path="/home/user/configs")  # YAML files in specific dir
            glob_search(pattern="test_*.py", sort_by_mtime=False)  # Faster, unsorted results

        Performance tip: For system-wide searches, use execute_bash with:
            find /path -name "*.py" -type f | wc -l  # Just count
            find /path -name "*.py" -type f -print | head -1000  # First 1000
        """
        if path is None:
            path = os.getcwd()

        # Expand ~ and tolerate shell escaping/quoting in the path.
        path = normalize_path(path)

        if not os.path.exists(path):
            return json.dumps(
                {"error": True, "message": f"Path does not exist: {path}"}
            )

        try:
            full_pattern = os.path.join(path, pattern)

            def _is_ignored(match: str) -> bool:
                # Match on path parts relative to the search root so an excluded
                # dir name in the base path itself doesn't filter everything.
                rel = os.path.relpath(match, path)
                return any(part in _DEFAULT_IGNORED_DIRS for part in rel.split(os.sep))

            def _mtime(p: str) -> float:
                # A file yielded by glob can vanish before we stat it; treat a
                # missing file as oldest rather than crashing the whole search.
                try:
                    return os.path.getmtime(p)
                except OSError:
                    return 0.0

            matches = []
            truncated = False
            scan_ceiling_hit = False

            if sort_by_mtime:
                # Collect ALL matches (up to a hard ceiling), THEN sort, THEN
                # slice — so a capped result is the truly-newest N, not the first
                # N glob happened to yield.
                for match in glob.iglob(full_pattern, recursive=True):
                    if not os.path.isfile(match):
                        continue
                    if not include_ignored and _is_ignored(match):
                        continue
                    matches.append(match)
                    if len(matches) >= _GLOB_SCAN_CEILING:
                        scan_ceiling_hit = True
                        logger.debug(
                            f"glob_search hit scan ceiling: {_GLOB_SCAN_CEILING}"
                        )
                        break
                matches.sort(key=_mtime, reverse=True)
                total = len(matches)
                if max_results > 0 and total > max_results:
                    matches = matches[:max_results]
                    truncated = True
            else:
                # Unsorted: stop lazily as soon as max_results is reached (fast).
                for match in glob.iglob(full_pattern, recursive=True):
                    if not os.path.isfile(match):
                        continue
                    if not include_ignored and _is_ignored(match):
                        continue
                    matches.append(match)
                    if max_results > 0 and len(matches) >= max_results:
                        truncated = True
                        break

            result = {
                "success": True,
                "matches": matches,
                "count": len(matches),
                "pattern": pattern,
                "search_path": path,
            }

            if truncated:
                result["truncated"] = True
                if sort_by_mtime and scan_ceiling_hit:
                    # Past the scan ceiling: `total` is only the buffered subset
                    # (not the true match count) and the newest-first ordering is
                    # over that subset, so don't claim either as global.
                    result["message"] = (
                        f"Showing {max_results} matches from the first "
                        f"{_GLOB_SCAN_CEILING} scanned (not globally newest — the "
                        "tree exceeds the scan ceiling). Use a more specific "
                        "pattern/path to narrow the search."
                    )
                elif sort_by_mtime:
                    result["message"] = (
                        f"Showing {max_results} of {total} matches (newest first). "
                        "Increase max_results or use max_results=0 for unlimited."
                    )
                else:
                    result["message"] = (
                        f"Results limited to {max_results} (unsorted). Use "
                        "sort_by_mtime=True for newest-first or max_results=0 for unlimited."
                    )
            if scan_ceiling_hit:
                result["scan_ceiling_hit"] = True

            return json.dumps(result)

        except Exception as e:
            logger.error(f"Error in glob_search: {str(e)}", exc_info=True)
            return json.dumps({"error": True, "message": str(e)})

    @mcp.tool()
    async def grep_search(
        pattern: str,
        path: str = None,
        file_pattern: str = None,
        case_insensitive: bool = False,
        output_mode: str = "files_with_matches",
        context_lines: int = 0,
        context_before: int = 0,
        context_after: int = 0,
        max_results: int = 100,
        offset: int = 0,
    ) -> str:
        """Fast content search using ripgrep.

        Use this to search FILE CONTENT, not filenames.
        This is 10-100x faster than traditional grep for large codebases.

        Args:
            pattern: Regex pattern to search for in file contents
            path: Directory to search (default: current directory)
            file_pattern: Filter files by glob (e.g., "*.py", "*.{ts,tsx}")
            case_insensitive: Case-insensitive search (default: False)
            output_mode: "files_with_matches" (default), "content", or "count"
            context_lines: Symmetric lines of context around each match, content mode (default: 0)
            context_before: Lines of context BEFORE each match (overrides context_lines; content mode)
            context_after: Lines of context AFTER each match (overrides context_lines; content mode)
            max_results: Max results to return, a TOTAL cap counting MATCHES only
                (context lines are extra); 0 = unlimited (default: 100)
            offset: Skip the first N results before applying max_results, for paging (default: 0)

        Returns:
            JSON string with search results

        Examples:
            grep_search(pattern="class Foo")  # Find class Foo definition
            grep_search(pattern="TODO|FIXME", file_pattern="*.py", case_insensitive=True)
            grep_search(pattern="import React", file_pattern="*.{ts,tsx}", output_mode="content")
        """
        if path is None:
            path = os.getcwd()

        # Expand ~ and tolerate shell escaping/quoting in the path.
        path = normalize_path(path)

        if not os.path.exists(path):
            return json.dumps(
                {"error": True, "message": f"Path does not exist: {path}"}
            )

        # Always run ripgrep in --json mode and derive every output mode from the
        # parsed match events. We deliberately do NOT pass --files-with-matches /
        # --count (they OVERRIDE --json and emit plain text this parser can't
        # read — which silently returned empty results) nor --max-count (a
        # PER-FILE cap, not the total the caller expects). The limit is applied
        # in Python after parsing, so max_results caps the true number of results
        # across all files in every mode. max_results <= 0 means unlimited.
        # --sort path forces a deterministic cross-file order (rg is parallel and
        # otherwise emits files in a nondeterministic order): the total cap and
        # offset paging are only meaningful over a STABLE order.
        cmd = ["rg", "--json", "--sort", "path"]

        if case_insensitive:
            cmd.append("-i")

        if file_pattern:
            cmd.append(f"--glob={file_pattern}")

        if output_mode == "content":
            # Asymmetric -A/-B (individual flags cover the symmetric case too);
            # context_before/after override the context_lines shorthand.
            before = context_before if context_before > 0 else context_lines
            after = context_after if context_after > 0 else context_lines
            if before > 0:
                cmd.append(f"-B{before}")
            if after > 0:
                cmd.append(f"-A{after}")

        # -e guards a pattern that begins with '-' from being read as a flag.
        cmd += ["-e", pattern, path]

        unlimited = max_results <= 0
        offset = max(0, offset)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # ripgrep gives an object its text as {"text": ...} for valid UTF-8,
            # or {"bytes": <base64>} for a line/path it couldn't decode (binary,
            # a stray byte). Prefer text; decode the bytes leniently otherwise so
            # a single non-UTF-8 match can't KeyError and abort the whole search.
            def _text(obj: dict) -> str:
                if not isinstance(obj, dict):
                    return ""
                if "text" in obj:
                    return obj["text"]
                b64 = obj.get("bytes")
                if b64:
                    try:
                        return base64.b64decode(b64).decode("utf-8", "replace")
                    except (ValueError, TypeError):
                        return ""
                return ""

            def _cap(text: str) -> str:
                # rg's --max-columns is a no-op under --json; cap here so a
                # minified/base64 line can't blow up the result.
                if len(text) > MAX_LINE_CHARS:
                    return text[:MAX_LINE_CHARS] + f"… [+{len(text) - MAX_LINE_CHARS} chars]"
                return text

            # Single pass over the events: an ordered row list (matches +, in
            # content mode, their -A/-B/-C context), the unique file order, and
            # per-file counts — enough to build any mode.
            all_matches = []
            file_order = []
            counts = {}
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Keep 'context' events too (rg emits them for -A/-B/-C) so the
                # model sees the surrounding lines in content mode; only real
                # matches count toward file_order/counts and the max_results cap.
                event = data.get("type")
                if event not in ("match", "context"):
                    continue
                ev_data = data["data"]
                file_path = _text(ev_data.get("path", {}))
                if not file_path:
                    continue  # can't identify the file — skip
                is_context = event == "context"
                all_matches.append(
                    {
                        "file": file_path,
                        "line": ev_data.get("line_number"),
                        "text": _cap(_text(ev_data.get("lines", {})).rstrip()),
                        "is_context": is_context,
                    }
                )
                if is_context:
                    continue
                if file_path not in counts:
                    counts[file_path] = 0
                    file_order.append(file_path)
                counts[file_path] += 1

            # ripgrep exits 0 (matches), 1 (no matches — a normal empty result),
            # or 2 (an error, e.g. an invalid regex OR a per-file permission
            # denial). Only treat exit >= 2 as FATAL when NO events were parsed:
            # a bad regex yields empty stdout + exit 2 (surface it), but an
            # unreadable file among readable ones still streams valid matches on
            # stdout with exit 2 (keep them, note the error non-fatally).
            if result.returncode >= 2 and not all_matches:
                msg = result.stderr.strip() or "ripgrep failed"
                return json.dumps(
                    {
                        "error": True,
                        "message": f"Search failed: {msg}",
                        "pattern": pattern,
                    }
                )
            partial_warning = (
                result.stderr.strip()
                if (result.returncode >= 2 and all_matches)
                else ""
            )

            if output_mode == "files_with_matches":
                files = (
                    file_order[offset:]
                    if unlimited
                    else file_order[offset : offset + max_results]
                )
                out = {
                    "success": True,
                    "files": files,
                    "count": len(files),
                    "pattern": pattern,
                    "search_path": path,
                }
                if not unlimited and len(file_order) > offset + max_results:
                    out["truncated"] = True
                    out["message"] = (
                        f"Showing {len(files)} of {len(file_order)} matching "
                        f"files (offset {offset}). Increase max_results/offset "
                        "or narrow the search."
                    )
                if partial_warning:
                    out["warnings"] = partial_warning
                return json.dumps(out)

            elif output_mode == "content":
                # Cap/offset count MATCHES only; context lines ride along with
                # THEIR match (window is [offset, offset+max_results)). Each
                # context row is owned by exactly ONE match — the nearest by line
                # number — so an off-window match's context can't leak in and the
                # same context line can't duplicate across pages (rg emits a
                # before-context row just before its match, an after-context row
                # just after; OR-ing both neighbours would double-attribute it).
                total_matches = sum(1 for r in all_matches if not r["is_context"])
                hi = None if unlimited else offset + max_results

                def _in_window(mi):
                    return mi is not None and mi >= offset and (hi is None or mi < hi)

                n = len(all_matches)
                # Ordinal of each match row (by appearance order).
                ordinal = {}
                c = -1
                for pos, r in enumerate(all_matches):
                    if not r["is_context"]:
                        c += 1
                        ordinal[pos] = c
                # Backward pass: for each position, the next match's ordinal+line.
                next_idx = [None] * n
                next_line = [None] * n
                ni = nl = None
                for pos in range(n - 1, -1, -1):
                    r = all_matches[pos]
                    if not r["is_context"]:
                        ni, nl = ordinal[pos], r["line"]
                    next_idx[pos], next_line[pos] = ni, nl
                # Forward pass: track the previous match; own each context row by
                # the nearer match (tie -> previous, both are correct).
                matches = []
                prev_idx = prev_line = None
                for pos, row in enumerate(all_matches):
                    if not row["is_context"]:
                        prev_idx, prev_line = ordinal[pos], row["line"]
                        if _in_window(prev_idx):
                            matches.append(row)
                        continue
                    rl = row["line"]
                    cand = []  # (line-distance, owning match ordinal)
                    if prev_idx is not None and prev_line is not None and rl is not None:
                        cand.append((abs(rl - prev_line), prev_idx))
                    if next_idx[pos] is not None and next_line[pos] is not None and rl is not None:
                        cand.append((abs(next_line[pos] - rl), next_idx[pos]))
                    if not cand:
                        continue  # no line info to attribute — drop the stray row
                    cand.sort(key=lambda t: t[0])
                    if _in_window(cand[0][1]):
                        matches.append(row)
                shown = sum(1 for r in matches if not r["is_context"])
                out = {
                    "success": True,
                    "matches": matches,
                    "count": shown,
                    "pattern": pattern,
                    "search_path": path,
                }
                if not unlimited and total_matches > offset + shown:
                    out["truncated"] = True
                    out["message"] = (
                        f"Showing {shown} of {total_matches} matches. "
                        "Increase max_results/offset or narrow the search."
                    )
                if partial_warning:
                    out["warnings"] = partial_warning
                return json.dumps(out)

            elif output_mode == "count":
                # One count line per file, so the total cap bounds the file set.
                capped = (
                    file_order[offset:]
                    if unlimited
                    else file_order[offset : offset + max_results]
                )
                capped_counts = {f: counts[f] for f in capped}
                out = {
                    "success": True,
                    "counts": capped_counts,
                    "total_matches": sum(capped_counts.values()),
                    "files_with_matches": len(capped_counts),
                    "pattern": pattern,
                    "search_path": path,
                }
                if not unlimited and len(file_order) > offset + max_results:
                    out["truncated"] = True
                    out["message"] = (
                        f"Showing counts for {len(capped)} of {len(file_order)} "
                        "matching files. Increase max_results/offset or narrow the search."
                    )
                if partial_warning:
                    out["warnings"] = partial_warning
                return json.dumps(out)

            else:
                return json.dumps(
                    {
                        "error": True,
                        "message": (
                            f"Invalid output_mode '{output_mode}'. Use "
                            "'files_with_matches', 'content', or 'count'."
                        ),
                    }
                )

        except subprocess.TimeoutExpired:
            return json.dumps(
                {
                    "error": True,
                    "message": "Search timed out after 30 seconds. Try narrowing your search with file_pattern.",
                }
            )
        except FileNotFoundError:
            return json.dumps(
                {
                    "error": True,
                    "message": "ripgrep (rg) not installed. Install with: brew install ripgrep (macOS) or apt install ripgrep (Linux)",
                }
            )
        except Exception as e:
            logger.error(f"Error in grep_search: {str(e)}", exc_info=True)
            return json.dumps({"error": True, "message": str(e)})
