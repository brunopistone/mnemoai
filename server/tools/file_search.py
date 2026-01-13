"""Fast search tools using glob and ripgrep."""

import glob
import json
import os
import subprocess
import sys
from mcp.server.fastmcp import FastMCP
from typing import Optional

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from utils.logger import logger


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
    ) -> str:
        """Fast file pattern matching.

        Use this to find files by name patterns, NOT for content search.
        For large searches (entire home dir or system-wide), use execute_bash with 'find' instead.

        Args:
            pattern: Glob pattern (e.g., "**/*.py", "src/**/*.ts", "*.yaml")
            path: Directory to search (default: current directory)
            max_results: Maximum number of results to return (default: 1000, use 0 for unlimited)
            sort_by_mtime: Sort by modification time (default: True, disable for faster large searches)

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

        # Expand user home directory
        path = os.path.expanduser(path)

        if not os.path.exists(path):
            return json.dumps(
                {"error": True, "message": f"Path does not exist: {path}"}
            )

        try:
            # Use glob with recursive support
            full_pattern = os.path.join(path, pattern)
            matches = []

            # Use iglob for lazy evaluation with max_results limit
            for match in glob.iglob(full_pattern, recursive=True):
                if os.path.isfile(match):
                    matches.append(match)
                    # Early termination if max_results reached (0 = unlimited)
                    if max_results > 0 and len(matches) >= max_results:
                        logger.debug(
                            f"glob_search reached max_results limit: {max_results}"
                        )
                        break

            # Sort by modification time only if requested and result set is reasonable
            if sort_by_mtime and len(matches) > 0:
                matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)

            result = {
                "success": True,
                "matches": matches,
                "count": len(matches),
                "pattern": pattern,
                "search_path": path,
            }

            # Add truncation warning if we hit the limit
            if max_results > 0 and len(matches) >= max_results:
                result["truncated"] = True
                result["message"] = (
                    f"Results limited to {max_results}. Use max_results=0 for unlimited or increase the limit."
                )

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
        max_results: int = 100,
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
            context_lines: Lines of context around matches (default: 0)
            max_results: Maximum number of results to return (default: 100)

        Returns:
            JSON string with search results

        Examples:
            grep_search(pattern="class Foo")  # Find class Foo definition
            grep_search(pattern="TODO|FIXME", file_pattern="*.py", case_insensitive=True)
            grep_search(pattern="import React", file_pattern="*.{ts,tsx}", output_mode="content")
        """
        if path is None:
            path = os.getcwd()

        # Expand user home directory
        path = os.path.expanduser(path)

        if not os.path.exists(path):
            return json.dumps(
                {"error": True, "message": f"Path does not exist: {path}"}
            )

        # Build ripgrep command
        cmd = ["rg", "--json", pattern]

        if case_insensitive:
            cmd.append("-i")

        if file_pattern:
            cmd.append(f"--glob={file_pattern}")

        if output_mode == "files_with_matches":
            cmd.append("--files-with-matches")
        elif output_mode == "count":
            cmd.append("--count")

        if context_lines > 0:
            cmd.append(f"-C{context_lines}")

        # Add max count to limit results
        cmd.append(f"--max-count={max_results}")

        # Add path at the end
        cmd.append(path)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if output_mode == "files_with_matches":
                # Parse file paths from JSON output
                files = []
                for line in result.stdout.strip().split("\n"):
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get("type") == "match":
                                file_path = data["data"]["path"]["text"]
                                if file_path not in files:
                                    files.append(file_path)
                        except json.JSONDecodeError:
                            continue

                return json.dumps(
                    {
                        "success": True,
                        "files": files,
                        "count": len(files),
                        "pattern": pattern,
                        "search_path": path,
                    }
                )

            elif output_mode == "content":
                # Parse matches with content
                matches = []
                current_file = None
                for line in result.stdout.strip().split("\n"):
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get("type") == "match":
                                match_data = data["data"]
                                file_path = match_data["path"]["text"]
                                line_number = match_data["line_number"]
                                match_text = match_data["lines"]["text"]

                                matches.append(
                                    {
                                        "file": file_path,
                                        "line": line_number,
                                        "text": match_text.rstrip(),
                                    }
                                )

                                # Limit results
                                if len(matches) >= max_results:
                                    break
                        except json.JSONDecodeError:
                            continue

                return json.dumps(
                    {
                        "success": True,
                        "matches": matches,
                        "count": len(matches),
                        "pattern": pattern,
                        "search_path": path,
                    }
                )

            elif output_mode == "count":
                # Parse count data
                counts = {}
                for line in result.stdout.strip().split("\n"):
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get("type") == "match":
                                file_path = data["data"]["path"]["text"]
                                if file_path in counts:
                                    counts[file_path] += 1
                                else:
                                    counts[file_path] = 1
                        except json.JSONDecodeError:
                            continue

                return json.dumps(
                    {
                        "success": True,
                        "counts": counts,
                        "total_matches": sum(counts.values()),
                        "files_with_matches": len(counts),
                        "pattern": pattern,
                        "search_path": path,
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
