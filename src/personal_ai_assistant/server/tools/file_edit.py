"""Precise file editing with exact string replacement."""

import json
import os
from mcp.server.fastmcp import FastMCP

from personal_ai_assistant.utils.logger import logger


def register_edit_tools(mcp: FastMCP) -> None:
    """Register file editing tools.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    async def file_edit(
        file_path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        """Perform exact string replacement in a file.

        PREFER THIS TOOL over fs_write for modifying existing files. Reserve
        fs_write for creating new files or wholesale rewrites; use file_edit for
        any targeted change to a file that already exists.

        WORKFLOW:
        1. Read the file first with fs_read
        2. Copy the exact text to replace, including whitespace and indentation
        3. Call file_edit(file_path, old_string, new_string)
        4. If you get a "not unique" error, include more surrounding context in
           old_string until it matches exactly one location (or set replace_all=True)

        CRITICAL REQUIREMENTS:
        1. You MUST read the file with fs_read BEFORE calling this tool
        2. The old_string must match EXACTLY as it appears in the file
        3. Include proper indentation (spaces/tabs) in old_string and new_string
        4. If the file uses tabs, use tabs. If spaces, use spaces.

        This tool provides safer editing than fs_write because:
        - It validates that old_string exists before modifying
        - It ensures old_string is unique (prevents accidental multiple replacements)
        - It provides clear error messages with guidance

        Args:
            file_path: Absolute path to the file to edit
            old_string: Exact text to replace (must exist and be unique unless replace_all=True)
            new_string: Replacement text (can be same length, longer, or shorter)
            replace_all: If True, replace ALL occurrences. If False (default), old_string must be unique.

        Returns:
            JSON string with success status and details

        Examples:
            # Replace a function definition
            file_edit(
                file_path="/path/to/file.py",
                old_string="def foo():\\n    return 1",
                new_string="def foo():\\n    return 2"
            )

            # Replace with more context for uniqueness
            file_edit(
                file_path="/path/to/config.yaml",
                old_string="MAX_TOKENS: 32768\\n  TEMPERATURE: 0.3",
                new_string="MAX_TOKENS: 8192\\n  TEMPERATURE: 0.3"
            )
        """
        # Expand user home directory
        file_path = os.path.expanduser(file_path)

        # Validate file exists
        if not os.path.exists(file_path):
            return json.dumps(
                {
                    "error": True,
                    "message": f"File not found: {file_path}",
                    "next_steps": [
                        "Check if the file path is correct",
                        "Verify the file exists with glob_search or execute_bash",
                        "Make sure you're using an absolute path, not a relative path",
                    ],
                }
            )

        # Validate it's a file (not directory)
        if not os.path.isfile(file_path):
            return json.dumps(
                {
                    "error": True,
                    "message": f"Path is not a file: {file_path}",
                    "next_steps": ["This path points to a directory, not a file"],
                }
            )

        # Read file content
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            return json.dumps(
                {
                    "error": True,
                    "message": f"File is not UTF-8 encoded: {file_path}",
                    "next_steps": [
                        "This file might be binary or use a different encoding",
                        "Try using execute_bash with 'file' command to check file type",
                    ],
                }
            )
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}", exc_info=True)
            return json.dumps(
                {"error": True, "message": f"Error reading file: {str(e)}"}
            )

        # Check if old_string exists
        if old_string not in content:
            # Provide helpful debugging info
            old_preview = old_string[:100] + ("..." if len(old_string) > 100 else "")
            return json.dumps(
                {
                    "error": True,
                    "message": "String to replace not found in file",
                    "old_string_preview": old_preview,
                    "next_steps": [
                        "Make sure you read the file with fs_read FIRST",
                        "Copy the EXACT text from the file (including spaces/tabs)",
                        "Check for invisible characters (tabs vs spaces)",
                        "Verify line endings (\\n vs \\r\\n)",
                        "Use a larger context snippet to ensure exact match",
                    ],
                }
            )

        # Count occurrences
        count = content.count(old_string)

        if count > 1 and not replace_all:
            # Show where the duplicates are (first few)
            lines = content.split("\n")
            occurrences = []
            current_pos = 0
            for i, line in enumerate(lines, 1):
                if old_string in line:
                    occurrences.append({"line": i, "preview": line.strip()[:80]})
                    if len(occurrences) >= 3:  # Show first 3
                        break

            return json.dumps(
                {
                    "error": True,
                    "message": f"Found {count} occurrences of old_string, but replace_all=False",
                    "occurrences_sample": occurrences,
                    "next_steps": [
                        f"Option 1: Set replace_all=True to replace all {count} occurrences",
                        "Option 2: Make old_string more specific by including more surrounding context",
                        "Example: Include the line before and after to make it unique",
                    ],
                }
            )

        # Perform replacement
        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
            new_content = content.replace(old_string, new_string, 1)

        # Validate the replacement actually changed something
        if new_content == content:
            return json.dumps(
                {
                    "error": True,
                    "message": "Replacement produced no changes (old_string and new_string are identical)",
                }
            )

        # Write back to file
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {str(e)}", exc_info=True)
            return json.dumps(
                {
                    "error": True,
                    "message": f"Error writing file: {str(e)}",
                    "next_steps": [
                        "Check if you have write permissions",
                        "Verify disk space is available",
                        "Make sure the file isn't locked by another process",
                    ],
                }
            )

        # Calculate change summary
        old_lines = len(old_string.split("\n"))
        new_lines = len(new_string.split("\n"))
        lines_delta = new_lines - old_lines

        return json.dumps(
            {
                "success": True,
                "file_path": file_path,
                "replacements": count if replace_all else 1,
                "old_lines": old_lines,
                "new_lines": new_lines,
                "lines_delta": lines_delta,
                "message": f"Successfully replaced {count if replace_all else 1} occurrence(s) in {file_path}",
            }
        )
