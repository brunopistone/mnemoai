"""File system writing tool with intelligent path resolution."""

import json
from mcp.server.fastmcp import FastMCP
import os
from mnemoai.utils.config import config

from mnemoai.utils.logger import logger
from ..error_handler import tool_error_handler


def _resolve_path(path: str) -> str:
    """Resolve local path.

    Args:
        path: Input path to resolve

    Returns:
        Resolved absolute path
    """
    # Get paths from config
    default_output = os.path.expanduser("~")

    # Already absolute path
    if path.startswith("/"):
        return path

    # Home directory expansion
    if path.startswith("~"):
        return os.path.expanduser(path)

    # Check if it's a project file (code, config, docs)
    project_extensions = {
        ".py",
        ".yaml",
        ".yml",
        ".json",
        ".md",
        ".txt",
        ".sh",
        ".toml",
    }
    file_ext = os.path.splitext(path)[1].lower()

    # If it's a project-related file, put it in project root or appropriate subdir
    if file_ext in project_extensions:
        if file_ext == ".py":
            # Python files go in project root or appropriate subdirectory
            return os.path.join(default_output, path)
        elif file_ext in {".yaml", ".yml", ".json", ".toml"}:
            # Config files go in project root
            return os.path.join(default_output, path)
        elif file_ext == ".md":
            # Documentation goes in project root
            return os.path.join(default_output, path)
        elif file_ext == ".sh":
            # Scripts go in project root
            return os.path.join(default_output, path)

    # For data files, reports, outputs -> use output directory
    # Create output directory if it doesn't exist
    os.makedirs(default_output, exist_ok=True)

    # If path contains subdirectories, preserve them
    if "/" in path:
        return os.path.join(default_output, path)
    else:
        return os.path.join(default_output, path)


def register_fs_write_tools(mcp: FastMCP) -> None:
    """Register file system writing tools.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    @tool_error_handler
    async def fs_write(
        path: str,
        command: str,
        file_text: str = "",
        old_str: str = "",
        new_str: str = "",
        insert_line: int = 0,
        summary: str = "",
        dry_run: bool = True,
        confirmed: bool = False,
    ) -> str:
        """Write or modify files on the filesystem.

        IMPORTANT: This tool requires a two-step process with user confirmation:
        1. Call with dry_run=True to preview changes (default)
        2. Ask user for explicit confirmation
        3. Call with dry_run=False AND confirmed=True to execute

        User confirmation is MANDATORY - execution will be blocked without it.

        This tool is for file operations only. Use it when the user explicitly asks to create, modify, or save a file with a specific path.

        Do not use this tool when:
        - User asks for code examples without mentioning a file path
        - User says "show me code", "write a function", "give me an example" (return code in markdown instead)
        - User wants to read or view a file (use fs_read instead)

        Use this tool when:
        - User specifies a file path: "create ~/script.py", "save to config.yaml"
        - User says "save this to a file", "create a file with this code"
        - User wants to modify an existing file: "update file.py", "change the config"

        For targeted edits to an EXISTING file, prefer file_edit (exact string
        replacement) over the str_replace command here — it validates the match
        and is safer. Use fs_write mainly for creating new files or full rewrites.

        This tool handles all file creation and modification operations with proper formatting and error handling.

        COMMANDS (choose the appropriate one):
        - create: Create a new file with content (overwrites if exists)
        - str_replace: Replace specific text in an existing file
        - insert: Insert new content after a specific line number
        - append: Add content to the end of an existing file

        Args:
            path: Full path where to create/modify the file (REQUIRED)
            command: Operation type - "create", "str_replace", "insert", or "append" (REQUIRED)
            file_text: Complete file content for "create" command
            old_str: Exact text to replace (for "str_replace" command)
            new_str: New text to insert/replace with
            insert_line: Line number to insert after (for "insert" command)
            summary: Brief description of what the change does
            dry_run: If True, preview the operation without executing (default: True)
            confirmed: If True, user has confirmed the operation (default: False, REQUIRED for execution)

        USAGE EXAMPLES:
        - Step 1 Preview: fs_write(path="~/script.py", command="create", file_text="...", dry_run=True)
        - Step 2 Ask user: "Should I proceed with creating this file?"
        - Step 3 Execute: fs_write(path="~/script.py", command="create", file_text="...", dry_run=False, confirmed=True)

        IMPORTANT: When users ask you to rewrite, reorganize, or create files, you MUST use this tool to actually perform the file operations. Do not just describe what you would do. You must not write files under "/"

        Returns:
            Preview of changes (dry_run=True) or success confirmation (dry_run=False with confirmed=True).
        """
        logger.debug(
            "Tool fs_write called with command: %s on path: %s (dry_run=%s, confirmed=%s)",
            command,
            path,
            dry_run,
            confirmed,
        )

        try:
            # Resolve path using intelligent logic
            resolved_path = _resolve_path(path)

            # CRITICAL: Block execution without user confirmation
            if not dry_run and not confirmed:
                return json.dumps(
                    {
                        "error": True,
                        "requires_confirmation": True,
                        "message": "User confirmation required before executing file operations. "
                        "You must ask the user for permission and then call with confirmed=True.",
                        "path": resolved_path,
                        "command": command,
                        "next_step": "Ask user: 'Should I proceed with this file operation?' "
                        "If user approves, call again with dry_run=False and confirmed=True",
                    }
                )

            # Preview mode - show what would happen without executing
            if dry_run:
                preview = {
                    "preview": True,
                    "path": resolved_path,
                    "command": command,
                    "summary": summary,
                    "file_exists": os.path.exists(resolved_path),
                }
                if command == "create":
                    preview["content_lines"] = len(file_text.splitlines())
                    preview["content_preview"] = file_text[:500] + (
                        "..." if len(file_text) > 500 else ""
                    )
                elif command == "str_replace":
                    preview["old_str_preview"] = old_str[:200] + (
                        "..." if len(old_str) > 200 else ""
                    )
                    preview["new_str_preview"] = new_str[:200] + (
                        "..." if len(new_str) > 200 else ""
                    )
                elif command == "insert":
                    preview["insert_line"] = insert_line
                    preview["content_preview"] = new_str[:200] + (
                        "..." if len(new_str) > 200 else ""
                    )
                elif command == "append":
                    preview["content_preview"] = new_str[:200] + (
                        "..." if len(new_str) > 200 else ""
                    )
                preview["message"] = (
                    "Preview only. Ask user to confirm, then call with dry_run=False and confirmed=True to execute."
                )
                return json.dumps(preview, indent=2)

            if command == "create":
                return await _create_file(resolved_path, file_text, summary)
            elif command == "str_replace":
                return await _str_replace(resolved_path, old_str, new_str, summary)
            elif command == "insert":
                return await _insert_line(resolved_path, insert_line, new_str, summary)
            elif command == "append":
                return await _append_file(resolved_path, new_str, summary)
            else:
                return json.dumps(
                    {
                        "error": True,
                        "message": f"Invalid command '{command}'. Use: create, str_replace, insert, append",
                    }
                )

        except Exception as e:
            logger.error(f"Error in fs_write: {str(e)}", exc_info=True)
            return json.dumps(
                {"error": True, "message": f"Error writing {path}: {str(e)}"}
            )


async def _create_file(path: str, content: str, summary: str) -> str:
    """Create a new file with content.

    Args:
        path: File path
        content: File content
        summary: Summary of operation

    Returns:
        JSON string with result
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            file.write(content)

        return json.dumps(
            {
                "success": True,
                "operation": "create",
                "path": path,
                "message": f"File created successfully",
                "summary": summary,
            }
        )

    except Exception as e:
        logger.error(f"Error during create file: {str(e)}", exc_info=True)

        return json.dumps(
            {"error": True, "message": f"Failed to create file: {str(e)}"}
        )


async def _str_replace(path: str, old_str: str, new_str: str, summary: str) -> str:
    """Replace specific text in a file.

    Args:
        path: File path
        old_str: Text to replace
        new_str: Replacement text
        summary: Summary of operation

    Returns:
        JSON string with result
    """
    if not os.path.exists(path):
        return json.dumps({"error": True, "message": f"File does not exist: {path}"})

    try:
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()

        # Check if old_str exists
        if old_str not in content:
            return json.dumps(
                {"error": True, "message": f"Text to replace not found in file"}
            )

        # Count occurrences
        occurrences = content.count(old_str)
        if occurrences > 1:
            return json.dumps(
                {
                    "error": True,
                    "message": f"{occurrences} occurrences of old_str were found when only 1 is expected",
                }
            )

        # Perform replacement
        new_content = content.replace(old_str, new_str)

        with open(path, "w", encoding="utf-8") as file:
            file.write(new_content)

        return json.dumps(
            {
                "success": True,
                "operation": "str_replace",
                "path": path,
                "message": "Text replaced successfully",
                "summary": summary,
            }
        )

    except Exception as e:
        logger.error(f"Error during str replace: {str(e)}", exc_info=True)

        return json.dumps(
            {"error": True, "message": f"Failed to replace text: {str(e)}"}
        )


async def _insert_line(path: str, line_number: int, content: str, summary: str) -> str:
    """Insert text after a specific line number.

    Args:
        path: File path
        line_number: Line number to insert after
        content: Content to insert
        summary: Summary of operation

    Returns:
        JSON string with result
    """
    if not os.path.exists(path):
        return json.dumps({"error": True, "message": f"File does not exist: {path}"})

    try:
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Validate line number
        if line_number < 0 or line_number > len(lines):
            return json.dumps(
                {
                    "error": True,
                    "message": f"Invalid line number {line_number}. File has {len(lines)} lines.",
                }
            )

        # Insert content after specified line
        if not content.endswith("\n"):
            content += "\n"

        lines.insert(line_number, content)

        with open(path, "w", encoding="utf-8") as file:
            file.writelines(lines)

        return json.dumps(
            {
                "success": True,
                "operation": "insert",
                "path": path,
                "message": f"Content inserted after line {line_number}",
                "summary": summary,
            }
        )

    except Exception as e:
        logger.error(f"Error during insert line: {str(e)}", exc_info=True)
        return json.dumps(
            {"error": True, "message": f"Failed to insert content: {str(e)}"}
        )


async def _append_file(path: str, content: str, summary: str) -> str:
    """Append content to end of existing file.

    Args:
        path: File path
        content: Content to append
        summary: Summary of operation

    Returns:
        JSON string with result
    """
    if not os.path.exists(path):
        return json.dumps({"error": True, "message": f"File does not exist: {path}"})

    try:
        # Check if file ends with newline
        with open(path, "r", encoding="utf-8") as file:
            existing_content = file.read()

        # Add newline if file doesn't end with one
        if existing_content and not existing_content.endswith("\n"):
            content = "\n" + content

        with open(path, "a", encoding="utf-8") as file:
            file.write(content)

        return json.dumps(
            {
                "success": True,
                "operation": "append",
                "path": path,
                "message": "Content appended successfully",
                "summary": summary,
            }
        )

    except Exception as e:
        logger.error(f"Error during append file: {str(e)}", exc_info=True)

        return json.dumps(
            {"error": True, "message": f"Failed to append content: {str(e)}"}
        )
