"""File system writing tool with intelligent path resolution."""

import io
import json
import os

from mcp.server.fastmcp import FastMCP

from mnemoai.utils.logger import logger
from mnemoai.utils.path_utils import clean_path_syntax

from ..error_handler import tool_error_handler
from .file_encoding import decode_to_lf, encode_from_lf
from .read_state import check_write_allowed, record_read
from .safety import classify_write_path


def _resolve_path(path: str) -> str:
    """Resolve a user-supplied path to an absolute one.

    A relative path resolves against the current working directory (like any
    normal tool), NOT the home directory — the model asked to write "here".
    Only an absolute path or an explicit ``~`` goes elsewhere. (An earlier
    version silently relocated relative paths into ``~`` by file extension, so
    ``fs_write("notes.txt", ...)`` clobbered ``~/notes.txt`` instead of writing
    in the working dir — a surprise-overwrite footgun.)

    Args:
        path: Input path to resolve

    Returns:
        Resolved absolute path
    """
    # Strip any shell quoting/escaping the user copied in (e.g. a path dragged
    # from a terminal as `/Users/me/My\ File.txt`) before resolving. The target
    # may not exist yet, so this is syntactic only (no filesystem probe).
    path = clean_path_syntax(path)

    # Already absolute.
    if os.path.isabs(path):
        return path

    # Explicit home-directory expansion.
    if path.startswith("~"):
        return os.path.expanduser(path)

    # Relative → resolve against the working directory.
    return os.path.abspath(os.path.join(os.getcwd(), path))


def register_fs_write_tools(mcp: FastMCP) -> None:
    """Register file system writing tools.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    @tool_error_handler
    def fs_write(
        path: str,
        command: str,
        file_text: str = "",
        old_str: str = "",
        new_str: str = "",
        insert_line: int = 0,
        summary: str = "",
    ) -> str:
        """Write or modify files on the filesystem.

        This tool is for file operations only. Use it when the user explicitly asks to create, modify, or save a file with a specific path. The client asks the user to confirm before the write runs, so just call it directly.

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

        IMPORTANT: When users ask you to rewrite, reorganize, or create files, you MUST use this tool to actually perform the file operations. Do not just describe what you would do. You must not write files under "/"

        Returns:
            Success confirmation, or an error object.
        """
        logger.debug(
            "Tool fs_write called with command: %s on path: %s", command, path
        )

        try:
            # Resolve path using intelligent logic
            resolved_path = _resolve_path(path)

            # Server-side hard floor: never write into a critical system
            # directory. Enforced here (not just in the docstring) because the MCP
            # server can be driven directly.
            path_verdict = classify_write_path(resolved_path)
            if path_verdict.blocked:
                logger.warning("fs_write blocked path: %s", resolved_path)
                return json.dumps(
                    {
                        "error": True,
                        "blocked": True,
                        "message": path_verdict.reason,
                        "path": resolved_path,
                    }
                )

            # Read-before-write gate (server-side, never prompts): refuse to
            # overwrite/modify an EXISTING file the model never read, or that
            # changed on disk since the last read. A brand-new file is allowed.
            read_verdict = check_write_allowed(resolved_path)
            if read_verdict is not None:
                return json.dumps(read_verdict)

            if command == "create":
                result = _create_file(resolved_path, file_text, summary)
            elif command == "str_replace":
                result = _str_replace(resolved_path, old_str, new_str, summary)
            elif command == "insert":
                result = _insert_line(resolved_path, insert_line, new_str, summary)
            elif command == "append":
                result = _append_file(resolved_path, new_str, summary)
            else:
                return json.dumps(
                    {
                        "error": True,
                        "message": f"Invalid command '{command}'. Use: create, str_replace, insert, append",
                    }
                )

            # Our own successful write becomes the new read baseline so a
            # follow-up edit in the same turn isn't falsely flagged stale.
            try:
                if not json.loads(result).get("error"):
                    record_read(resolved_path)
            except (ValueError, TypeError):
                pass
            return result

        except Exception as e:
            logger.error(f"Error in fs_write: {str(e)}", exc_info=True)
            return json.dumps(
                {"error": True, "message": f"Error writing {path}: {str(e)}"}
            )


def _create_file(path: str, content: str, summary: str) -> str:
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


def _str_replace(path: str, old_str: str, new_str: str, summary: str) -> str:
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
        # Decode to LF-normalized text, capturing the file's shape so the write
        # below preserves its encoding/BOM/line-ending (in-place edit).
        with open(path, "rb") as file:
            content, shape = decode_to_lf(file.read())

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

        with open(path, "wb") as file:
            file.write(encode_from_lf(new_content, shape))

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


def _insert_line(path: str, line_number: int, content: str, summary: str) -> str:
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
        # Decode to LF text + capture shape; split via StringIO.readlines() so
        # the line count/index matches the old readlines() EXACTLY (str.splitlines
        # also breaks on \v, \f, \x1c-\x1e and Unicode separators — readlines
        # does not, so a file containing those would get a wrong insert index).
        with open(path, "rb") as file:
            text, shape = decode_to_lf(file.read())
        lines = io.StringIO(text).readlines()

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

        with open(path, "wb") as file:
            file.write(encode_from_lf("".join(lines), shape))

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


def _append_file(path: str, content: str, summary: str) -> str:
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
        # Decode to LF text + capture shape; rewrite the whole file (instead of
        # mode "a") so the appended content re-emits the original BOM/encoding
        # and line ending (mode "a" would append UTF-8 LF into a UTF-16/CRLF file).
        with open(path, "rb") as file:
            existing_content, shape = decode_to_lf(file.read())

        # Add newline if file doesn't end with one
        if existing_content and not existing_content.endswith("\n"):
            content = "\n" + content

        with open(path, "wb") as file:
            file.write(encode_from_lf(existing_content + content, shape))

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
