"""File system reading tool with multiple modes."""

import json

from mcp.server.fastmcp import FastMCP

from mnemoai.utils.logger import logger
from mnemoai.utils.path_utils import normalize_path

from ..error_handler import tool_error_handler
from .read_state import record_read
from .readers import (
    read_csv,
    read_directory,
    read_docx,
    read_json,
    read_lines,
    read_pdf,
    search_file,
)


def register_fs_read_tools(mcp: FastMCP) -> None:
    """Register file system reading tools.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    @tool_error_handler
    async def fs_read(
        path: str,
        mode: str = "Line",
        start_line: int = 1,
        end_line: int = -1,
        pattern: str = "",
        context_lines: int = 2,
        depth: int = 0,
    ) -> str:
        """Read and analyze file content in various formats.

        Use this tool when users ask to read, examine, analyze, view, or check file content.

        READ vs WRITE: Read-intent commands (read, show, view, check, display) use
        this tool only. Write-intent commands (create, modify, update, fix, change)
        belong to file_edit (existing files) or fs_write (new files). If the intent
        is unclear, ask for clarification rather than guessing.

        Important: For data files (JSON, JSONL, CSV): Before reading, check the file size with `ls -lh` and line count with `wc -l`. If the file has more than 1000 lines or is larger than 1MB, For large datasets, use execute_bash with "head -n 3 <path>" to show samples instead of reading the entire file.

        For large files (>1MB), check file size first using execute_bash with "ls -lh <path>".

        This tool efficiently reads and processes various file types including text files, code files, CSV data, JSON data, PDF documents, and DOCX documents.

        MODES (choose the appropriate one):
        - Line: Read text files, code files, markdown, etc. (DEFAULT - use this for most files)
        - Search: Find specific patterns or text within files
        - Directory: List contents of folders/directories
        - CSV: Parse and structure CSV/spreadsheet data (⚠️ CHECK SIZE FIRST!)
        - JSON/JSONL: Parse and structure JSON data files (⚠️ CHECK SIZE FIRST!)
        - PDF: Extract text content from PDF documents
        - DOCX: Extract text content from Word documents

        Args:
            path: Full path to the file or directory (REQUIRED)
            mode: Reading mode - "Line" for most files, "CSV" for spreadsheets, "JSON" for data files, "PDF" for PDFs, "DOCX" for Word docs
            start_line: Starting line number (1-indexed, default: 1)
            end_line: Ending line number (-1 for entire file, default: -1)
            pattern: Search text when using Search mode
            context_lines: Lines of context around search matches (default: 2)
            depth: Directory depth for Directory mode (default: 0)

        USAGE EXAMPLES:
        - Read entire Python file: fs_read(path="~/script.py", mode="Line")
        - Read PDF document: fs_read(path="~/document.pdf", mode="PDF")
        - Parse CSV data: fs_read(path="~/data.csv", mode="CSV")
        - Search in code: fs_read(path="~/file.py", mode="Search", pattern="function_name")
        - Parse JSON: fs_read(path="~/data.json", mode="JSON")
        - Parse JSONL: fs_read(path="~/data.jsonl", mode="JSONL")
        - Parse PDF: fs_read(path="~/data.pdf", mode="PDF")
        - Parse DOCX: fs_read(path="~/data.docx", mode="DOCX")

        Returns:
            Structured JSON with file content, metadata, and processing information.
            depth: Directory depth for Directory mode (0 = current level only)
        """
        logger.debug(f"fs_read called with path: {path}")

        try:
            # Normalize path, tolerating shell escaping/quoting (a path the user
            # copied from a terminal as `/Users/me/My\ File.txt` or quoted).
            normalized_path = normalize_path(path)

            if mode == "Directory":
                # A directory listing isn't a file read; nothing to gate later.
                return await read_directory(normalized_path, depth)
            elif mode == "Line":
                result = await read_lines(normalized_path, start_line, end_line)
            elif mode == "Search":
                result = await search_file(normalized_path, pattern, context_lines)
            elif mode == "CSV":
                result = await read_csv(normalized_path)
            elif mode in ["JSON", "JSONL"]:
                result = await read_json(normalized_path, start_line, end_line)
            elif mode == "PDF":
                result = await read_pdf(normalized_path)
            elif mode == "DOCX":
                result = await read_docx(normalized_path)
            else:
                return json.dumps(
                    {
                        "error": True,
                        "message": f"Invalid mode '{mode}'. Use 'Line', 'Search', 'Directory', 'CSV', or 'JSON'",
                    }
                )

            # Record read-state (current mtime) so a later fs_write/file_edit on
            # this file is allowed. Skip on a failed read so we never bless a
            # file the model didn't actually see.
            try:
                if not json.loads(result).get("error"):
                    record_read(normalized_path)
            except (ValueError, TypeError):
                pass
            return result

        except Exception as e:
            logger.error(f"Error in fs_read: {str(e)}", exc_info=True)

            return json.dumps(
                {"error": True, "message": f"Error reading {path}: {str(e)}"}
            )
