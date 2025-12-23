"""File system reading tool with multiple modes."""

from .readers import (
    read_directory,
    read_lines,
    search_file,
    read_csv,
    read_json,
    read_pdf,
    read_docx,
)
import json
import os
from mcp.server.fastmcp import FastMCP
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from utils.logger import logger


def register_fs_read_tools(mcp: FastMCP) -> None:
    """Register file system reading tools.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
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
            # Normalize path
            normalized_path = os.path.expanduser(path.strip())

            if mode == "Directory":
                return await read_directory(normalized_path, depth)
            elif mode == "Line":
                return await read_lines(normalized_path, start_line, end_line)
            elif mode == "Search":
                return await search_file(normalized_path, pattern, context_lines)
            elif mode == "CSV":
                return await read_csv(normalized_path)
            elif mode in ["JSON", "JSONL"]:
                return await read_json(normalized_path, start_line, end_line)
            elif mode == "PDF":
                return await read_pdf(normalized_path)
            elif mode == "DOCX":
                return await read_docx(normalized_path)
            else:
                return json.dumps(
                    {
                        "error": True,
                        "message": f"Invalid mode '{mode}'. Use 'Line', 'Search', 'Directory', 'CSV', or 'JSON'",
                    }
                )

        except Exception as e:
            logger.error(f"Error in fs_read: {str(e)}", exc_info=True)

            return json.dumps(
                {"error": True, "message": f"Error reading {path}: {str(e)}"}
            )
