"""Search functionality for files."""

from .. import validate_file_path
import json
import os
import re
import sys

# Add parent directory to path to allow imports from root
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from utils.logger import logger


async def search_file(path: str, pattern: str, context_lines: int) -> str:
    """Search for pattern in file with context.

    Args:
        path: File path
        pattern: Search pattern
        context_lines: Number of context lines around matches

    Returns:
        JSON string with search results
    """
    if not pattern:
        return json.dumps({"error": True, "message": "Search pattern cannot be empty"})

    is_valid, normalized_path, error_dict = validate_file_path(path)
    if not is_valid:
        return json.dumps(error_dict)

    try:
        with open(normalized_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        matches = []
        pattern_re = re.compile(re.escape(pattern), re.IGNORECASE)

        for line_num, line in enumerate(lines, 1):
            if pattern_re.search(line):
                # Get context lines
                start_context = max(0, line_num - 1 - context_lines)
                end_context = min(len(lines), line_num + context_lines)

                context = []
                for i in range(start_context, end_context):
                    context.append(
                        {
                            "number": i + 1,
                            "content": lines[i].rstrip("\n\r"),
                            "is_match": i + 1 == line_num,
                        }
                    )

                matches.append(
                    {
                        "line_number": line_num,
                        "line_content": line.rstrip("\n\r"),
                        "context": context,
                    }
                )

        return json.dumps(
            {
                "path": normalized_path,
                "pattern": pattern,
                "matches": matches,
                "total_matches": len(matches),
                "context_lines": context_lines,
            }
        )

    except UnicodeDecodeError as e:
        logger.error(f"Error during search file: {str(e)}", exc_info=True)

        return json.dumps(
            {"error": True, "message": f"Cannot decode file as text: {path}"}
        )
