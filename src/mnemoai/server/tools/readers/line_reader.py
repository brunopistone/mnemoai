"""Line-based file reading functionality."""

from .. import validate_file_path, count_tokens
import json

from mnemoai.utils.config import config
from mnemoai.utils.logger import logger


async def read_lines(path: str, start_line: int, end_line: int) -> str:
    """Read specific lines from a file.

    Args:
        path: File path
        start_line: Starting line number
        end_line: Ending line number

    Returns:
        JSON string with line data
    """
    is_valid, normalized_path, error_dict = validate_file_path(path)
    if not is_valid:
        return json.dumps(error_dict)

    try:
        with open(normalized_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        total_lines = len(lines)

        # Handle negative indices (from end of file)
        if start_line < 0:
            start_line = total_lines + start_line + 1
        if end_line < 0:
            end_line = total_lines + end_line + 1

        # Validate line numbers
        start_line = max(1, start_line)
        end_line = min(total_lines, end_line) if end_line > 0 else total_lines

        if start_line > end_line:
            return json.dumps(
                {
                    "error": True,
                    "message": f"Invalid line range: {start_line}-{end_line}. File has {total_lines} lines.",
                }
            )

        # Extract requested lines (convert to 0-indexed)
        selected_lines = lines[start_line - 1 : end_line]

        full_content = ""
        lines_processed = 0

        for i, line in enumerate(selected_lines):
            line_content = line.rstrip("\n\r") + "\n"

            # Check token limit before adding this line
            test_content = full_content + line_content
            if count_tokens(test_content) > config.get("DOC_MAX_TOKENS"):
                # Try to fit partial line if we have room
                remaining_tokens = config.get("DOC_MAX_TOKENS") - count_tokens(
                    full_content
                )
                if remaining_tokens > 50:  # Only if we have reasonable space left
                    words = line_content.split()
                    partial_line = ""
                    for word in words:
                        test_partial = full_content + partial_line + word + " "
                        if count_tokens(test_partial) > config.get("DOC_MAX_TOKENS"):
                            break
                        partial_line += word + " "

                    if partial_line.strip():
                        full_content += (
                            partial_line + "\n[TRUNCATED - Content exceeds token limit]"
                        )
                break

            full_content += line_content
            lines_processed = i + 1

        was_truncated = (
            lines_processed < len(selected_lines) or "[TRUNCATED" in full_content
        )
        final_tokens = count_tokens(full_content)

        return json.dumps(
            {
                "path": normalized_path,
                "content": full_content.strip(),
                "start_line": start_line,
                "end_line": end_line,
                "total_lines": total_lines,
                "lines_processed": lines_processed,
                "lines_requested": len(selected_lines),
                "tokens": final_tokens,
                "max_tokens": config.get("DOC_MAX_TOKENS"),
                "truncated": was_truncated,
                "message": f"Read {lines_processed}/{len(selected_lines)} lines ({final_tokens} tokens)"
                + (" - truncated due to token limit" if was_truncated else ""),
            }
        )

    except UnicodeDecodeError as e:
        logger.error(f"Error during read lines: {str(e)}", exc_info=True)

        return json.dumps(
            {"error": True, "message": f"Cannot decode file as text: {path}"}
        )
