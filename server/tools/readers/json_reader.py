"""JSON file reading functionality - simple text-based approach."""

from .. import validate_file_path, count_tokens
import json
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from utils.config import config
from utils.logger import logger


async def read_json(path: str, start_line: int = 1, end_line: int = -1) -> str:
    """Read JSON/JSONL file as text with token limit enforcement.

    Args:
        path: Path to JSON or JSONL file
        start_line: Starting line number (1-indexed, default: 1)
        end_line: Ending line number (-1 for entire file, default: -1)

    Returns:
        JSON string with file data
    """
    is_valid, normalized_path, error_dict = validate_file_path(path)
    if not is_valid:
        return json.dumps(error_dict)

    # Detect JSONL format
    is_jsonl = normalized_path.endswith(".jsonl")

    try:
        if is_jsonl:
            # Efficient line-by-line reading for JSONL
            with open(normalized_path, "r", encoding="utf-8") as file:
                # First pass: count total lines
                total_lines = sum(1 for _ in file)

                # Handle negative indices
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

                # Second pass: read only requested lines
                file.seek(0)
                selected_lines = []
                for i, line in enumerate(file, 1):
                    if i < start_line:
                        continue
                    if i > end_line:
                        break
                    selected_lines.append(line)

                content = "".join(selected_lines).strip()
        else:
            # For regular JSON, must read entire file
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
            content = "".join(selected_lines).strip()

        if not content:
            return json.dumps(
                {
                    "path": normalized_path,
                    "type": "jsonl" if is_jsonl else "json",
                    "content": "",
                    "start_line": start_line,
                    "end_line": end_line,
                    "total_lines": total_lines,
                    "message": "File is empty or selected range is empty",
                }
            )

        # Validate JSON syntax
        try:
            if is_jsonl:
                # Validate each line separately for JSONL
                for line in content.split("\n"):
                    if line.strip():
                        json.loads(line)
            else:
                json.loads(content)
        except json.JSONDecodeError as e:
            return json.dumps(
                {"error": True, "message": f"Invalid JSON format: {str(e)}"}
            )

        # Check token count and truncate if needed
        content_tokens = count_tokens(content)
        max_tokens = config.get("DOC_MAX_TOKENS", 1024 * 8)
        was_truncated = False
        lines_included = len(selected_lines)

        if content_tokens > max_tokens:
            # Truncate content to fit token limit
            lines = content.split("\n")
            truncated_content = ""
            lines_included = 0

            for line in lines:
                test_content = truncated_content + line + "\n"
                if count_tokens(test_content) > max_tokens:
                    break
                truncated_content = test_content
                lines_included += 1

            content = (
                truncated_content.strip()
                + "\n... [TRUNCATED - Content exceeds token limit]"
            )
            was_truncated = True
            content_tokens = count_tokens(content)

        return json.dumps(
            {
                "path": normalized_path,
                "type": "jsonl" if is_jsonl else "json",
                "content": content,
                "start_line": start_line,
                "end_line": end_line,
                "total_lines": total_lines,
                "lines_requested": len(selected_lines),
                "lines_included": lines_included,
                "tokens": content_tokens,
                "max_tokens": max_tokens,
                "truncated": was_truncated,
                "message": f"""
                    Read {'JSONL' if is_jsonl else 'JSON'} file lines {start_line}-{end_line} ({content_tokens} tokens).
                    TRUNCATED at token limit {max_tokens}. File has {total_lines} total lines.
                    Inform the of the file's total size, structure, and how many lines were actually included. 
                    Then ask what specific information they're looking for. 
                    Do not attempt to read the entire file in chunks unless the user explicitly requests it. 
                    The correct response to truncation is to ask for guidance, not to try to fulfill the original request completely.
                """,
            }
        )

    except Exception as e:
        logger.error(f"Error during read json: {str(e)}", exc_info=True)
        return json.dumps({"error": True, "message": f"Error reading JSON: {str(e)}"})
