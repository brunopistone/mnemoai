"""Line-based file reading functionality."""

import json

from mnemoai.utils.config import config
from mnemoai.utils.logger import logger

from .. import binary_file_error, count_tokens, looks_like_binary, validate_file_path
from ..file_encoding import bom_encoding


def read_lines(path: str, start_line: int, end_line: int) -> str:
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

    # Fail fast on binary/image files with a message that steers the model to
    # the right tool (describe_image for images), instead of choking on the
    # UTF-8 decode and dumping a stack trace.
    if looks_like_binary(normalized_path):
        return json.dumps(binary_file_error(normalized_path))

    try:
        # A BOM'd UTF-16/UTF-32 file is real text (not binary): decode with the
        # BOM's codec so it's readable — and therefore editable (the write gate
        # can only be satisfied once fs_read succeeds). Default UTF-8 otherwise.
        with open(normalized_path, "rb") as raw_file:
            encoding = bom_encoding(raw_file.read(4)) or "utf-8"

        # Stream the file so a huge (multi-hundred-MB) read never loads the whole
        # thing into memory. Pass 1 counts lines — needed for total_lines and for
        # negative-index (from-EOF) resolution; a binary file that slipped past
        # looks_like_binary raises UnicodeDecodeError here and is steered below.
        with open(normalized_path, "r", encoding=encoding) as file:
            total_lines = sum(1 for _ in file)

        if total_lines == 0:
            # An empty file is a successful read of NOTHING, not an invalid range:
            # no request can be out of bounds when the file has no bounds. As an
            # error it also made the file unwritable — the write gate only clears a
            # file whose read succeeded, so an empty one could be neither read nor
            # filled (read_state exempts it now; this is the other half).
            return json.dumps(
                {
                    "path": normalized_path,
                    "content": "",
                    "start_line": 0,
                    "end_line": 0,
                    "total_lines": 0,
                    "lines_processed": 0,
                    "lines_requested": 0,
                    "tokens": 0,
                    "max_tokens": config.get("DOC_MAX_TOKENS"),
                    "truncated": False,
                    "message": "File is empty (0 lines)",
                }
            )

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

        # Pass 2: materialize ONLY the requested [start_line, end_line] slice,
        # streamed — stop once past end_line so trailing lines are never buffered.
        # Byte-identical to lines[start_line - 1 : end_line] from readlines().
        selected_lines = []
        with open(normalized_path, "r", encoding=encoding) as file:
            for line_no, line in enumerate(file, start=1):
                if line_no < start_line:
                    continue
                if line_no > end_line:
                    break
                selected_lines.append(line)

        # Token accounting is INCREMENTAL: a running total plus the cost of the
        # line being added. Re-counting `full_content + line` on every line made
        # the read quadratic in tokenizer work — measured 9.8s for a 2839-line
        # file against 0.007s for a single count of the same text — and with
        # DOC_MAX_TOKENS sized for a large-context model the loop runs to EOF, so
        # a big file cost minutes of CPU. Per-line counts sum slightly HIGH (the
        # merges a tokenizer would make across a line boundary are lost), which
        # is the safe direction for a limit.
        max_tokens = config.get("DOC_MAX_TOKENS")
        pieces = []
        used_tokens = 0
        lines_processed = 0

        for i, line in enumerate(selected_lines):
            # cat -n style gutter: right-aligned file line number + TAB, so the
            # model can cite lines precisely. file_edit strips a matching gutter
            # from old_string before its exact match, so a pasted numbered block
            # still edits cleanly (provider-agnostic).
            line_no = start_line + i
            line_content = f"{line_no:>6}\t" + line.rstrip("\n\r") + "\n"

            # Check token limit before adding this line
            line_tokens = count_tokens(line_content)
            if used_tokens + line_tokens > max_tokens:
                # Try to fit partial line if we have room
                remaining_tokens = max_tokens - used_tokens
                if remaining_tokens > 50:  # Only if we have reasonable space left
                    # Split the RAW line (not the guttered line_content) so the
                    # line number isn't pulled in as a leading word and the TAB
                    # gutter is preserved on the truncated fragment.
                    words = line.rstrip("\n\r").split()
                    gutter = f"{line_no:>6}\t"
                    partial_words = []
                    partial_tokens = count_tokens(gutter)
                    for word in words:
                        word_tokens = count_tokens(word + " ")
                        if used_tokens + partial_tokens + word_tokens > max_tokens:
                            break
                        partial_words.append(word)
                        partial_tokens += word_tokens

                    if partial_words:
                        pieces.append(
                            gutter
                            + "".join(f"{word} " for word in partial_words)
                            + "\n[TRUNCATED - Content exceeds token limit]"
                        )
                break

            pieces.append(line_content)
            used_tokens += line_tokens
            lines_processed = i + 1

        full_content = "".join(pieces)
        was_truncated = (
            lines_processed < len(selected_lines) or "[TRUNCATED" in full_content
        )
        # One exact count of the final content — the running total above is only
        # the budget guard, and summed per line it reads a little high.
        final_tokens = count_tokens(full_content)

        return json.dumps(
            {
                "path": normalized_path,
                "content": full_content.rstrip("\n"),
                "start_line": start_line,
                "end_line": end_line,
                "total_lines": total_lines,
                "lines_processed": lines_processed,
                "lines_requested": len(selected_lines),
                "tokens": final_tokens,
                "max_tokens": max_tokens,
                "truncated": was_truncated,
                "message": f"Read {lines_processed}/{len(selected_lines)} lines ({final_tokens} tokens)"
                + (" - truncated due to token limit" if was_truncated else ""),
            }
        )

    except UnicodeDecodeError:
        # Expected for a binary/image file the up-front check didn't catch —
        # not an internal error, so log calmly (debug) and steer the model.
        logger.debug(f"read_lines: {path} is not valid UTF-8 text; treating as binary")
        return json.dumps(binary_file_error(normalized_path))
