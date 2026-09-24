"""CSV file reading functionality."""

import csv
import json

from mnemoai.utils.config import config
from mnemoai.utils.logger import logger

from .. import count_tokens, validate_file_path


def read_csv(path: str, *, _encoding: str = "utf-8-sig") -> str:
    """Read and parse CSV file with token-based truncation.

    Args:
        path: Path to CSV file

    Returns:
        JSON string with CSV data
    """
    is_valid, normalized_path, error_dict = validate_file_path(path)
    if not is_valid:
        return json.dumps(error_dict)

    try:
        with open(normalized_path, "r", encoding=_encoding, newline="") as file:
            sample = file.read(1024)
            file.seek(0)

            # Detect delimiter
            delimiter = ","
            if "," not in sample:
                sniffer = csv.Sniffer()
                try:
                    delimiter = sniffer.sniff(sample).delimiter
                except csv.Error:
                    for test_delim in [";", "\t", "|"]:
                        if test_delim in sample:
                            delimiter = test_delim
                            break

            file.seek(0)
            reader = csv.DictReader(file, delimiter=delimiter)
            columns = reader.fieldnames or []

            # Read rows with token limit
            max_tokens = config.get("DOC_MAX_TOKENS", 1024 * 8)
            rows = []
            current_tokens = count_tokens(f"Columns: {', '.join(columns)}\n")

            total_rows = 0
            truncated = False
            for row in reader:
                total_rows += 1
                if truncated:
                    continue
                row_tokens = count_tokens(json.dumps(row))
                if current_tokens + row_tokens > max_tokens:
                    truncated = True
                    continue
                rows.append(row)
                current_tokens += row_tokens

            # Count total rows
            was_truncated = total_rows > len(rows)

            return json.dumps(
                {
                    "path": normalized_path,
                    "type": "csv",
                    "columns": columns,
                    "delimiter": delimiter,
                    "encoding": _encoding,
                    "total_rows": total_rows,
                    "rows_returned": len(rows),
                    "rows": rows,
                    "tokens": current_tokens,
                    "max_tokens": max_tokens,
                    "truncated": was_truncated,
                    "message": f"Read CSV with {len(columns)} columns. Showing {len(rows)} of {total_rows} rows ({current_tokens} tokens). {'TRUNCATED at token limit.' if was_truncated else ''}",
                }
            )

    except UnicodeDecodeError:
        # The fallback goes through the same bounded reader and row accounting.
        return read_csv(normalized_path, _encoding="latin-1")
    except csv.Error as e:
        logger.error(f"Error during read csv: {str(e)}", exc_info=True)

        return json.dumps({"error": True, "message": f"CSV parsing error: {str(e)}"})
    except Exception as e:
        logger.error(f"Error during read csv: {str(e)}", exc_info=True)

        return json.dumps({"error": True, "message": f"Error parsing CSV: {str(e)}"})
