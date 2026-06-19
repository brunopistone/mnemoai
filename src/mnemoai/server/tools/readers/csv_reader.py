"""CSV file reading functionality."""

from .. import validate_file_path, count_tokens
import csv
import json

from mnemoai.utils.logger import logger
from mnemoai.utils.config import config


async def read_csv(path: str) -> str:
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
        with open(normalized_path, "r", encoding="utf-8") as file:
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

            for row in reader:
                row_tokens = count_tokens(json.dumps(row))
                if current_tokens + row_tokens > max_tokens:
                    break
                rows.append(row)
                current_tokens += row_tokens

            # Count total rows
            total_rows = len(rows) + sum(1 for _ in reader)
            was_truncated = total_rows > len(rows)

            return json.dumps(
                {
                    "path": normalized_path,
                    "type": "csv",
                    "columns": columns,
                    "delimiter": delimiter,
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
        # Try alternative encodings
        for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
            try:
                with open(normalized_path, "r", encoding=encoding) as file:
                    sample = file.read(1024)
                    file.seek(0)

                    delimiter = ","
                    if "," not in sample:
                        for test_delim in [";", "\t", "|"]:
                            if test_delim in sample:
                                delimiter = test_delim
                                break

                    file.seek(0)
                    reader = csv.DictReader(file, delimiter=delimiter)
                    columns = reader.fieldnames or []
                    rows = list(reader)

                    return json.dumps(
                        {
                            "path": normalized_path,
                            "type": "csv",
                            "columns": columns,
                            "rows": rows,
                            "total_rows": len(rows),
                            "rows_returned": len(rows),
                            "delimiter": delimiter,
                            "encoding": encoding,
                        }
                    )
            except (UnicodeDecodeError, csv.Error):
                continue

        logger.error(f"Error during read csv: {str(e)}", exc_info=True)

        return json.dumps(
            {
                "error": True,
                "message": "Could not decode file with any supported encoding",
            }
        )
    except csv.Error as e:
        logger.error(f"Error during read csv: {str(e)}", exc_info=True)

        return json.dumps({"error": True, "message": f"CSV parsing error: {str(e)}"})
    except Exception as e:
        logger.error(f"Error during read csv: {str(e)}", exc_info=True)

        return json.dumps({"error": True, "message": f"Error parsing CSV: {str(e)}"})
