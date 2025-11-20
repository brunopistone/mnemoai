"""CSV file reading functionality."""

from .. import validate_file_path, count_tokens
import csv
import json
import sys
import os

# Add parent directory to path to allow imports from root
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from utils.logger import logger
from .chunking_helper import process_large_content


async def read_csv(path: str) -> str:
    """Read and parse CSV file with automatic chunking for large files.

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

            # Read all rows
            rows = list(reader)

            # Convert to text format for chunking
            csv_text = f"Columns: {', '.join(columns)}\n\n"
            for i, row in enumerate(rows):
                csv_text += f"Row {i+1}: {json.dumps(row)}\n"

            # Process with chunking if needed
            processed_content, metadata = await process_large_content(csv_text)

            return json.dumps(
                {
                    "path": normalized_path,
                    "type": "csv",
                    "columns": columns,
                    "total_rows": len(rows),
                    "content": processed_content,
                    "processing_metadata": metadata,
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
