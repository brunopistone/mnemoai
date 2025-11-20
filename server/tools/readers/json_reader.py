"""JSON file reading functionality."""

from .. import validate_file_path
import json
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from utils.logger import logger
from .chunking_helper import process_large_content


async def read_json(path: str) -> str:
    """Read and parse JSON file with automatic chunking for large files.

    Args:
        path: Path to JSON file

    Returns:
        JSON string with file data
    """
    is_valid, normalized_path, error_dict = validate_file_path(path)
    if not is_valid:
        return json.dumps(error_dict)

    try:
        with open(normalized_path, "r", encoding="utf-8") as file:
            content = file.read().strip()

            if not content:
                return json.dumps(
                    {
                        "path": normalized_path,
                        "type": "json",
                        "content": "",
                        "message": "File is empty",
                    }
                )

            # Process with chunking if needed
            processed_content, metadata = await process_large_content(content)

            return json.dumps(
                {
                    "path": normalized_path,
                    "type": "json",
                    "content": processed_content,
                    "processing_metadata": metadata,
                }
            )

    except json.JSONDecodeError as e:
        logger.error(f"Error during read json: {str(e)}", exc_info=True)
        return json.dumps({"error": True, "message": f"Invalid JSON format: {str(e)}"})

    except Exception as e:
        logger.error(f"Error during read json: {str(e)}", exc_info=True)
        return json.dumps({"error": True, "message": f"Error reading JSON: {str(e)}"})
