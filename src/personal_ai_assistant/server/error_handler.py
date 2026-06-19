"""Standardized error handling for tools with helpful troubleshooting guidance."""

import json
from functools import wraps
from typing import Callable

from personal_ai_assistant.utils.logger import logger


def tool_error_handler(func: Callable) -> Callable:
    """Decorator for standardized tool error handling.

    Wraps tool functions to provide consistent error messages with:
    - Clear description of what went wrong
    - Specific error type for debugging
    - Next steps and troubleshooting guidance
    - Original error details for transparency

    Usage:
        @mcp.tool()
        @tool_error_handler
        async def my_tool(param: str) -> str:
            # Tool implementation
            pass
    """

    @wraps(func)
    async def wrapper(*args, **kwargs) -> str:
        try:
            return await func(*args, **kwargs)

        except FileNotFoundError as e:
            error_path = str(e).split("'")[1] if "'" in str(e) else str(e)
            return json.dumps(
                {
                    "error": True,
                    "error_type": "FileNotFoundError",
                    "message": f"File or directory not found: {error_path}",
                    "next_steps": [
                        "Verify the file path is correct and complete",
                        "Use glob_search to find files by pattern",
                        "Check if the file exists with: execute_bash('ls -la <parent_directory>')",
                        "Ensure you're using an absolute path, not a relative path",
                        "Check for typos in the filename or path",
                    ],
                    "original_error": str(e),
                },
                indent=2,
            )

        except PermissionError as e:
            error_path = str(e).split("'")[1] if "'" in str(e) else str(e)
            return json.dumps(
                {
                    "error": True,
                    "error_type": "PermissionError",
                    "message": f"Permission denied: {error_path}",
                    "next_steps": [
                        "Check file permissions with: execute_bash('ls -la <file>')",
                        "Ask user to grant read/write permissions if needed",
                        "User may need to run: chmod +r <file> (for read)",
                        "User may need to run: chmod +w <file> (for write)",
                        "Verify you're not trying to write to a protected system directory",
                        "Check if the file is owned by another user",
                    ],
                    "original_error": str(e),
                },
                indent=2,
            )

        except IsADirectoryError as e:
            error_path = str(e).split("'")[1] if "'" in str(e) else str(e)
            return json.dumps(
                {
                    "error": True,
                    "error_type": "IsADirectoryError",
                    "message": f"Path is a directory, not a file: {error_path}",
                    "next_steps": [
                        "This path points to a directory, not a file",
                        "If you want to list directory contents, use: execute_bash('ls <directory>')",
                        "If you want to find files in the directory, use: glob_search(pattern='*', path='<directory>')",
                        "Verify the complete file path including filename",
                    ],
                    "original_error": str(e),
                },
                indent=2,
            )

        except UnicodeDecodeError as e:
            return json.dumps(
                {
                    "error": True,
                    "error_type": "UnicodeDecodeError",
                    "message": "File is not UTF-8 encoded or contains invalid characters",
                    "next_steps": [
                        "This file might be binary (e.g., images, executables, PDFs)",
                        "Check file type with: execute_bash('file <path>')",
                        "If it's a text file with different encoding, ask user about encoding",
                        "For binary files, consider using specialized tools",
                        "Try reading with different encoding if user confirms file type",
                    ],
                    "original_error": str(e),
                    "encoding_attempted": "utf-8",
                },
                indent=2,
            )

        except ValueError as e:
            return json.dumps(
                {
                    "error": True,
                    "error_type": "ValueError",
                    "message": f"Invalid value provided: {str(e)}",
                    "next_steps": [
                        "Check that all required parameters are provided",
                        "Verify parameter types match expected types",
                        "Review parameter values for validity (e.g., positive numbers, valid paths)",
                        "Check the tool documentation for correct parameter format",
                        "Ensure numeric values are within acceptable ranges",
                    ],
                    "original_error": str(e),
                },
                indent=2,
            )

        except TypeError as e:
            return json.dumps(
                {
                    "error": True,
                    "error_type": "TypeError",
                    "message": f"Type mismatch or incorrect arguments: {str(e)}",
                    "next_steps": [
                        "Check that parameters are the correct type (string, int, bool, etc.)",
                        "Verify all required parameters are provided",
                        "Ensure optional parameters use correct default values",
                        "Review the tool signature for expected parameter types",
                        "Check for None values where they're not allowed",
                    ],
                    "original_error": str(e),
                },
                indent=2,
            )

        except OSError as e:
            # Catch-all for OS-related errors (disk full, broken pipe, etc.)
            return json.dumps(
                {
                    "error": True,
                    "error_type": "OSError",
                    "message": f"Operating system error: {str(e)}",
                    "next_steps": [
                        "Check available disk space with: execute_bash('df -h')",
                        "Verify the filesystem is not read-only",
                        "Check if the path or filename is too long",
                        "Ensure the device/drive is accessible",
                        "Look for filesystem issues or corrupted files",
                        "Try a simpler operation to isolate the issue",
                    ],
                    "original_error": str(e),
                    "error_code": getattr(e, "errno", None),
                },
                indent=2,
            )

        except json.JSONDecodeError as e:
            return json.dumps(
                {
                    "error": True,
                    "error_type": "JSONDecodeError",
                    "message": f"Invalid JSON format: {str(e)}",
                    "next_steps": [
                        "Check for missing quotes, commas, or brackets",
                        "Verify JSON structure is well-formed",
                        "Use a JSON validator to identify syntax errors",
                        "Check for trailing commas (not allowed in JSON)",
                        "Ensure all strings use double quotes, not single quotes",
                        "Validate JSON at: https://jsonlint.com",
                    ],
                    "original_error": str(e),
                    "error_location": (
                        f"Line {e.lineno}, Column {e.colno}"
                        if hasattr(e, "lineno")
                        else "Unknown"
                    ),
                },
                indent=2,
            )

        except TimeoutError as e:
            return json.dumps(
                {
                    "error": True,
                    "error_type": "TimeoutError",
                    "message": f"Operation timed out: {str(e)}",
                    "next_steps": [
                        "The operation took too long to complete",
                        "Try breaking the operation into smaller parts",
                        "Check network connectivity if accessing remote resources",
                        "Increase timeout if the operation legitimately needs more time",
                        "Look for infinite loops or blocking operations",
                    ],
                    "original_error": str(e),
                },
                indent=2,
            )

        except KeyError as e:
            missing_key = str(e).strip("'\"")
            return json.dumps(
                {
                    "error": True,
                    "error_type": "KeyError",
                    "message": f"Missing required key: {missing_key}",
                    "next_steps": [
                        f"The key '{missing_key}' was not found in the data structure",
                        "Check that the configuration/data file is complete",
                        "Verify you're accessing the correct nested structure",
                        "Check for typos in the key name",
                        "Ensure the data source provides all required fields",
                    ],
                    "missing_key": missing_key,
                    "original_error": str(e),
                },
                indent=2,
            )

        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(
                f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True
            )
            return json.dumps(
                {
                    "error": True,
                    "error_type": type(e).__name__,
                    "message": f"Unexpected error occurred: {str(e)}",
                    "next_steps": [
                        "This is an unexpected error that wasn't specifically handled",
                        "Review the error message for clues about what went wrong",
                        "Try the operation again with different parameters",
                        "Check logs for more detailed error information",
                        "Simplify the operation to isolate the issue",
                        "If this persists, it may be a bug that needs reporting",
                    ],
                    "original_error": str(e),
                    "function": func.__name__,
                },
                indent=2,
            )

    return wrapper


def create_error_response(
    error_type: str, message: str, next_steps: list, **extra_fields
) -> str:
    """Helper function to create standardized error responses.

    Args:
        error_type: Type of error (e.g., 'ValidationError', 'ConfigError')
        message: Clear description of what went wrong
        next_steps: List of actionable troubleshooting steps
        **extra_fields: Additional fields to include in the response

    Returns:
        JSON string with standardized error format
    """
    error_data = {
        "error": True,
        "error_type": error_type,
        "message": message,
        "next_steps": next_steps,
    }

    # Add any extra fields
    error_data.update(extra_fields)

    return json.dumps(error_data, indent=2)
