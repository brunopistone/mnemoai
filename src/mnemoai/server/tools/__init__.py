"""Tools package for MCP server."""

from .tools_manager import ToolManager

# Image extensions that the vision tool (describe_image) handles. A text reader
# encountering one of these should steer the model there instead of trying to
# decode bytes as UTF-8.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".ico")


def looks_like_binary(path: str) -> bool:
    """Heuristically detect a binary (non-text) file.

    Checks a known image extension first, then sniffs the first chunk for NUL
    bytes / a high ratio of non-text bytes. Used by the text readers to fail
    fast with a helpful message rather than a raw UnicodeDecodeError.
    """
    import os

    if path.lower().endswith(IMAGE_EXTENSIONS):
        return True
    try:
        with open(path, "rb") as f:
            chunk = f.read(2048)
    except OSError:
        return False
    if not chunk:
        return False
    if b"\x00" in chunk:
        return True
    # Bytes that are unlikely in UTF-8 text (control chars outside tab/nl/cr).
    text_chars = bytes(range(32, 127)) + b"\t\n\r\f\b"
    nontext = sum(1 for b in chunk if b not in text_chars and b < 128)
    return nontext / len(chunk) > 0.30


def binary_file_error(path: str) -> dict:
    """Standard error payload steering the model away from text-reading a binary.

    For images it points at ``describe_image``; for other binaries it just
    explains the file isn't text.
    """
    is_image = path.lower().endswith(IMAGE_EXTENSIONS)
    if is_image:
        message = (
            f"'{path}' is an image, not a text file. Use the describe_image "
            "tool to inspect it (it answers questions about images), not a "
            "file-reading tool."
        )
    else:
        message = (
            f"'{path}' appears to be a binary file and can't be read as text."
        )
    return {"error": True, "message": message, "binary": True, "is_image": is_image}

# Create global tool manager instance
tool_manager = ToolManager()

# Export commonly used functions and constants for backward compatibility
register_tools = tool_manager.register_tools
validate_file_path = tool_manager.validate_file_path
count_tokens = tool_manager.count_tokens
vision_model = tool_manager.get_vision_model()
vision_model_controller = tool_manager.vision_model_controller
