"""Image description tool using Ollama vision model."""

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from mnemoai.utils.logger import logger

from . import tool_manager, validate_file_path


def register_image_tools(mcp: FastMCP) -> None:
    """Register image description tools.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    def describe_image(
        image_path: str, question: str = "Describe this image in detail."
    ) -> str:
        """Read the content of an image using AI vision model.

        Args:
            image_path: Path to the image file (supports PNG, JPG, JPEG, GIF, BMP, WEBP)
            question: Question about the image (optional, default: "Describe this image in detail.")

        Returns:
            JSON with image description or error message.
        """
        logger.debug(
            f"Tool describe_image called with image_path: {image_path} and question: {question}"
        )

        try:
            # Validate and normalize path
            is_valid, normalized_path, error_dict = validate_file_path(image_path)
            if not is_valid:
                return json.dumps(error_dict)

            # Check if it's a supported image format
            path = Path(normalized_path)
            supported_formats = [
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".bmp",
                ".webp",
            ]
            if path.suffix.lower() not in supported_formats:
                return json.dumps(
                    {
                        "error": True,
                        "message": f"Unsupported image format: {path.suffix}. Supported formats: {', '.join(supported_formats)}",
                        "file_path": normalized_path,
                    }
                )

            # Load image as raw bytes
            with open(normalized_path, "rb") as f:
                image_bytes = f.read()

            image_ext = normalized_path.split(".")[-1]

            # Resolved per CALL, never bound at import: at module level these
            # names run the vision build (BaseChatModel→transformers→torch) while
            # the server is still registering tools, and then hold that one model
            # for the life of the process. Deliberately after the cheap
            # validations above, so a bad path or an unsupported format costs no
            # model build at all. See ToolManager._ensure_vision.
            controller = tool_manager.vision_model_controller
            model = tool_manager.vision_model
            if model is None or controller is None:
                return json.dumps(
                    {
                        "error": True,
                        "message": "No vision model is configured "
                        "(set VISION_MODEL_ID in config.yaml).",
                        "file_path": normalized_path,
                    }
                )

            # Create message with image and question using LangChain format
            message = controller.format_request(question, image_bytes, image_ext)

            # Use LangChain model invoke
            response = model.invoke([message])
            content = response.content if hasattr(response, "content") else response
            # Normalize: some protocols (e.g. OpenAI Responses, Anthropic) return
            # content as a list of blocks rather than a plain string.
            description = controller._content_to_text(content)

            return json.dumps(
                {
                    "description": description.strip(),
                    "file_path": normalized_path,
                    "question": question,
                    "image_format": path.suffix.lower(),
                }
            )

        except Exception as e:
            logger.error(f"\nError describing image {image_path}: {e}")
            return json.dumps(
                {
                    "error": True,
                    "message": f"Failed to describe image: {str(e)}",
                    "file_path": image_path,
                }
            )
