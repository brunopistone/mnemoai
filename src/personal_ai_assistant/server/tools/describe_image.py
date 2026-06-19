"""Image description tool using Ollama vision model."""

from . import validate_file_path, vision_model, vision_model_controller
import json
from mcp.server.fastmcp import FastMCP
from pathlib import Path

from personal_ai_assistant.utils.logger import logger


def register_image_tools(mcp: FastMCP) -> None:
    """Register image description tools.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    async def describe_image(
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
            supported_formats = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"]
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

            # Create message with image and question using LangChain format
            message = vision_model_controller.format_request(question, image_bytes, image_ext)

            # Use LangChain model invoke
            response = vision_model.invoke([message])
            content = response.content if hasattr(response, 'content') else response
            # Normalize: some protocols (e.g. OpenAI Responses, Anthropic) return
            # content as a list of blocks rather than a plain string.
            description = vision_model_controller._content_to_text(content)

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
