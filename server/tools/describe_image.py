"""Image description tool using Ollama vision model."""

from . import validate_file_path, vision_model, vision_model_controller
import base64
import json
from mcp.server.fastmcp import FastMCP
import os
from pathlib import Path
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from utils.logger import logger


def register_image_tools(mcp: FastMCP) -> None:
    """Register image description tools.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    async def describe_image(
        image_path: str, question: str = "Describe this image in detail."
    ) -> str:
        """Describe an image using AI vision model.

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

            # Create messages with image and question
            messages = [
                vision_model_controller.format_request(question, image_bytes, image_ext)
            ]

            # Stream response and collect text
            description = ""
            async for event in vision_model.stream(messages):
                if (
                    "contentBlockDelta" in event
                    and "delta" in event["contentBlockDelta"]
                ):
                    if "text" in event["contentBlockDelta"]["delta"]:
                        description += event["contentBlockDelta"]["delta"]["text"]

            return json.dumps(
                {
                    "description": description.strip(),
                    "file_path": normalized_path,
                    "question": question,
                    "model": "qwen2.5vl:3b",
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
