"""Tools manager that handles common functions and objects across tools"""

import os
import sys
import tiktoken
from typing import Any, Optional

# Add parent directory to path to allow imports from root
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from models.vision_model_controller import VisionModelController
from utils.config import config

# Load configuration from centralized config
BRAVE_API_KEY = config.get("BRAVE_API_KEY", None)

if BRAVE_API_KEY:
    os.environ["BRAVE_API_KEY"] = BRAVE_API_KEY

# Tiktoken parameters
MODEL_ID = "gpt-4"  # Default model for token counting


class ToolManager:
    def __init__(self) -> None:
        """Initialize tool manager."""
        self.model_id = config.get("VISION_MODEL_ID")
        self.encoder = tiktoken.encoding_for_model(MODEL_ID)

        if self.model_id:
            self.vision_model_controller = VisionModelController()
            self.vision_model_controller.initialize_model()
            self.vision_model = self.vision_model_controller.get_model()
        else:
            self.vision_model = None

    def get_encoder(self) -> tiktoken.Encoding:
        """Get the tiktoken encoder.

        Returns:
            Tiktoken encoder instance
        """
        return self.encoder

    def set_encoder(self, encoder: tiktoken.Encoding) -> None:
        """Set the tiktoken encoder.

        Args:
            encoder: Tiktoken encoder instance
        """
        self.encoder = encoder

    def get_vision_model(self) -> Optional[Any]:
        """Get the vision model instance.

        Returns:
            Vision model instance or None
        """
        return self.vision_model

    def count_tokens(self, text: str) -> int:
        """Count tokens with model-specific approximation.

        For Ollama models, uses character-based approximation.
        For OpenAI/Bedrock models, uses tiktoken encoder.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        model_type = config.get("MODEL_ID", {}).get("TYPE", "ollama")

        if model_type == "ollama":
            # Ollama approximation: ~1.3 chars per token (configurable)
            multiplier = (
                config.get("LLM", {})
                .get("TOKEN_COUNTING", {})
                .get("OLLAMA_APPROXIMATION", 1.3)
            )
            return int(len(text) / multiplier)
        else:
            # Use tiktoken for OpenAI/Bedrock/SageMaker
            return len(self.encoder.encode(text))

    def register_tools(self, mcp: Any) -> None:
        """Register all tools with the MCP server.

        Args:
            mcp: MCP server instance
        """
        from .describe_image import register_image_tools
        from .edit import register_edit_tools
        from .execute_bash import register_execute_bash_tools
        from .fs_read import register_fs_read_tools
        from .fs_write import register_fs_write_tools
        from .rag import register_rag_tools
        from .search import register_search_tools
        from .todo import register_todo_tools
        from .web_crawler import register_web_crawler_tools
        from .web_search import register_web_search_tools

        # Register all tool categories
        register_execute_bash_tools(mcp)
        register_edit_tools(mcp)
        register_fs_read_tools(mcp)
        register_fs_write_tools(mcp)
        register_search_tools(mcp)
        register_todo_tools(mcp)

        if self.get_vision_model() is not None:
            register_image_tools(mcp)

        if config.get("ENABLE_RAG", False):
            register_rag_tools(mcp)

        if config.get("ENABLE_WEB_CRAWL", None):
            register_web_crawler_tools(mcp)

        if BRAVE_API_KEY:
            register_web_search_tools(mcp)

    def validate_file_path(self, file_path: str) -> tuple[bool, str, dict]:
        """Validate and normalize file path.

        Args:
            file_path: Path to validate

        Returns:
            tuple: (is_valid, normalized_path, error_dict_or_empty)
        """
        # Handle escape characters and normalize path
        normalized_path = os.path.expanduser(file_path.strip())

        # Check if path exists
        if not os.path.exists(normalized_path):
            return (
                False,
                normalized_path,
                {
                    "error": True,
                    "message": f"Path doesn't exist: {normalized_path}",
                    "original_path": file_path,
                    "normalized_path": normalized_path,
                },
            )

        # Check if it's a file (not directory)
        if not os.path.isfile(normalized_path):
            return (
                False,
                normalized_path,
                {
                    "error": True,
                    "message": f"Path exists but is not a file: {normalized_path}",
                    "original_path": file_path,
                    "normalized_path": normalized_path,
                },
            )

        return True, normalized_path, {}
