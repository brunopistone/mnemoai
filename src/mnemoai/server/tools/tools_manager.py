"""Tools manager that handles common functions and objects across tools"""

import os
from typing import Any, Optional

import tiktoken

from mnemoai.utils.config import config
from mnemoai.utils.path_utils import normalize_path
from mnemoai.utils.tokenization import count_tokens

# Load configuration from centralized config
BRAVE_API_KEY = config.get("BRAVE_API_KEY", None)

if config.get("ENABLE_WEB_SEARCH", None) and BRAVE_API_KEY:
    os.environ["BRAVE_API_KEY"] = BRAVE_API_KEY

# Tiktoken parameters
MODEL_ID = "gpt-4"  # Default model for token counting


class ToolManager:
    def __init__(self) -> None:
        """Initialize tool manager."""
        self.model_id = config.get("VISION_MODEL_ID")
        self.encoder = tiktoken.encoding_for_model(MODEL_ID)

        self._vision_model_controller = None
        self._vision_model = None
        self._vision_ready = False

    def _ensure_vision(self) -> None:
        """Initialize the vision model on first use, at most once.

        Constructing this eagerly would make *importing* the tools package load
        BaseChatModel→transformers→torch whenever a VISION_MODEL_ID is
        configured. Besides costing ~3s, torch vendors its own OpenMP runtime,
        which aborts the process (``OMP: Error #15``) as soon as faiss — which
        vendors another — runs a search in the same interpreter. Import-time
        side effects here must stay config-independent.

        The "done" flag is set only AFTER a successful build, so a failure stays
        loud on every subsequent access. Setting it first made failure sticky
        and SILENT: ``describe_image`` binds these names through the package's
        ``__getattr__``, and ``_handle_fromlist`` probes with ``hasattr``, which
        swallows an ``AttributeError`` — a pre-set flag then let the retry
        return None, permanently binding a dead model and dropping the tool
        from the registered set with no error anywhere.
        """
        if self._vision_ready:
            return
        if not self.model_id:
            self._vision_ready = True
            return
        from mnemoai.models.controllers.vision_model_controller import (
            VisionModelController,
        )

        # Build into locals first: a raise must not leave a half-initialized
        # controller behind for the next caller to find.
        controller = VisionModelController()
        controller.initialize_model()
        self._vision_model_controller = controller
        self._vision_model = controller.get_model()
        self._vision_ready = True

    @property
    def vision_model_controller(self) -> Optional[Any]:
        """The vision controller, initialized on first access."""
        self._ensure_vision()
        return self._vision_model_controller

    @property
    def vision_model(self) -> Optional[Any]:
        """The vision model, initialized on first access."""
        self._ensure_vision()
        return self._vision_model

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
        return self.vision_model  # property: initializes on first access

    def count_tokens(self, text: str) -> int:
        """Count tokens with model-specific approximation.

        Delegates to :func:`mnemoai.utils.tokenization.count_tokens` — the single
        shared implementation used by both the client and the server — so the
        two layers can't drift apart.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        return count_tokens(text)

    def register_tools(self, mcp: Any) -> None:
        """Register all tools with the MCP server.

        Args:
            mcp: MCP server instance
        """
        from .ask_user_question import register_ask_user_tools
        from .background_tasks import register_background_tasks_tools
        from .execute_bash import register_execute_bash_tools
        from .file_edit import register_edit_tools
        from .file_search import register_search_tools
        from .fs_read import register_fs_read_tools
        from .fs_write import register_fs_write_tools
        from .git_safety import register_git_safety_tools
        from .memory_tool import register_memory_tools
        from .plan_mode_exit import register_plan_mode_exit_tools
        from .skill_tool import register_skill_tools
        from .subagent_tool import register_subagent_tools
        from .thread_offload import ThreadedToolServer
        from .todo_manager import register_todo_tools
        from .web_crawler import register_web_crawler_tools
        from .web_search import register_web_search_tools

        # Every group registers through this proxy, so a tool with a blocking
        # (sync) body runs on a worker thread instead of the server's single
        # event loop — where it would freeze every other agent's in-flight tool
        # call until the client's MCP_CALL_TIMEOUT killed it. See thread_offload.
        mcp = ThreadedToolServer(mcp)

        # Register all tool categories
        register_ask_user_tools(mcp)
        register_background_tasks_tools(mcp)
        register_execute_bash_tools(mcp)
        register_edit_tools(mcp)
        register_fs_read_tools(mcp)
        register_fs_write_tools(mcp)
        register_git_safety_tools(mcp)
        register_plan_mode_exit_tools(mcp)
        register_search_tools(mcp)
        register_subagent_tools(mcp)
        register_todo_tools(mcp)

        # The two heavy groups are imported INSIDE their gate, not above with the
        # rest. describe_image reaches transformers/torch and .rag reaches faiss,
        # each vendoring an OpenMP runtime, and a process holding both aborts
        # (``OMP: Error #15``) once faiss searches. Importing them
        # unconditionally made the gates dead weight for that cost: the module
        # object — and its OpenMP registration — is created by the import, so
        # only a gated import can decline to pay it.
        if self.get_vision_model() is not None:
            from .describe_image import register_image_tools

            register_image_tools(mcp)

        if config.get("ENABLE_MEMORY", True):
            register_memory_tools(mcp)

        if config.get("ENABLE_SKILLS", True):
            register_skill_tools(mcp)

        if config.get("ENABLE_RAG", False):
            from .rag import register_rag_tools

            register_rag_tools(mcp)

        if config.get("ENABLE_WEB_CRAWL", None):
            register_web_crawler_tools(mcp)

        if config.get("ENABLE_WEB_SEARCH", None) and BRAVE_API_KEY:
            register_web_search_tools(mcp)

    def validate_file_path(self, file_path: str) -> tuple[bool, str, dict]:
        """Validate and normalize file path.

        Args:
            file_path: Path to validate

        Returns:
            tuple: (is_valid, normalized_path, error_dict_or_empty)
        """
        # Normalize the path, tolerating shell escaping/quoting (e.g. a path the
        # user copied from a terminal as `/Users/me/My\ File.png` or quoted).
        normalized_path = normalize_path(file_path)

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
