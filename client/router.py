"""Query router for classifying user queries into workflow categories."""

import re
from typing import Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from utils.config import config
from utils.logger import logger


# Route definitions: maps route names to tool name lists.
# None means all tools (fallback).
ROUTE_TOOLS: Dict[str, Optional[List[str]]] = {
    "simple_qa": [],
    "code": [
        "fs_read",
        "fs_write",
        "file_edit",
        "glob_search",
        "grep_search",
        "execute_bash",
        "git_safe",
        "git_status_safe",
        "git_commit_safe",
        "describe_image",
        "todo_write",
        "todo_read",
        "enter_plan_mode",
        "add_plan_step",
        "add_plan_file",
        "add_plan_risk",
        "present_plan",
        "approve_plan",
        "exit_plan_mode",
        "start_background_task",
        "get_task_status",
        "get_task_output",
        "wait_for_task",
    ],
    "research": [
        "web_search",
        "web_crawler",
    ],
    "knowledge": [
        "fs_read",
        "glob_search",
        "read_csv",
        "read_json",
        "read_pdf",
        "read_docx",
        "list_documents",
        "search_in_documents",
    ],
    "full": None,
}


def get_classifier_prompt() -> str:
    """Get the classifier prompt from config, falling back to the default.

    Returns:
        Classifier prompt string
    """
    return config.get("ROUTING_PROMPT", None)


class QueryRouter:
    """Routes queries to appropriate tool subsets based on classification."""

    def __init__(self, model: BaseChatModel) -> None:
        """Initialize the query router.

        Args:
            model: LangChain chat model for classification
        """
        self.model = model
        self._valid_routes = set(ROUTE_TOOLS.keys())

    def classify(
        self,
        query: str,
        conversation_context: str = "",
    ) -> str:
        """Classify a query into a route category.

        Args:
            query: The user's query
            conversation_context: Recent conversation for context

        Returns:
            Route name (one of ROUTE_TOOLS keys)
        """
        messages: List[BaseMessage] = [
            SystemMessage(content=get_classifier_prompt()),
        ]

        if conversation_context:
            messages.append(
                HumanMessage(
                    content=(f"[Recent conversation context]\n{conversation_context}")
                )
            )

        messages.append(HumanMessage(content=query))

        try:
            # Temporarily suppress instance-level callbacks so the
            # spinner keeps running during classification
            saved_callbacks = getattr(self.model, "callbacks", None)
            self.model.callbacks = None
            try:
                response = self.model.invoke(messages, config={"callbacks": []})
            finally:
                self.model.callbacks = saved_callbacks
            route = self._parse_route(response.content)

            logger.debug(f"Router classified query as: {route}")
            return route

        except Exception as e:
            logger.error(f"Router classification failed: {e}")
            return "full"

    def _parse_route(self, content: str) -> str:
        """Parse the model's classification response into a valid route.

        Handles thinking tags, extra whitespace, quotes, etc.

        Args:
            content: Raw model response

        Returns:
            Valid route name
        """
        # Handle Bedrock-style list content blocks (thinking enabled)
        if isinstance(content, list):
            text = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            text = content or ""

        # Strip thinking tags if present
        text = re.sub(
            r"<think(?:ing)?>.*?</think(?:ing)?>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        route = text.strip().lower().strip("\"'.,!").strip()

        if route not in self._valid_routes:
            logger.warning(
                f"Router returned unknown route '{route}', " "falling back to 'full'"
            )
            return "full"

        return route
