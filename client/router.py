"""Query router for classifying user queries into workflow categories."""

import asyncio
import re
from typing import Dict, List, Optional

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
    """Get the classifier prompt from config.

    Returns:
        Classifier prompt string
    """
    return config.get("ROUTING_PROMPT", None)


class QueryRouter:
    """Routes queries to appropriate tool subsets based on classification."""

    def __init__(self, model) -> None:
        """Initialize the query router.

        Args:
            model: Strands model instance with async stream() method
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
        classifier_prompt = get_classifier_prompt()
        if not classifier_prompt:
            logger.debug("No ROUTING_PROMPT configured, falling back to 'full'")
            return "full"

        messages = []

        if conversation_context:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "text": f"[Recent conversation context]\n{conversation_context}"
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "text": "I understand the context. Please provide the query to classify."
                        }
                    ],
                }
            )

        messages.append(
            {
                "role": "user",
                "content": [{"text": query}],
            }
        )

        try:
            response_text = asyncio.run(
                self._stream_classify(messages, classifier_prompt)
            )
            route = self._parse_route(response_text)
            logger.debug(f"Router classified query as: {route}")
            return route

        except Exception as e:
            logger.error(f"Router classification failed: {e}")
            return "full"

    async def _stream_classify(self, messages: list, system_prompt: str) -> str:
        """Stream the classification response from the model.

        Args:
            messages: Strands-format messages
            system_prompt: Classifier system prompt

        Returns:
            Full response text
        """
        response_text = ""
        async for event in self.model.stream(
            messages, system_prompt=system_prompt, think=False
        ):
            if (
                "contentBlockDelta" in event
                and "delta" in event["contentBlockDelta"]
                and "text" in event["contentBlockDelta"]["delta"]
            ):
                response_text += event["contentBlockDelta"]["delta"]["text"]
        return response_text

    def _parse_route(self, content: str) -> str:
        """Parse the model's classification response into a valid route.

        Handles thinking tags, extra whitespace, quotes, etc.

        Args:
            content: Raw model response

        Returns:
            Valid route name
        """
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
                f"Router returned unknown route '{route}', falling back to 'full'"
            )
            return "full"

        return route
