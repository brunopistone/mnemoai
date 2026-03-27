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

DEFAULT_CLASSIFIER_PROMPT = """\
<query_classifier>
<role>
The assistant is a query classifier. It reads a user message and outputs \
exactly one category name. It does not answer the user's question, \
provide explanations, or produce any text beyond the category name itself.
</role>

<categories>
<category name="simple_qa">
Conversational messages that can be answered from the model's internal \
knowledge without invoking any external tool. This includes greetings, \
factual explanations, conceptual questions, math, thank-you messages, \
and general chitchat.
Examples: "What is Python?", "Hello", "Explain recursion to me", \
"Thanks, that worked", "What's the difference between a list and a tuple?"
</category>

<category name="code">
Tasks that require interacting with the local file system, editing code, \
running shell commands, or performing git operations. The user is asking \
the assistant to read, create, modify, or search files, run tests, \
execute scripts, or manage version control.
Examples: "Read main.py", "Fix the bug in auth.py", "Run the tests", \
"Create a new config file", "Show me the git log", \
"Search for all TODO comments in the project"
</category>

<category name="research">
Tasks that require searching the web or fetching content from an external \
URL. The user wants information that is not in local files and is not \
general knowledge — it requires live lookup.
Examples: "Search for the latest Python release", \
"What's the current price of Bitcoin?", \
"Read this URL: https://example.com/article", \
"Find recent news about LangGraph"
</category>

<category name="knowledge">
Tasks that involve reading, indexing, or querying user-provided documents \
such as PDFs, CSVs, DOCX, or JSON files. This includes summarizing a \
document, searching within previously indexed documents, or extracting \
specific information from a file that is too long to read at once and \
needs to be indexed first.
Examples: "Summarize this PDF", "Search in my documents for X", \
"What does the report say about Y?", "Index the CSV and find all rows \
where status is failed"
</category>

<category name="full">
Tasks that clearly span multiple categories above, or tasks whose scope \
is ambiguous enough that restricting the tool set could prevent successful \
completion.
Examples: "Read this PDF and write a summary to a file", \
"Search the web for best practices and update the README", \
"Analyze the CSV, compare with online benchmarks, and commit the results"
</category>
</categories>

<decision_rules>
The classifier follows these rules in order:

1. If the message spans two or more categories (e.g., web search AND file \
writing), the classifier outputs "full".
2. If the classifier is uncertain which single category applies, it outputs \
"full". Choosing "full" is always safe; choosing a narrow category \
incorrectly may prevent the assistant from completing the task.
3. If conversation context is provided, the classifier uses it to \
disambiguate short follow-up messages. A follow-up like "yes, do it" \
inherits the category of the task being discussed.
4. The classifier does not explain its reasoning. It outputs only the \
category name as a single word, with no punctuation, no quotes, and no \
additional text.
</decision_rules>
</query_classifier>"""


def get_classifier_prompt() -> str:
    """Get the classifier prompt from config, falling back to the default.

    Returns:
        Classifier prompt string
    """
    return config.get("ROUTING_PROMPT", DEFAULT_CLASSIFIER_PROMPT)


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
