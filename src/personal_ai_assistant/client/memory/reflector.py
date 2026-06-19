"""ACE Reflector - Analyzes tool executions and extracts actionable strategies."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from personal_ai_assistant.utils.logger import logger


class PlaybookEntry:
    """A structured strategy entry for the playbook."""

    def __init__(
        self,
        context: str,
        strategy: str,
        source: str,
        outcome: str = "success",
        tools: List[str] = None,
        confidence: float = 1.0,
    ):
        self.context = context  # When to apply this strategy
        self.strategy = strategy  # What to do
        self.source = source  # Where this was learned from
        self.outcome = outcome  # success or failure
        self.tools = tools or []
        self.confidence = confidence
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context": self.context,
            "strategy": self.strategy,
            "source": self.source,
            "outcome": self.outcome,
            "tools": self.tools,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

    def to_prompt_text(self) -> str:
        """Format for injection into system prompt."""
        prefix = "✓" if self.outcome == "success" else "✗"
        return f"{prefix} [{self.context}]: {self.strategy}"


class Reflector:
    """Analyzes execution trajectories and extracts strategies."""

    def __init__(self, persist_path: str = None):
        self.pending_reflections: List[Dict[str, Any]] = []
        self.persist_path = persist_path
        self.metrics_file = (
            os.path.join(persist_path, "metrics.json") if persist_path else None
        )
        self.metrics = self._load_metrics()

    def _load_metrics(self) -> Dict[str, Any]:
        """Load metrics from disk."""
        default = {
            "total_tool_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "strategies_extracted": 0,
            "failure_types": {},
            "daily_stats": {},
        }
        if self.metrics_file and os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return default

    def _save_metrics(self) -> None:
        """Persist metrics to disk."""
        if self.metrics_file:
            try:
                os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
                with open(self.metrics_file, "w") as f:
                    json.dump(self.metrics, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save metrics: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics with success rate."""
        success_rate = 0.0
        if self.metrics["total_tool_calls"] > 0:
            success_rate = (
                self.metrics["successful_calls"] / self.metrics["total_tool_calls"]
            )
        return {
            **self.metrics,
            "success_rate": round(success_rate, 3),
        }

    # Patterns that indicate specific failure types
    FAILURE_PATTERNS = {
        "string_not_found": [
            "no occurrences of",
            "string not found",
            "could not find",
            "not unique",
        ],
        "file_not_found": [
            "file not found",
            "no such file",
            "does not exist",
            "path not found",
        ],
        "permission_denied": [
            "permission denied",
            "access denied",
            "not permitted",
            "operation not allowed",
        ],
        "syntax_error": [
            "syntax error",
            "invalid syntax",
            "unexpected token",
            "parse error",
        ],
        "timeout": [
            "timeout",
            "timed out",
            "took too long",
            "deadline exceeded",
        ],
        "api_error": [
            "api error",
            "rate limit exceeded",
            "authentication failed",
            "unauthorized",
            "403",
            "401",
            "429",
        ],
        "command_failed": [
            "command not found",
            "exit code",
            "non-zero exit",
            "returned error",
        ],
        "json_error": [
            "json decode",
            "invalid json",
            "expecting value",
            "unterminated string",
        ],
    }

    # Patterns that indicate actual tool errors (not content)
    ERROR_INDICATORS = [
        "error:",
        "failed:",
        "exception:",
        "traceback",
        "could not",
        "unable to",
    ]

    # Strategy templates for common failure patterns
    FAILURE_STRATEGIES = {
        "string_not_found": "Read the file first to get the exact string including whitespace before using str_replace",
        "file_not_found": "Use glob_search or ls to verify file path exists before reading/writing",
        "permission_denied": "Check file permissions with ls -la before attempting write operations",
        "syntax_error": "Validate code syntax before writing; consider using a linter",
        "timeout": "Break long operations into smaller chunks or use background tasks",
        "api_error": "Check API status and rate limits; implement retry with backoff",
        "command_failed": "Verify command exists and arguments are correct; check PATH if needed",
        "json_error": "Validate JSON structure before parsing; check for trailing commas or missing quotes",
    }

    def _is_actual_error(self, result_lower: str) -> bool:
        """Check if the result indicates an actual tool error, not just content."""
        return any(indicator in result_lower for indicator in self.ERROR_INDICATORS)

    def _track_metric(self, success: bool, failure_type: str = None) -> None:
        """Track a tool execution metric and persist."""
        today = datetime.now().strftime("%Y-%m-%d")

        self.metrics["total_tool_calls"] += 1
        if success:
            self.metrics["successful_calls"] += 1
        else:
            self.metrics["failed_calls"] += 1
            if failure_type:
                self.metrics["failure_types"][failure_type] = (
                    self.metrics["failure_types"].get(failure_type, 0) + 1
                )

        # Track daily stats
        if today not in self.metrics.get("daily_stats", {}):
            self.metrics["daily_stats"][today] = {"success": 0, "failure": 0}
        if success:
            self.metrics["daily_stats"][today]["success"] += 1
        else:
            self.metrics["daily_stats"][today]["failure"] += 1

        self._save_metrics()

    def analyze_tool_execution(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: str,
        task_context: str,
    ) -> Optional[PlaybookEntry]:
        """Analyze a single tool execution and extract strategy if notable.

        Args:
            tool_name: Name of the tool executed
            tool_args: Arguments passed to the tool
            tool_result: Result/output from the tool
            task_context: The user's original task

        Returns:
            PlaybookEntry if a notable pattern was found, None otherwise
        """
        # Skip if no result to analyze
        if not tool_result:
            return None

        result_lower = tool_result.lower()

        # Check for failures first
        if self._is_actual_error(result_lower):
            # Check for specific failure patterns
            for failure_type, patterns in self.FAILURE_PATTERNS.items():
                if any(p in result_lower for p in patterns):
                    self._track_metric(success=False, failure_type=failure_type)
                    entry = self._create_failure_entry(
                        failure_type, tool_name, tool_args, tool_result, task_context
                    )
                    if entry:
                        self.metrics["strategies_extracted"] += 1
                        self._save_metrics()
                    return entry

            # Generic failure (no specific pattern matched)
            self._track_metric(success=False)
            return None

        # Success case
        self._track_metric(success=True)

        # Check for notable successes worth remembering
        if self._is_notable_success(tool_name, tool_result):
            entry = self._create_success_entry(
                tool_name, tool_args, tool_result, task_context
            )
            if entry:
                self.metrics["strategies_extracted"] += 1
                self._save_metrics()
            return entry

        return None

    def _create_failure_entry(
        self,
        failure_type: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: str,
        task_context: str,
    ) -> PlaybookEntry:
        """Create a playbook entry from a failure."""
        strategy = self.FAILURE_STRATEGIES.get(
            failure_type, f"Verify preconditions before using {tool_name}"
        )

        # Add specific context from the failure
        context = self._extract_context(tool_name, tool_args, task_context)

        source = f"Failed {tool_name} on {datetime.now().strftime('%Y-%m-%d')}: {failure_type}"

        return PlaybookEntry(
            context=context,
            strategy=strategy,
            source=source,
            outcome="failure",
            tools=[tool_name],
            confidence=0.9,
        )

    def _create_success_entry(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: str,
        task_context: str,
    ) -> Optional[PlaybookEntry]:
        """Create a playbook entry from a notable success."""
        strategy = self._extract_success_strategy(tool_name, tool_args, tool_result)

        # Skip if no specific strategy
        if not strategy:
            return None

        context = self._extract_context(tool_name, tool_args, task_context)
        source = f"Successful {tool_name} on {datetime.now().strftime('%Y-%m-%d')}"

        return PlaybookEntry(
            context=context,
            strategy=strategy,
            source=source,
            outcome="success",
            tools=[tool_name],
            confidence=0.8,
        )

    def _extract_context(
        self, tool_name: str, tool_args: Dict[str, Any], task_context: str
    ) -> str:
        """Extract the context/situation for when this strategy applies."""
        # Tool-specific context extraction
        if tool_name in ["fs_write", "file_edit", "str_replace"]:
            path = tool_args.get("path", tool_args.get("file_path", ""))
            ext = path.split(".")[-1] if "." in path else "file"
            return f"editing {ext} files"

        if tool_name in ["fs_read"]:
            path = tool_args.get("path", "")
            ext = path.split(".")[-1] if "." in path else "file"
            return f"reading {ext} files"

        if tool_name in ["execute_bash"]:
            cmd = tool_args.get("command", "")[:30]
            return f"running bash commands ({cmd}...)"

        if tool_name in ["glob_search", "grep_search"]:
            return "searching files"

        if tool_name in ["web_search", "web_crawler"]:
            return "web operations"

        # Default: use task context
        return task_context[:50] if task_context else tool_name

    def _extract_success_strategy(
        self, tool_name: str, tool_args: Dict[str, Any], tool_result: str
    ) -> str:
        """Extract a reusable strategy from a successful execution."""
        if tool_name == "file_edit":
            return "Include sufficient context in old_string to ensure uniqueness"

        if tool_name == "execute_bash":
            cmd = tool_args.get("command", "")
            if "grep" in cmd:
                return "Use grep with context (-B/-A flags) for better matches"
            if "find" in cmd:
                return "Use glob_search instead of find for better performance"
            if "curl" in cmd:
                return "Use -s flag for silent mode, -f to fail on HTTP errors"

        if tool_name == "fs_read" and "pdf" in str(tool_args).lower():
            return "For large PDFs, read specific page ranges rather than entire file"

        if tool_name == "web_crawler":
            url = tool_args.get("url", "")
            if url:
                return f"web_crawler works for fetching content from URLs"

        # No specific strategy - return None to skip
        return None

    def _is_notable_success(self, tool_name: str, tool_result: str) -> bool:
        """Determine if a success is worth remembering."""
        # Only remember successes for tools with specific strategies
        notable_tools = ["file_edit", "execute_bash"]
        return tool_name in notable_tools

    def reflect_on_trajectory(
        self, messages: List[Any], task: str
    ) -> List[PlaybookEntry]:
        """Analyze a full execution trajectory and extract all strategies.

        Args:
            messages: Full conversation messages including tool calls/results
            task: The original user task

        Returns:
            List of PlaybookEntry objects
        """
        entries = []

        for msg in messages:
            # Extract tool calls and results from messages
            tool_calls = self._extract_tool_calls(msg)

            for call in tool_calls:
                tool_name = call.get("name", "")
                tool_args = call.get("args", {})
                tool_call_id = call.get("id", "")

                # Find corresponding result
                result = self._find_tool_result(tool_call_id, tool_name, messages)

                entry = self.analyze_tool_execution(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=result,
                    task_context=task,
                )

                if entry:
                    entries.append(entry)
                    logger.debug(
                        f"Reflector: extracted strategy for {tool_name} ({entry.outcome})"
                    )

        return entries

    def _extract_tool_calls(self, msg: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from a message."""
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            return [
                {
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                    "id": tc.get("id", ""),
                }
                for tc in msg.tool_calls
            ]
        return []

    def _extract_tool_results(self, msg: Any) -> List[str]:
        """Extract tool results from a message."""
        if hasattr(msg, "type") and msg.type == "tool":
            content = getattr(msg, "content", "")
            return [content] if content else []
        return []

    def _find_tool_result(
        self, tool_call_id: str, tool_name: str, messages: List[Any]
    ) -> str:
        """Find the result for a specific tool call."""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "tool":
                # Match by tool_call_id if available
                msg_tool_call_id = getattr(msg, "tool_call_id", "")
                if msg_tool_call_id and msg_tool_call_id == tool_call_id:
                    return getattr(msg, "content", "")
                # Fallback: match by name
                if hasattr(msg, "name") and msg.name == tool_name:
                    return getattr(msg, "content", "")
        return ""
