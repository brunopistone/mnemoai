"""The shared model↔tool execution loop (agent-arg helper).

One tool call, start to finish: normalize the args, short-circuit the client-side
stubs, resolve the tool, then the gates in their required order — plan mode blocks
BEFORE anything else (a blocked tool must never even ask), then the user's
``PreToolUse`` hooks, then the confirmation prompt — and finally ``_invoke_tool``,
where the post-call gate lives. A hook's ``deny`` ends the call; its ``allow``
satisfies only the confirmation prompt, never the block above it. Every branch
emits exactly one ``ToolMessage`` per ``tool_call_id``: an unanswered id is a
provider-level error, so there is no path out of here that skips the reply.

This existed as two near-identical copies, in ``_execute_tools`` (main loop) and
``_run_worker_loop`` (sub-agents + orchestrator waves) — ~70 lines each, ~39 of
them byte-identical. Both carry the plan-mode hard block and the destructive-tool
confirmation gate, which is exactly the logic that must not drift between the
paths: the ``str(e) or repr(e)`` fix for a bare ``TimeoutError``'s empty ``str()``
was applied to the foreground copy only, so for a release the worker logged a
blank error, and the worker never logged the tool-not-found warning at all.

The remaining differences between the two paths are real parameters, not
branches: ``quiet`` (a sub-agent suppresses display), ``activity`` (the TUI panel
sink; None makes those calls dead), ``spawn_results`` (the main loop batches
parallel ``spawn_agent`` calls, a worker runs its one spawn inline) and
``log_label``, which only keeps the two paths distinguishable in a log file when
sub-agents run concurrently.

``messages`` is appended to IN PLACE rather than returned. The worker's list is
the same one the next iteration re-streams and that ``saveable`` slices at every
exit, and a ``KeyboardInterrupt`` (the UI's cancel — a ``BaseException``, so
deliberately not caught below) must leave the calls already answered in place;
returning a list would discard the aborted round.

Takes the agent as the first arg and dispatches back through its own methods, so
an override on a bare ``__new__`` test stub still intercepts — the
``plan_policy``/``confirmation_gate`` collaborator pattern.
"""

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage, ToolMessage

from mnemoai.client import hooks
from mnemoai.client.agent.agent_activity import ActivitySink
from mnemoai.utils.logger import logger


def run_tool_calls(
    agent,
    tool_calls: Sequence[Dict[str, Any]],
    tools: Sequence[Any],
    messages: List[BaseMessage],
    quiet: bool = False,
    activity: Optional[ActivitySink] = None,
    spawn_results: Optional[Dict[str, str]] = None,
    log_label: str = "Tool execution",
) -> None:
    """Run ``tool_calls``, appending one ToolMessage each to ``messages``."""
    for call in tool_calls:
        name = call["name"]
        tool_id = call["id"]
        args = agent._normalize_tool_args(call["args"])

        # exit_plan_mode / spawn_agent / resume_agent / ask_user_question are
        # handled client-side, not via MCP, so they're never looked up below.
        client_msg = agent._client_side_tool_message(name, args, tool_id, spawn_results)
        if client_msg is not None:
            messages.append(client_msg)
            continue

        if activity is not None:
            activity.tool_call(name, args)

        # The caller's subset first, then every tool (a worker's subset is narrow).
        tool = next((t for t in tools if t.name == name), None)
        if not tool:
            tool = next((t for t in agent.tools if t.name == name), None)

        if not tool:
            logger.warning(f"Tool not found: {name}")
            _reply(messages, tool_id, name, f"Tool not found: {name}")
            continue

        # Plan mode: hard-block mutating/exec tools, ABOVE the confirm gate.
        if agent._is_blocked_by_plan_mode(name, args):
            _reply(messages, tool_id, name, agent._plan_mode_block_message(name))
            continue

        # User hooks, BELOW plan mode and ABOVE the prompt: a deny is final, but an
        # allow only satisfies the confirmation gate — it can't reach past the
        # block above it or the server-side floors.
        pre = agent._run_hooks(hooks.PRE_TOOL_USE, name, args, quiet=quiet)
        if pre.denied:
            _reply(messages, tool_id, name, agent._hook_deny_message(name, pre.reason))
            continue

        # Hard gate: confirm destructive tools before running.
        if not (pre.allowed or agent._confirm_tool(name, args)):
            _reply(messages, tool_id, name, "User declined to run this command.")
            continue

        try:
            logger.debug(f"{log_label}: {name} with args: {args}")
            result = agent._invoke_tool(tool, name, args, quiet=quiet)
            content = agent._truncate_tool_result(str(result))
            post = agent._run_hooks(hooks.POST_TOOL_USE, name, args, str(result), quiet=quiet)
            _reply(messages, tool_id, name, _with_context(content, post))
            if activity is not None:
                activity.tool_result(name, str(result))
        except Exception as e:
            # `str(e) or repr(e)` — a bare TimeoutError has an empty str(), so this
            # logged the label and nothing else. (`e or …` would NOT work: an
            # exception is truthy.)
            logger.error(f"{log_label} error: {str(e) or repr(e)}")
            if activity is not None:
                activity.tool_error(name, str(e))
            failed = agent._run_hooks(
                hooks.POST_TOOL_USE_FAILURE, name, args, str(e) or repr(e), quiet=quiet
            )
            _reply(messages, tool_id, name, _with_context(agent._tool_error_message(name, e), failed))


def _with_context(content: str, outcome) -> str:
    """Append a hook's ``additionalContext`` to a tool result, if it added any."""
    return f"{content}\n\n[Hook] {outcome.context}" if outcome.context else content


def _reply(messages: List[BaseMessage], tool_id: str, name: str, content: str) -> None:
    """Answer one tool call. Every branch above goes through here — an
    unanswered tool_call_id makes the provider reject the next turn."""
    messages.append(ToolMessage(content=content, tool_call_id=tool_id, name=name))
