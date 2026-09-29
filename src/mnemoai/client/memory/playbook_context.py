"""Refresh generated playbook blocks without rewriting conversation history."""

import copy
import re

from langchain_core.messages import SystemMessage

from mnemoai.utils.logger import logger

PLAYBOOK_BLOCK_MARKER = "[Tool-use notes from past sessions]"
PLAYBOOK_END_MARKER = "[End tool-use notes]"
_LEGACY = re.compile(
    r"(?m)^" + re.escape(PLAYBOOK_BLOCK_MARKER) + r"\n"
    r"(?:(?:Noted after past (?:errors|successes):|  · [^\n]*)\n?)*"
)
_BOUNDED = re.compile(
    r"(?ms)^" + re.escape(PLAYBOOK_BLOCK_MARKER) + r"\n.*?^"
    + re.escape(PLAYBOOK_END_MARKER) + r"(?:\n|$)"
)


def split_prompt(text):
    prefix, marker, summary = text.partition("<conversation_summary>")
    prefix = _BOUNDED.sub("", prefix)
    return _LEGACY.sub("", prefix).strip(), marker + summary


def has_block(text):
    prefix = text.partition("<conversation_summary>")[0]
    return bool(_BOUNDED.search(prefix) or _LEGACY.search(prefix))


def refresh_messages(messages, provider):
    """Recheck before a model send, including retries and quiet worker streams."""
    def texts(message):
        if not isinstance(message, SystemMessage):
            return []
        if isinstance(message.content, str):
            return [message.content]
        return [
            block["text"] for block in message.content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]

    if provider is None or not any(has_block(text) for msg in messages for text in texts(msg)):
        return messages
    try:
        current = provider()
        if not isinstance(current, str):
            raise ValueError("Invalid playbook context")
    except Exception as exc:
        logger.warning("Playbook refresh unavailable (%s); stale notes omitted", type(exc).__name__)
        current = ""

    def replace(text):
        if not has_block(text):
            return text
        base, summary = split_prompt(text)
        return "\n\n".join(filter(None, (base, current, summary)))

    result = []
    for message in messages:
        if not any(has_block(text) for text in texts(message)):
            result.append(message)
            continue
        content = message.content
        if isinstance(content, str):
            content = replace(content)
            if not content:
                continue
        else:
            refreshed = []
            for block in copy.deepcopy(content):
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    generated = has_block(block["text"])
                    block["text"] = replace(block["text"])
                    if generated and not block["text"]:
                        continue
                refreshed.append(block)
            content = refreshed
            if not content:
                continue
        result.append(message.model_copy(update={"content": content}))
    return result
