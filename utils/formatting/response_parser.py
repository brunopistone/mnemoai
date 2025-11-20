"""Utilities for parsing and formatting AI responses."""

from typing import Optional


def extract_answer(response: str) -> str:
    """Extract the answer from a response that contains <answer> tags.

    Args:
        response: The raw response from the AI

    Returns:
        The extracted answer or the original response if no tags are found
    """
    if "<answer>" in response and "</answer>" in response:
        answer_start = response.find("<answer>") + len("<answer>")
        answer_end = response.find("</answer>")
        return response[answer_start:answer_end].strip()

    # Handle closing thinking tags (with or without opening tags)
    if "</think>" in response:
        thinking_end = response.find("</think>") + len("</think>")
        answer = response[thinking_end:].strip()
        return answer if answer else response

    if "</thinking>" in response:
        thinking_end = response.find("</thinking>") + len("</thinking>")
        answer = response[thinking_end:].strip()
        return answer if answer else response

    return response


def extract_thinking(response: str) -> Optional[str]:
    """Extract the thinking process from a response that contains <think> tags.

    Args:
        response: The raw response from the AI

    Returns:
        The extracted thinking process or None if no tags are found
    """
    if "<thinking>" in response and "</thinking>" in response:
        think_start = response.find("<thinking>") + len("<thinking>")
        think_end = response.find("</thinking>")
        return response[think_start:think_end].strip()
    if "<think>" in response and "</think>" in response:
        think_start = response.find("<think>") + len("<think>")
        think_end = response.find("</think>")
        return response[think_start:think_end].strip()
    return None


def format_response(response: str) -> tuple[Optional[str], str]:
    """Format the AI response for display.

    Args:
        response: The raw response from the AI

    Returns:
        A tuple containing (thinking, answer) where thinking may be None
    """
    thinking = extract_thinking(response)
    answer = extract_answer(response)

    return thinking, answer
