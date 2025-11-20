"""DPO data collector for generating preference pairs."""

from datetime import datetime
import json
import os
import re
import textwrap
from typing import Any, Dict, List
from utils.config import config
from utils.logger import logger


class DPOCollector:
    """Collects preference pairs for DPO training."""

    def __init__(self, session_timestamp: str = None) -> None:
        """Initialize DPO collector.

        Args:
            session_timestamp: Optional session timestamp for file naming
        """
        # Use profile-based path
        user_home = os.path.expanduser("~")
        profile_name = config.get("PROFILE", {}).get("NAME", "default")
        self.save_dir = os.path.join(user_home, "agent-conversations", profile_name)
        self.dpo_dir = os.path.join(self.save_dir, "dpo_pairs")
        os.makedirs(self.dpo_dir, exist_ok=True)

        # Use session timestamp for single file per session
        self.session_timestamp = session_timestamp or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        self.session_file = os.path.join(
            self.dpo_dir, f"dpo_pairs_{self.session_timestamp}.jsonl"
        )

    @staticmethod
    def build_conversation(messages: List[Dict]) -> List[Dict]:
        """Build conversation array from agent messages with separated reasoning and tool calls.

        Args:
            messages: Agent messages list

        Returns:
            Formatted conversation array with role, content, reasoning_content, tool_calls
        """
        conversation = []

        for msg in messages:
            if msg["role"] == "user":
                content = msg.get("content", [])
                # Check if this is a tool result
                if isinstance(content, list) and any(
                    "toolResult" in str(c) for c in content
                ):
                    # Add as tool role
                    for item in content:
                        if "toolResult" in item:
                            tool_content = item.get("toolResult", {}).get("content", [])
                            if isinstance(tool_content, list):
                                tool_text = "\n".join(
                                    [str(c.get("text", c)) for c in tool_content]
                                )
                            else:
                                tool_text = str(tool_content)
                            conversation.append({"role": "tool", "content": tool_text})
                else:
                    # Regular user message
                    if isinstance(content, list):
                        text = content[0].get("text", "") if content else ""
                    elif isinstance(content, str):
                        text = content
                    else:
                        continue
                    if text:
                        conversation.append({"role": "user", "content": text})

            elif msg["role"] == "assistant":
                content_items = msg.get("content", [])
                if not isinstance(content_items, list):
                    content_items = [{"text": content_items}] if content_items else []

                reasoning = ""
                text_content = ""
                tool_calls = []

                for item in content_items:
                    if "text" in item:
                        full_text = item["text"]
                        # Extract reasoning from <thinking> tags (both closed and unclosed)
                        thinking_match = re.search(
                            r"<thinking>(.*?)</thinking>", full_text, re.DOTALL
                        )
                        if thinking_match:
                            # Closed thinking tag
                            reasoning += thinking_match.group(1).strip() + "\n"
                            cleaned_text = re.sub(
                                r"<thinking>.*?</thinking>",
                                "",
                                full_text,
                                flags=re.DOTALL,
                            ).strip()
                            if cleaned_text:
                                text_content += cleaned_text + " "
                        elif "<thinking>" in full_text:
                            # Unclosed thinking tag - extract everything after it
                            parts = full_text.split("<thinking>", 1)
                            before_tag = parts[0].strip()
                            if before_tag:
                                text_content += before_tag + " "
                            if len(parts) > 1:
                                reasoning += parts[1].strip() + "\n"
                        else:
                            text_content += full_text + " "
                    elif "toolUse" in item:
                        tool_calls.append(
                            {
                                "name": item["toolUse"].get("name", ""),
                                "arguments": item["toolUse"].get("input", {}),
                            }
                        )

                # Clean up accumulated text
                text_content = text_content.strip()
                reasoning = reasoning.strip()

                if text_content or tool_calls:
                    assistant_msg = {"role": "assistant", "content": text_content}
                    if reasoning:
                        assistant_msg["reasoning_content"] = reasoning
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls
                    conversation.append(assistant_msg)

        return conversation

    async def generate_alternative_response(
        self, messages: List[Dict], model: Any, temperature: float = 0.9
    ) -> str:
        """Generate an alternative response with higher temperature and degraded prompt.

        Args:
            messages: List of conversation messages
            model: Model instance for generating response
            temperature: Temperature for response generation

        Returns:
            Alternative response text
        """
        response_text = ""

        # Temporarily increase temperature for diversity
        original_temp = model.temperature if hasattr(model, "temperature") else None
        if hasattr(model, "temperature"):
            model.temperature = temperature

        # Use degraded system prompt - explicitly instruct to generate bad/wrong answers
        degraded_prompt = """
            You are generating a REJECTED response for DPO training.

            CRITICAL RULES:
            1. Keep <think> tags EMPTY or with only 1 brief sentence
            2. Provide a response that is WRONG, INCOMPLETE, or only PARTIALLY CORRECT
            3. Give MINIMAL information (2-3 sentences maximum)
            4. Make INCORRECT assumptions or provide MISLEADING information
            5. Be VAGUE and avoid specific details
            6. Do NOT use available tools even when they would solve the problem
            7. Sound confident but provide low-value or incorrect guidance

            Examples of bad responses:
            - Giving generic advice instead of using a tool to get actual data
            - Providing outdated or incorrect information
            - Missing the main point of the question
            - Answering a different question than what was asked
            - Being overly brief and unhelpful

            Your goal: Generate a response that seems like an attempt to help but is clearly wrong, incomplete, or unhelpful.
        """

        # Clean up prompt
        degraded_prompt = textwrap.dedent(degraded_prompt).strip()

        try:
            async for event in model.stream(messages, system_prompt=degraded_prompt):
                if (
                    "contentBlockDelta" in event
                    and "delta" in event["contentBlockDelta"]
                    and "text" in event["contentBlockDelta"]["delta"]
                ):
                    response_text += event["contentBlockDelta"]["delta"]["text"]
        finally:
            # Restore original temperature
            if original_temp is not None and hasattr(model, "temperature"):
                model.temperature = original_temp

        return response_text.strip()

    def save_preference_pair(
        self,
        system_prompt: str,
        chosen: list,
        rejected: list,
        metadata: Dict = None,
    ) -> None:
        """Append a preference pair to the session file.

        Args:
            system_prompt: System prompt used for the conversation
            chosen: Conversation array with preferred response
            rejected: Conversation array with rejected response
            metadata: Optional metadata dictionary
        """
        pair_data = {
            "system": system_prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": metadata or {},
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

        try:
            with open(self.session_file, "a") as f:
                f.write(json.dumps(pair_data) + "\n")
        except Exception as e:
            logger.error(f"Failed to save DPO pair: {e}")

    def export_to_dpo_format(self, output_file: str = None) -> str:
        """Export all pairs to standard DPO training format.

        Args:
            output_file: Optional output file path

        Returns:
            Output file path or None on error
        """
        if not output_file:
            output_file = os.path.join(self.dpo_dir, "dpo_dataset.jsonl")

        pairs = []
        for filename in os.listdir(self.dpo_dir):
            if filename.startswith("dpo_pair_") and filename.endswith(".json"):
                filepath = os.path.join(self.dpo_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        pairs.append(json.load(f))
                except Exception as e:
                    logger.error(f"Failed to load {filename}: {e}")

        # Write in JSONL format
        try:
            with open(output_file, "w") as f:
                for pair in pairs:
                    f.write(json.dumps(pair) + "\n")
            print(f"\033[37mExported {len(pairs)} pairs to {output_file}\033[0m")
            return output_file
        except Exception as e:
            logger.error(f"Failed to export DPO dataset: {e}")
            return None
