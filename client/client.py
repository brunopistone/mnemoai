"""Strands client implementation."""

import asyncio
from datetime import date, datetime
from client.managers.agent_conversation_manager import AgentConversationManager
from client.managers.user_profile_manager import UserProfileManager
from client.managers.dpo_collector import DPOCollector
from client.ui.spinner import Spinner
import json
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from models.llm_controller import LLMController
import os
import re
from server.tools import count_tokens
import shutil
import sqlite3
from strands import Agent
from strands.tools.mcp import MCPClient
import sys
import traceback
from utils.formatting.code_formatter import CodeFormatter
from utils.config import config
from utils.logger import logger


class StrandsClient:

    def __init__(
        self,
        server_path: str = "server/server.py",
        verbose: bool = False,
    ) -> None:
        """
        Initialize the Strands client with a server configuration.

        Args:
            messages: Initial conversation messages
            server_path: Path to the server.py file to run the MCP server
        """
        self.verbose_mode = verbose  # Track verbose mode
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_path],
            env=None,
        )

        # Initialize profile manager first
        self.profile_manager = UserProfileManager()

        self.system_prompt = config.system_prompt

        if self.system_prompt:
            current_date = date.today().strftime("%Y-%m-%d")
            self.system_prompt = self.system_prompt.format(current_date=current_date)

            # Add user profile to system prompt
            profile_summary = self.profile_manager.get_profile_summary()
            if profile_summary:
                self.system_prompt = f"{self.system_prompt}\n\n{profile_summary}"

        # Initialize MCP client
        self.mcp_client = MCPClient(lambda: stdio_client(self.server_params))

        # Initialize session ID (used by RAG and chat interface)
        profile_name = config.get("PROFILE", {}).get("NAME", "default")
        self.session_id = f"{profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.agent = None
        self.tools = None
        self.llm_controller = LLMController(verbose=self.verbose_mode)
        self.llm_controller.initialize_model()
        self.model = None

        self.conversation_manager = AgentConversationManager(
            max_tokens=config.get("MAX_CONVERSATION_TOKENS", 1024 * 4)
        )
        self.dpo_collector = DPOCollector()
        self.dpo_mode = False  # Toggle for DPO collection
        self.spinner = Spinner()
        self.first_token_received = False
        self.visible_content = None

    # Custom callback handler to control verbosity
    def __minimal_callback_handler(self, **kwargs) -> None:
        """Handle streaming events without showing thinking content.

        Args:
            **kwargs: Event data from streaming response
        """
        # Initialize code formatter on messageStart
        if "event" in kwargs and (
            "contentBlockStart" in kwargs["event"] or "messageStart" in kwargs["event"]
        ):
            if hasattr(self, "_code_formatter_verbose"):
                self._code_formatter_verbose = CodeFormatter()

        # Stop spinner only when first actual data arrives
        if not self.first_token_received:
            if "data" in kwargs and kwargs["data"]:
                self.spinner.stop()
                self.first_token_received = True

        # Stop spinner when tool call starts
        if "message" in kwargs and kwargs["message"].get("role") == "assistant":
            content = kwargs["message"].get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("toolUse"):
                        self.spinner.stop()
                        break

        # Restart spinner after tool execution completes
        if (
            "message" in kwargs
            and kwargs["message"].get("role") == "user"
            and "toolResult" in str(kwargs["message"].get("content", ""))
        ):
            # Tool result received, restart spinner for final response
            self.first_token_received = False
            self.spinner.start()

        if "data" in kwargs:
            data = kwargs["data"]

            # Initialize state if not exists
            if not hasattr(self, "_in_thinking_minimal"):
                self._in_thinking_minimal = False
                self._code_formatter_minimal = CodeFormatter()

            # Check for thinking tags
            if "<thinking>" in data:
                self._in_thinking_minimal = True
                before_thinking = data.split("<thinking>")[0]
                if before_thinking:
                    self._code_formatter_minimal.process_chunk(before_thinking)

                after_thinking = (
                    data.split("<thinking>", 1)[1] if "<thinking>" in data else ""
                )

                if "</thinking>" in after_thinking:
                    # Skip thinking content, only process after
                    after_thinking_end = after_thinking.split("</thinking>", 1)[1]
                    if after_thinking_end:
                        self._code_formatter_minimal.process_chunk(after_thinking_end)
                    self._in_thinking_minimal = False

            elif "</thinking>" in data and self._in_thinking_minimal:
                # Skip thinking content, only process after
                after_thinking = data.split("</thinking>", 1)[1]
                if after_thinking:
                    self._code_formatter_minimal.process_chunk(after_thinking)
                self._in_thinking_minimal = False

            elif self._in_thinking_minimal:
                # Inside thinking, skip this chunk
                pass

            else:
                self._code_formatter_minimal.process_chunk(data)

    # Custom callback handler that shows all content including reasoning
    def __verbose_callback_handler(self, **kwargs) -> None:
        """Handle streaming events showing all content including thinking.

        Args:
            **kwargs: Event data from streaming response
        """
        # Initialize code formatter on messageStart
        if "event" in kwargs and (
            "contentBlockStart" in kwargs["event"] or "messageStart" in kwargs["event"]
        ):
            if hasattr(self, "_code_formatter_verbose"):
                self._code_formatter_verbose = CodeFormatter()

        # Stop spinner only when first actual data arrives
        if not self.first_token_received:
            if "data" in kwargs and kwargs["data"]:
                self.spinner.stop()
                self.first_token_received = True

        # Stop spinner when tool call starts
        if "message" in kwargs and kwargs["message"].get("role") == "assistant":
            content = kwargs["message"].get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("toolUse"):
                        self.spinner.stop()
                        break

        # Restart spinner after tool execution completes
        if (
            "message" in kwargs
            and kwargs["message"].get("role") == "user"
            and "toolResult" in str(kwargs["message"].get("content", ""))
        ):
            # Tool result received, restart spinner for final response
            self.first_token_received = False
            self.spinner.start()

        if "data" in kwargs:
            data = kwargs["data"]

            # Initialize state if not exists
            if not hasattr(self, "_in_thinking_verbose"):
                self._in_thinking_verbose = False
                self._code_formatter_verbose = CodeFormatter()

            # Check for thinking tags
            if "<thinking>" in data:
                self._in_thinking_verbose = True
                before_thinking = data.split("<thinking>")[0]
                if before_thinking:
                    self._code_formatter_verbose.process_chunk(before_thinking)

                after_thinking = (
                    data.split("<thinking>", 1)[1] if "<thinking>" in data else ""
                )

                if "</thinking>" in after_thinking:
                    thinking_content = after_thinking.split("</thinking>")[0]
                    after_thinking_end = after_thinking.split("</thinking>", 1)[1]

                    if thinking_content:
                        print(f"\033[90m{thinking_content}\033[0m", end="", flush=True)

                    if after_thinking_end:
                        self._code_formatter_verbose.process_chunk(after_thinking_end)

                    self._in_thinking_verbose = False
                else:
                    if after_thinking:
                        print(f"\033[90m{after_thinking}\033[0m", end="", flush=True)

            elif "</thinking>" in data and self._in_thinking_verbose:
                thinking_content = data.split("</thinking>")[0]
                after_thinking = data.split("</thinking>", 1)[1]

                if thinking_content:
                    print(f"\033[90m{thinking_content}\033[0m", end="", flush=True)

                # Ensure color is fully reset and add newline for clean state
                print("\033[0m\n", end="", flush=True)

                if after_thinking:
                    self._code_formatter_verbose.process_chunk(after_thinking)

                self._in_thinking_verbose = False

            elif self._in_thinking_verbose:
                print(f"\033[90m{data}\033[0m", end="", flush=True)

            else:
                self._code_formatter_verbose.process_chunk(data)

    def __count_context_tokens(self) -> int:
        """Count total tokens in the current conversation context.

        Returns:
            Total token count
        """
        total_tokens = 0

        # Count system prompt tokens
        if self.system_prompt:
            total_tokens += count_tokens(self.system_prompt)

        # Count conversation messages tokens by converting to JSON string
        if self.agent and hasattr(self.agent, "messages"):
            total_tokens += count_tokens(json.dumps(self.agent.messages, default=str))

        return total_tokens

    def clear_context(self) -> None:
        """Clear conversation history but keep system prompt."""
        system_msg = config.get("SYSTEM_PROMPT")
        profile_name = config.get("PROFILE", {}).get("NAME", "default")

        self.agent.messages.clear()
        self.session_id = f"{profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if system_msg:
            current_date = date.today().strftime("%Y-%m-%d")
            system_msg = system_msg.format(current_date=current_date)
            self.system_prompt = system_msg
            self.agent.system_prompt = system_msg

        # Flush RAG database when clearing context
        if config.get("ENABLE_RAG", False):
            self._flush_rag_store()

        self._flush_chunk_cache_store()

    def _initialize_rag_session(self) -> None:
        """Initialize RAG session at application startup."""
        try:
            user_home = os.path.expanduser("~")
            profile_name = config.get("PROFILE", {}).get("NAME", "default")
            rag_dir = os.path.join(user_home, "agent-conversations", profile_name)
            os.makedirs(rag_dir, exist_ok=True)

            session_file = os.path.join(rag_dir, "rag_session_id.txt")
            with open(session_file, "w") as f:
                f.write(self.session_id)

            logger.debug(f"RAG session initialized: {self.session_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize RAG session: {e}")

    def _initialize_chunk_cache(self) -> None:
        """Initialize chunk cache DB at application startup."""
        try:
            user_home = os.path.expanduser("~")
            profile_name = config.get("PROFILE", {}).get("NAME", "default")
            rag_dir = os.path.join(user_home, "agent-conversations", profile_name)
            os.makedirs(rag_dir, exist_ok=True)

            # Write session_id to file for MCP subprocess to read (same as RAG)
            session_file = os.path.join(rag_dir, "chunk_session_id.txt")
            with open(session_file, "w") as f:
                f.write(self.session_id)

            # Create session-specific DB
            db_path = os.path.join(rag_dir, f"chunk_cache_{self.session_id}.db")
            conn = sqlite3.connect(db_path)
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                CREATE TABLE IF NOT EXISTS chunk_cache (
                    key TEXT PRIMARY KEY,
                    summary TEXT,
                    updated_at TEXT
                )
                """
                )
                conn.commit()
                logger.debug(f"Chunk cache initialized: {os.path.basename(db_path)}")
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"Failed to initialize chunk cache: {e}")

    def _flush_chunk_cache_store(self) -> None:
        """Flush the RAG database and session-specific chunk cache."""
        try:
            from server.tools.readers.chunking_helper import reset_session_chunk_cache

            reset_session_chunk_cache()

            # Delete persisted session files
            user_home = os.path.expanduser("~")
            profile_name = config.get("PROFILE", {}).get("NAME", "default")
            rag_dir = os.path.join(user_home, "agent-conversations", profile_name)

            # Remove session-specific files (rag_store and chunk_cache with session_id)
            if os.path.exists(rag_dir):
                for file in os.listdir(rag_dir):
                    # Delete RAG store files and session-specific chunk cache
                    if file.startswith("chunk_cache_"):
                        file_path = os.path.join(rag_dir, file)
                        try:
                            os.remove(file_path)
                            logger.debug(f"Deleted session file: {file}")
                        except Exception as e:
                            logger.debug(f"Failed to delete {file}: {e}")

            logger.debug("Session reset - Chunk cache store and chunk cache cleared")
        except Exception as e:
            logger.warning(f"Failed to reset session: {e}")

    def _flush_rag_store(self) -> None:
        """Flush the RAG database and session-specific chunk cache."""
        try:
            if config.get("ENABLE_RAG", False):
                from server.tools.rag import reset_session_rag

                reset_session_rag()

            # Delete persisted session files
            user_home = os.path.expanduser("~")
            profile_name = config.get("PROFILE", {}).get("NAME", "default")
            rag_dir = os.path.join(user_home, "agent-conversations", profile_name)

            # Remove session-specific files (rag_store and chunk_cache with session_id)
            if os.path.exists(rag_dir):
                for file in os.listdir(rag_dir):
                    # Delete RAG store files/directories and session-specific chunk cache
                    if file.startswith("rag_store_"):
                        file_path = os.path.join(rag_dir, file)
                        try:
                            if os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                                logger.debug(f"Deleted session directory: {file}")
                            else:
                                os.remove(file_path)
                                logger.debug(f"Deleted session file: {file}")
                        except Exception as e:
                            logger.debug(f"Failed to delete {file}: {e}")

            logger.debug("Session reset - RAG store and chunk cache cleared")
        except Exception as e:
            logger.warning(f"Failed to reset session: {e}")

    def save_conversation(self, timestamp: str = None) -> None:
        """Save conversation to file.

        Args:
            timestamp: Optional timestamp for filename (default: current time)
        """
        if self.agent and self.agent.messages:
            # Use profile-based path with conversations subdirectory
            user_home = os.path.expanduser("~")
            profile_name = config.get("PROFILE", {}).get("NAME", "default")
            save_dir = os.path.join(
                user_home, "agent-conversations", profile_name, "conversations"
            )
            os.makedirs(save_dir, exist_ok=True)

            if not timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"conversation_{timestamp}.json"
            filepath = os.path.join(save_dir, filename)

            try:
                # Create conversation data with metadata
                conversation_data = {"messages": [], "tools": []}

                # Add system prompt as first message if it exists
                if self.system_prompt:
                    system_message = {
                        "role": "system",
                        "content": [{"text": self.system_prompt}],
                    }
                    conversation_data["messages"].append(system_message)

                # Add all conversation messages
                conversation_data["messages"].extend(self.agent.messages)

                # Add tools information if available
                if self.tools:
                    for tool in self.tools:
                        tool_info = {}

                        # Try different possible attribute names
                        if hasattr(tool, "name"):
                            tool_info["name"] = tool.name
                        elif hasattr(tool, "tool_name"):
                            tool_info["name"] = tool.tool_name
                        elif hasattr(tool, "__name__"):
                            tool_info["name"] = tool.__name__
                        else:
                            tool_info["name"] = str(tool)

                        # Try to get description
                        if hasattr(tool, "description"):
                            tool_info["description"] = tool.description
                        elif hasattr(tool, "__doc__"):
                            tool_info["description"] = tool.__doc__

                        # Try to get arguments/parameters
                        if hasattr(tool, "input_schema"):
                            tool_info["arguments"] = tool.input_schema
                        elif hasattr(tool, "parameters"):
                            tool_info["arguments"] = tool.parameters
                        elif hasattr(tool, "args"):
                            tool_info["arguments"] = tool.args
                        elif hasattr(tool, "schema"):
                            tool_info["arguments"] = tool.schema

                        conversation_data["tools"].append(tool_info)

                with open(filepath, "w") as f:
                    json.dump(conversation_data, f, indent=2, default=str)
                print(f"Conversation saved to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save conversation: {e}")

    def save_conversation_with_quality(
        self, timestamp: str = None, quality_markers: list = None
    ) -> None:
        """Save conversation with quality markers for training data.

        Args:
            timestamp: Optional timestamp for filename (default: current time)
            quality_markers: List of quality labels for each message (e.g., ['good', 'unlabeled'])
        """
        if self.agent and self.agent.messages:
            # Use profile-based path with conversations subdirectory
            user_home = os.path.expanduser("~")
            profile_name = config.get("PROFILE", {}).get("NAME", "default")
            save_dir = os.path.join(
                user_home, "agent-conversations", profile_name, "conversations"
            )
            os.makedirs(save_dir, exist_ok=True)

            if not timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"conversation_{timestamp}.json"
            filepath = os.path.join(save_dir, filename)

            try:
                conversation_data = {
                    "messages": [],
                    "tools": [],
                    "quality_markers": quality_markers or [],
                }

                if self.system_prompt:
                    system_message = {
                        "role": "system",
                        "content": [{" text": self.system_prompt}],
                    }
                    conversation_data["messages"].append(system_message)

                conversation_data["messages"].extend(self.agent.messages)

                # Get tools metadata from self.tools
                if self.tools:
                    for tool in self.tools:
                        try:
                            # Access the underlying MCP tool
                            mcp_tool = (
                                tool.mcp_tool if hasattr(tool, "mcp_tool") else tool
                            )

                            # Get parameters with better formatting
                            parameters = {}
                            if hasattr(mcp_tool, "inputSchema"):
                                schema = mcp_tool.inputSchema
                                properties = schema.get("properties", {})

                                # Clean up parameter descriptions
                                for param_name, param_info in properties.items():
                                    # Use description from schema if available, otherwise use title
                                    desc = param_info.get("description", "")
                                    if desc.startswith("Property "):
                                        # Generic description, try to get from title or just use param name
                                        desc = f"{param_name}: {param_info.get('type', 'any')}"

                                    parameters[param_name] = {
                                        "type": param_info.get("type", "string"),
                                        "description": desc,
                                        "default": (
                                            param_info.get("default")
                                            if "default" in param_info
                                            else None
                                        ),
                                    }

                            tool_info = {
                                "name": (
                                    mcp_tool.name
                                    if hasattr(mcp_tool, "name")
                                    else tool.tool_name
                                ),
                                "description": (
                                    mcp_tool.description
                                    if hasattr(mcp_tool, "description")
                                    else ""
                                ),
                                "parameters": parameters,
                            }
                            conversation_data["tools"].append(tool_info)
                        except Exception as e:
                            logger.error(f"Error extracting tool info: {e}")
                            continue

                with open(filepath, "w") as f:
                    json.dump(conversation_data, f, indent=2, default=str)
                print(f"Conversation saved to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save conversation: {e}")

    def load_conversation(self, file_path: str) -> bool:
        """Load conversation from file, excluding system prompt and tools.

        Args:
            file_path: Path to the conversation JSON file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Expand user path and check if file exists
            normalized_path = os.path.expanduser(file_path.strip())
            if not os.path.exists(normalized_path):
                logger.error(f"File not found: {normalized_path}")
                return False

            # Load conversation data
            with open(normalized_path, "r") as f:
                conversation_data = json.load(f)

            # Handle both old format (list) and new format (dict)
            if isinstance(conversation_data, list):
                messages = conversation_data
            else:
                messages = conversation_data.get("messages", [])

            # Filter out system messages and load only user/assistant messages
            conversation_messages = []
            for message in messages:
                if message.get("role") != "system":
                    conversation_messages.append(message)

            # Clear current conversation and load the saved one
            if self.agent:
                self.agent.messages.clear()
                self.agent.messages.extend(conversation_messages)
                logger.info(
                    f"Loaded {len(conversation_messages)} messages from {normalized_path}"
                )

                token_count = self.__count_context_tokens()
                print(f"\n\033[90m[Context: {token_count} tokens]\033[0m")

                return True
            else:
                logger.error("Agent not initialized")
                return False

        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return False

    def start(self, verbose: bool = False) -> None:
        """Start the client and initialize the agent.

        Args:
            verbose: Enable verbose mode to show thinking process
        """
        try:
            self.verbose_mode = verbose  # Store verbose mode

            with self.mcp_client:
                # Get tools from MCP server
                self.tools = self.mcp_client.list_tools_sync()

                # Initialize RAG session after MCP server is ready (if enabled)
                if config.get("ENABLE_RAG", False):
                    self._initialize_rag_session()

                # Initialize chunk cache DB
                self._initialize_chunk_cache()

                self.model = self.llm_controller.get_model()

                if not verbose:
                    additional_params = {
                        "callback_handler": self.__minimal_callback_handler
                    }
                else:
                    additional_params = {
                        "callback_handler": self.__verbose_callback_handler
                    }

                # Create agent with tools
                self.agent = Agent(
                    model=self.model,
                    tools=self.tools,
                    system_prompt=self.system_prompt,
                    **additional_params,
                )
        except Exception as e:
            stacktrace = traceback.format_exc()
            logger.error(stacktrace)

            raise e

    def query(self, prompt: str) -> str:
        """
        Send a query to the Strands agent.

        Args:
            prompt: User's query

        Returns:
            Agent's response
        """
        if not self.agent:
            raise RuntimeError(
                "Client not started. Call start() or use with-statement first."
            )

        # Reset first token flag and start spinner
        self.first_token_received = False
        self.spinner.start()

        try:
            with self.mcp_client:
                # Call agent with prompt
                response = self.agent(prompt)

                # Flush any remaining buffered backticks
                if hasattr(self, "_code_formatter_minimal"):
                    self._code_formatter_minimal.flush()
                if hasattr(self, "_code_formatter_verbose"):
                    self._code_formatter_verbose.flush()

                asyncio.run(
                    self.conversation_manager.manage_messages(
                        self, self.model, self.agent
                    )
                )

                # Update user profile with conversation
                self.profile_manager.analyze_conversation(self.agent.messages)

                # Check if response is only thinking tags (no visible content)
                response_text = str(response)  # Convert AgentResult to string
                visible_content = re.sub(
                    r"<thinking>.*?</thinking>",
                    "",
                    response_text,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                visible_content = re.sub(
                    r"<think>.*?</think>",
                    "",
                    visible_content,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                visible_content = visible_content.strip()

                if not visible_content:
                    # Response has only thinking, add a visible message
                    response_text += "\n\nI apologize, but I need to provide a visible response. Could you please rephrase your request?"
                    print(
                        "\n\033[91m⚠️  Model provided only thinking without visible response\033[0m"
                    )

                    print(
                        "I apologize, but I need to provide a visible response. Could you please rephrase your request?"
                    )

                # Print token count in a clean format
                token_count = self.__count_context_tokens()
                print(f"\n\033[90m[Context: {token_count} tokens]\033[0m")

                return response_text
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully - cancel all pending tasks
            self.first_token_received = False
            self.spinner.stop()

            try:
                loop = asyncio.get_event_loop()
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
            except:
                pass

            # Reset MCP client to clean state
            try:
                self.mcp_client = MCPClient(lambda: stdio_client(self.server_params))
            except:
                pass

            return "Operation was cancelled."
        except Exception as e:
            # Handle other exceptions - MCP client will be closed by context manager
            self.first_token_received = False
            self.spinner.stop()
            raise e
        finally:
            # Ensure spinner is stopped
            self.first_token_received = False
            self.spinner.stop()
