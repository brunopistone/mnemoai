"""Tools package for MCP server."""

from .tools_manager import ToolManager

# Create global tool manager instance
tool_manager = ToolManager()

# Export commonly used functions and constants for backward compatibility
register_tools = tool_manager.register_tools
validate_file_path = tool_manager.validate_file_path
count_tokens = tool_manager.count_tokens
vision_model = tool_manager.get_vision_model()
vision_model_controller = tool_manager.vision_model_controller
