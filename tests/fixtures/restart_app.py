"""Subprocess fixture: real pinned input, dialog teardown and os.execv, no LLM."""

import faulthandler
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from mnemoai import main
from mnemoai.client.agent.agent_activity import AgentActivityStore
from mnemoai.client.client import LangGraphClient
from mnemoai.client.managers.agent_conversation_manager import AgentConversationManager
from mnemoai.client.session_log import SessionLog
from mnemoai.client.ui import chat_interface
from mnemoai.client.ui.chat_interface import ChatInterface
from mnemoai.utils.config import config

config._config_data = {
    "MODEL_ID": {"TYPE": "ollama", "NAME": "unused-test-model"},
    "ENABLE_USER_PROFILE": False, "ENABLE_PLAYBOOK": False,
}
faulthandler.dump_traceback_later(15)


def make_client():
    client = LangGraphClient(verbose=False)
    client.agent = SimpleNamespace(messages=[], system_prompt="Fresh instructions", session_log=SessionLog())
    client.agent._activity = AgentActivityStore()
    client.system_prompt = client.agent.system_prompt
    client.conversation_manager = AgentConversationManager(max_tokens=8000)
    client.conversation_manager._session_blocks = lambda *_: []
    client.current_conversation_path = None
    client.auto_approve_mode = "off"
    client.plan_mode_active = False
    client.reviewer = None
    client.tools = []
    client.mcp_client = SimpleNamespace(shutdown=lambda: None)
    client._print_context_size = lambda: None
    return client


result_path = Path(os.environ["RESTART_TEST_RESULT"])
client = make_client()

if "--restart-state" in sys.argv:
    client.start = lambda *a, **kw: None
    import mnemoai.client.client

    mnemoai.client.client.LangGraphClient = lambda **kw: client
    main.StartupLoader = Mock()

    class RestoredUI(ChatInterface):
        def run_chat_loop(self, **kwargs):
            result_path.write_text(json.dumps({
                "pid": os.getpid(),
                "history": [m.content for m in self.client.agent.messages],
                "summary": self.client.conversation_manager.summary_text,
                "save_path": self.client.current_conversation_path,
                "draft": self._restart_draft,
                "argv": sys.argv,
                "tokens": getattr(self.client.agent, "_last_input_tokens", None),
            }))

    chat_interface.ChatInterface = RestoredUI
    main.cli()
else:
    client.agent.messages = [HumanMessage(content="saved question"), AIMessage(content="unsaved answer")]
    client.conversation_manager.summary_text = "Keep the original task."
    client.current_conversation_path = str(result_path.with_suffix(".conversation.json"))
    client.agent._last_input_tokens = 9000
    status = [1, 0] if os.environ.get("RESTART_TEST_DEFER") else [0, 0]
    client.agent.background_status = lambda: tuple(status)
    client.agent.has_undelivered_background = lambda: status[1] > 0

    class InitialUI(ChatInterface):
        def _dispatch(self, line):
            if not line.strip():
                client.agent.messages.append(AIMessage(content="delivered background report"))
                status[:] = [0, 0]
                return None
            return super()._dispatch(line)

    ui = InitialUI(client)

    def select_model():
        ui._pinned_reader.restore_draft("unsent draft")
        return SimpleNamespace(
            section="MODEL_ID", label="Chat model", model_name="updated model",
            model_type="bedrock", enabled_feature=False, follows_chat=False, parameters_reset=False,
        )

    chat_interface.run_model_override = select_model
    with create_pipe_input() as pipe:
        def type_command():
            while getattr(ui, "_pinned_reader", None) is None:
                time.sleep(0.01)
            pipe.send_text("/model\r")
            if status[0]:
                while getattr(ui, "_pending_restart_notice", None) is None:
                    time.sleep(0.01)
                status[:] = [0, 1]
                ui._pinned_reader.notify_background_complete()

        threading.Thread(target=type_command, daemon=True).start()
        with create_app_session(input=pipe, output=DummyOutput()):
            ui._run_pinned_loop()
