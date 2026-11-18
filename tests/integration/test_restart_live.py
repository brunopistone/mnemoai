"""Fresh provider/MCP clients continue the handoff after model configuration changes."""

import copy
import json
import uuid

import pytest
import yaml
from langchain_core.messages import AIMessage, HumanMessage

from mnemoai.client import restart_state
from mnemoai.client.client import LangGraphClient
from mnemoai.utils.config import config

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("section", ["MODEL_ID", "VISION_MODEL_ID"])
def test_new_model_client_continues_loaded_and_unsaved_context(
    live_client, tmp_path, monkeypatch, section, record_testsuite_property,
):
    alternate = config.get("AREA_MODELS", {}).get("ROUTER")
    if not isinstance(alternate, dict) or not alternate.get("NAME"):
        pytest.skip("requires a configured alternate router model for the model-change check")
    old_config = copy.deepcopy(config._config_data)
    new_config = copy.deepcopy(old_config)
    new_config[section] = {**old_config[section], **alternate}
    if new_config[section] == old_config[section]:
        pytest.skip("configured router and target model are identical")

    source = tmp_path / "loaded.json"
    source.write_text(json.dumps({"messages": [
        {"role": "user", "content": [{"text": "We are checking conversation continuity."}]},
        {"role": "assistant", "content": [{"text": "Ready."}]},
    ]}))
    original = source.read_bytes()
    assert live_client.load_conversation(str(source))
    codeword = "PINE-" + uuid.uuid4().hex[:10]
    live_client.agent.messages.extend([
        HumanMessage(content=f"The project's codeword is {codeword}. Keep it for our next exchange."),
        AIMessage(content="I will retain the codeword."),
    ])
    summary = "Earlier project decision: the required export format is CSV."
    assert live_client.conversation_manager.apply_restored_summary(live_client, live_client.agent, summary)
    snapshot = restart_state.write(live_client.snapshot_for_restart())

    new_config.update(ENABLE_ROUTING=False, ENABLE_ORCHESTRATION=False, ENABLE_REVIEW=False)
    isolated_config = tmp_path / "new-model.yaml"
    isolated_config.write_text(yaml.safe_dump(new_config))
    isolated_config.chmod(0o600)
    monkeypatch.setattr(config, "_config_data", new_config)
    monkeypatch.setenv("MNEMOAI_CONFIG", str(isolated_config))
    monkeypatch.setenv("MNEMOAI_INSTANCE_ID", uuid.uuid4().hex)
    fresh = LangGraphClient(verbose=False)
    try:
        fresh.start()
        assert fresh.restore_after_restart(restart_state.read(snapshot))
        assert fresh.agent._last_input_tokens is None
        assert fresh.current_conversation_path == str(source)
        assert fresh.conversation_manager.summary_text == summary
        assert source.read_bytes() == original
        assert fresh.snapshot_for_restart()["messages"] == restart_state.read(snapshot)["messages"]

        # A real tool call proves the newly started MCP connection works.
        document = tmp_path / "restart-mcp.txt"
        document.write_text("MCP is available after restoration.")
        reader = next(tool for tool in fresh.agent.tools if tool.name == "fs_read")
        assert "MCP is available" in str(reader.invoke({"path": str(document)}))
        # Neither the codeword nor the summary fact is repeated in this request.
        answer = fresh.query(
            "What is the project's codeword and required export format from our conversation? "
            "Return those two values only. Do not use tools."
        )
        assert codeword in answer and "CSV" in answer, answer
        assert fresh.agent._last_input_tokens
        assert fresh.agent.usage.totals()["calls"] > 0
        record_testsuite_property(f"restart_{section}", {
            "previous": old_config[section]["NAME"],
            "selected": new_config[section]["NAME"],
            "chat": new_config["MODEL_ID"]["NAME"],
            "live_input_tokens": fresh.agent._last_input_tokens,
        })
    finally:
        fresh.mcp_client.shutdown()
        snapshot.unlink(missing_ok=True)
