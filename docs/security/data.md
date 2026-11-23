# Data and privacy

MnemoAI runs on your computer. Model requests go to the providers you configure,
and connected tools can contact their own services.

## Local state

Configuration normally lives in `~/.mnemoai/config/`. Conversations, session
transcripts, memories, logs, and other state live under `~/.mnemoai/`, with
profile and model scoping where applicable.

See the [directory guide](../getting-started/mnemoai-directory.md) for the exact
paths and which files you or the assistant maintain.

- [Session recording](../features/sessions.md#resuming-a-session) is automatic
  unless disabled. Its retention is controlled by `SESSION_MAX_AGE_DAYS`.
- `/save` creates a conversation you keep deliberately; saved conversations do
  not expire with recorded sessions.
- [Log retention](../development/troubleshooting.md#read-the-logs) is configured
  separately.
- [Playbook learning](../guides/memory.md#ace-playbook-agentic-context-engineering)
  is local-only and makes no reflection or embedding requests.

## Model and service requests

Your selected chat provider receives the context supplied for a request. That
can include conversation history, instructions, attached files, and tool results.
Vision and embedding requests use their configured model providers.

For local inference, configure the relevant chat, vision, and embedding models
with local providers. Choosing a local chat model alone does not change the
providers used by other model areas.

[Model providers](../configuration/models.md),
[models for specific tasks](../configuration/area-models.md), and
[embedding models](../configuration/embeddings.md) explain those settings.

Optional web search, crawling, external MCP servers, and tool hooks have their
own connections and behavior. Review them in `/features`, `/mcp`, and `/hooks`.

## Share deliberately

[`/export`](../features/sharing.md#exporting-a-transcript) creates a readable
transcript. Review an export before sharing it. A conversation rewind changes
conversation state; it is not a redaction of the recorded transcript or a rollback
of files.

See [taking back a prompt](../features/sessions.md#taking-back-your-last-prompt)
and [permissions and safety](../guides/safety.md).
