# Quickstart { #getting-started }

Get MnemoAI running in your terminal and complete a first task.

## 1. Install

On macOS or Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/brunopistone/mnemoai/main/install.sh | sh
```

Follow any PATH hint printed by the installer, then open a terminal in a project:

```bash
cd /path/to/your/project
mnemoai
```

For pip, pipx, updates, and uninstall instructions, see
[Installation](installation.md).

## 2. Choose a model

The first-run wizard asks for a model provider and saves your configuration to
`~/.mnemoai/config/config.yaml`.

- **Local:** install [Ollama](https://ollama.com), download a chat model, and
  select it in the wizard. See [Ollama setup](installation.md#5-ollama-quick-setup).
- **Hosted:** select your provider and configure its credentials.
  See [Model providers](../configuration/models.md).

You can change settings later with `/config`, `/model`, `/params`, and `/features`.

## 3. Try a small task

```text
Read the README in this directory and explain how to run the project.
```

MnemoAI can use tools to inspect the project. When an action needs your approval,
the terminal asks before proceeding.

Use `/help` for commands, or type `/` to search them.
Continue with [Your first conversation](first-chat.md) and
[Working with MnemoAI](workflows.md).

## Next steps

- [Queue a task or steer work already running](../features/queue.md)
- [Resume a conversation](../features/sessions.md)
- [Understand permissions and plan mode](../guides/safety.md)
- [Check your installation](../resources/diagnostics.md)
