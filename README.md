<p align="center">
  <img src="https://raw.githubusercontent.com/brunopistone/mnemoai/main/images/mnemoai-logo.png" alt="Mnemo AI" width="120">
</p>

<h1 align="center">Mnemo AI</h1>

[![PyPI](https://img.shields.io/pypi/v/mnemoai-assistant.svg)](https://pypi.org/project/mnemoai-assistant/)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A local agentic AI assistant with MCP (Model Context Protocol) integration, RAG capabilities, and intelligent conversation management. Built on LangGraph with LangChain for multi-provider LLM support (Ollama, MLX, Amazon Bedrock, Bedrock Mantle, OpenAI, Anthropic, Amazon SageMaker AI, LiteLLM).

<p align="center">
  <video src="https://github.com/user-attachments/assets/344440f1-2c7a-4f2f-a022-99cda7be5adb" controls width="720">
    <a href="https://github.com/user-attachments/assets/344440f1-2c7a-4f2f-a022-99cda7be5adb"><img src="https://raw.githubusercontent.com/brunopistone/mnemoai/main/images/demo-poster.png" alt="▶️ Watch the demo" width="720"></a>
  </video>
</p>

## 📖 Documentation

Full documentation is available at **https://brunopistone.github.io/mnemoai/**

- [Getting Started](https://brunopistone.github.io/mnemoai/getting-started/)
- [Usage & commands](https://brunopistone.github.io/mnemoai/guides/usage/)
- [Configuration](https://brunopistone.github.io/mnemoai/configuration/)
- [Orchestration & sub-agents](https://brunopistone.github.io/mnemoai/guides/orchestration/)
- [Memory & learning](https://brunopistone.github.io/mnemoai/guides/memory/)
- [Everyday tools](https://brunopistone.github.io/mnemoai/guides/productivity-tools/)
- [Architecture](https://brunopistone.github.io/mnemoai/getting-started/architecture/)
- [Development](https://brunopistone.github.io/mnemoai/development/)

## 🚀 Quick Start

```bash
pip install mnemoai-assistant   # or: uv tool install mnemoai-assistant
mnemoai                          # verbose (shows thinking); --no-verbose to hide
```

On first run, if no config is found, an interactive configurator launches and walks you through picking a provider, model, and feature toggles — then writes `~/.mnemoai/config/config.yaml`.

→ See the [Getting Started guide](https://brunopistone.github.io/mnemoai/getting-started/) for full setup.

## ✨ Key Features

- **🤖 Multi-Model Support**: Ollama (local), MLX (local, Apple Silicon), Amazon Bedrock, Bedrock Mantle, OpenAI, Anthropic (Claude), Amazon SageMaker AI (chat only — no tool calling), LiteLLM (100+ providers)
- **🔧 MCP Tool System**: Extensible tool architecture via Model Context Protocol
- **📚 RAG**: Automatic document indexing and semantic (hybrid) search
- **🧠 User Profile Learning**: Personalized responses learned from interactions
- **🧩 Episodic Memory**: Learns from successful task completions and retrieves similar solutions
- **📖 ACE Playbook**: Learns strategies from successes AND failures (Agentic Context Engineering)
- **🔍 Web Search & 🌐 Crawler**: Brave Search API + web page extraction with RAG ingestion
- **🖼️ Vision Support**: Image analysis with vision models
- **📁 File Operations & ✏️ Precise Editing**: Read/write/edit text, CSV, JSON, PDF, DOCX
- **🔎 Fast Search**: Glob + ripgrep content search (10-100x faster)
- **📋 Todo Tracking, 📝 Plan Mode & 🔄 Background Tasks**: Multi-step task management
- **⚡ Bash Execution & 🛡️ Git Safety**: Shell commands with smart error handling and guardrails
- **🪝 Tool Hooks**: Your own commands run before/after any tool call — format writes, deny paths, auto-approve safe commands ([docs](https://brunopistone.github.io/mnemoai/guides/hooks/))
- **⌨️ Your Own Slash Commands**: A markdown file per prompt you retype — `commands/review.md` becomes `/review <path>` ([docs](https://brunopistone.github.io/mnemoai/guides/usage/#your-own-slash-commands))
- **📎 `@`-File Mentions**: Type `@` to complete a path anywhere in your prompt; the file is read and sent with the question ([docs](https://brunopistone.github.io/mnemoai/guides/usage/#attaching-a-file-with))
- **⟲ Take Back a Prompt**: `/rewind` drops your last prompt and everything the turn produced — the conversation only, files on disk untouched ([docs](https://brunopistone.github.io/mnemoai/guides/usage/#taking-back-your-last-prompt))
- **🗂️ Workspace Reports**: `/files` lists what the session read, changed or attached; `/diff` shows uncommitted changes with this session's edits marked; `/copy` puts the last answer on the clipboard without the terminal's wrapping ([docs](https://brunopistone.github.io/mnemoai/guides/usage/#what-this-session-touched))

## 📄 License

Licensed under the MIT License — see the LICENSE file for details.

## 🤝 Contributing

This is a personal development project. Feel free to fork and adapt it to your needs; attribution to the original repository is appreciated but not required.
