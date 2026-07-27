<p align="center">
  <img src="https://raw.githubusercontent.com/brunopistone/mnemoai/main/images/mnemoai-logo.png" alt="Mnemo AI" width="120">
</p>

<h1 align="center">Mnemo AI</h1>

[![PyPI](https://img.shields.io/pypi/v/mnemoai-assistant.svg)](https://pypi.org/project/mnemoai-assistant/)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A local agentic AI assistant with MCP (Model Context Protocol) integration, RAG capabilities, and intelligent conversation management. Built on LangGraph with LangChain for multi-provider LLM support (Ollama, Amazon Bedrock, Bedrock Mantle, OpenAI, Anthropic, Amazon SageMaker AI, LiteLLM).

▶️ **[Watch the demo](https://github.com/brunopistone/mnemoai/blob/main/images/assistant-demo.gif)**

## 📖 Documentation

Full documentation is available at **https://brunopistone.github.io/mnemoai/**

- [Getting Started](https://brunopistone.github.io/mnemoai/getting-started/)
- [Usage & commands](https://brunopistone.github.io/mnemoai/guides/usage/)
- [Configuration](https://brunopistone.github.io/mnemoai/configuration/)
- [Orchestration & sub-agents](https://brunopistone.github.io/mnemoai/guides/orchestration/)
- [Memory & learning](https://brunopistone.github.io/mnemoai/guides/memory/)
- [Productivity Tools](https://brunopistone.github.io/mnemoai/guides/productivity-tools/)
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

- **🤖 Multi-Model Support**: Ollama (local), Amazon Bedrock, Bedrock Mantle, OpenAI, Anthropic (Claude), Amazon SageMaker AI (chat only — no tool calling), LiteLLM (100+ providers)
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

## 📄 License

Licensed under the MIT License — see the LICENSE file for details.

## 🤝 Contributing

This is a personal development project. Feel free to fork and adapt it to your needs; attribution to the original repository is appreciated but not required.
