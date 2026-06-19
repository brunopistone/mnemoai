<p align="center">
  <img src="https://raw.githubusercontent.com/brunopistone/mnemoai/main/images/mnemoai-logo.png" alt="Mnemo AI" width="120">
</p>

<h1 align="center">Mnemo AI</h1>

[![PyPI](https://img.shields.io/pypi/v/mnemoai-assistant.svg)](https://pypi.org/project/mnemoai-assistant/)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A local agentic AI assistant with MCP (Model Context Protocol) integration, RAG capabilities, and intelligent conversation management. Built on LangGraph with LangChain for multi-provider LLM support (Ollama, Amazon Bedrock, OpenAI, Anthropic, Amazon SageMaker AI, LiteLLM).

![Demo](https://raw.githubusercontent.com/brunopistone/mnemoai/main/images/assistant-demo.gif)

## 📖 Documentation

Full documentation is available at **https://brunopistone.github.io/mnemoai/**

- [Getting Started](https://brunopistone.github.io/mnemoai/getting-started/)
- [Usage](https://brunopistone.github.io/mnemoai/usage/)
- [Configuration](https://brunopistone.github.io/mnemoai/configuration/)
- [Advanced Features](https://brunopistone.github.io/mnemoai/advanced-features/)
- [Productivity Tools](https://brunopistone.github.io/mnemoai/productivity/)
- [Architecture](https://brunopistone.github.io/mnemoai/architecture/)
- [Development](https://brunopistone.github.io/mnemoai/development/)

## 🚀 Quick Start

```bash
pip install mnemoai-assistant   # or: uv tool install mnemoai-assistant
mnemoai                          # verbose (shows thinking); --no-verbose to hide
```

On first run, if no config is found, an interactive configurator launches and walks you through picking a provider, model, and feature toggles — then writes `~/.mnemoai/config/config.yaml`.

→ See the [Getting Started guide](https://brunopistone.github.io/mnemoai/getting-started/) for full setup.

## ✨ Key Features

- **🤖 Multi-Model Support**: Ollama (local), Amazon Bedrock, OpenAI, Anthropic (Claude), Amazon SageMaker AI, LiteLLM (100+ providers)
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
