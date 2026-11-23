# Development setup { #dependencies }

All Python dependencies are listed in `requirements.txt`. The new productivity tools use only standard library features:

| Tool             | Python Packages                 | External Tools     |
| ---------------- | ------------------------------- | ------------------ |
| TodoWrite        | Standard library only           | None               |
| Edit Tool        | Standard library only           | None               |
| Glob Search      | Standard library (`glob`)       | None               |
| Grep Search      | Standard library (`subprocess`) | ripgrep (required) |
| Error Handler    | Standard library (`functools`)  | None               |
| Git Safety       | Standard library (`subprocess`) | git                |
| Plan Mode        | Standard library (`json`, `os`) | None               |
| Background Tasks | Standard library (`threading`)  | None               |

**External Tools:**

- **ripgrep**: **Required** by the `grep_search` tool — there is no fallback. Install via system package manager (see [Recommended optional tools](../getting-started/installation.md#7-recommended-optional-tools)); without it `grep_search` returns an error telling you to install it. Every other tool works without it.

**Core Python Packages:**

- `langgraph`: Agent orchestration framework
- `langchain`, `langchain-core`: LLM abstraction layer
- `langchain-ollama`: Ollama integration
- `langchain-aws`: AWS Bedrock integration
- `langchain-openai`: OpenAI integration (also used for Bedrock Mantle OpenAI/Responses protocols)
- `langchain-anthropic`: Anthropic integration (Bedrock Mantle `anthropic` protocol)
- `aws-bedrock-token-generator`: Bearer-token auth for Bedrock Mantle
- `mcp`, `mcp[cli]`: Model Context Protocol
- `ollama`: Local LLM support
- `boto3`: AWS Bedrock/SageMaker
- `tiktoken`: Token counting
- `chromadb`, `faiss-cpu`: Vector stores for RAG
- `pypdf`, `python-docx`: Document readers
- `Pygments`: Code syntax highlighting
- `prompt_toolkit`: Interactive CLI
- `brave-search-python-client`: Web search
- `crawl4ai`: Web crawling
