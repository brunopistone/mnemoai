# Check your installation { #checking-your-install }

`/doctor` inspects this install and reports what's wrong with it — the first thing
to run when something behaves strangely, and the thing to paste into an issue:

```
Doctor — 1 problem, 1 warning

  Install
    · mnemoai  1.12.0
    · python  3.12.14 (/opt/homebrew/opt/python@3.12/bin/python3.12)
    · platform  Darwin 25.6.0
    ✓ app home  ~/.mnemoai

  Configuration
    ✓ config.yaml  ~/.mnemoai/config/config.yaml
    ✓ prompts.yaml  ~/.mnemoai/config/prompts.yaml

  Provider
    ✓ model  bedrock: us.anthropic.claude-sonnet-4-5-20250929-v1:0
    ✓ aws credentials  resolved, region us-east-1
    ✓ prompt cache  on, 1h TTL

  Tools
    ✗ rg  not on PATH — grep_search will not work
      → Install rg.
    ✓ git  /opt/homebrew/bin/git
    ✓ bash  /opt/homebrew/bin/bash
    ✓ MCP tools  41 registered

  Features
    ✓ chromadb (chromadb)  installed

  State
    ✓ sessions here  10 recorded, kept 30 days
    · logs  ~/.mnemoai/logs/mnemoai.log — 34 KB, kept 7 days
    ! MEMORY.md  2192 / 2200 chars (~/.mnemoai/bpistone/MEMORY.md)
      → Nearly full; the next entry may push an older one out.
    · steering  ~/dev/project/CLAUDE.md — 14240 chars, injected every turn
```

Each line is `✓` fine, `!` worth knowing, `✗` broken, or `·` informational, and a
problem comes with the command that fixes it. What it covers:

- **Configuration** — which `config.yaml` and `prompts.yaml` are **actually loaded**.
  ("I edited `config.yaml` and nothing changed" is nearly always a different file
  being live — an `MNEMOAI_CONFIG` override or a checkout fallback — and this is the
  only place that tells you.)
- **Provider** — the model and protocol in use, whether its credentials resolve,
  whether the endpoint answers (a 2-second local probe for Ollama and
  OpenAI-compatible servers), and whether prompt caching applies to it. Keys are
  never printed, only whether one was found.
- **Tools** — `rg` (required by `grep_search`, with no fallback), `git`, `bash`, how
  many MCP tools loaded, and whether every server declared in `mcp.json` actually
  connected. A server that **failed** is a warning rather than being reported as "not
  configured", which is the distinction that sends you looking in the right place.
- **Features** — for each feature you switched on, whether what it needs is present:
  the vector store package for RAG, a `BRAVE_API_KEY` for web search.
- **State** — the files that grow and then bite: `MEMORY.md` near its cap (it starts
  dropping older entries), steering files and the size each one injects **every**
  turn, how many sessions this directory has recorded, and where the app log is
  (the terminal shows one line per error; the traceback behind it is only there).
  Your own [slash commands](../configuration/slash-commands.md#your-own-slash-commands) are listed here too, along
  with any file that was **skipped** and why — otherwise a rejected command is
  indistinguishable from a feature that doesn't work.

It's local, cheap and read-only: no model call, nothing written, no network beyond
that one local port probe.
