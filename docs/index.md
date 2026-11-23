---
hide:
  - toc
---

<div class="docs-hero" markdown>
<div markdown>

<p class="docs-eyebrow">Mnemo AI documentation</p>

# Overview { #mnemo-ai }

Bring a task to your terminal. MnemoAI can gather context, work with your files,
run tools, and carry useful knowledge into the next conversation.

Use a local model or connect a hosted provider. You choose the models,
the context they receive, and the actions they can take.

[Quickstart](getting-started/index.md){ .md-button .md-button--primary }
[Explore use cases](use-cases/index.md){ .md-button }

</div>
<figure class="docs-preview" id="walkthrough">
  <video src="https://raw.githubusercontent.com/brunopistone/mnemoai/main/images/demo.mp4" poster="https://raw.githubusercontent.com/brunopistone/mnemoai/main/images/demo-poster.png" controls playsinline preload="none" aria-label="MnemoAI terminal walkthrough">
    <a href="https://github.com/brunopistone/mnemoai/blob/main/images/demo.mp4">Watch the MnemoAI terminal walkthrough</a>
  </video>
  <figcaption>Planning, code and tests, memory, queued messages, and cancellation.</figcaption>
</figure>
</div>

<a id="quick-start"></a>
<a id="typical-workflow"></a>

## Get started

Install MnemoAI, choose a provider, and try a first task in your project.

```bash
curl -fsSL https://raw.githubusercontent.com/brunopistone/mnemoai/main/install.sh | sh
mnemoai
```

The first-run wizard creates your configuration. For a local setup, install
[Ollama](https://ollama.com) and download a chat model first.
See [installation](getting-started/installation.md) for requirements and alternatives.

<a id="what-mnemo-ai-can-do"></a>
<a id="choose-your-path"></a>

## Explore the docs

<div class="grid cards" markdown>

- **Features**

    Manage conversations, queue work, use files and tools, and learn from past tasks.

    [Explore features →](features/index.md)

- **Configuration**

    Choose providers and models, set defaults, and add skills, hooks, or MCP servers.

    [Configure MnemoAI →](configuration.md)

- **Developers**

    Understand the architecture, run tests, and add a tool or model provider.

    [Build with MnemoAI →](development/index.md)

- **Security**

    Understand permissions, plan mode, local data, and what reaches a provider.

    [Review security controls →](security/index.md)

- **Use cases**

    Follow practical workflows for coding, document research, and ongoing work.

    [Find a workflow →](use-cases/index.md)

- **Resources**

    Find commands, exact tool parameters, troubleshooting, and release notes.

    [Browse resources →](resources/index.md)

</div>

## What's new

**Queue the next task while the current one runs.** Use Tab to queue,
Enter to steer, and Ctrl+Q to manage pending messages.
[Learn queue and cancellation controls](features/queue.md).

For published changes, see [releases and project links](resources/releases.md).

<a id="where-configuration-lives"></a>

## Make it yours

Your usual configuration file is `~/.mnemoai/config/config.yaml`.
Use `MNEMOAI_CONFIG=/path/to/config.yaml mnemoai` to select another one.
The [configuration guide](configuration.md) explains providers, feature controls,
and the files created during setup.
