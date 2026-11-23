---
hide:
  - toc
---

# Developers { #development }

Understand how MnemoAI works, set up a development checkout, and extend the
assistant without changing its public behavior unexpectedly.

<div class="grid cards" markdown>

- **Build and test**

    Install development dependencies, run the test tiers, and validate changes.

    [Development setup →](setup.md)

- **Understand the architecture**

    Follow the client, model controllers, memory, and MCP tool execution paths.

    [Read the architecture →](../getting-started/architecture.md)

- **Add capabilities**

    Implement a tool, file reader, or provider using the existing extension points.

    [Add tools and readers →](tools.md)

- **Maintain and release**

    Understand the public contract, versioning, and the release checklist.

    [Versioning and releases →](releases.md)

</div>

## Build and test

- [Development setup and dependencies](setup.md)
- [Unit and live integration testing](testing.md)
- [Contributing and license](contributing.md)

## Internals and extensions

- [Architecture](../getting-started/architecture.md)
- [Add tools and file readers](tools.md)
- [Model provider integration](providers.md)
- [Detailed source map](https://github.com/brunopistone/mnemoai/blob/main/ARCHITECTURE.md)

## Project maintenance

- [Versioning and release checklist](releases.md)
- [Archived review from October 2026](review-2026-10-13.md)

The archived review records a past validation; it is not a claim about the
current checkout or release.

## Find a previous section

Developer topics now have dedicated pages. Existing section links below
lead to the corresponding topic.

<a id="dependencies"></a>
- [📦 Dependencies](setup.md#dependencies)

<a id="development_1"></a>

<a id="testing"></a>
- [Testing](testing.md#testing)

<a id="stability-versioning"></a>
- [Stability & Versioning](releases.md#stability-versioning)

<a id="release-checklist"></a>
- [Release Checklist](releases.md#release-checklist)

<a id="adding-new-tools"></a>
- [Adding New Tools](tools.md#adding-new-tools)

<a id="adding-new-file-readers"></a>
- [Adding New File Readers](tools.md#adding-new-file-readers)

<a id="switching-model-providers"></a>
- [Switching Model Providers](providers.md#switching-model-providers)

<a id="adding-new-model-providers"></a>
- [Adding New Model Providers](providers.md#adding-new-model-providers)

<a id="license"></a>
- [📄 License](contributing.md#license)

<a id="contributing"></a>
- [🤝 Contributing](contributing.md#contributing)

<a id="acknowledgments"></a>
- [🙏 Acknowledgments](contributing.md#acknowledgments)
