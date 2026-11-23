# Research and write with documents

Use files as evidence, then ask MnemoAI to produce a document with explicit
requirements and sources.

## Start with the material

For small text files, attach them with [`@`](../features/files.md).
For PDF or DOCX files, give the file path and ask MnemoAI to read it.

```text
Read the project brief and the meeting notes in this directory. Draft a decision
memo with the agreed requirements, open questions, and sources for each claim.
Do not treat unresolved questions as decisions.
```

## Search a larger collection

For repeated questions across documents, enable RAG through `/features`,
configure an embedding model, and ask the assistant to ingest the relevant files.
See [Documents and the web](../guides/knowledge-and-web.md) and
[Embedding models](../configuration/embeddings.md).

Web search and crawling are separate optional capabilities. Enable and configure
them when external sources are part of the task.

## Define the output

Name the target file, required headings, and what the assistant should verify:

```text
Save the memo as docs/decision.md. Include Summary, Evidence, and Open questions.
Read the saved file back and check that every factual claim has a source.
```

Use [peer review](../guides/peer-review.md) for an optional second assessment.
Inspect the saved document yourself before sharing it.

## Keep useful context

Put stable project guidance in [steering files](../guides/memory.md#steering-steeringmd).
Use [memory and local learning](../guides/memory.md) for useful context that
should inform later work.
