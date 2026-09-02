# Changelog

All notable changes to **Mnemo AI** (PyPI: `mnemoai-assistant`) are documented
here. The format follows [Keep a Changelog](https://keepachangelog.com/), and
the project aims to follow [Semantic Versioning](https://semver.org/): until
1.0.0, minor versions may introduce features and occasional breaking changes;
from 1.0.0 on, breaking changes to the public surface (config keys, the
`mcp.json` schema, CLI commands, the package/CLI name) bump the major version.

## [Unreleased]

### Changed

- **The app starts on a fresh screen.** Launching in a terminal that already had
  output in it appended the banner directly under whatever was there — the tail
  of a build, the `Exiting...` line of the session you just left — so a new run
  read as a continuation of the old one, with the wordmark buried mid-screen
  instead of introducing anything. The launch now begins at the top of a blank
  screen. **Nothing is erased:** the previous output is scrolled away, so it is
  still there a flick of the wheel up — which is the difference from `/clear`,
  where discarding the conversation means discarding the scrollback behind it
  too. Only in a real terminal; a piped or redirected run writes no escape
  sequences, so logs and CI output stay clean.

## [1.19.1] — 2026-09-02

### Fixed

- **Describing an image works on Amazon Bedrock again — including with a model
  that already worked for chat.** With `VISION_MODEL_ID` pointing at an OpenAI GPT
  model on Bedrock, every `describe_image` call failed instantly with
  `Unsupported parameter: 'max_tokens' is not supported with this model` — the
  provider rejecting the request before it ever looked at the picture, on a
  configuration the app had accepted without complaint and that the very same
  model id handled fine as the chat model. Vision was the one path still talking
  to Bedrock through its older, per-model interface, where settings like
  `MAX_TOKENS` travel inside the request body and each model family defines that
  body differently: what Claude reads there, a GPT model refuses outright. It now
  uses the same unified interface the chat model talks to, which carries those
  settings as standard fields every family understands. Also brought in line
  with chat: the request timeout from `LLM.REQUEST_TIMEOUT` now applies here
  too, so a large image no longer runs into the AWS SDK's own 60s default, and
  the stream guard from 1.17.2 is applied for consistency. Verified live
  end-to-end on Bedrock with a GPT and a Claude vision model, across PNG, JPEG,
  GIF, WebP and BMP.
- **A vision model that can't be built no longer takes every other tool with
  it.** The tools server decided whether to offer `describe_image` by _building_
  the vision model and seeing whether one came back — so anything that made that
  build fail raised while the server was still registering, and that server is
  what holds **all** the tools. A mistyped `TYPE`, a local vision server that
  wasn't running, a region that doesn't carry the model, credentials that had
  expired: any of them turned one line in an optional config section into a
  session with no file reads, no shell, no search, and nothing on screen
  explaining why. The gate reads the **configuration** now instead of a built
  model: `describe_image` is offered whenever `VISION_MODEL_ID` is set, the model
  is built the first time an image is actually described, and a build that fails
  then reports itself as that one tool returning an error — which is all that was
  ever wrong.
- **Startup is ~2.5s shorter, and the tools server no longer loads two
  conflicting math runtimes at once.** Building the vision model pulls in a deep
  numerical stack (a tokenizer library and the tensor framework under it), which
  is why it has been deferred to first use since 1.8.7. It wasn't actually
  deferred: the registration gate above built it on every launch, and the image
  tool then bound its model at import time as well, so the cost was paid twice
  over before the first prompt appeared — with the deferral machinery in place
  and doing nothing. That stack is now genuinely absent until an image is
  described, which also closes a real hazard rather than just a delay: the tensor
  framework and the vector-search library each bundle their own copy of the same
  parallel runtime, and a process holding both aborts outright the moment a
  search runs. With document search enabled and a vision model configured — a
  perfectly ordinary pair of settings — every tools server was starting up
  holding both. Two smaller gates were leaking the same way and are closed too:
  web crawling and web search now load their dependencies only when those
  features are switched on.
- **Two parallel image descriptions no longer build two vision models.** First
  use is now a tool call, and tool calls run on concurrent threads, so a wave of
  sub-agents describing several images at once could each start their own build.
  One is built, once, and the rest wait for it.

### Security

- **The pinned dependency set is current again, closing 11 published
  advisories.** `uv.lock` — the exact versions a development checkout and CI
  install — had drifted behind on two indirect dependencies: `nltk` (pulled in by
  the web-crawling stack) is now 3.10.3, resolving ten advisories including a
  critical one, and `cryptography` is now 50.0.1, resolving one rated high. This
  is lag rather than exposure: nothing in the project caps either package, so a
  fresh `pip install mnemoai-assistant` already resolved to these versions — the
  lock was simply describing an older resolution. Two companions moved with them
  because they were what held `cryptography` back (`langchain-litellm` 0.7.1,
  `pyopenssl` 26.4.0), and `langchain-core` came along at 1.6.1, which is what a
  new install has been getting for some time. The full suite, both tiers, passes
  unchanged against the new set. Five advisories remain open because **no fixed
  version exists upstream**: one in `nltk` (its model-artifact loaders can reach
  outside the directory they are given) and four in ChromaDB, whose latest
  release is still in range. The ChromaDB four all require its **HTTP server** —
  two are code injection through the collections endpoint, two concern
  permissions across tenants — and this app never starts one, using only the
  embedded on-disk client, so there is no endpoint to reach and no tenant
  boundary to cross.

## [1.19.0] — 2026-09-01

### Added

- **The models for the internal calls are set from the app now, not by hand.**
  `AREA_MODELS` shipped as YAML you had to know existed and write yourself, which
  for optional config is close to not shipping it: nothing in the setup flow
  mentioned it, and the one place you go to change a model — `/model` — listed chat,
  vision and embeddings only. All three areas are now offered where the models
  already are. The first-run configurator asks for a **router** model when query
  routing is on and an **orchestrator** model when orchestration is on — only when
  the toggle is on, since a model for a feature that never runs is dead config —
  and afterwards `/model` lists Router, Orchestrator and Summary beside the existing
  sections, with `/params` tuning an area that has its own block.
  Each one opens with the same question, **"use the same model as Chat?"**, and
  answering yes writes **nothing at all**. That is the design rather than a
  shortcut: an area with no entry follows `MODEL_ID`, so it goes on following it
  the next time you change the chat model — copying the block would instead freeze
  today's model in place under a second name. Answering no runs the ordinary
  provider flow (provider menu, model name, connection details, optional max output
  tokens), so an area can be set up on any provider the chat model could use,
  including a different one — a local model classifying while a hosted one answers.
  Removing an override is the same question answered yes, or a blank model name.
  An area whose feature is switched off is still listed, tagged
  `[query routing is off]`, and picking it offers to switch that feature on first;
  hiding the row would be indistinguishable from the feature not existing. `SUMMARY`
  is never gated, because compaction runs whatever the toggles say. Your
  `config.yaml` keeps its comments either way — including the commented
  `AREA_MODELS` example, which survives an override being added and removed again.

## [1.18.0] — 2026-09-01

### Added

- **`/auto` — a session mode that stops the `Proceed?` prompt from interrupting.**
  The confirmation gate is the right default and the wrong one for a task made of
  thirty small edits: the prompt is the only thing standing between the model and
  a tree you are watching change anyway, and answering `y` thirty times is not a
  decision, it is a keystroke tax. The existing escape hatches both overshoot —
  `a` at a prompt trusts a whole category for the rest of the session with no way
  back, and setting `REQUIRE_WRITE_CONFIRMATION: false` in `config.yaml` is
  permanent, global, and outlives the task by months. `/auto` is a **tiered,
  session-scoped, reversible** middle: `off` (the shipped behavior), `edits` (file
  writes whose target resolves inside the working directory), `writes` (any path,
  plus curated-memory updates), `all` (the above plus shell commands). Bare `/auto`
  steps to the next tier, `/auto <tier>` sets one, and the active tier is shown as
  a colored badge on the input line — green, amber, red as the ladder widens — so
  a mode you turned on for one task can't be forgotten silently. For a run that is
  meant to go uninterrupted from the first turn there is **`mnemoai --auto`**,
  which opens the session at the top tier; it is a bare switch, with the
  granularity left to the command.
  The tier that matters is `edits`, and its scoping is the point: it resolves
  symlinks on both sides and compares on a separator boundary, so a link pointing
  out of the tree cannot smuggle a write past the scope and a sibling directory
  sharing a name prefix (`/repo-old` next to `/repo`) is not inside it. A target
  it cannot resolve is treated as outside — whenever the scope is unknowable the
  prompt is the safe answer.
  **It replaces the keypress and nothing else.** Every gate above the prompt is
  decided before auto-approve is consulted, so no tier can reach past them: the
  server-side safety floor still refuses a catastrophic command or a system-path
  write, plan mode still hard-blocks the mutating tools (and `/plan` resets the
  tier to `off`, since "read-only" and "don't ask" cannot both be in force), and a
  tool hook's `deny` still wins. One category is auto-approved by **no** tier: the
  `git` gate, which is how a tool's own `requires_confirmation` refusal gets
  overridden — that is a server-side safety check the model is asking to waive, so
  it stays a human decision at every tier, and the switch-on notice says so rather
  than leaving you to infer it.
- **`AREA_MODELS` — a different model for the internal calls beside the answer.**
  A turn is not one request. Before the answer there is a classification (one
  route label), sometimes a decomposition (a JSON list of subtasks), and when
  history outgrows the window a compaction summary — none of them user-visible,
  and none of them necessarily deserving the model that writes prose. Picking a
  route is a job for a 1.7B; splitting a hard task is arguably worth a **bigger**
  model than the answer, because a bad split wastes the whole task; a summary
  wants throughput, not reasoning. Until now all three ran on `MODEL_ID`, so the
  cheap calls paid the expensive model's rate and its time-to-first-byte.
  Each area now takes an optional model of its own:

  ```yaml
  AREA_MODELS:
    ROUTER: qwen3.5:1.7b # query classification
    ORCHESTRATOR: qwen3.5:32b # task decomposition
    SUMMARY: # conversation compaction
      NAME: qwen3.5:1.7b
      TEMPERATURE: 0.3
  ```

  A value is either a bare model name or a **partial `MODEL_ID` block** merged over
  the main one and rebuilt through the ordinary provider dispatch — so an area may
  override `TYPE` and run on a different provider entirely (a local model for
  classification, a hosted one for the answer), and any param the target provider
  doesn't support is dropped for it exactly as it is for the main model.
  Absence is the default: **no section means the main model everywhere**, so no
  existing install changes and no config edit is required. A failure is a
  non-event by design — an area whose model can't be built logs once and falls back
  to the main model, because an unreachable side model must degrade to a working
  turn rather than end it. `/usage` attributes each area's tokens to the model that
  actually served them, and `/doctor` lists the areas running on their own model
  (plus any misspelled area name, which otherwise looks exactly like the setting
  having no effect). `/params` re-derives them in place, so an override can be
  added or removed without restarting. The **aggregator is deliberately not an
  area**: its output is the answer you read, so a model swap there would change the
  answer's voice rather than the cost of an internal step.

## [1.17.2] — 2026-08-31

### Fixed

- **A thinking model served over the OpenAI protocol now shows its thinking.**
  On a local MLX / llama-server / vLLM / LM Studio server with a reasoning parser
  configured, the answer appeared out of nowhere: no `Thought for Ns…` block, no
  gray inline reasoning, nothing in the transcript — while the same model on
  Ollama or Bedrock showed it. The reasoning was arriving and being thrown away
  one layer below the app. A reasoning parser splits the model's `<think>` block
  out of the answer and reports it in a _separate response field_, but that field
  is not part of the OpenAI schema, and the adapter for that protocol documents
  that it drops every non-standard field: only `tool_calls` and `function_call`
  survive. So the thinking was parsed, transmitted, and discarded — the one
  outcome worse than not separating it at all, since with the parser turned off
  the thinking at least reaches the screen as (mislabeled) prose.
  Recovered now, and **not for one server**: the field name is not standardized
  and differs per server _and_ per reasoning parser — `reasoning_content` (MLX,
  vLLM, llama-server, SGLang, DeepSeek), `reasoning` (mlx*lm's own server,
  OpenRouter), a structured `reasoning_details` list (OpenRouter), or
  `thinking`/`thinking_blocks` (Anthropic-shaped proxies), each of which may hold
  a plain string, a list of blocks, or a nested summary. All of them are read,
  in whatever shape, and normalized to the ONE key the turn view already consumes
  for Ollama and LiteLLM, so nothing downstream needs to know which server
  produced the turn. Encrypted/redacted reasoning blocks are skipped (they carry
  ciphertext, not prose), and a provider that sends the same thinking twice under
  two names yields it once rather than doubled. Both the streamed and
  non-streamed paths are covered, plus Bedrock Mantle's OpenAI-shaped protocols.
  Purely additive: a response with no reasoning field is untouched, so real
  OpenAI behaves exactly as before, and the recovered text is never echoed back
  to the server on the next turn — the adapter's outbound direction drops it,
  which matters for a chat template that would reject it.
  One deliberate side effect: reasoning now counts as the stream having \_started*
  (it is the model producing tokens), so the wide first-token window from 1.17.1
  narrows to the per-chunk one as soon as thinking begins, tightening dead-socket
  detection on exactly the turns that used to look idle the longest.
- **An OpenAI GPT model on Amazon Bedrock now completes a turn at all.** Every
  turn died immediately with `IndexError: list index out of range` — before a word
  of the answer, on any prompt, so those models were effectively unusable. Those
  models put their reasoning in the _encrypted_ form of Bedrock's reasoning block
  and emit it as the first block of every answer; the Converse adapter has a
  branch for both readable forms but none for the encrypted one, and it feeds
  each streamed block into an unguarded index of its own converter's result — so
  the block it cannot convert took the turn down. Since the failure is
  deterministic it was never retried either, and it only appears when streaming,
  which this app enables on purpose. An event the adapter cannot convert is now
  skipped rather than fatal — which is what the non-streamed path already did
  with the same block — so the answer, the token-by-token stream and tool calls
  all work. Encrypted reasoning still cannot be _displayed_: it is ciphertext,
  and the provider returns no readable text at any reasoning effort or summary
  setting, so no `Thought for Ns…` block appears for these models. The guard
  applies to every Converse model (a no-op for one that never sends such a
  block), and a genuinely unknown block type still surfaces as an error instead of
  being silently dropped. It asks the adapter what it can convert rather than
  keeping a list of its own, so it retires itself where upstream catches up:
  `langchain-aws` 1.7.4 (2026-08-26) grew the missing branch, and on it nothing is
  dropped at all — while an install on 1.7.3 or older, which this project still
  supports, keeps working.

## [1.17.1] — 2026-08-31

### Fixed

- **A turn against a local MLX (or any OpenAI-shaped) server no longer dies at
  `No stream data for 120s` and then retries forever.** Those servers accept a
  request and _immediately_ send a contentless `{"delta": {"role": "assistant"}}`
  — before prefill has begun — and the stream watchdog counted it as the first
  token. That is the one thing it must never do: the long window that covers
  prefill and reasoning (`REQUEST_TIMEOUT` + 30s) was handed back for the short
  per-chunk one while the model still hadn't produced a word, so a large prompt
  timed out against a perfectly healthy server, and since every retry re-sent the
  same prompt and re-paid the same doomed prefill, the turn could never complete —
  the exact trap the first-token window was added to fix, one layer down. Measured
  against a local MLX server on a 43k-token prompt: the priming chunk at +0.01s,
  the first real token at +300s, against a 120s window; on a 6k-token one, 28
  contentless chunks (the first at +12s) before the first token at +63s. The window
  now narrows only when a chunk carries something the model actually produced —
  content, a tool-call fragment, reasoning, a finish reason, real usage numbers —
  and it is a wall-clock deadline rather than a per-poll counter, so a server that
  emits keep-alives faster than the poll interval can't hold a dead turn open
  either. A chunk shape we can't classify still counts as a start, so nothing
  _delays_ dead-socket detection for an unknown provider.

- **A provider's error now reads as a sentence instead of the raw response body it
  arrived in.** A warning or error on screen is meant to be one line of the
  interface, with the whole record kept in `~/.mnemoai/logs/mnemoai.log` — but when
  the text comes from an HTTP provider it isn't prose, it's a serialized response:
  `Stream connection failed (Error code: 503 - {'detail': {'error': {'message':
"Failed to load on-demand model …", 'type': 'model_load_error', 'code': 503}}});
retrying turn on a fresh connection in 1.1s` — three levels of braces, two keys
  nobody reads, and no pointer to the log file, because the 500-character cap that
  exists for exactly this reason never engaged on a message this short. The
  envelope is now unwrapped down to the `message` inside it, in the one place every
  warning, error and UI line built from an exception passes through. The reason
  it's an unwrap rather than a tighter cap: **the actionable part sits at the end
  of the body** — in the case above, `No module named 'aiohttp'`, the whole
  explanation — so shortening the line by length would delete precisely the fact
  worth showing. Since something was dropped, the dim
  `(traceback → ~/.mnemoai/logs/mnemoai.log)` pointer now appears on these lines
  too, which is where the untouched body still is. Conservative by construction: a
  provider that words its failures in prose (botocore) and a body carrying no
  `message` are left exactly as they were.

## [1.17.0] — 2026-08-30

### Added

- **`TYPE: mlx` — a local MLX server as a first-class provider.** On Apple Silicon
  the fast way to run a quantized model is MLX, and reaching one already half-worked:
  `TYPE: openai` with `API_BASE: http://127.0.0.1:8000/v1` speaks the right protocol.
  What it could not do is anything MLX-specific. `TOP_K`, `MIN_P` and
  `REPETITION_PENALTY` are real knobs on that server but are not `ChatOpenAI` fields,
  so they were dropped on the way out with nothing said; `KEEP_ALIVE` — how long the
  server keeps a model resident after a request — had nowhere to go at all. `mlx` is
  now its own `TYPE` across **all three sections** (chat, vision, and
  `RAG.EMBED_MODEL_ID`): the connection is `HOST`/`PORT` (default `127.0.0.1:8000`, no
  hand-written `/v1` suffix), no key is required — a placeholder is sent, so a real
  `OPENAI_API_KEY` in your environment is never forwarded to a local server — and the
  MLX-only params travel in the request body where the server actually reads them.
  `API_BASE`/`API_KEY` stay available for a server behind a proxy or an auth layer.
  `KEEP_ALIVE` takes `30m`, `1h30m`, `500ms`, bare seconds, `0` to unload the moment
  the request finishes, or `-1` to pin, and it is set per section — which is what makes
  one server hosting chat + vision + embeddings practical: pin what you talk to
  constantly and let a rarely-used model fall out of memory instead of holding RAM.
  `/config` and `/model` prompt for host/port with MLX's own wording and defaults
  rather than Ollama's, `/params` offers `MIN_P` and `KEEP_ALIVE` (validated as a
  duration, so a typo is caught at the prompt instead of on the first request), and
  `/doctor` TCP-probes the server and tells you to start it rather than reporting a
  credential that does not exist. One knob is deliberately **absent**: `TOP_K` on
  `mlx` vision, because that server's multimodal handler never passes it to the
  sampler — config that looks effective and does nothing is worse than a knob that
  isn't offered. Token counting has no MLX-specific multiplier for the same reason:
  one server can serve any tokenizer family, so it takes the conservative 1.35
  fallback (override with `LLM.TOKEN_COUNTING.MLX_MULTIPLIER`).

### Fixed

- **The arrow keys move the selection in every picker, not just some of them.** In
  `--resume` and `/load` the `(*)` marker follows ↑/↓, so ↓↓Enter opens the third
  conversation. In the `/model` and `/config` dialogs it did not: the marker stayed
  on the row the dialog opened at while the arrows moved only the cursor, and Enter
  confirmed the marker — so arrowing down two rows and pressing Enter silently acted
  on the **first** row (measured on "Which model to change?": `↓↓Enter` returned
  _Chat_). Pressing Space committed the highlighted row first, but nothing on screen
  said so — and the keys reference has documented these dialogs as "arrow keys move,
  `Enter` confirms" all along. A single-choice list tracks its highlighted row
  separately from its committed value and only its own Enter/Space binding
  reconciles the two, which these dialogs override so that Enter confirms directly
  instead of needing a Tab-to-OK step. The conversation pickers were fixed for this
  in 1.8.4; the settings dialogs are built in a different module and kept the bug.
  Both opt into commit-on-move now, and a test reads the source so a picker added in
  a third module later can't reintroduce the split. Multi-select (`/features`) keeps
  Space to toggle on purpose: there, moving past a row must not tick it.

## [1.16.2] — 2026-08-29

### Changed

- **The demo is a video instead of a 10 MB GIF.** The README showed a 10.5 MB
  animated GIF, and since 1.0.0 it showed it as a bare link: GitHub serves `.gif`
  from `raw.githubusercontent.com` as `application/octet-stream`, which PyPI's
  image proxy refuses, so inline it rendered broken. It is now an H.264 video —
  the same session at 4x, 6.0 MB — embedded with `<video>` and a clickable poster
  (`images/demo-poster.png`) as the fallback child, so GitHub and the docs site
  play it inline while PyPI, which strips `<video>` but keeps its children, shows
  the poster linked to the video. The video is deliberately not committed: one
  served out of the repository cannot play on GitHub at all, because
  `raw.githubusercontent.com` is absent from the `media-src` of github.com's
  Content-Security-Policy. It is a repository attachment instead, which also
  keeps it out of every clone.

  Maintainer note: an attachment is served to anonymous visitors only while its
  URL is referenced from issue or pull-request content — a reference from a
  committed markdown file is not enough. Issue #2 holds that reference and must
  not be deleted, or the video 404s for everyone who is not logged in.

## [1.16.1] — 2026-08-28

### Changed

- **Launch is ten lines instead of a full screen.** Every start printed the whole
  command reference — 26 commands in a framed box, 31 of the 41 lines on screen
  before you had typed anything — and the box's own last row told you that `/help`
  brings it back. So the list now lives where that row said it did: **`/help` prints
  the full reference** (and `/` searches it as you type, `@` completes a path), while
  launch shows the wordmark, what this is, the **version**, and one line naming those
  three ways in. Nothing was removed — every command is still in `/help`, which also
  documents the keys that aren't commands — and the version is new there, since until
  now only `/doctor` could tell you which one you were running. Resuming a session
  gains the most: the conversation you came back for no longer sits behind a
  screenful of commands you weren't reading.

## [1.16.0] — 2026-08-28

### Added

- **`/rewind` — take back your last prompt.** Sometimes the prompt was the mistake:
  the wrong file, the wrong framing, a question that sent the whole turn somewhere
  you don't want it. Until now the only ways out were `/clear` (throw the session
  away) and `/compact` (summarize it) — neither of which undoes _one_ thing. This
  drops the last prompt **and everything the turn produced** from the live
  conversation and from the transcript, leaving the context exactly as it was the
  moment before you pressed Enter, and echoes the prompt it withdrew so you can see
  which turn went. It moves the **conversation only — files on disk are untouched**,
  the same boundary `/branch` draws, and the notice says so every time: a command
  that rolled back half a turn would be worse than one that tells you which half it
  does. Two rewinds in a row walk back two turns. It finds the last thing _you_
  typed, not the last message with your name on it, so a turn of tool results or an
  auto-delivered sub-agent report can't be mistaken for a prompt; and it refuses,
  with the reason, when the conversation compacted during or right after that turn —
  a summary that already stands for the turn cannot be un-summarized. The transcript
  stays append-only, so the withdrawal is recorded rather than cut out: a `/branch`
  can no longer fork at a withdrawn turn, but nothing you said is scrubbed from
  disk. What the session _learned_ from the turn does stay, and the notice names
  which stores that means — episodic memory keeps entries by similarity and the
  playbook merges a repeat strategy into an existing one, so there is nothing
  precise left to delete.
- **Your own slash commands.** A prompt you retype is now a file: every `*.md` in
  `~/.mnemoai/commands/` becomes a command you can type, and the file name is the
  command (`commands/review.md` → `/review`). The body is the prompt that gets
  sent, with `$ARGUMENTS` (or `$1` … `$9`) filled in from whatever you typed after
  the name — and if the body references no placeholder, your arguments are appended
  rather than dropped, so a one-line instruction plus a target needs no markup at
  all. Optional frontmatter (`description`, `argument_hint`) documents the command
  in the `/` menu and in a "Yours" group in `/help`; without it the first line of
  the body is used, because a plain markdown file must work with no ceremony.
  Expansion happens after every built-in, so a file can never shadow `/save` (one
  that tries is skipped, and `/doctor` names it with the reason instead of leaving
  you with a command that silently does nothing); an unknown `/thing` still reaches
  the model as prose. The model never learns a command was involved — the expansion
  _is_ the prompt — so a dim `⌘ /name · file.md` line records which file ran, and
  the turn is an ordinary one after that. Edits apply to the next line you type, no
  restart; files beginning with `_` or `.` are ignored so notes can live beside the
  commands. Read from the app home only, deliberately: what a name you type expands
  to must not be redefinable by a repo you clone. A worked `explain.md` and an
  authoring `_README.md` are installed for you, and refreshed on upgrade unless
  you've edited them.
- **Point at a file with `@`, and it comes with the question.** Typing `@` anywhere
  in your prompt completes paths as you type — a bare name searches the whole
  project by basename (`@chat_int` finds `src/mnemoai/client/ui/chat_interface.py`),
  while a fragment containing `/` or starting at `~` completes directory by
  directory, so a file outside the project is reachable too. On submit, every
  `@path` in the line is read and attached to the prompt, which is the difference
  that matters: the contents are there whether or not the model would have decided
  to go and look, so "does `@config.py` handle this" doesn't start with a round of
  searching. A mentioned directory contributes its listing rather than its files.
  The syntax is the one steering files already use, including `@`-references from
  inside a mentioned file's own text, and a reference that isn't a readable path is
  simply left in the prose — so `@staticmethod`, an `@handle` and an email address
  keep meaning what they say. Each mention is announced as a dim `@path · 412 lines`
  line, because attaching nothing looks exactly like attaching the right file and
  the second leaves the model guessing; a typo says `no such file` and the question
  is still asked. Bounded on purpose, since what rides in your own message can be
  summarized by compaction but never shrunk the way a tool result is: 10 files per
  line, a 60k-character total, and 20 000 characters per file (`MENTIONS.MAX_FILE_CHARS`,
  `0` to lift it) with a visible note where a file was cut. A mention is not a
  read — an edit to a mentioned file still needs the tool that reads it first.
- **The terminal now tells you when it wants you back.** A turn that ran longer
  than 30 seconds rings the **bell** and raises an **OSC 9 desktop notification**
  when it finishes (`mnemoai · done in 4m12s · project` — the directory is there
  because a popup arrives with no idea which terminal it came from). The same goes
  out when a confirmation prompt or a question is waiting on you, which is sent
  however short the turn has been: that one has the work stopped behind it. And
  when a background sub-agent's report lands while you're idle, since until now the
  only sign was the report itself appearing in a window you'd stopped watching.
  Inside tmux or screen the sequence is wrapped in a passthrough so it reaches the
  outer terminal rather than being swallowed by the multiplexer. Nothing is sent
  off a terminal (a pipe, CI) or for a turn **you** cancelled — your hand is
  already on the keyboard — and two notifications closer together than 10 seconds
  collapse into one, so a task confirming eight writes is one interruption rather
  than eight. Tunable and fully silenceable via `NOTIFY` (`AFTER_SECONDS`, `BELL`,
  `DESKTOP`).
- **`/files` — what this session actually touched.** A long turn scrolls its own
  record away, so after twenty minutes of work "which files did it change?" is a
  question you can only answer by re-reading the transcript. The report groups every
  file the conversation read, wrote or attached with `@`, newest first, with how many
  times each one came up — and it counts the quiet paths too: a sub-agent's edits and
  a parallel wave's reads are in there, because it is fed from the one place every
  tool call passes through. Two spellings of the same file (`./src/x.py`,
  `~/proj/src/x.py`) are one row, and the list survives compaction — a file
  summarized out of the context is still a file you touched. Reset by `/clear`.
- **`/diff` — the uncommitted changes, with yours marked.** `git status` cannot tell
  you which of the twelve dirty files came from this conversation and which were
  already dirty when you started; this can, because it cross-references the same
  ledger `/files` reads and puts a `✎` on the ones the session wrote. Staged,
  unstaged and untracked files in one list with `+`/`−` counts and a total; `/diff
<path>` shows that one file's unified diff, colored, with an untracked file
  rendered as the all-additions diff it effectively is. Read-only by construction —
  it runs `rev-parse`, `diff` and `ls-files`, and nothing else — so it can never
  stage, stash or check anything out; bounded, and when a list or a diff is cut it
  prints the exact git command that shows the rest.
- **`/copy` — the last answer, on the clipboard, without the line wrapping.**
  Selecting a streamed answer with the mouse takes the terminal's wrapping with it,
  which is why a copied code block so often has to be reflowed by hand. `/copy`
  takes the message as the model wrote it; `/copy code` narrows to its last fenced
  block (and names the language it copied), `/copy 2` reaches the answer before
  last. It uses a local helper when there is one (`pbcopy`, `wl-copy`, `xclip`,
  `xsel`, `clip.exe`) and falls back to **OSC 52**, so a terminal that supports it
  works with no helper installed at all — and **over SSH the terminal goes first**,
  since a helper on the far end would set the clipboard of a machine nobody is
  sitting at. Inside tmux or screen the sequence is wrapped in a passthrough. The
  notice says what was copied, how big it was and which transport carried it,
  because a clipboard operation is otherwise invisible until you paste.

### Fixed

- **A conversation you compacted and then closed without asking anything else no
  longer comes back at its full size.** A session file is deleted at exit when it
  holds no turn of its own — it was written the moment you launched, so quitting
  straight away would otherwise leave an empty conversation in the `--resume` list.
  But resuming a chat, running `/compact` and then quitting hit exactly that rule:
  the only thing in the new file was the compaction, and deleting it threw away the
  one record of the smaller context. The next resume picked up the parent instead
  and re-inflated the whole pre-compaction history, so the summary you had already
  paid for was summarized again. A file whose records _shrank_ the context is now
  kept and offered, however few turns it has.

## [1.15.1] — 2026-08-28

### Fixed

- **A running plan now ticks the steps you're looking at, instead of listing them
  again underneath.** A multi-step task printed its checklist once per wave — and
  since a wave is printed _before_ its steps start, all of its rows were green
  `[ ]` and the header sat at `Steps 0/5` for as long as the work took. Progress
  therefore arrived as extra lines below the block (`[✓] 1/5 …`, `[✓] 2/5 …`),
  which is not where anyone was looking: the natural reading is that the five
  rows above are still unfinished and five _more_ steps are being added. The plan
  is now one block in the pinned region below your output, where every repaint
  replaces it, so a finished step is checked off **in its own row** and the count
  climbs in the header; when the plan ends, a single all-checked block is printed
  to scrollback as its permanent record. The dead checklist is also cleared when a
  turn is cancelled or a step fails, rather than staying pinned above the prompt.
  Off a TTY (a pipe, CI, `--no-verbose`'s plain loop) there is no region to update,
  so the per-wave block and its tick lines stay exactly as they were.
- **Closing a terminal tab no longer leaves a week of scratch indexes behind.** The
  per-session RAG artifacts under your profile (`rag_store_…`, `chunk_cache_….db`,
  `rag_session_id_….txt`, `chunk_session_id_….txt`) are deleted when the app shuts
  down, so closing the window instead of quitting left them sitting there until a
  7-day sweep collected them. Each name carries the process id that created it, so
  startup now asks the system whether that process is still running: the leftovers
  of an instance that is provably gone are reclaimed at once, and age remains the
  fallback for a name no owner can be read from. The same fact fixes the opposite
  case — an instance that is still **open** keeps its files however old they are,
  so a session you have left open for weeks can no longer have its index deleted
  by another one starting up.

## [1.15.0] — 2026-08-28

### Added

- **A turn now ends with a line that says so** — `· done in 7m22s · 11:08`, dim,
  just above the prompt. A streamed answer simply stops: the last chunk looks like
  every other one, so nothing separated "the model is finished" from "the next
  paragraph is still coming", and the idle prompt looked identical either way. The
  spinner that means _working_ lives in the toolbar, and a toolbar is not
  scrollback — it can't answer the question once it's gone, or ten minutes later
  when you come back to the terminal and want to know whether the long turn ever
  landed. So the line also carries the two facts you'd have to have watched for:
  how long the turn took, and the clock time it finished. It is printed for **every**
  turn, fast ones included — a terminator you can only sometimes rely on doesn't
  terminate anything — and for the delivery-only turn a finished background
  sub-agent triggers on its own, which is the output least likely to be expected.
  A cancelled turn reads `⊘ stopped after 12s · 11:08`, replacing the bare
  `⊘ Stopped`; a failed one still ends with its `✗` line, which already says the
  turn is over and why.

## [1.14.3] — 2026-08-28

### Fixed

- **A provider answering "service unavailable" or "throttled" is retried instead
  of being treated as final.** The transient-failure classifier matches an error's
  wording, and those two arrive named after their exception class — a class name
  has no spaces, so `ServiceUnavailableException` never contained the
  `service unavailable` marker and `ThrottlingException` matched nothing at all.
  Both were classified deterministic and retried **zero** times; because this
  layer deliberately keeps botocore's own retries off, the message even said so
  (`reached max retries: 0`). On a streamed turn that ended the answer outright; on
  an auxiliary call it was quieter and worse — route classification took its
  fallback on the first rejection (`! Router classification failed (…); binding the
full toolset`), so nothing broke and the turn just got dumber. Both class-name
  forms are now transient, as are `too many requests` / `rate limit` / `429` (a
  429 is transient by definition, and it's the case where a provider most often
  attaches its own `retry-after`, which the backoff already honors) and
  `ModelNotReadyException`. Deterministic failures — validation, access denied,
  `model not found` — stay fatal.

## [1.14.2] — 2026-08-27

### Fixed

- **An existing empty file can be written again.** The read-before-write gate asks
  that a file be read before it's modified, and an empty file had no read to give:
  the reader resolved the whole-file range to `1-0`, called it `Invalid line range`
  and returned an error, so no read was ever recorded and every `fs_write` /
  `file_edit` on the file came back with "read it first" — advice that could be
  followed forever without effect. A placeholder created by `touch` or `install`
  (an empty `Dockerfile`, an `__init__.py`) was therefore impossible to fill: the
  model bounced between the two tools and gave up, offering to delete the file or
  to run a shell workaround instead. Reading an empty file is now a successful read
  of empty content, and the gate exempts a zero-byte file outright — it holds no
  content to clobber, so it is a create in every way that matters, and the
  exemption also covers the modes whose reader legitimately errors on an empty file
  (`JSON` on an empty `.json`).

## [1.14.1] — 2026-08-27

### Fixed

- **A multi-step task's checklist now marks steps as they finish.** The block was
  printed once, before the wave of steps started — so the usual case, a task whose
  steps are all independent and therefore run as a single wave, showed `Steps 0/3`
  with every row live for the whole turn and then ended with nothing checked, right
  as the answer arrived. Each step in a wave of several now ticks its own
  `[✓] 2/3 <step>` line the moment it lands (in completion order, with the spinner
  counting down what's still running), and the plan closes on a full block:

  ```
  Steps 3/3
    [✓] Research how to push an image to the container registry
    [✓] Research the CLI command reference
    [✓] Write a concise practical guide
  ```

- **A warning from a dependency no longer prints four raw lines into the chat.**
  `warnings.warn` bypasses logging entirely and writes the path, line number,
  category, message and the offending line of source straight to the terminal —
  which is how `RuntimeWarning: Tool messages were passed without toolConfig` landed
  in the middle of a conversation, twice per turn, beside the one-line failures
  1.14.0 introduced. A warning is now one `! …` line like any other, with where it
  came from recorded in `~/.mnemoai/logs/mnemoai.log`, and a repeat of the same
  warning goes to the file without taking another line on screen.

## [1.14.0] — 2026-08-27

### Added

- **A log file, so the terminal can stay a conversation.**
  `~/.mnemoai/logs/mnemoai.log` now records every diagnostic in full — traceback,
  thread name, and the `INFO` lifecycle lines leading up to it — for the app's own
  errors **and** for anything a library or the standard library logs. On screen a
  problem is one line ending in a pointer to that file
  (`✗ Query failed: division by zero (traceback → ~/.mnemoai/logs/mnemoai.log)`),
  which is where a stack trace belongs: printed into the chat it buries the answer
  above it and asks you to debug the app. `LOG_LEVEL=DEBUG` still shows the whole
  trace on screen — at that point the terminal _is_ the debugger.
- **A failure now reads as part of the interface, not as a log line.** A problem
  on screen is `✗ <what failed>` in red (`!` amber for a warning, `·` for anything
  lower) — the same mark the rest of the app already uses — with no timestamp,
  logger name or level word, and capped to one line of at most 500 characters, so a
  provider error carrying a JSON body can't flood the window the way the traceback
  used to. A turn that dies mid-flight prints the failure and what to do about it,
  and the details are one path away:

  ```
  ✗ ConnectionClosedError: Connection was closed before we received a valid response
    Your conversation is intact — try again or rephrase.  Details: ~/.mnemoai/logs/mnemoai.log
  ```

- **One report per failure.** A single exception used to be announced twice — once
  as the app's own message and again as a red diagnostic line about the same thing.
  A diagnostic can now be marked file-only (`extra={"console": False}`), so the
  record and its traceback still reach the log while the screen keeps the one
  report that was written for you to read.
- **The log is bounded two ways, so it can't grow without end.** `mnemoai.log`
  rotates at 2 MB keeping two generations, and every file under `logs/` — including
  the MCP subprocess's `mcp.log` — is deleted after **`LOG_MAX_AGE_DAYS`** days at
  startup (new root-level config key, default `7`; `0` keeps them forever).
  Rotation bounds one noisy run; only age bounds a year of quiet ones.
- **`/doctor` reports the log.** A `logs` row under **State** prints the path, its
  current size and the retention in force — the traceback behind a one-line error
  exists nowhere else, so the report that describes this install now says where to
  find it.

### Fixed

- **Cancelling during setup no longer prints a Python traceback into the chat.**
  Pressing Esc twice while `/model` (or any dialog) was open could surface
  `Exception in worker` followed by a full `KeyboardInterrupt` traceback through
  `concurrent.futures` and `asyncio` — alarming, and about nothing: the interrupt
  _was_ the cancellation, working as intended. Two independent causes, both closed:
  the second press could land inside the thread pool's own bookkeeping (the worker
  now clears its cancellation target on its own thread and swallows a late
  interrupt), and the report reached the screen through the **root** logger, which
  had no handler and so fell back to printing on stderr. With the log file attached
  to the root logger as well, no library — ours or the standard library's — can put
  a traceback on screen again.

## [1.13.0] — 2026-08-27

### Added

- **A status footer under the prompt.** One dim line pinned below the input,
  carrying the three facts that apply to whatever you type next: the **model**
  (and provider) the turn will go to, the **directory** the session runs in — what
  the file and shell tools act on — and a **context meter**,
  `▓░░░░░░░ 90.1k · 9%`, showing how large the next prompt is and how much of the
  model's window it fills. The meter turns **amber past 70%** and **red past 90%**,
  so a long session says it's getting long before a compaction interrupts it
  (compaction starts at 80% by default). The count is the provider's own number for
  the last turn; before the first turn — and right after a `--resume`, where none
  has run yet — it is a local estimate, marked with a `~` and usually revised
  downward by the first real turn. A narrow terminal drops the path, then the
  provider, keeping the meter. Off an interactive terminal (a pipe, CI) nothing
  paints a footer, so the `[Context: N tokens]` line prints after each turn as
  before.
- **A multi-step task shows a checklist.** While the orchestrator works through a
  decomposed task, the steps render as `[✓]` for finished and a **green `[ ]`** for
  the ones executing right now, with pending steps in gray — so a parallel wave
  shows both of its steps as live, and it's clear at a glance what is done and what
  is left. A long plan is windowed around the current step with a count of what's
  elided, so it can't bury the answer it's working toward. This replaces the
  `[Step i/N: …]` line printed per subtask, which said nothing about progress.

### Changed

- **The per-turn `[Context: N tokens]` line is gone on an interactive terminal** —
  the footer shows the same number permanently, instead of one that scrolled away
  with the turn that printed it. That includes a `--resume`, `/load` or `/branch`,
  which printed it too. It still prints off-TTY, where there is no footer.
- **A tool call's name is now colored** rather than the same white as the model's
  prose, so tool activity and the answer no longer blur together. The accent used
  for it and for the file-operation headers is a **lighter blue** than before — it
  sits inline among gray arguments all turn long, where the darker indigo of the
  launch banner read as low-contrast.
- **A resume prints a compact `⟲ resumed <id>` marker** instead of
  `[Resumed session <id>]`. The full session id is kept, since `--resume <id>`
  matches on any suffix of it.
- **Nothing is printed about a restored compaction any more.** A resumed session no
  longer opens with a note saying how many earlier messages are carried as a
  summary: it described an internal detail nothing could be done about, sitting
  above a transcript that had just replayed in full. A resume reads the same whether
  or not the session had been compacted — what came back is visible in the footer's
  context meter.

## [1.12.7] — 2026-08-27

### Fixed

- **A resume no longer gives back the tokens tool-result trimming had reclaimed.**
  Before summarizing anything, mnemoai trims the bodies of older tool results — a
  cheap pass with no model call that often frees enough on its own. That pass wrote
  nothing to the session transcript, which keeps every result at its original size,
  so `--resume` (and `/load`, `/branch`) replayed the full-size results and handed
  back exactly the context the trimming had freed. It is the same defect 1.12.6
  fixed for summaries, in the layer that runs more often — and it was invisible,
  because nothing about a trim shows up as "compacted". Both kinds of reduction are
  now checkpointed, and a trim recorded now can never overwrite a summary an earlier
  compaction left standing. Sessions recorded by earlier versions still restore in
  full, as they always did.

## [1.12.6] — 2026-08-27

### Fixed

- **Resuming a compacted conversation no longer undoes the compaction.** A session
  the provider had just reported at **235,793 tokens** came back after `--resume`
  at **1,166,221** — five times bigger, past the model's window, before a single
  new message. The transcript keeps every turn's full text (deliberately: nothing
  said is ever lost from disk), but restoring it replayed the raw history a
  `/compact` had already summarized away. So every resume re-inflated the context
  and the first turn back had to summarize the whole conversation again, paying for
  a summary that had already been paid for and thrown away. A compaction now
  records what replaced that history — the summary plus the messages that stayed
  live — and a restore rebuilds **the state the session ended in**, for `--resume`,
  `/load` and `/branch` alike. The conversation still replays on screen in full,
  with a note saying how many earlier messages are carried as a summary.
- **`/save` now records the active summary too.** Saving after a `/compact` wrote
  only the messages left in the window: the summary lived inside the system prompt,
  which `/load` rebuilds from scratch, so a loaded conversation resumed mid-thread
  with the earlier history silently gone. Files saved before this release load
  exactly as before.
- **`/clear` now drops the compaction summary.** It rebuilt the system prompt
  without the summary block but left the manager's copy in place, so a cleared
  conversation's history could still reach the next compaction's summary — and, with
  the change above, a later `/save`.

## [1.12.5] — 2026-08-27

### Fixed

- **Resuming a session no longer inflates it.** Every resume re-saved each tool
  result wrapped inside a copy of itself, with all the quotes escaped again — so
  the same results roughly **doubled in size on every resume**, silently, without
  a single character of new content. A 31-character result reached 4,735
  characters after ten resumes; in a real conversation, file reads reached **1.07
  million characters each**, and a 1,929-message session weighed 21M characters of
  which about 90% was backslashes. That is what produced impossible readings like
  `[Context: 12650351 tokens]` right after `--resume`, followed by a context
  overflow on the first message. The conversion is now stable — a result is
  byte-identical after fifty round trips — and `MAX_TOOL_RESULT_CHARS` can no
  longer be exceeded after the fact.
- **Sessions already bloated by this are repaired when you open them.** Resuming
  or `/load`ing an affected conversation unwraps the nested copies as it reads,
  so the history is restored at its real size and re-saved clean; the reported
  session went from 21M to 2.1M characters (10× smaller) with no content lost.
  Nothing needs to be deleted or migrated by hand.

## [1.12.4] — 2026-08-26

### Fixed

- **A conversation that outgrows the context window mid-turn now recovers instead
  of dead-ending.** The turn compacted, retried once, and then hit
  `Context overflow … could not compact` on the very next model call — a few
  thousand tokens later, over a history that had just been reduced to two
  messages. The compaction was real but the running turn never saw it: it keeps
  re-reading the history it started with, so every later call in that turn re-sent
  the same oversized prompt, and by the second overflow there was nothing left to
  compact. The compacted history is now substituted into the rest of the turn, and
  the tool calls and results the turn already produced are carried across the retry
  so that work isn't redone.
- **A compaction mid-turn is no longer undone when the turn ends.** Everything the
  compaction had summarized away was appended back to the conversation — and
  written to the session transcript as if this turn had produced it — so the next
  turn immediately compacted the same history again. Visible as two compactions in
  a row over almost the same message count (`summarized 1209 older messages`, then
  `summarizing 1165 older messages`). Only what the turn actually produced is
  committed now.
- **`--resume`, `/load` and `/branch` no longer skip the pre-flight compaction
  check.** The check uses the exact context size the provider reported for the
  last turn, but restoring a conversation replaces history wholesale — and the
  transcript keeps every message compaction had summarized away, so the restored
  history can be far larger than the number left over from before. The stale count
  read too low, the check passed, and the first turn after a `/branch` went
  straight to a provider-side context overflow. The cached size is now dropped
  whenever live history is replaced, so the next turn measures what is really
  there.

## [1.12.3] — 2026-08-26

### Fixed

- **A dropped connection no longer ends the turn.** `Connection was closed before
we received a valid response from endpoint URL …` was reported as an error the
  app "can't recover from automatically" — yet a closed socket is the single most
  retryable failure there is. The classifier matches provider wordings, and
  botocore phrases this one as "Connection **was** closed", which the existing
  `connection closed` entry didn't match. Only its two errors containing the word
  "timeout" were ever retried; the other five — a closed connection, an
  unreachable endpoint, a failed response stream — ended the turn on the first
  occurrence. All of them now retry on a fresh connection.
- **A long conversation is no longer mistaken for a dead stream.**
  `LLM.STREAM_IDLE_TIMEOUT` (default 120s) exists to abandon a stream that goes
  silent mid-response, but it was also policing the wait for the **first** token —
  which is the model reading your whole prompt before it starts answering, and on
  a large context takes minutes, not seconds. So a perfectly healthy turn was
  aborted at 120s with `No stream data for 120s (connection likely dropped)`, and
  because every retry re-sent the same prompt and re-paid the same wait, the turn
  could never complete: the bigger the conversation, the more reliably it failed.
  The wait for the first token now gets its own budget, derived from
  `LLM.REQUEST_TIMEOUT` (default 600s), while `STREAM_IDLE_TIMEOUT` continues to
  guard the running stream — it narrows back the moment data arrives, so a socket
  that dies mid-response is still caught as quickly as before. No new config key,
  and the log now says which of the two tripped (`No first token for Ns` vs
  `No stream data for Ns`).

## [1.12.2] — 2026-08-26

### Fixed

- **A provider overload (`529 overloaded_error`) no longer silently costs you
  routing and task decomposition.** When the provider is busy it rejects
  everything in flight at once, and the streamed turn handled that correctly —
  `retrying turn on a fresh connection (attempt 1/6)` — while the two smaller
  calls beside it, query classification and task decomposition, gave up on their
  **first** rejection. Both have a graceful fallback (bind every tool; treat the
  whole request as one subtask), which is exactly why the failure was easy to
  miss: nothing broke, the work just quietly got dumber for that turn, and the
  only trace was `Router classification failed` in the log. They now retry the
  overload the same way the turn does — fewer attempts, because a working
  fallback makes a quick second chance better than a long stall — and fall back
  only once the attempts are spent.
- **Compaction no longer loses a slice of your history to an overload.** The
  summary summarizes history in batches fanned out concurrently, so an overloaded
  provider rejects several at once — and a rejected batch was simply dropped from
  the summary, permanently, since the messages it covered are removed either way.
  Batches (and the final merge) now retry a transient failure first.
- Retry backoff is **jittered**, and honors the provider's own `retry-after`
  header when it sends one. Orchestrator waves, spawned sub-agents and the
  compaction map all fan out together, so a purely deterministic backoff had them
  all come back at the same instant and collide again.
- The router's give-up message is now a warning that names the consequence
  (`binding the full toolset`) instead of an `ERROR` for something the app
  recovers from.

## [1.12.1] — 2026-08-26

### Fixed

- **`'glob_search' did not respond within 300s` — the same class of failure as
  1.10.3, this time hiding behind an `await`.** Reading files was still stalling
  every other agent's tool call: with a reviewer sub-agent reading the same files
  in parallel, a `glob_search` whose real work is milliseconds died at
  `LLM.MCP_CALL_TIMEOUT`. `fs_read` looked innocent — it was `async def` and it
  awaited its readers — but every reader was itself an `async def` with no await
  anywhere (one of them making a blocking model call to summarize a large PDF), so
  the entire chain ran start-to-finish on the server's single event loop. **An
  `await` on a coroutine that never suspends buys nothing; only the leaf matters.**
  `fs_read` and all seven readers are now plain functions and run on a worker
  thread like every other blocking tool.
- **The stall was measured in minutes, not milliseconds, because line reading was
  quadratic.** Each line's token count re-counted the whole accumulated text, so
  the tokenizer re-did all its previous work on every line: **9.0 seconds for one
  2839-line source file, against 0.007 seconds to count the same text once** — and
  since the read cap is derived from the context window, a large-context model
  reads to the end of the file rather than stopping early, so the cost kept
  growing. Counting is now incremental (a running total plus the new line), with
  one exact count of what is actually returned. Same for JSON/JSONL truncation.
  The same file now reads in **0.007s**, and a `glob_search` issued while three
  agents read files in parallel returns in **0.008s**.
- **`glob_search` now bounds its work, not just its output.** It filtered
  `node_modules`, `build`, `dist` and friends out of the _results_ — after walking
  every one of them — never looked at the clock, and followed directory symlinks,
  so a link back to a parent directory made the walk effectively endless. It now
  walks the tree itself: noise directories are skipped **before** descending, and
  the scan stops after 30 seconds with the matches it has and `timed_out` set,
  instead of running until the client gives up on it. Two deliberate changes that
  follow: symlinked _directories_ are no longer traversed (symlinks to files still
  match), and a pattern that explicitly names an ignored directory —
  `node_modules/**/*.js` — is now honored rather than silently returning nothing.
- Summarizing a large PDF or DOCX in chunks is **actually** concurrent now (a
  bounded thread pool; the previous asyncio semaphore never yielded, so chunks ran
  one after another _and_ on the event loop), and its progress lines go to the log
  file instead of `print` — standard output in the server subprocess is the
  protocol pipe, so printing there wrote text into the message stream.
- The guard that was supposed to prevent all of this only inspected the
  **tools**, which is exactly how `fs_read` slipped through it. It now also fails
  on any **helper** under `server/tools/` declared `async def` without a real
  await, and a new troubleshooting entry explains that the tool named in a 300s
  timeout is usually the victim rather than the cause.

## [1.12.0] — 2026-08-25

### Added

- **Tool hooks: your own commands run around every tool call.** A prompt is a
  request, not a rule — "always format what you write", "never touch that
  directory", "stop asking me to confirm `git status`" hold right up until the
  model decides otherwise. `~/.mnemoai/hooks/hooks.json` declares shell commands
  that run **before** a tool (`PreToolUse`), **after it succeeds**
  (`PostToolUse`), or **after it fails** (`PostToolUseFailure`), matched against
  the tool name with a glob (`fs_*`, `execute_bash`, `*`). The hook gets the call
  as JSON on stdin (event, tool name, arguments, and the result or error where
  there is one), and answers with its exit code or a JSON object: **exit 2 blocks
  the call** and its stderr becomes the reason the model is told, `additionalContext`
  hands the model a note alongside the tool result, `{"decision": "allow"}` waves
  a call past the confirmation prompt. Anything printed plainly is shown to you
  and never sent to the model. `/hooks` lists what is loaded, from where, and what
  didn't parse.
- **Where a hook sits is the whole design: below the existing gates, never above
  them.** The server-side safety floor runs first, then the plan-mode block, then
  the hook, then the confirmation prompt. So a `deny` is honored anywhere, but an
  `allow` reaches exactly one gate — the prompt for that single call. It cannot
  unblock a tool plan mode blocked, and it cannot reach the floors inside the MCP
  server, which still refuse a catastrophic command. A config file must not become
  a way to widen what the app is allowed to do, and a `deny` wins over an `allow`
  regardless of which one you wrote first.
- Hooks are read from the **app home only** and **snapshotted at startup** — both
  deliberate, both unlike `STEERING.md`. A `hooks.json` arriving with a `git clone`
  would be remote code execution on your first edit in that repo, so nothing
  outside `~/.mnemoai/` is read; and hooks are code, so editing the file
  mid-session cannot change what is already running (restart to apply). A hook
  that crashes, exits non-zero for any other reason, or overruns its `timeout` is
  **reported and skipped** — a broken hook must never be able to wedge a turn, and
  hooks fire from sub-agent and parallel-wave threads too, so none of them can ask
  you a question. They run under real bash, and a commented `hooks.json.example`
  is seeded next to the live file.
- **`/doctor` — is anything about this install broken, or about to be.** Every
  other report describes the conversation; this one describes the machine, because
  this app fails in places you cannot see: the MCP server is a piped subprocess,
  so a missing `ripgrep` surfaces as one tool erroring mid-task; a feature toggle
  is a line in a YAML file, so `ENABLE_RAG: true` with no vector store installed
  looks like the model ignoring you; prompt caching is silently provider-gated, so
  a config that cannot cache reads as one that does. It reports the config and
  prompts files that actually **loaded** (not the ones you expected — four
  resolution tiers make "I edited config.yaml and nothing changed" a real
  outcome), the provider with its credentials resolved or its local port probed,
  the external binaries, declared-versus-connected MCP servers, the optional
  dependency behind each switched-on feature, and the two files re-sent every turn
  — `MEMORY.md` against its cap and each steering file at its injected size. Every
  check is local, cheap, and read-only, with the fix printed under the failure.
- **`/rename <title>` names a session in the `--resume` picker.** The picker
  labelled every row with its first prompt — the one thing a resumed conversation
  and its parent have in common — so after a few sessions in one project the rows
  stopped being distinguishable. A name is recorded like every other entry
  (append-only, last one wins) and survives a resume, since that continues the
  same conversation in a new file. A `/branch` fork deliberately does _not_ inherit
  it: a fork already shares its parent's opening prompt, and inheriting the name
  too would put the two rows right back to identical. `/rename` alone shows the
  current name, `/rename clear` drops it.
- **Docs for all three:** a new [Tool hooks](docs/guides/hooks.md) guide (the
  file, the events, the stdin/exit-code contract, the gate order and why the file
  is app-home-only), plus `/doctor` and `/rename` sections in the usage guide, a
  `hooks/` row in the `~/.mnemoai` map, and a "run `/doctor` first" opener on the
  troubleshooting page.

### Fixed

- `/params` no longer ends with "Reload to apply" — it applies the change in
  place and says so on the next line. The note was left behind when `/params`
  stopped restarting the app.

## [1.11.0] — 2026-08-25

### Added

- **Prompt caching: a turn stops re-paying for the prefix it already sent.** One
  turn is many model calls over a prompt that only grows — system prompt, tool
  definitions, steering, memory, every prior message and tool result — and each
  call re-billed and re-prefilled all of it from scratch. The stable prefix now
  carries a cache breakpoint, so subsequent calls read it at a fraction of the
  input price and the provider skips prefill it has already done. That second
  part is latency, not just cost: prefill is what dominates time-to-first-byte
  (a ~440k-token turn at `REASONING_EFFORT: max` measured ~123s before its first
  byte). Enabled where the provider and model family support it — `bedrock`,
  `anthropic`, and `mantle` on `API_PROTOCOL: anthropic`, for the Claude and Nova
  families — and deliberately nowhere else, because the marker is a per-call
  keyword an OpenAI-shaped endpoint rejects outright. Below the provider's cache
  minimum (~1024 tokens) a request is simply not cached, so there is nothing to
  configure for short sessions. `PROMPT_CACHE: false` opts out per model,
  `PROMPT_CACHE_TTL` chooses `5m` or `1h`, and both are tunable live from
  `/params` (they survive a `/model` provider switch instead of being pruned).
  `/usage` shows reads and writes.
- **The system prompt carries its own breakpoint on the Anthropic-shaped
  transports** — the difference between caching that looks enabled and caching
  that lands. The per-turn injections (`<steering>`, the episodic block, the plan
  reminder) ride the newest user message and are stripped before storage, so the
  tail of one request is never a prefix of the next and a single end-of-prompt
  marker matches nothing on the following turn. Measured on the same request: **0
  cached tokens read with only the tail marker, the whole 4812-token system
  prefix read with the system one.** Bedrock's Converse format places its own
  breakpoint after the system blocks and rejects an Anthropic-style key inside
  one, so it deliberately doesn't get the extra marker.
- **A steering file can pull in another with an `@path` reference.** A long
  ruleset can now be split into focused files (`See @rules/testing.md`) instead of
  one wall of text, and referenced files are appended after the file that
  mentioned them under their own header — the reference stays in the prose, so the
  sentence still reads. Resolved against the referencing file's own directory (not
  the process cwd, so a project's rules mean its neighbours regardless of where
  the app was launched); `~` expands, absolute paths are taken as-is. A reference
  that doesn't name a readable file is left alone, which is what makes a Python
  decorator, an `@handle` or an email address harmless. Bounded three ways, since
  every byte here is paid on every turn and compaction can never reclaim it:
  3 levels deep, 20 files, and the remaining share of the referencing file's
  `STEERING.MAX_CHARS` budget — with anything dropped stated in the block rather
  than silently missing.
- **`/context` — where the context window actually goes.** `/usage` answers what
  the session has spent; this answers what the _next_ turn is paying for and which
  part of it can be shrunk. A gauge plus a breakdown: the system prompt segmented
  into its real parts (learned profile, MEMORY.md, skills listing, sub-agent
  types, playbook strategies, compaction summary), one row per steering file at
  the size actually injected, the tool schemas, and the conversation. It exists
  for the two costs nothing else surfaces — a large steering file, re-sent
  verbatim every turn and never reclaimable, and the tool schemas, bound on every
  call before a word of the conversation. The total is the provider's exact
  reported input size (the same number as `[Context: N]`) with the parts scaled
  onto it, and the system prompt is segmented from the live string rather than
  re-derived, so the report can't drift from what is really being sent.
- **`/help`** prints the full command reference plus the keys that aren't
  commands — Esc to interrupt, Ctrl+J for a newline, Ctrl+A for the agents panel.
  The launch banner shows the same box (one renderer, so they can't drift), but it
  scrolls away, and until now nothing brought it back.

### Changed

- **`execute_bash` runs under bash, and remembers where it left off.**
  `shell=True` runs `/bin/sh`, which is not bash everywhere — dash on
  Debian/Ubuntu, bash in POSIX mode on macOS — so `[[ ]]`, arrays, `source` and
  process substitution failed or behaved differently depending on the host, on a
  tool documented as bash. Bash is now what runs them, falling back to the default
  shell only where no bash exists. And each call used to be a fresh shell at the
  launch directory, so a `cd` evaporated: the model would `cd project` and the next
  command silently operated on the wrong tree. The directory a command ends in is
  now where the next one starts, as in an interactive session, and is reported back
  as `cwd`. A killed (timed-out) command doesn't move it — it didn't finish
  anywhere — and a tracked directory that has since been deleted falls back to the
  launch directory instead of failing every command. Background tasks
  (`start_background_task`) start in the same tracked directory when no
  `working_directory` is given.
- **`execute_bash` and background tasks no longer inherit the server's stdin.**
  It is the MCP protocol pipe: a command that reads stdin (`cat`, an interactive
  prompt) could consume the client's JSON-RPC stream. With stdin closed such a
  command sees EOF and fails immediately instead of burning the full timeout
  waiting for input that can never arrive.
- **A file edit shows a line-level diff.** The old→new block was printed whole —
  every removed line in red, then every added line in green — so a one-word fix
  inside a 40-line replacement left the reader to diff it by eye. The two sides
  are now paired line-by-line, only the lines that actually changed are marked,
  the surrounding lines stay as gray context, and a long unchanged run is elided
  to its ends (`… 12 unchanged lines`). A pure insertion or deletion still prints
  whole, since there is nothing to pair.

### Fixed

- **`/usage` reported `0` cache writes on providers that break them out per
  TTL.** Both Bedrock and Mantle report a cache write of several thousand tokens
  under `ephemeral_5m_input_tokens` while leaving LangChain's normalized
  `cache_creation` at `0`, so a session that was caching correctly displayed as
  having written nothing — the one number that proves caching engaged. Writes are
  now read from the normalized field, a nested per-TTL breakdown, or the per-TTL
  keys, whichever the provider filled.

## [1.10.3] — 2026-08-25

### Fixed

- **One agent's tool call no longer freezes every other agent's.** With several
  agents in flight — an orchestrator wave, `spawn_agent` sub-agents — tool calls
  failed in bursts with `'<tool>' did not respond within 300s`, including ones
  whose real work takes milliseconds (`glob_search`, `memory`), leaving the run to
  be cancelled by hand. The MCP server is one subprocess with one event loop and
  the SDK dispatches requests concurrently, but 24 of the 31 tools were declared
  `async def` with no `await` anywhere in the body, so each ran start-to-finish on
  that loop: while one blocked, the server could not dispatch another call, read a
  request, or even hand back a response it had already finished. A single
  `execute_bash` was enough to kill every other in-flight call at
  `LLM.MCP_CALL_TIMEOUT`, and a long wait the client explicitly supports —
  `wait_for_task(timeout_seconds=1500)`, which polls with `time.sleep` — made it
  certain. A tool with a blocking body is now a plain `def` and runs on a worker
  thread, applied once where the tools are registered so one added later is
  covered by construction; only genuinely-awaiting tools (`fs_read`, `web_search`,
  `web_crawler`) still run on the loop. Measured on the real server subprocess,
  a `glob_search` issued during an 8s `execute_bash`: **7.03s before, 0.00s after**.
- The one-time Playwright Chromium download (~260 MB, triggered by the first
  `web_crawler` call after a fresh install) also ran on the event loop, blocking
  every other agent's tool call for the length of the download. It now runs on a
  thread.

### Security

- **`PyPDF2` replaced by `pypdf`.** PyPDF2 was retired at 3.0.1 (the project was
  renamed to `pypdf`), so its open advisory — an infinite loop on a malformed
  comment, reachable by reading a hostile PDF through `fs_read` — can never be
  patched there. `pypdf` carries the fix and the reader's API use is unchanged;
  extraction on real PDFs is byte-identical, and it additionally preserves the
  line breaks PyPDF2 flattened into runs of spaces.
- `h2` raised to 4.4.1 (duplicate-`Host` request-smuggling advisory, reached
  through `httpx`'s HTTP/2 support).
- `cryptography` deliberately **not** raised, and `langchain-litellm` floored at
  0.7.0 to keep a resolver from doing it silently: 0.7.0 requires
  `cryptography<49`, so pulling cryptography past its advisory range (`<50.0.0`)
  downgrades that provider integration by two minor versions instead. The
  advisory concerns PKCS#7 `EnvelopedData` decryption, which nothing in this
  project calls.
- The open `chromadb` advisories have **no patched release** — 1.5.9 is the
  latest and every vulnerable range includes it. All of them require a running
  Chroma HTTP server: its `/api/v2` collection endpoints, cross-tenant
  permissions, or `SimpleRBACAuthorizationProvider`. Both call sites here
  construct `chromadb.PersistentClient(path=…)` — an embedded, single-user store
  on the local filesystem — and nothing starts a server or configures an
  authorization provider, so none of them is reachable in this application. To be
  raised when upstream ships a fix.

## [1.10.2] — 2026-08-24

### Fixed

- **The agents panel no longer drops older sub-agents — it scrolls.** With more
  runs than the panel's 6 rows, the renderer sliced off everything but the last
  6, so in an 11-agent fan-out the earlier ones couldn't be selected, opened or
  stopped, and a still-running early agent was invisible with no way to reach it.
  The panel is now a viewport over every retained run: ↑/↓ in nav-mode moves the
  cursor through the whole list and scrolls the view. Idle it still shows the
  newest rows, but the header says what's hidden (`+5 more (1 running)`), and
  Ctrl+A jumps the cursor straight to the oldest agent still working.
- **Ctrl+A reopens the list after the panel hides.** The panel disappears once
  every agent has finished, which also blocked Ctrl+A — leaving the finished
  agents' reports unreadable. Their runs are still retained, so the key now works
  whenever any run exists.

### Changed

- Retained sub-agent runs raised from 32 to 64, since that limit is now what
  decides how far back the panel's list reaches. A running agent is still never
  evicted.

## [1.10.1] — 2026-08-10

### Fixed

- **A bold word next to a markdown link no longer prints stray `1m` / `[0m`.**
  `**Bold** of [a link](https://example.com)` rendered as
  `1mBold of [a link (https://example.com[0m)`, losing both the emphasis and the
  link. Link formatting runs after emphasis is already ANSI, and its patterns
  treated escapes as ordinary text: the markdown-link pattern matched the `[`
  _inside_ `ESC[1m` as a link start, stranding the ESC; and the plain-URL pass
  re-matched an already-wrapped URL with a trailing class that didn't exclude
  ESC, eating the ESC of the following reset. Links and plain URLs are now
  rewritten in one alternation pass, ESC is excluded from every URL class, and a
  lookbehind makes re-formatting a no-op. Emphasis inside link text still
  renders, and the OSC 8 clickable path (which was double-wrapping plain URLs)
  is fixed too.

## [1.10.0] — 2026-08-04

Tuning a parameter no longer costs you the conversation. `/params` used to restart
the app to apply a temperature — throwing away the chat you were tuning it for,
and worst right after a `--resume`, where the restored history lived only in the
file the restart abandoned. It now applies in place, and the turn after it is the
next turn of the same conversation. The restart the other settings commands rely
on is unchanged, and it no longer leaves an abandoned session file behind.

The source distribution also stops shipping the 10.5 MB demo GIF, which was 96%
of the download and read by nothing.

### Changed

- **`/params` no longer restarts the app — the conversation continues.** Tuning
  temperature (or top_p, reasoning effort, stop, stream, …) used to re-exec the
  process, which threw away the chat you were tuning it for. That restart existed
  because `/config`, `/model` and `/features` genuinely need one — they can change
  the provider, the model, or a toggle that gates MCP tool registration at
  subprocess boot — but `/params` edits inference knobs only, and nothing the MCP
  server fixed at boot can have changed. It now rebuilds the model in place and
  the turn after it is the next turn of the same conversation. The rebuild is a
  fresh controller, not a re-`initialize_model()` on the existing one, because the
  controller snapshots every parameter at construction; every holder of a built
  model is re-pointed with it (the tool-bound model, each per-route binding, the
  router, and the cached summary/sub-agent variants), so the new value can't
  half-apply. If the rebuild fails, the old restart still happens rather than
  leaving the session running on a config that was only partly applied.

  Worst before this change: right after `--resume`. Resuming writes a _new_
  session file seeded with the restored history, so a `/params` restart at that
  point discarded the live conversation _and_ left the only copy of it in a file
  the restart abandoned — the way out was to quit and `--resume` again. `/config`,
  `/model` and `/features` are deliberately unchanged.

### Fixed

- **A restart no longer leaves an abandoned session file behind.**
  `_restart_in_place` re-execs with `os.execv`, which _replaces_ the process — no
  `atexit`, no `finally` — so the end-of-run cleanup that unlinks a turn-less
  transcript never ran. Every restart before the user had typed anything (the
  normal case for a `/config` or `/model` right after startup, and for every
  `--resume` followed immediately by a settings change) left an empty session file
  to sit on disk until it aged out. The restart now discards it itself, and a
  failure to do so is logged rather than allowed to block the restart.

### Packaging

- **The source distribution no longer ships the `images/` directory** (12.87 MB →
  1.93 MB). The demo GIF alone was 10.5 MB — 96% of the archive — for a file
  nothing in the distribution reads: the README and docs reference it by absolute
  `raw.githubusercontent.com` URL, so the rendered PyPI page is unchanged and the
  file still lives in the repository. Wheels never contained it, so
  `pip install mnemoai-assistant` was already unaffected; this only shrinks the
  sdist. The documentation site uses its own copy of the logo under
  `docs/assets/`, which is untouched.

## [1.9.1] — 2026-08-03

### Fixed

- **The shipped-`prompts.yaml` hash guard no longer fails on the release that
  cuts it.** `test_previously_shipped_prompts_hashes_are_tracked` enumerates every
  release tag and requires each shipped `prompts.yaml` to be registered as
  pristine — but the CURRENTLY bundled content is deliberately not in that set
  (it's compared against the bundle at runtime). The moment a release tag exists
  it ships the current file, so the guard failed on its own release, every time.
  It now skips the currently-bundled hash, the exemption the sibling `SKILL.md`
  guard already had. Test-only; no runtime behavior changes, and the guard still
  fails on a genuinely unregistered hash (verified by removing one).

## [1.9.0] — 2026-08-03

An always-on instructions file may now be named `CLAUDE.md`, so a repository that
already keeps its agent instructions under that name is picked up with nothing to
write and nothing to configure. Two rules make the second name safe rather than
surprising: within one directory `STEERING.md` wins, and that choice is per
directory rather than global.

Accepting a filename other tools also write turned four latent weaknesses into
likely ones, so they are fixed here too: a large file corrupted query routing, a
non-UTF-8 file broke every turn, an unreadable file shadowed a readable one, and
the injected block had no size ceiling despite being re-sent verbatim forever.
Separately, the walk no longer treats a file in your home directory as always-on
instructions for everything beneath it — and the test tier that was quietly
reading this project's own instructions file (in both of its processes) now runs
isolated.

### Added

- **An always-on instructions file may be named `CLAUDE.md`.** Discovery accepts
  either name at every level it already searched — the app home, and each
  directory from the CWD up to the repo root — so a project whose agent
  instructions already live in a `CLAUDE.md` is picked up with no second file to
  write and nothing to configure. Two rules define the precedence, and both are
  deliberate:
  - **Within one directory `STEERING.md` wins**, and a `CLAUDE.md` beside it is
    skipped rather than appended. That is what lets a repo keep both: shared
    instructions in one file, the parts meant for this assistant in the other,
    without the two being concatenated into contradictions.
  - **The choice is per directory, never global.** A global `CLAUDE.md` still
    applies alongside a project `STEERING.md`, and a subdirectory's `CLAUDE.md`
    still applies when nothing shadows it. Finding one name in one directory
    must not switch the other name off everywhere else — the alternative reads
    identically in the common case and diverges exactly where a user has both.

  The global tier reads the app home only (`~/.mnemoai/STEERING.md`, else
  `~/.mnemoai/CLAUDE.md`); files belonging to other tools elsewhere on the
  machine are never read. Reading accepts both names, **writing** still has one
  target (`STEERING.md`), so the bundled `steering-creator` skill can't be talked
  into authoring the fallback name. Files reached twice under different spellings
  (a symlinked app home inside the walked chain) are de-duplicated by real path,
  so one file is never injected — or billed — twice.

- **`STEERING.MAX_CHARS` caps each instructions file's contribution** (default
  45000 chars per file; `0` disables). This block is re-sent verbatim every turn
  and is deliberately kept out of stored history so compaction can never reclaim
  it — an oversized file is therefore a permanent per-turn cost, not a one-off
  (~14k tokens/turn for a 56 KB file, every turn, forever). Accepting `CLAUDE.md`
  made that reachable in practice, since those files already exist and are
  routinely tens of KB. The cut is never silent: the injected text says the file
  was truncated and names it, so the model reads the rest with its file tools
  instead of assuming the omitted part said nothing. **Note if you already run a
  `STEERING.md` larger than 45000 characters:** it was previously injected whole
  and is now truncated at that boundary. Raise `STEERING.MAX_CHARS`, or set it to
  `0`, to keep the old behavior.

### Fixed

- **A large instructions file corrupted query routing.** Every routing decision
  reads the last message's text, which carries the injected block — so an
  always-on file of any size buried the actual question: `"Hello"` exceeded the
  word-count gate and the file's own paths and extensions tripped the
  deterministic content signals, flipping it from trivial chit-chat to a
  decomposed multi-worker task. Both the classifier and the trivial-query gate
  now read the user's text with the ephemeral blocks stripped. Latent since
  `STEERING.md` shipped; accepting `CLAUDE.md` is what made it common.
- **A non-UTF-8 instructions file broke every turn.** `UnicodeDecodeError` is a
  `ValueError`, not an `OSError`, so it escaped the read guard, propagated out of
  the per-turn injection, and surfaced as the generic "something went wrong" on
  every single turn with nothing pointing at the file. Files are now read with
  `errors="replace"` behind a widened guard: a stray byte degrades one character
  instead of ending the conversation. More likely with the second name, which is
  authored by other tooling.
- **An unreadable `STEERING.md` no longer shadows a readable `CLAUDE.md`.**
  Readability is part of the per-directory choice, so a permission-blocked file
  falls through to the other name instead of leaving that directory contributing
  nothing.
- **The walk survives an unreadable directory.** One `PermissionError` mid-chain
  previously abandoned the whole resolution, which silently dropped the MOST
  specific files — the deeper ones are collected last. Each directory is now
  guarded on its own, and the project root is detected with `.exists()` so a git
  worktree or submodule (where `.git` is a FILE) still terminates the walk.
- **Prompt and skill improvements can now reach installs from the last four
  releases.** `_PRISTINE_BUNDLED_PROMPTS_HASHES` was missing the `prompts.yaml`
  shipped by 1.8.4–1.8.7 (and, further back, 0.8.17–1.4.5), so those installs
  read as "user-customized" and the in-place refresh skipped them — meaning an
  edit to an EXISTING prompt key would never have arrived (the bundled fallback
  only fills MISSING keys). Both guard tests now **enumerate tags from git**
  instead of a hardcoded list, which is precisely how the gap went unnoticed for
  four releases while the test stayed green, and the same guard now exists for
  bundled skills, which had none.
- **An instructions file in your home directory is no longer always-on
  everywhere.** With no `.git` in any ancestor the walk ran to the filesystem
  root, so a single `~/CLAUDE.md` (a file other tooling readily creates) became
  permanent instructions for every non-repository directory beneath it —
  contradicting the rule that only the app home is global. The unrooted walk now
  stops below the home directory; an explicit `.git` **at** `$HOME` is still an
  ordinary project root, so a dotfiles repo keeps working. Reachable before with
  one filename, but a stray `STEERING.md` there was unlikely in a way that a
  `CLAUDE.md` is not.
- **The integration tier no longer runs from the checkout — in either process.**
  `$MNEMOAI_HOME` is redirected for that tier, but project discovery walks
  `Path.cwd()`, which no env var touches — so the tier read this project's own
  56 KB instructions file and prepended it to every live query: real tokens
  against the configured provider, and the routing guards quietly stopped
  exercising the paths they were written to protect while still passing. The fix
  needs two halves, because the MCP server is a **subprocess** that inherits cwd
  once at spawn and keeps it for life: the client is moved per test, and the
  client/server pair is now started from the neutral directory too — previously
  the server ran with the checkout as its cwd, so a relative-path `fs_write`
  created files inside the developer's repository. The per-test chdir is
  deliberately function-scoped: a session-scoped version applied to the whole
  pytest session, moving the **unit** tier as well, where `git` then found no
  repository and both shipped-hash guards skipped themselves — green while
  checking nothing. Those guards now locate the repo from their own file rather
  than the process cwd, so no future fixture can disarm them, and the tier's
  isolation is asserted by tests of its own.

## [1.8.7] — 2026-07-31

The lazy vision initialization shipped in 1.8.6 was correct but ineffective: it
fixed the one import path that never needed fixing. An adversarial re-check of
that release — agents told to refute its claims rather than confirm them —
found that the MCP server still loaded both OpenMP runtimes on every start,
through three import chains nobody had traced. Also a silent-failure mode in
the same code, plus the documentation and reporting gaps the review turned up.
No public-surface change.

### Fixed

- **A disabled tool group no longer pays for its dependencies.**
  `register_tools` gates `describe_image` (which reaches transformers/torch) and
  the RAG tools (which reach faiss) behind `VISION_MODEL_ID` and `ENABLE_RAG`,
  but both were imported unconditionally at the top of the function — and an
  import is what creates the module and registers its OpenMP runtime, so
  declining to _register_ the group bought nothing. torch and faiss each vendor
  their own OpenMP, and a process holding both aborts (`OMP: Error #15`) as soon
  as faiss searches, which is the whole reason for the gate. Two further chains
  defeated it independently of that: `pdf_reader`/`docx_reader` imported
  `..rag` at module scope purely to set an `_rag_available` boolean — and
  `fs_read` imports every reader unconditionally, so that was every process
  touching the tools package — and `web_crawler` did the same. Measured with
  both features off: previously torch, faiss, and transformers all loaded;
  now none does, and the 25 tools that don't need them still register. The
  probes became a `_rag_session()` helper resolved at call time.
- **A failed vision initialization is no longer silent.** `_ensure_vision` set
  its "done" flag _before_ building, so a raising provider was remembered as
  complete. For one exception type that lost the error entirely: `describe_image`
  binds the model through the package's `__getattr__`, and importlib's
  `_handle_fromlist` probes with `hasattr`, which swallows an `AttributeError`
  — the pre-set flag then let the retry return `None`, permanently binding a
  dead model and dropping the tool from the registered set with nothing logged
  anywhere. The flag now goes up only after a successful build, and the
  controller is assembled in a local so a raise can't leave a half-built one
  behind for the next caller. No reachable trigger existed in the shipped code
  (every misconfiguration raises `TypeError`/`KeyError`/`ValueError`/
  `RuntimeError`, all of which were already loud), so this was latent — but the
  cost of being wrong was a capability that fails invisibly.
- **`fs_read`'s invalid-mode error listed five of its eight modes.** A model
  told to use `Line`, `Search`, `Directory`, `CSV`, or `JSON` had no way to learn
  that `JSONL`, `PDF`, and `DOCX` also work — the error taught it a smaller
  tool than it has. All eight are listed now, noted as case-sensitive.

### Changed

- **`tool_loop.py` is documented.** The module introduced in 1.8.6 to unify the
  two tool-execution chokepoints was absent from `ARCHITECTURE.md` and
  `CLAUDE.md`, which still described a shape the code no longer had — the
  failure mode that produced the drift it was written to fix. Both now record
  it, including the two properties that are load-bearing rather than incidental:
  plan-mode blocking sits _above_ the confirmation prompt, and every branch
  answers through one helper so each `tool_call_id` gets exactly one
  `ToolMessage`. The collaborator sections now also distinguish the pure modules
  from the two that take the agent as their first argument.
- **The `fs_read` "JSON" mode row said validate-then-slice; the code slices
  first.** The page contradicted itself four lines later, where the correct
  order was already explained. Corrected, and the note about the misleading
  error message was dropped along with the message itself.
- **`.ruff_cache/` is ignored.** It was excluded only by the directory's own
  self-ignoring file, so any tooling not honoring that saw it as untracked.

### Internal

- Regression tests for each fix, verified by mutation rather than by passing:
  reverting any one of the six changes (each of the three import chains, the
  flag ordering, the local-assignment, the hoisted imports) fails the suite.
  The gate tests assert on `sys.modules` rather than on where the `import`
  statement sits, and stay meaningful whether or not torch and faiss are
  installed — an absent module can't be in `sys.modules`, so the "stays out"
  direction cannot false-pass, and the "loads when enabled" direction is
  skipped rather than asserted vacuously when the package is missing.

## [1.8.6] — 2026-07-31

Four latent bugs, all found by asking a different question than "do the tests
pass?" — mutation-testing the safety-critical paths, i.e. breaking each one on
purpose and checking whether anything failed. Two gates and one whole tool loop
turned out to be unprotected, and the reasoning-disable helper was mutating a
model nobody was about to call. No public-surface change.

### Fixed

- **Turning reasoning off for an internal call mutated a model that was often
  not the one being called.** Five call sites — the query classifier, task
  decomposition, both empty-turn salvages, and the main model call — disabled
  reasoning in place on `self.model`, then invoked something else: a worker's
  clone, or the tool-bound binding that actually receives the request. On the
  providers where reasoning is a plain attribute (Ollama's `reasoning`,
  `reasoning_effort`) the disable therefore never reached the object being
  invoked, so the retry ran with reasoning still on — the exact failure the
  disable exists to prevent, since a reasoning model leaves `content` empty and
  the salvage produces nothing to show. On `ChatBedrockConverse` it was worse
  than a no-op: applied to a bound model it raised **after** already popping
  `thinking`, so the saved state was discarded and reasoning stayed off for the
  rest of the session. All five now build a reasoning-disabled **twin**
  (`reasoning_utils.without_reasoning`) and invoke that, leaving the shared model
  untouched — which also removes a real thread-safety hazard, since parallel
  orchestrator waves and background sub-agents ran these on pool threads while
  the main turn was mid-call. The twin deep-copies `model_kwargs` /
  `additional_model_request_fields` before disabling, because `model_copy()` is
  shallow and would otherwise reach through into the shared model's own dicts;
  a model that cannot be copied falls back to the previous save/restore.
- **A sub-agent's tool failure could log an empty error.** The fix for an
  exception whose `str()` is empty — a bare `TimeoutError` — had been applied to
  the foreground tool loop only, so for a release a worker's timed-out tool call
  logged `Worker tool error:` and nothing after it. The same tool loop also
  never logged the tool-not-found warning the foreground logged, so a sub-agent
  calling a tool it had not been given left no trace in the log at all. Both
  paths now run one implementation, so neither can drift again.
- **Importing the tools package could abort the interpreter.** With a
  `VISION_MODEL_ID` configured, `server/tools/__init__` initialized the vision
  model at **import** time, pulling `BaseChatModel`→transformers→torch into any
  process that touched `mnemoai.server.tools`. Beyond costing ~3s, torch and
  faiss each vendor their own OpenMP runtime, and loading both aborts the
  process outright (`OMP: Error #15`) as soon as faiss runs a search — which is
  what episodic-memory search does. Vision initialization is now deferred to
  first use, so the import stays light regardless of what the config says.
  Its regression test has to hold where there is neither torch nor a
  `config.yaml` — every CI runner and every fresh clone — which ruled out the two
  obvious ways to write it: asserting `"torch" not in sys.modules` proves nothing
  where torch was never installed (it is not a dependency of this project), and
  the initialization is skipped entirely unless `VISION_MODEL_ID` is set, so a
  run with no config never reaches the path under test. Each subprocess therefore
  injects a vision config plus a sentinel controller that records its own
  construction, needing no heavy packages; confirmed by restoring the eager
  initialization and checking the suite fails for it under those conditions.

### Changed

- **The two tool-execution chokepoints are one loop.** Running a tool existed as
  two near-identical ~70-line copies — one for the main graph, one for
  sub-agents and orchestrator waves — of which ~39 lines were byte-identical.
  Both carried the plan-mode hard block and the destructive-tool confirmation
  gate, the two pieces of logic that must never differ between the paths. They
  now share `client/agent/tool_loop.py`; the remaining differences between the
  callers are parameters (`quiet`, the activity sink, batched spawns, the log
  label), not branches. `agent.py` is 160 lines shorter.
- **The safety gates now have tests that would notice them being removed.**
  Neither chokepoint had ever been _called_ by a test — both were pinned only by
  a source-text substring check, so deleting the confirmation gate from either
  one left the entire suite green. Verified the way the hole was found: each
  gate was broken on purpose and the suite now fails for every one, from both
  paths. Two of those checks had also been asserting against captured log
  output that could never arrive, since the application logger does not
  propagate to the root handler.
- **Three documentation corrections.** The tools reference said fourteen of the
  31 tools are always available; the real split is twenty-three always
  available and eight behind a feature toggle. `fs_read` accepts eight `mode`
  values, not seven. The README's link to the everyday-tools guide now matches
  the page's own title.

## [1.8.5] — 2026-07-30

Documentation release: the capability docs are rewritten task-first, every tool
the model can call is now documented, and three statements that were simply wrong
are corrected. One shipped config template changed, which is why this is a release
and not just a site rebuild.

### Added

- **A tools reference — all 31 tools, with their real parameters and limits.**
  Previously the largest documentation gap: `fs_read`'s seven modes
  (`Line`/`Search`/`Directory`/`CSV`/`JSON`/`JSONL`/`PDF`/`DOCX`) and `fs_write`'s
  four commands (`create`/`str_replace`/`insert`/`append`) appeared nowhere, so the
  only way to learn what they accepted was to read the source. Also newly
  documented: the read-before-write gate and its two errors (`must_read_first`,
  `stale_read`), `clear_completed_tasks`, the `use_skill` and `resume_agent` tool
  names, `glob_search`'s `include_ignored`, `grep_search`'s `offset` /
  `context_before` / `context_after`, and `web_search`'s six parameters.
- **A safety page** collecting the confirmation prompt, plan mode, the server-side
  safety floor, the git protections, and the read-before-write gate — previously
  scattered across a ten-topic page, or (for the URL/SSRF policy) absent.
- **A `~/.mnemoai` directory map**, with a "choose the right file" table. The
  answer to "where does my config go?" was spread across five pages and
  `paths.py`; `MEMORY.md`, `STEERING.md`, skills, agents, plans, task logs and the
  per-model stores now have one home.

### Changed

- **`FALLBACK_MODEL` removed from the three shipped `config.yaml` templates.** The
  key was read by nothing — it named a tiktoken _model_ where the code wants an
  _encoding_, and the counter it belonged to was replaced. Removing it is not
  breaking: an existing `config.yaml` that still lists it keeps working, since an
  unknown key is ignored. `OLLAMA_APPROXIMATION` stays — it is still read, though
  only for the episodic-memory size budget and not for conversation token
  counting, which the docs now say.
- **Capability docs lead with the task, not the subsystem.** Headings name what
  you're trying to do; each capability states the situation that should make you
  reach for it, what to type, and what you'll see. The ten-capability
  "Productivity tools" grab-bag is now everyday tools only, with safety, plan mode,
  and git split out.
- Docs gained a **Reference** section, separating exhaustive tables from the
  task-shaped guides.

### Fixed

- **The docs promised a ripgrep fallback that does not exist.** Two pages stated
  that `grep_search` "automatically falls back to a slower built-in search"
  without ripgrep. It does not: it returns `ripgrep (rg) not installed` and
  searches nothing. Both pages now say ripgrep is required, and the installation
  page no longer files it under "recommended".
- **`LLM.MAX_RETRIES` was documented as `3`**; the shipped template says `5`.
- Two pages still dated their behavior "as of 0.8.16".
- Malformed Markdown that silently degraded to plain text: `!!!` admonition bodies
  without the 4-space indent Material requires, and an unescaped `|` inside a table
  cell that collapsed an entire table into literal text.

## [1.8.4] — 2026-07-30

Three ways to get a conversation unstuck: let the assistant ask you when it's
genuinely blocked, fork a session that went the wrong way, and get a readable
transcript out of one.

### Added

- **`ask_user_question` — the model can put a decision back to you.** When it's
  blocked on a choice that is genuinely yours to make, it now offers 2–8 concrete
  options in a picker and continues from your answer, instead of guessing or
  writing out every alternative for you to sort through. Thin MCP stub +
  client-side interception (the same split as `exit_plan_mode`): the server is a
  piped subprocess and can't prompt a terminal.
  - **It can't be used where nobody would see it.** A sub-agent doesn't get the
    tool bound at all, and refuses it even if it asks anyway — a background one
    runs on a daemon thread with no terminal, so a picker there would block on a
    prompt that never paints. Off-TTY it reports itself unavailable rather than
    stalling a scripted run. Every refusal tells the model to decide and state its
    assumption, so a blocked question never becomes a hung turn.
  - Dismissing the picker (Esc) is a first-class answer: the model proceeds on its
    own judgment and says what it assumed, rather than re-asking.
  - The prompt guidance pushes toward acting on a sensible default — one question
    the user must answer costs more than a choice they can correct.

- **`/usage` — token totals for the session, per model.** Reports input, output and
  (where the provider offers them) cache tokens, accumulated across **every** model
  call — including the ones you never see: sub-agents, orchestrator workers, and the
  query router. Those are the point: a single delegated task measured **91,587 tokens
  across 8 calls** in testing, none of it previously visible anywhere.
  - **Distinct from `[Context: N]`.** That's the size of the prompt the next turn
    re-sends; `/usage` is cumulative spend for the conversation. The report says so
    explicitly, because conflating the two is the obvious misreading.
  - **No dollar cost, deliberately.** Pricing doesn't apply uniformly across the
    supported providers — Ollama and a local OpenAI-compatible server are
    marginal-free, SageMaker bills by endpoint-hour, and LiteLLM proxies an open set
    of models at prices this process can't know. A confidently wrong figure is worse
    than none, so it reports tokens and says whose numbers they are.
  - **A partial total can't masquerade as a complete one.** `usage_metadata` isn't
    populated by every provider, so a call that reports nothing is counted separately
    and the report flags the total as a lower bound rather than folding in zeros.
  - Resets on `/clear`, since that starts a fresh conversation.
- **`/export [md|txt] [path]` — a shareable transcript.** Writes the conversation
  as readable Markdown or plain text into the **current directory** (not the
  profile), for pasting into a bug report, a PR description, or a message. This is
  deliberately NOT `/save`: that writes re-importable JSON for `/load`, while an
  export is a one-way artifact optimized for a human reader. Tool _calls_ are kept
  as one-line summaries (a file body passed as an argument is replaced by its
  size); tool **results** are dropped, since a few thousand lines of file content
  is the single biggest source of noise. Injected context — the steering block and
  the prepended episodic-memory block — is stripped, because the user never typed
  it. Reasoning is opt-in (`/export reasoning`). The filename is derived from the
  opening prompt, so exports are identifiable in a directory listing.
- **`/branch [turn]` — fork a session and continue in the copy.** With no argument
  it shows a turn picker; `/branch 3` branches directly. The transcript up to that
  turn is copied to a new session, the live history is truncated to match, and this
  run continues writing into the fork. **The original is never modified** — that's
  the whole safety property: it stays resumable exactly as it was, so a branch that
  goes nowhere costs nothing. Forks are tagged `(branch @ turn N)` in the `--resume`
  picker, since a fork inherits its parent's opening prompt and the two rows would
  otherwise be indistinguishable.

### Fixed

- **`--resume` listed one conversation as several rows.** Resuming writes a _new_
  session file seeded with the whole prior conversation (so each file is
  self-contained and a resume-of-a-resume can't truncate the chain). But the picker
  listed one row per _file_, so a chat resumed three times appeared as **four
  near-identical rows** — each a strict superset of the last, all sharing the same
  opening prompt and therefore indistinguishable. On a real machine this turned 14
  conversations into 18 rows. A session that another session resumed is now hidden:
  its content lives entirely inside its successor, and the successor is the one you
  want to continue. Nothing is deleted, and naming a superseded id explicitly
  (`--resume <id>`) still resumes that exact point.
- **The longest conversation in the list reported the fewest turns.** The count came
  from turns taken in that _file_, which excludes inherited history — so a resumed
  chat holding 243 messages displayed as **"1 turn"** and read as the shortest entry
  in the list. Rows are now sized by the whole restorable conversation (that same
  row now reads "26 turns"), and a continued session is tagged `(continued)`.
- A `/branch` fork is deliberately **not** collapsed: it diverges from its parent
  rather than superseding it, so both remain real conversations and both are offered.
- **A resumed conversation opened with a wall of injected context.** The replay
  printed each stored user message _raw_, but a stored prompt still carries what the
  client prepended to it — so every `--resume` and `/load` began with a ~30-line dump
  of the episodic-memory block (a paragraph of comma-separated tool names and a
  similarity score), and the first real prompt lost its `>` marker inside it. The
  replay now shows only what you typed. The stripping rule (episodic block,
  `<steering>`/`<plan-mode-active>` reminders, auto-delivered sub-agent reports) moved
  into one shared `turn_view.user_prompt_text` used by the replay, the `--resume`
  picker label, and `/export` — it previously existed as three separate copies, and
  the replay was the one that never got it.
- **Arrow keys in every picker selected the wrong row.** A `RadioList` tracks the
  highlighted row separately from its committed value, and only its own
  enter/space binding reconciles the two — but the dialog overrides Enter to
  confirm directly (skipping the Tab-to-OK step). So ↓↓Enter returned the **first**
  entry: `--resume` and `/load` opened a different conversation than the one
  highlighted, and the `/load` Delete button deleted the wrong file. Pressing
  Space before Enter happened to work, which is why it went unnoticed. Found while
  driving the new picker through a real pty — no unit test could have caught it,
  since the bug lives in the widget's two-value split.

### Security

Four holes closed in the tool-safety layer. Each was reachable by the model
itself, so none depended on a hostile user.

- **The model could approve its own dangerous git operation.** A tool that can only
  detect danger once it inspects the repo (is this branch pushed? is main
  protected?) refuses and returns `requires_confirmation` — but that payload went to
  the MODEL, whose documented next step was to re-call with
  `allow_dangerous=True, reason="user confirmed"`. Nobody ever asked a human. The
  refusal is now resolved with the **user** at the tool chokepoint, and
  `allow_dangerous=True` is itself confirmed **before** the call, tool-agnostically —
  so setting it on the first call prompts too, not just on a retry. A declined
  operation returns a refusal that tells the model not to override it.
- **`rm -rf / --no-preserve-root` was not blocked.** The root-target check required
  the path at the END of the command, and `--no-preserve-root` is precisely the flag
  GNU `rm` needs to actually erase `/`. Now blocked in any flag order, with or
  without `sudo`, and for `-fr` as well as `-rf`.
- **`execute_bash` bypassed the write-path policy entirely.** `sh -c 'echo x >
/etc/hosts'` isn't a "write tool" call, so neither the system-path floor nor the
  read-before-write gate applied. Shell write targets are now extracted
  (redirections, `tee`, `cp`/`mv`, `dd of=`, `chmod`, nested `sh -c`) and run through
  the **same** `classify_write_path` the file tools use — one floor, not a second
  copy. Ordinary writes to your home, project or temp dir are unaffected.
- **Quoting hid a dangerous git flag.** The danger patterns matched the raw string
  while execution used `shlex` argv, so `push origin main --for"ce"` reached git as
  `--force` completely unwarned. Both now derive from the same tokenization, which
  also removed the symmetric false positive (a commit message mentioning
  `reset --hard` no longer trips the check).

### Fixed (correctness)

Six bugs found by auditing rather than by hitting them — several had been failing
silently since they were written.

- **`file_edit`'s confirmation prompt showed no filename.** The server takes
  `file_path` but every client-side reader looked for `path`, so the gate asked you
  to approve a bare "edit" with no target — and `is_plan_file("")` was false, which
  meant plan mode blocked **every** `file_edit`, including one writing the plan
  itself. All readers now go through one `plan_policy.write_target()` accepting both
  spellings, and fail safe (no recognized key → still blocked).
- **User profiling was recording garbage.** It received the whole conversation after
  every turn, so turn N re-counted every earlier prompt: `interaction_count` grew as
  N²/2 (a real profile reached **62,977**) and each trait's EMA washed out —
  `technical_level` had collapsed to 0.0002 and `abstraction` sat pinned at 0.5.
  Now scoped to the current turn via the same `current_turn_messages` the reflector
  already used for this exact bug. An inflated profile is repaired once at load: the
  count is re-estimated, saturated traits reset so they can re-learn, and
  mid-range traits kept since they still carry signal.
- **Reading a file counted as a tool failure.** The reflector flagged any result
  containing `error:` / `failed:` / `traceback`, so a successful `fs_read` of almost
  any source file was logged as a failure — inflating the failure metrics and writing
  junk strategies into the playbook, which is injected into the system prompt. It now
  trusts the structured `{"error": true}` that tools already return, and only falls
  back to text matching when the result _is_ an error message (short, led by the
  indicator) rather than merely contains one.
- **Chunk token counting never worked.** `tiktoken.get_encoding("gpt-4")` — a model
  name, not an encoding name — raised on every call, and a bare `except` returned
  `len(text)//4`. Every RAG chunk boundary was ~12% off, invisibly. Now uses the same
  encoding as the rest of the app, so chunk sizes can't drift from the context budget
  they're measured against.
- **`clear_documents` could never work.** It assigned to `store.metadatas` and
  `store.index`, which are getter-only properties — so it raised `AttributeError` on
  every call, swallowed into a generic error message. Both backends already had a
  correct `clear()`; the tool calls it.
- **A renderer fallback called a method that doesn't exist.** When the markdown
  parser raised, the recovery path called `self._render_text_block()` — turning a
  recoverable render failure into an `AttributeError` that lost the whole answer. It
  now renders the tail the same way the normal path does.

Follow-ups from a second review of those fixes — three were incomplete and one of
them was a regression I introduced:

- **A failed shell command was being recorded as a success.** Trusting the JSON
  envelope meant `execute_bash` — which reports the command's failure as an
  `exit_status` and carries no `error` key — read as a win. So a `pytest` run exiting
  2 with a stderr full of tracebacks taught the playbook nothing, losing the most
  common real failure in a coding session. A non-zero `exit_status`/`return_code` is
  now a failure. (Introduced by the reflector fix above; caught before release.)
- **A turn's tool calls were cut off mid-analysis.** The turn boundary was "the last
  message with role `user`", but an encoded tool RESULT also carries that role — so a
  turn with tool calls was truncated at its last result. The boundary is now a real
  prompt.
- **Tool results counted as interactions.** Same root cause on the profile side: one
  turn with three tool calls scored four, which also tripped the "enough data" gate in
  the profile summary after a single tool-heavy turn. `record_tool_outcome` and
  episodic storage were fed the whole session too, so `tool_patterns` totals kept
  growing quadratically — both are now scoped to the turn.
- **`clear_documents` left the keyword index intact.** It dropped the vectors but not
  the separately-built BM25 corpus, which kept a tokenized copy of every "cleared"
  document for the life of the process. Nothing was served from it only because a
  bounds check discarded indices past the emptied metadata list — isolation resting
  on an accident. `SessionRAG.clear()` now drops both, and clearing before any ingest
  says "nothing to clear" instead of claiming the backend doesn't support it.
- **An ambiguous write target now fails closed.** `file_edit(path=<plan>, file_path=<other>)`
  had plan mode check one path while the server wrote the other. Conflicting spellings
  resolve to "no target", which the gate blocks.
- **A legacy profile could replace an answer with an error.** A profile predating
  `interaction_count` raised `KeyError` during the post-turn hook — after the answer
  had already streamed — surfacing as "Something went wrong". Profiling failures are
  now contained.
- **Your writing style was being profiled from text you didn't write, with the sign
  inverted.** Every trait is scored from the message text, and that text was the RAW
  stored message — so with a five-entry episodic block prepended, "fix it" (6 chars)
  measured 496 and pushed `verbosity` toward "detailed". A terse user was profiled as
  verbose on every turn episodic memory injected, and `technical_level` /
  `abstraction` were keyword-scored over a paragraph of tool names. Traits are now
  scored on what you actually typed.
- **An inflated `interaction_count` resets instead of being estimated.** Inverting
  N²/2 assumed the old increment was one per turn when it was one per _message_, so
  the estimate overshot by √(1+tools) — 2-3× for a tool-heavy user. Its only consumer
  is the "enough data to profile you" gate, and a confidently wrong number there is
  worse than starting over, so the count resets to 0 and re-accrues honestly.

## [1.8.3] — 2026-07-29

MCP transport fixes, all traced back to one real session where a long-running
tool call failed four times with a blank error message and Esc appeared to do
nothing.

### Fixed

- **A tool's own timeout is now honored by the transport.** Every call was capped
  at `LLM.MCP_CALL_TIMEOUT` (default 300s) regardless of what the tool was asked
  to wait for, so `wait_for_task(timeout_seconds=1500)` could **never** complete —
  the client gave up while the server was still dutifully waiting, once every five
  minutes. The per-call deadline is now derived from the tool's own
  `timeout_seconds` / `timeout` argument plus headroom, treating the configured
  value as a floor rather than a ceiling — and bounded above (1 hour), since that
  argument is model-supplied and validated nowhere, so an absurd value would
  otherwise become an effectively unbounded wait.
- **Esc now cancels a tool call in progress instead of at its deadline.** The
  worker waited in a single `Future.result(timeout=…)`, which parks in
  `threading.Condition.wait` — a C-level acquire that only notices the injected
  `KeyboardInterrupt` when it _returns_. So cancelling during a tool call showed
  `(cancelling…)` and then hung for the whole deadline, with the next message
  stuck in the queue behind it (measured: a cancel 1s into a 600s wait landed at
  600s). The wait is now sliced and consults the agent's existing cooperative
  cancel event between slices, so a cancel lands within a tick — verified at 1.0s
  against a 630s deadline. Teardown is deliberately exempt: `shutdown()` often
  runs right after a cancelled turn, and honoring that still-set flag there would
  abort the disconnect and orphan the server subprocess.
- **A timed-out call says what happened.** `concurrent.futures.TimeoutError`
  stringifies to the empty string, and every log and re-raise site interpolated
  it — so a timeout surfaced as a bare `Tool execution error:` with nothing after
  the colon, indistinguishable from a crash. Timeouts now carry the tool name, the
  deadline that applied, and the knob to change; the model-facing formatter falls
  back to the exception class rather than emitting a bare `Error:`. The call is
  deliberately **not** retried: the request was already delivered, so the tool may
  well have run, and repeating it could duplicate a commit, a file edit, or a
  background build.
- **Recovering from a dead MCP server no longer dumps a traceback.** The stdio and
  session contexts were entered in one coroutine and exited in another; since each
  runs as its own asyncio task, `anyio`'s task-affine cancel scopes rejected it
  with `RuntimeError: Attempted to exit cancel scope in a different task than it
was entered in`. Reconnection still worked, but printed an alarming stack trace
  and left the dead subprocess's pipes unreaped. A single long-lived task now owns
  both contexts for the life of the connection, so entry and exit are the same
  task by construction, and teardown is bounded so a wedged server can't hang exit.
- **Compaction records its boundary in the session transcript.** The marker was
  defined but never called from the compaction path. Purely informational — the
  transcript is append-only, so `--resume` already restored the full conversation
  either side of a compaction — but the boundary is otherwise invisible in a log
  that never loses anything.

## [1.8.2] — 2026-07-28

### Fixed

- **A startup warning no longer looks like the model failed to connect.** A
  message printed while the boot spinner was animating landed on the spinner's
  own line with nothing erasing it first, so an unrelated warning rendered as
  `⠸ Connecting model.✗ MCP server 'time' failed to start; skipping.` — reading
  as though connecting the model was what went wrong. The message's newline then
  pushed the animation onto a fresh line, and since only the final line is
  cleared on exit, a stale `⠿ Connecting model…` was left stranded above the
  welcome banner. Console output now suspends the spinner, clears its line, emits
  the message, and resumes — so the warning stands alone and the spinner leaves
  nothing behind. Applied at the `utils.console` chokepoint, so every startup
  message gets it rather than just the one that surfaced the bug.

## [1.8.1] — 2026-07-28

A bug-fix release, mostly about session transcripts: three separate defects made
`--resume` show conversations that were never yours, label the real ones
unreadably, and restore only part of them. Plus an episodic-retrieval fix and two
internal consolidations. No public-surface change.

### Fixed

- **`mcp` is capped below 2.0.** `mcp 2.0.0` removed `mcp.server.fastmcp` — the
  server API every tool in `server/tools/` imports (renamed to
  `mcp.server.mcpserver` with a different surface) — so a fresh install resolving
  onto the 2.x line failed at import time and took the whole test suite with it.
  The requirement is `mcp[cli]>=1.26.0,<2` in `pyproject.toml` and, as an
  explicit exception to the single-source rule, restated in `requirements.txt` /
  `requirements-dev.txt` so a `pip install -r` can't drift onto 2.x either. The
  cap lifts with the port to the 2.x server API.
- **A conversation is no longer partially recorded when a turn fails.** A turn
  killed by a mid-flight error (a dropped provider connection, an MCP failure)
  left its work in the live conversation but never reached the session
  transcript, so recording silently stopped at the last _successful_ turn while
  the chat carried on — one 408-message conversation had recorded only 64
  messages, and `--resume` restored just that fragment even though `/save` held
  everything. The failure was invisible because transcript writes are
  best-effort by design. Every way a turn can end now writes before propagating
  the error, so the transcript can't drift from the history on screen.
- **Resuming a session now carries the whole conversation forward.** A session
  file only held the turns that happened after it was created, so continuing a
  resumed (or `/load`-ed) conversation produced a transcript starting mid-thread
  — resuming _that_ replayed a stump, and each resume-of-a-resume truncated the
  chain further. The restored history is now copied into the new session's file,
  making every file a complete record; the session resumed _from_ is never
  modified, so the same point can be resumed again.
- **`--resume` labels show what you actually typed.** Retrieved episodic memory,
  steering instructions, the plan-mode banner and auto-delivered sub-agent
  reports are prepended to a prompt before it is stored, and the picker showed
  them verbatim — so unrelated sessions all rendered as
  `[Episodic Memory - Similar Past Tasks] 1. "hello" → …` and none could be told
  apart. Labels are now extracted with injected context stripped, by the same
  code the `/load` dialog uses. A resumed session you never typed into is also
  no longer offered as a duplicate row of the session it restored.
- **Running the integration test tier no longer writes into your own app home.**
  That tier drives a real client, so each run recorded its test prompts as
  genuine sessions in `~/.mnemoai` — they appeared in `--resume` as
  conversations reading `Hello` or `What is the capital of France?` — and could
  also leave an episode behind in episodic memory (a trivial `"hello"` episode
  then got retrieved and injected as a "similar past task"). The tier now runs
  against a throwaway `MNEMOAI_HOME` while still using the configured provider.
  Existing stray entries from earlier runs are not removed automatically; delete
  the leftover files under `{profile}/sessions/` if you want a clean picker.
- **Episodic retrieval no longer silently drops episodes.** The ChromaDB backend
  keyed hybrid-search candidates by an episode's searchable text
  (`task + solution + tools`), which is not unique: two runs of the same task
  collapsed into one candidate and only one could survive the merge, so asking
  the same thing twice with different outcomes kept just one of them. Candidates
  are now keyed by the episode's unique id — a store with three matching
  episodes returns three (it returned two).

### Changed

- **Hybrid (semantic + BM25) ranking is now one shared implementation**
  (`utils/hybrid_search.py`) instead of three near-copies in the episodic Chroma
  store, the episodic FAISS store, and the session RAG store — a scoring change
  had to land three times before, and the copies had already drifted. Ranking is
  also deterministic now: the previous copies iterated a `set`, so equally scored
  results could come back in a different order run to run.
- **`BaseModelController` holds the behavior its subclasses were duplicating** —
  shared config reads and provider dispatch, so the LLM, vision, and embeddings
  controllers declare their supported providers instead of repeating an
  `if/elif` chain each.

## [1.8.0] — 2026-07-27

A correctness + hardening release from a full codebase audit. **Minor bump, not a
patch:** two changes are user-visible beyond a bugfix — existing episodic memory
stores are reset once, and two internal agent methods were removed. Read
"Upgrade notes" before installing.

### Upgrade notes

- **Episodic memory is reset once on first run.** Both backends now produce
  cosine-in-[0,1] scores (see Fixed → retrieval thresholds), and the scoring
  version rides in the embedding fingerprint, so an existing store is rebuilt
  through the already-tested migration path rather than silently mixing two
  incomparable scales. Learned episodes are re-learnable scratch, but they ARE
  lost. The ACE playbook, `MEMORY.md`, saved conversations, and `--resume`
  sessions are unaffected.
- **Removed `agent.steer()` / `agent.clear_steering()`** and the mid-turn
  steering queue (dead since the intake was retired in 1.7.x — the UI had no
  producer, so the drain always returned empty). `client/agent/steering.py` is
  now **`cancellation.py`**, holding only the cancel primitives it always
  really provided. Recovery path: `git log -- src/mnemoai/client/agent/steering.py`.
  (Unrelated to **STEERING.md**, the user-authored always-on instructions
  feature, which is untouched and live — the two only ever shared a name.)

### Fixed

- **Long sessions silently lost their learned context.** The compaction rebuild
  re-fetched the base system prompt and re-injected only skills + sub-agents, so
  after the FIRST compaction `MEMORY.md`, the user-profile block, and the ACE
  playbook were gone for the rest of the session — defeating "learns and
  remembers" precisely in the long sessions where it matters. All session-start
  blocks now come from one `context_injection.build_session_blocks()` used by
  **both** the session-start and compaction paths, so they can't drift again (a
  prior fix had patched only the skills instance of this same bug). A
  memory-read failure degrades the prompt instead of aborting the compaction —
  losing a block beats overflowing the window. **`/clear` had the same defect**
  and is fixed with it.
- **Hitting the safety step limit destroyed the whole turn.** `GraphRecursionError`
  carries no state and `graph.invoke()` returns nothing, so every tool call and
  assistant message the turn produced was discarded: the user's next message
  ("continue", "run the tests") arrived with no record that the work had ever
  happened, and the reply shown was the **previous** turn's answer — a confident
  response to a different question. The graph is now streamed
  (`stream_mode="values"`), so the last snapshot before the limit is committed
  exactly like a completed turn's, and it reaches the `--resume` transcript too.
  A turn that produced nothing is closed with the interrupted marker instead of
  being left dangling.
- **`git_safe` was broken for its own documented example, and injectable.** It
  split the command with `str.split()`, so `commit -m 'Add feature'` — the
  example in the tool's own docstring — committed with the message `Add`. The
  same split let git-level options through: `-c core.pager=…`,
  `-c alias.x='!sh …'`, `--exec-path`, `--upload-pack` are arbitrary-command
  execution that no danger pattern inspected. Now parsed with `shlex` (quoted
  arguments survive; unbalanced quotes are refused) and those option families
  are rejected before anything runs. `git_commit_safe` also shlex-splits
  `add_files`, so a quoted path with spaces stages as one file.
- **Retrieval thresholds weren't comparable across backends.** One knob,
  `EPISODIC_MEMORY.RETRIEVAL_THRESHOLD`, gated a `1/(1+L2)` mapping on ChromaDB
  and a raw inner product on FAISS, so switching `VECTOR_STORE` silently changed
  recall. Both now return cosine rescaled to [0, 1] (0.5 = orthogonal) with
  vectors L2-normalized on write and query. Verified identical on both backends
  for identical / orthogonal / opposite inputs.
- **Reflection re-analyzed the whole session every turn.** The reflector received
  all of `agent.messages` each turn, so earlier tool calls were re-counted:
  `total_tool_calls` grew triangularly (1, 3, 6, 10, 15 over five turns instead
  of 1, 2, 3, 4, 5) and duplicate strategies kept re-bumping their confidence.
  It now analyzes only the current turn, falling back to the full list when
  history was compacted away. Also fixed `_find_tool_result`, which could
  attribute the first result of a repeated tool to every later call of it.
- **SageMaker silently dropped every tool.** `bind_tools` was a no-op returning
  `self`, so an agentic assistant ran with zero tools and no signal. It now logs
  a loud one-time WARNING naming the endpoint and the consequence; the README no
  longer advertises SageMaker without that caveat.
- **A broken embedding model degraded retrieval silently.** The
  embedding→Jaccard similarity fallback was an `except: pass`, so word-overlap
  scoring replaced semantic search with no indication. It now warns once.
- **Symlinks escaped the write policy.** `path_policy` normalized with
  `abspath`, not `realpath`, so a symlink into a protected system directory
  wrote straight through. Now resolved before classification, and the policy is
  applied to `execute_bash` / `start_background_task` working directories too.
- **`web_crawler` could reach internal services (SSRF).** It validated only the
  URL scheme, so cloud metadata endpoints (`169.254.169.254`), `localhost`, and
  RFC1918 addresses were crawlable — and fetched text becomes model input. A new
  `safety/url_policy` resolves the host and refuses non-public addresses
  (loopback, link-local, private, reserved) and non-http(s) schemes.
- **The RAG store unpickled untrusted data.** `rag/faiss_store` used
  `pickle.load` on a sidecar file, with a fallback to a world-writable
  `/tmp/rag_store*` path — a planted file was code execution on a shared host.
  Metadata is now JSON under a new `.meta.json` name (so an old pickle is never
  read at all, rather than needing a migration), and the `/tmp` fallback is gone.
- **Learned state could be corrupted by a crash or a second terminal tab.**
  `MEMORY.md`, `playbook.json`, `metrics.json`, and the user profile were
  truncate-then-write with no lock. All now use one shared
  `utils/atomic_write` helper (temp file + `os.replace`); a failed serialize
  leaves the previous file intact. `todo_manager` delegates to it instead of
  keeping its own copy.

### Changed

- **CI installs once and enforces a coverage floor.** `requirements-dev.txt` is
  now just `-e .[dev,docs]`, so `pyproject.toml` is the single source for
  dependencies and the two can't drift; the workflow drops its duplicate
  runtime install. Unit tests run with `--cov-fail-under=60` (measured 64%) as a
  regression guard, not a target.
- **Fixed four broken documentation links and a stale `black` badge** in the
  README (the project uses ruff), plus stale claims in the development docs.
  A new `test_doc_links.py` checks documented routes against the mkdocs nav, so
  these can't rot silently again — root docs aren't in the nav, so `--strict`
  never caught them.

## [1.7.9] — 2026-07-27

### Added

- **Resume a previous session from the directory you launched in
  (`--resume` / `--continue`).** Every session is now recorded automatically to an
  append-only transcript, scoped to the launch directory, so resuming inside a
  project only ever offers that project's sessions:
  `mnemoai --resume` (pick from a list showing how long ago each ran, its turn
  count, and its opening prompt), `mnemoai --resume <session-id>` (restore one
  directly; a partial id suffix resolves), `mnemoai --continue` (most recent, no
  prompt). Picking one replays the conversation into the terminal and continues
  where you left off, reusing the same decode + render path as `/load` so a
  resumed chat looks identical to a loaded one.
  - **`/save` and `/load` are untouched and unaffected.** They remain the
    user-curated path — a conversation you deliberately keep, under a name you
    choose, in `conversations/` — and are **never** expired or deleted by session
    cleanup. Resuming never adopts a saved file either: `resume_session`
    deliberately does not set `current_conversation_path`, so a bare `/save`
    after a resume writes a new file rather than overwriting one of yours.
  - Storage: `~/.mnemoai/{profile}/sessions/{sanitized-cwd}/session_<ts>_<pid>_<rand>.jsonl`,
    one JSON object per line (`meta` / `turn` / `compact`). Deep sibling paths
    can't collide — a long directory name is truncated with a hash of the full
    path. **Expiry is age-based**: `SESSION_MAX_AGE_DAYS` (root-level key,
    default **30**; `0` disables recording entirely), swept across every project
    directory at startup. The picker's 20-entry cap bounds only what is
    _offered_, never what is kept, so a cap can't silently drop a session.
  - **Append-only by design, not a mirror of the live history.** Compaction
    _replaces_ `agent.messages` wholesale, so a mirror would lose the
    summarized-away turns and resume a stub; each turn is buffered and flushed
    once, and a `compact` marker records that the live context shrank without
    discarding the transcript. Sub-agent runs are deliberately excluded (they run
    on their own isolated message list; logging them would interleave a second
    conversation). A **cancelled** turn is recorded too, including its
    interrupted marker, so a resumed session shows exactly what the live one did.
    Every write is best-effort — a transcript can never break the turn you're
    waiting on.

### Fixed

- **Cancelling the resume picker started a fresh session instead of exiting.**
  Pressing `Esc` fell through into a brand-new conversation — surprising, since
  the app was launched purely to resume, and it left another empty session behind.
  Cancelling (or naming a session that doesn't exist) now exits cleanly.
  Relatedly, a single available session no longer auto-resumes without showing the
  picker, which had denied any chance to back out.
- **A resumed transcript rendered above the logo instead of above the prompt.**
  The welcome banner was printed by the chat loop _after_ the restore, pushing the
  entire replayed conversation off the top. The banner is now shown before the
  replay, so it reads top-to-bottom: logo → commands → your conversation → prompt.
- **Turn-less session files accumulated on disk.** The `meta` record is written at
  startup, before we know whether you'll type anything, so a launch you
  immediately quit (or a cancelled `--resume`) left a file with no turns. Those
  were already hidden from the picker — which is why a directory could show three
  files but only two entries — and are now removed at exit, after confirming the
  file really has zero turns so it can never delete a session another process
  wrote.

## [1.7.8] — 2026-07-27

### Fixed

- **Large-context Bedrock turns timed out and could never recover.** No read
  timeout was configured anywhere, so **botocore's 60-second default** applied —
  and that default bounds the wait for the **first response byte**, not idle time.
  On a large conversation the model spends that entire budget on prefill and
  reasoning before emitting anything: measured against live Bedrock, a
  ~440k-token turn at `REASONING_EFFORT: max` took **123 seconds to first byte**,
  so every attempt died at 60s. Worse, `read timed out` classifies as a transient
  network error, so all `MAX_RETRIES` attempts re-sent the same oversized prompt
  and re-paid the same doomed prefill — minutes of retries that could not
  possibly succeed, ending in "Stream failed after retries (connection issue)".
  `STREAM_IDLE_TIMEOUT` never applied, because the connection died before the
  stream began. The Bedrock client is now built with an explicit botocore
  `Config`: `read_timeout` from the new **`LLM.REQUEST_TIMEOUT`** (default
  **600s**, an order of magnitude above boto's default) plus a short
  `connect_timeout` (**`LLM.CONNECT_TIMEOUT`**, default 30s — connecting is fast
  and must not be inflated to the request ceiling). botocore's **own** retries are
  disabled (`max_attempts: 0`) so this layer's retry/backoff isn't multiplied by
  boto's. Verified end-to-end through the app's own model object: the identical
  workload that failed with `ReadTimeoutError` now streams 2259 chunks to
  completion. Both keys are documented in the Bedrock config template.
- **A read-timeout failure now names the knob instead of blaming the network.**
  The message said the connection was lost and to just send again — misleading
  when the real cause is a request that timed out before the model's first token,
  which resending reproduces exactly. A read timeout now adds a line pointing at
  `LLM.REQUEST_TIMEOUT` (or `/compact` to shrink the prompt).

## [1.7.7] — 2026-07-26

### Changed

- **Cancelling a turn no longer erases what you asked.** Since 1.5.1 a cancel
  (Esc/Ctrl+C) rolled the **whole** turn out of history — the user message plus any
  partial work. That fixed a dangling _unanswered_ question being picked up out of
  context on the next turn, but it over-corrected: the cancelled request vanished
  entirely, so following a cancel with "sorry, continue" left the model with no
  idea what it had been asked (it replied that there was no pending work), and any
  partial result it had already reported was lost. The user message now **stays**
  and an explicit `[Turn interrupted by the user before it completed.]` assistant
  marker is appended instead, which fixes both failure modes at once: the turn
  reads as explicitly terminated (so the model doesn't silently resume it or
  re-run the tool calls it was making), while a follow-up "continue" still has the
  original request in context. `SYSTEM_PROMPT` gains matching guidance for how to
  read the marker (resume on "continue"; otherwise answer the new question and
  leave the interrupted work alone). The marker is bookkeeping, never surfaced as
  an answer — `_last_visible_from` skips it when salvaging a cut-short reply.

### Fixed

- **A prompt change could silently never reach installs from 1.6.3–1.7.6.**
  `prompts.yaml` changed in 1.6.3 but that version's hash was never appended to
  `_PRISTINE_BUNDLED_PROMPTS_HASHES`, so an otherwise-pristine installed copy was
  treated as user-customized and left untouched — and since the bundled-fallback
  loader only fills **missing** keys, an edit to an EXISTING prompt key (like the
  interruption guidance above) would never have arrived. The missing hash is now
  tracked, and the guard test was strengthened: it previously only asserted the set
  was non-empty (which is why this slipped), and now verifies each shipped
  version's actual hash is recognized.

## [1.7.6] — 2026-07-25

### Fixed

- **A failed scrollback notice on the cancel path was silent, and its awaitable
  was dropped.** The two remaining fire-and-forget `run_in_terminal` calls — the
  `(cancelling…)` line in `_request_cancel` and the
  `(press Ctrl+C again to force-quit)` hint — discarded the returned awaitable, so
  a failed terminal write produced no notice and surfaced only as an unretrieved-
  task warning. `run_in_terminal` also chains on `app._running_in_terminal_f`, so
  an un-awaited call is exactly the shape that can stall a _later_ in-terminal
  write. Both now go through a shared `_notice()` helper that awaits the write in
  a task with its own error trap (best-effort: a notice can never raise into a key
  handler or affect control flow), and is a no-op when the loop is missing or
  closing. No `run_in_terminal` call in the UI is un-awaited any more. This is the
  hardening companion to the 1.7.5 confirm-prompt hang fix — these were the prime
  suspects for that hang, and although the mechanism was never reproduced, they
  are no longer able to hide a failure or stall the terminal chain.

## [1.7.5] — 2026-07-25

### Fixed

- **The app could freeze permanently at a bare cursor when a tool asked for
  confirmation.** The confirmation prompt is painted by the pinned app while the
  worker thread blocks on an `Event`, and two defects combined into an
  unrecoverable hang: the scrollback echo was dispatched **fire-and-forget**
  (`call_soon_threadsafe(lambda: run_in_terminal(echo))` discarded the returned
  awaitable), so an echo that failed or never completed was entirely silent and
  could leave the pinned prompt unpainted — and the wait itself was an unbounded
  `done.wait()`, so with no prompt on screen there was nothing to press and
  **Esc/Ctrl+C cannot reach a thread parked in `Event.wait()`**. The result was a
  turn stuck with no `▶ Run shell command?` visible and no way out. The echo is
  now awaited as a real task with its own error trap and always invalidates the
  app in a `finally` (a broken echo can no longer block the prompt), and the wait
  polls the cancel token so Esc/Ctrl+C always releases the worker. A cancelled
  wait **always denies** rather than returning the caller's `default` — plan
  approval passes `default="approve"`, and a prompt nobody saw must never count
  as approval (it maps to "keep planning" instead). Pending confirm state is now
  fully torn down, so no stale, unanswerable question lingers in the status line.
  Covered by regression tests that **hang against the previous code**.

## [1.7.4] — 2026-07-25

### Fixed

- **A `RuntimeWarning` about `toolConfig` leaked into the terminal after a
  cancelled turn, and the conversation history was silently rewritten.** The
  orchestrator's task-decomposition call (`_decompose_task`) binds **no tools** —
  it uses the raw model, not the tool-bound one — but it was handed the _real_
  prior conversation, including replayable `tool_use`/`tool_result` blocks. With no
  matching tool schema, Bedrock Converse logged `Tool messages (toolUse/toolResult)
detected without toolConfig. Converting tool blocks to text format…`, raised a
  `RuntimeWarning` that printed over the pinned UI, and **rewrote the messages
  itself**; a stricter provider can reject the request outright. It surfaced most
  visibly right after cancelling a turn, because a cancelled turn leaves tool
  blocks in history that the next turn's decomposer then replays. Tool blocks are
  now flattened to plain text before that call via a new pure
  `message_sanitizer.flatten_tool_blocks()` — an assistant tool call becomes
  `[called tools: name(args)]` and a result becomes `[tool result from name]: …`,
  so the decomposer still knows what ran without replaying it. Tool-block-free
  messages pass through untouched (same object) and the input is never mutated.
  The other tool-less auxiliary calls were audited and were already safe: the
  router sends a rendered text context block, compaction rebuilds clean
  string-content messages, and the aggregator builds fresh messages.

## [1.7.3] — 2026-07-25

Streaming was silently off on the standard Bedrock (Converse) path for newer
Claude models, and `STREAM` wasn't tunable there at all.

### Fixed

- **Newer Claude on `TYPE: bedrock` streamed nothing — one blob after the full
  generation.** `ChatBedrockConverse` auto-derives `disable_streaming` from a
  **hardcoded model-id allowlist** that lags new releases (langchain-aws 1.6.0
  matches `claude-3`/`claude-sonnet-4`/`claude-opus-4`/`claude-haiku-4` but NOT
  `claude-opus-5` / `claude-sonnet-5`), and LangChain's `stream()` **silently
  defers to `invoke()`** when that flag is set — so a newer Claude produced a
  single chunk with no error and no warning. The Bedrock path now sets
  `disable_streaming` explicitly from `STREAM` (the library's validator only
  auto-derives it when the key is absent, so an explicit value wins). Verified
  against live Bedrock: `global.anthropic.claude-opus-5` went from 1 chunk to
  **22 chunks with a ~1.3 s first token**, including with tools bound and adaptive
  thinking. `EXTRA_PARAMS` is still applied last, so a deliberate override wins.
- **`STREAM` is now tunable for Bedrock via `/params`.** It was missing from the
  provider registry, so it was both ignored by the Converse path and never offered
  by `/params` (every other streaming provider already exposed it). Also documents
  `STREAM` in the Bedrock config example and adds the `xhigh` effort level to that
  example's `REASONING_EFFORT` comment.

### Changed

- **Removed every unused import across `src/` and `tests/`** (25 in total; the
  repo-wide `ruff --select I,F401` check is now clean). Two were deliberate and
  were preserved rather than deleted: the `make_urls_clickable` package
  re-export (documented + imported from the package) is kept with an explicit
  `noqa`, and the two `try: import langchain … except ImportError` availability
  probes keep working (one now imports the module rather than unused classes).

## [1.7.2] — 2026-07-25

Reasoning-effort fixes on the Bedrock / Mantle Anthropic paths — no
public-surface change (the fix is behavioral: correct requests per provider).

### Fixed

- **Non-Anthropic Bedrock/Mantle models no longer get Anthropic-only thinking
  fields.** The extended-thinking injection was gated only on
  `_claude_version(name)` returning `None`/`≥(4,6)`, but `None` also means "not a
  Claude model at all" — so a standard-Bedrock (`ChatBedrockConverse`) model like
  `amazon.nova-pro-v1:0`, `mistral.*`, `meta.llama*`, `us.deepseek.r1-v1:0`, or
  `qwen.*` with `REASONING_EFFORT`/`REASONING` set had Anthropic's
  `{thinking:{type:adaptive}, output_config:{effort}}` (or the legacy
  `budget_tokens` form) injected into its request, which the Converse API rejects.
  A new `is_anthropic_model(name)` predicate (substring-based, robust to the
  `anthropic.` provider prefix and `us./eu./apac./global.` inference-profile
  prefixes and the bare `claude-` id) now gates every Anthropic-thinking injection
  site (standard Bedrock, Mantle `anthropic` protocol, direct Anthropic). A
  non-Claude model gets no injection (many families — e.g. DeepSeek-R1 — reason
  automatically); `EXTRA_PARAMS` stays the escape hatch for a deliberate
  provider-specific reasoning field. Claude behavior is unchanged (all id shapes
  still get adaptive thinking + `output_config.effort`). `_claude_version` is now a
  version probe only, never the is-Claude signal.

### Added

- **`xhigh` reasoning-effort support** (added by Anthropic with Opus 4.7; between
  `high` and `max`, the recommended coding/agentic effort on Opus 4.7+/Sonnet 5/
  Fable 5). It already passed through to `output_config.effort` on the adaptive
  path; `_EFFORT_TO_TOKENS` now includes it (`xhigh: 24576`) so the `max_tokens`
  headroom bump is consistent, and the two duplicated per-effort maps in
  `llm_controller` (standard Bedrock + SageMaker) were de-duplicated onto the
  single `mantle_factory._EFFORT_TO_TOKENS`.

## [1.7.1] — 2026-07-24

Internal structural refactor of the `client/` package — no behavior change, no
public-surface change. The two largest modules were decomposed into cohesive,
individually-testable collaborators behind thin delegators, paying down the
long-standing "God object" debt in the agent loop and the client facade.

### Changed

- **`client/agent/agent.py` shrank 3314 → 2672 lines** by extracting five sibling
  modules of pure/near-pure logic, each with the class keeping thin delegating
  methods (and class-attribute aliases for the marker/category constants) so the
  full method surface is preserved: `response_parsing.py` (provider-agnostic
  thinking/visible-text extraction), `stream_policy.py` (network/overflow/empty
  classifiers + backoff), `steering.py` (mid-turn steer queue + cancel token),
  `confirmation_gate.py` (the destructive-tool confirm gate), and
  `subagent_runner.py` (the `spawn_agent`/`resume_agent`/background envelope
  around the stationary worker loop).
- **`client/client.py` shrank 1373 → 933 lines** by extracting `context_injection.py`
  (system-prompt assembly + episodic/steering/plan-mode injection + similarity +
  context-token count) and `session_artifacts.py` (per-instance session-id minting
  - RAG/chunk-cache init/flush), and relocating `StreamingCallbackHandler` to
    `client/ui/streaming_callback.py` (re-exported from `client.py`).
- **Deduplicated the sub-agent run lifecycle.** The identical "finish a run that
  returned normally — mark _stopped_ if cancelled mid-flight, else record the
  final answer and mark _done_" transition, previously repeated across the
  foreground/background spawn and orchestrator-subtask paths, is now a single
  `ActivitySink.finish_ok()`.

All 1376 unit tests pass, the import-sort gate is clean, and the extractions were
verified behavior-identical against the prior code (AST-level equivalence plus a
live end-to-end run of the agent + sub-agent paths).

## [1.7.0] — 2026-07-24

Live visibility and control over sub-agents in the pinned TUI, a provider-
agnostic reasoning-block replay fix, and a faster/less-frozen startup.

### Added

- **Live "agents" panel + navigation (pinned TUI).** While hidden sub-agents run
  — foreground `spawn_agent` batches, background spawns, AND orchestrator subtask
  workers (all previously invisible) — a panel pins below the input showing each
  agent's status dot (● running / ✓ done / ✗ stopped-or-failed), type,
  description, tool count, and live elapsed time. **Ctrl+A** enters nav-mode
  (↑/↓ select, **Enter** opens a full-screen scrollable detail view of that
  agent's tool calls / results / errors / final answer rendered like the main
  thread, **Esc** exits). The panel shows while any agent runs and hides once all
  finish. A new thread-safe `AgentActivityStore` (`client/agent/agent_activity.py`)
  captures the per-agent activity feed the panel reads.
- **Per-agent stop, foreground OR background.** In nav-mode, **`x`** stops the
  selected agent; **Ctrl+X Ctrl+K** stops all running agents from anywhere
  (not just nav-mode). Stopping is cooperative (a per-run cancel the worker loop
  polls each iteration), so a background agent can be stopped across turns.
  Stopping _all_ also cancels the turn (a foreground batch blocks the turn, so
  the turn must stop too — otherwise it resumed with the agents' partial
  reports). A stopping agent shows an animated `cancelling…` label until it
  actually finishes.
- **Startup progress spinner.** Typing `mnemoai` now shows an animated
  `⠦ Loading libraries… → Starting tools server… → Connecting model…` line
  (dependency-free `utils/startup_loader.py`) instead of a frozen terminal during
  the multi-second boot, clearing before the welcome banner.

### Fixed

- **Reasoning-block replay 400s in sub-agents (provider-agnostic).** A sub-agent
  re-feeding its accumulated assistant messages could send a malformed reasoning
  block the provider rejects on a later turn — most visibly Anthropic extended
  thinking's `messages.N.content.0.thinking.thinking: Field required` (a
  streamed `signature_delta` accumulates a `thinking` block that keeps its
  signature but loses the inner text). The main agent was immune (it scrubs every
  turn); the quiet worker/sub-agent path was not. Now a shared egress guard
  (`message_sanitizer.strip_malformed_reasoning`, wired into `_stream_response`
  and the invoke fallbacks) **normalizes** the block (re-injects the empty inner
  field, keeping the signature and block ordering) rather than dropping it —
  dropping would break Anthropic's thinking-first ordering on a tool-use turn.
  Provider-aware: preserves OpenAI Responses `reasoning` items carrying
  `id`/`encrypted_content` and Bedrock signed `reasoning_content`, so no provider
  regresses. Sub-agent tool errors are now attributed to the specific agent
  instead of surfacing as anonymous log lines.

### Changed

- **Faster, less-frozen startup.** Deferred `langchain_litellm` (which pulls
  litellm→transformers→torch, ~1.5s of dead weight for every non-litellm
  provider) to a lazy import in `llm_controller`, and moved the server's
  `VisionModelController` / summarization `LangChainLLMController` imports off the
  unconditional tool-registration path — trimming seconds from both the client
  and the MCP server subprocess. The heavy client stack now imports under the
  startup spinner rather than blocking a blank terminal.
- The agent-detail viewer scrolls with ↑/↓ · PgUp/PgDn · g/G and shows a
  scrollbar; `format_duration` now renders hours (`1h2m5s`).

## [1.6.4] — 2026-07-23

Internal cleanup of the `client/` tree (no public-surface change): a verified
dead-code sweep + behavior-preserving simplifications, net ≈ −120 LOC. Every
change was adversarially verified to preserve behavior.

### Fixed

- **Episodic ChromaDB lost its embedding-model fingerprint on clear/reconnect.**
  `ChromaEpisodicStore.clear()` and the moved-DB `_reconnect()` recreated the
  collection with only a description, dropping the `embed_fingerprint` stamp that
  the normal create path adds — so a cleared or reconnected store silently
  reverted to the legacy un-fingerprinted path on the next open. Both now recreate
  through `_create_collection()`, keeping the stamp consistent.

### Removed

- **Dead methods with no remaining callers** (verified against dynamic dispatch,
  framework contracts, tests, and prompts): `LangGraphAgent.get_thinking` /
  `has_pending_background`; `LangGraphClient._inject_playbook_context` (superseded
  by `_get_playbook_context`); `PinnedPromptReader._echo_steered` (orphaned by the
  retired UI-steering path); `MCPClientWrapper.get_tools` / `MultiMCPClient.get_tools`
  (superseded by `list_tools_sync`); `Reflector.get_metrics` /
  `PlaybookEntry.to_prompt_text` / `Reflector._extract_tool_results`; and the
  write-only `_context_depth` counter in `MCPClientWrapper`.

### Changed

- **De-duplicated client-side code without changing behavior.** The client-side
  tool-interception (`exit_plan_mode`/`spawn_agent`/`resume_agent`) was copy-pasted
  across both tool chokepoints — now one `_client_side_tool_message` helper (the
  `_execute_tools` path still reuses a batched parallel-spawn result; the worker
  path still runs inline). The last-HumanMessage-query scan (3 copies), the
  sub-agent result footer (2 copies), the block-on-Event confirm state machine
  (`confirm_ui`/`plan_approval_ui` → `_await_confirm`), the double-Ctrl+C exit
  counter (both REPL loops → `_note_interrupt`), the episodic-store success tail
  (`_store_success_episode`), the compaction per-message loop (`_summary_texts`),
  the prior-session sweep (`_repoint_session`), the system-prompt block append,
  the TTY-detection predicate, the spinner frames, and the `turn_view` Update-diff
  header were each consolidated to a single source. Also corrected a misleading
  comment in `subagents.py` (the denylist uses `_parse_denylist` with inverse
  `*`/`all` semantics, not `_parse_tools`).

## [1.6.3] — 2026-07-23

### Fixed

- **Orchestrated tasks lost the conversation history.** A task routed to the
  orchestrator (decompose → workers) only ever saw the current query — the prior
  conversation was discarded. So a context-dependent follow-up ("write the issue
  to a file", "fix it") was decomposed and executed with no idea what "the
  issue"/"it" referred to, and the worker fabricated content (e.g. a placeholder
  file instead of the just-drafted GitHub issue). The decomposer and every worker
  now receive the **real prior messages** — the same conversation the main agent
  sees — inserted between their system prompt and the current subtask. The
  history is **uncapped** (it's already bounded to the model window by the
  compaction layer, so an extra cap would be redundant and could drop the very
  turn a follow-up refers to) and tool-pair-repaired so strict providers don't
  reject it. Spawned sub-agents (`spawn_agent`/`resume_agent`) keep their
  deliberate context isolation — only the orchestrator threads history in.

### Changed

- **The orchestrator no longer runs a degenerate single-subtask worker.**
  Decomposition now happens in its own graph node (`decompose`) placed between
  the classifier and the agent-vs-orchestrator branch, and the route is chosen on
  the **actual subtask count**: a task that decomposes to one atomic step (the
  old "Step 1/1") falls back to the normal streaming `agent` — full toolset,
  native conversation history, live token stream — instead of a hidden quiet
  worker whose output only surfaced at the end. Only a genuine multi-step plan
  (≥2 subtasks) is owned by the orchestrator. Trivial and plan-execution turns
  skip decomposition entirely (no extra LLM call), and direct `_orchestrate`
  callers still decompose internally, so nothing that reaches the orchestrator
  another way changes.
- **Clarified in the system prompt that decomposition is framework-driven.** The
  model is now told that multi-part requests are split into steps automatically
  before its turn — there is no "orchestrator" tool to call — so `spawn_agent`
  stays clearly framed as the model's own delegation/parallelism tool, distinct
  from the framework's task decomposition.

## [1.6.2] — 2026-07-22

Compaction was firing far too early and taking minutes on a large-window,
reasoning-heavy model. Four fixes, all provider-agnostic:

### Fixed

- **Compaction triggered ~2× too early.** The high-water check compared against
  an _estimate_ (tiktoken over the serialized message JSON × a per-provider
  safety multiplier, e.g. 1.5× for Anthropic/Mantle) that inflates the real
  prompt ~2×, so it fired while the true context was well under the window. It
  now triggers on the provider's **exact `input_tokens`** (the same figure shown
  as `[Context: N]`) when a turn has run, using the estimate only as a fallback
  before any turn — so the trigger and the displayed count agree.
- **Compaction ran the summary on the main reasoning model with extended
  thinking ON** (the dominant cost — minutes per call on a max-reasoning model).
  Summaries now use a **reasoning-disabled variant of the same model**, built via
  the controller by clearing `REASONING`/`REASONING_EFFORT` (and, for Ollama,
  `verbose`), so it works for every provider and no-ops where thinking doesn't
  exist. Set `LLM.SUMMARIZATION_THINK: true` to keep thinking on.
- **The map-reduce summary was sequential.** It folded each batch into a rolling
  summary one call at a time, so wall-clock was the SUM of all batch calls. It's
  now a **parallel map + single reduce**: batches are summarized concurrently
  (bounded by `LLM.SUBAGENT_MAX_CONCURRENCY`), then one reduce call folds the
  ordered partials (and any prior summary) into a coherent whole. A single batch
  skips the reduce. Wall-clock drops to ~one batch + one reduce.
- **The cheap tool-result eviction ran only on the proactive path.** It now runs
  first on **every** compaction path (manual `/compact`, post-turn auto, and the
  overflow backstop) — in tool-heavy sessions this shrinks old grep/read/web
  dumps (no LLM call) so the summary that follows has far less to read, and often
  avoids the LLM summary entirely.

## [1.6.1] — 2026-07-22

### Fixed

- **`grep_search` result order was nondeterministic**, so the new `offset`
  pagination could return a different page on repeated runs (ripgrep searches
  files in parallel and doesn't order them). It now runs with `--sort path` for a
  stable cross-file order — a prerequisite for meaningful paging. (CI: the Tests
  workflow now installs ripgrep so the `grep_search` tests run there; they skip
  gracefully where `rg` is absent.)

## [1.6.0] — 2026-07-22

Tool-quality pass: sharpen the built-in MCP tools where they lagged,
**without removing any capability mnemoai's tools already offer** (multi-mode
`fs_read`, multi-command `fs_write`, the server-side
catastrophic-command/path floor, `git_safe`, RAG offload, structured search
JSON, glob's no-ripgrep fallback, etc. all preserved). Adds read-before-write
safety, line-numbered reads, richer search (context/paging/width-cap), web
recency/region + crawl timeout, per-instance todos, per-agent sub-agent
tool-denylist + model overrides, line-numbered + streamed file reads,
encoding/BOM/line-ending-preserving edits, and skill argument substitution —
plus several latent-bug fixes surfaced along the way (two `grep_search` modes
that silently returned nothing, unbounded `execute_bash` output, orphaned
background children, a footgun path-relocation, whole-file read OOM).

### Added

- **Read-before-write / staleness gate (server-side).** A new in-process
  read-state registry (`server/tools/read_state.py`) records the on-disk mtime of
  every file the model reads via `fs_read`, and `fs_write`/`file_edit` now refuse
  to modify an **existing** file that was never read, or that **changed on disk
  since it was last read** — the model would otherwise clobber content it never
  saw. Creating a brand-new file needs no prior read; a successful write
  re-baselines the file so a chained edit in the same turn isn't flagged. This is
  a separate layer from the client confirmation gate and the `path_policy` floor,
  never prompts (returns a normal tool error).
- **Line-numbered file reads.** `fs_read` Line mode now prefixes each line with a
  `cat -n` style number gutter so the model can cite lines precisely. `file_edit`
  is resilient to it: if `old_string` isn't found it strips a pasted gutter and
  retries against the raw content (a raw `new_string` is written verbatim, so a
  legitimate leading `digits<TAB>` — e.g. TSV — is never mangled).
- **`grep_search`: asymmetric context + pagination + long-line cap.** New
  `context_before`/`context_after` (with `context_lines` as the symmetric
  shorthand) — and context lines are now actually **returned** (flagged
  `is_context`), each owned by exactly one match so nothing leaks or duplicates
  across pages. New `offset` for paging (counts matches only). Over-long matching
  lines are capped (`MAX_LINE_CHARS`) so a minified/base64 line can't blow up the
  result.
- **`glob_search`: skips noise dirs + honest newest-first.** Excludes common
  vendored/build dirs by default (`.git`, `node_modules`, `.venv`, `__pycache__`,
  `build`, `dist`, …; opt out with `include_ignored=True`) — kept on stdlib glob
  so it still works with **no ripgrep installed** (glob's differentiator).
- **`web_search`: recency + region.** Optional `freshness` (pd/pw/pm/py or a date
  range), `country`, and `ui_lang` (forwarded only when set); a citation +
  current-year instruction in the tool description; applied filters + current year
  echoed in the result.
- **`web_crawler`: explicit page timeout.** Configurable via
  `WEB_CRAWL.PAGE_TIMEOUT_MS` (code default 60 s), so a slow site can't hang the
  crawl.
- **Per-agent sub-agent `disallowed-tools` + `model`.** A custom
  `~/.mnemoai/agents/*.md` can now declare a tool **denylist** (`disallowed-tools`,
  applied after the allowlist; `*`/`all` = deny everything) and a per-agent
  **model** override (a same-provider model with only the name swapped — a cheap
  agent can run a cheaper model). Fixes a latent gap where the loader advertised
  `model` but ignored it.
- **Skill argument substitution.** `use_skill(name, arguments=…)` now substitutes
  `$ARGUMENTS` and `${SKILL_DIR}`/`${CLAUDE_SKILL_DIR}` (and the unbraced forms) in
  the skill body — so a skill can take input and reference its own bundled scripts
  by absolute path. Substitution is a single word-bounded pass (a value that
  contains a `${SKILL_DIR}` literal, or a `$SKILL_DIRECTORY` token, is left intact).
- **Encoding-aware editing.** `file_edit` and `fs_write` (str_replace/insert/append)
  now detect and **preserve a file's encoding, BOM, and line ending** on an
  in-place edit — a CRLF file stays CRLF, a UTF-16/BOM file keeps its BOM (and is
  now editable at all, where it previously bounced to binary-steering). Creating a
  brand-new file still writes plain UTF-8 LF.

### Fixed

- **`grep_search` was silently broken in two of its three modes.** Passing
  `--files-with-matches` / `--count` **overrides** ripgrep's `--json`, so those
  modes emitted plain text the parser couldn't read and returned **empty results**
  (and `files_with_matches` is the default mode). `grep_search` now always runs
  `--json` and derives every mode from the parsed match events.
- **`grep_search` `max_results` was a PER-FILE cap, not a total.** It mapped to
  ripgrep `--max-count` (N matches _per file_); `files_with_matches`/`count` had no
  total limit at all. It is now a true **total** cap across all files in every
  mode, applied after parsing, with a `truncated` flag (`max_results=0` =
  unlimited, matching `glob_search`).
- **`grep_search` crashed on a non-UTF-8 matching line.** A match on a line (or
  path) ripgrep couldn't decode carries `bytes` (base64), not `text`, which raised
  `KeyError` and **aborted the entire search** in all modes. Such lines are now
  decoded leniently (replacement chars) so one bad byte can't kill the search.
- **`grep_search` reported an invalid regex as "0 matches".** ripgrep exits 2 on a
  malformed pattern; the tool ignored the exit code and returned success/empty.
  Exit 2 now surfaces as an error — but **only when no matches were parsed**, so
  an unreadable file among readable ones (which also exits 2) keeps its valid
  matches and reports the problem non-fatally in a `warnings` field.
- **`glob_search` truncated before sorting.** With a result cap it returned the
  first N files glob happened to yield (arbitrary order) then sorted only those,
  so "newest first" was false when capped. It now collects, sorts, then slices —
  the returned N really are the newest (bounded by a scan ceiling on huge trees).
- **`execute_bash` returned unbounded output.** A chatty command's full
  stdout/stderr went into the tool result verbatim (token/OOM risk). Output is now
  middle-truncated to a 30 KB ceiling (head + tail kept, with a truncation
  marker), at the source — ahead of the client's downstream compaction.
- **`cancel_background_task` orphaned grandchild processes.** It sent
  `os.kill(pid, 9)` to the shell only; background tasks now spawn with
  `start_new_session=True` and cancel kills the whole **process group** (`killpg`),
  matching `execute_bash`.
- **A cancelled background task was reported as `failed`.** After the reliable
  group-kill, the reader thread's terminal-status write clobbered the `cancelled`
  status with `failed` (return code −9); the worker now leaves an
  already-`cancelled` task alone. Also: `get_task_output`'s `total_lines` (a
  `"lines" in dir()` bug that returned 0) is fixed, and cancelling a task in its
  brief startup window returns a clear "still starting, retry" message instead of
  "No PID found".
- **`fs_write` silently relocated relative paths into `~`.** `_resolve_path`
  moved a relative path (e.g. `notes.txt`) into the home directory based on its
  extension — a surprise-overwrite footgun. Relative paths now resolve against the
  current working directory (like any normal tool); only an absolute path or an
  explicit `~` goes elsewhere.
- **`fs_read` loaded the whole file into memory.** Line mode did `readlines()`
  before slicing, so reading a few lines of a multi-hundred-MB file could OOM. It
  now streams — one pass to count lines (for `total_lines` + negative-index
  resolution), a second that materializes only the requested range and stops
  early — with byte-identical results. All token-budgeting/truncation preserved.
- **Skill listing re-scanned + re-parsed every turn.** `SkillStore` now memoizes
  the scan, invalidated by a cheap per-skill `SKILL.md` mtime signature (edits
  still apply next turn), and the always-on `<available_skills>` block is bounded
  by a token budget (with the char cap as a hard secondary), not chars alone.

## [1.5.9] — 2026-07-21

### Added

- **Delete saved conversations from the `/load` picker.** The `/load` dialog now
  has a **Delete** button (next to OK/Cancel); pressing it asks "Delete this
  conversation? Yes/No", removes the highlighted conversation on Yes, then reopens
  the refreshed picker — so you can prune saved chats without leaving the dialog.
  Deletion is guarded to only remove a `*.json` file inside the profile's
  `conversations/` dir, and clears the open-conversation pointer if you delete the
  one currently loaded.

### Fixed

- **No more silent turns — every answer path now displays.** The app's display
  contract was "streaming prints the answer", but several paths produce the final
  text WITHOUT streaming and so showed nothing (only `[Context: N]`): the
  **orchestrator single-subtask** result (the reported case — a `use the skill`
  turn ended blank), the aggregation-fallback concatenation, and the
  context-overflow / stream-error / recursion terminal messages, and the empty-
  final salvage. A turn now tracks whether it streamed a visible answer and, at
  the single point where every path returns its final text, emits anything that
  wasn't streamed (via the same `●`-marker + `CodeFormatter` rendering, so it
  looks identical) — a central safety net that closes all of these at once and
  can't regress on a future path. Normal streamed turns are unaffected (the flag
  prevents double-printing). The `client.query()` error handler also now prints
  its failure message instead of only logging it. As part of centralizing on this
  one display point, three pre-existing ad-hoc `print()`s (the truncated-token and
  reasoning-only fallbacks in `_call_model`, and the worker-salvage fallback) were
  removed — they returned the same text that the net now renders, so they'd have
  printed it twice; `_emit_answer` is now the single chokepoint for every
  non-streamed answer.

## [1.5.8] — 2026-07-21

### Fixed

- **Submitting a large paste no longer garbles the scrollback.** The 1.5.5
  "expand the paste in scrollback on submit" echo printed the entire expanded
  body (e.g. an 18 KB / 757-line file) as one monolithic write into the
  pinned-input terminal; that overran the non-full-screen app's cursor-relative
  repaint (which tracks how many rows it last drew), so its refresh emitted
  cursor-up/erase-line sequences over the just-scrolled text and the paste showed
  as a garbled single line of overlapping fragments. Now the scrollback echo of a
  large paste is **capped to a head+tail preview** (first ~12 and last ~6 lines
  with a `… +N lines …` marker), dimmed **per line**, and written with **CRLF**
  line endings (raw mode leaves `ONLCR` off, so a bare `\n` staircased and
  desynced the renderer). The **model still receives the full, untruncated
  paste** — only the on-screen echo is capped. Applies to both immediately-sent
  and **queued** messages (both flow through the same echo path); a queued
  paste's live `> … (queued)` line already showed the compact placeholder. Small
  pastes are still echoed in full.

## [1.5.7] — 2026-07-21

### Fixed

- **Switching the embedding model (or its dimension) no longer crashes every
  turn — the episodic store migrates.** An existing episodic vector store is only
  comparable to vectors from the SAME embedding model: a different dimension makes
  every query raise (`Collection expecting embedding with dimension of X, got Y`),
  and even a **same-dimension** switch (e.g. Ollama `qwen3-embedding`@1024 →
  Cohere `embed-v4`@1024) yields semantically incompatible vectors. Each store now
  records a **model fingerprint** (`type|name|endpoint|dimension`) and **resets**
  when it changes — for both ChromaDB (stamped in collection metadata) and FAISS
  (a sidecar `episodic_fingerprint.txt`). Episodic memory is model-scoped,
  re-learnable scratch, so a reset is the safe migration (old vectors can't be
  reused across models). Triggered by a new model OR a changed `DIMENSION` (via
  `/model` or `/params`). Also hardens both stores against a moved/removed persist
  dir (ChromaDB reconnect on SQLite code 1032; FAISS dir-recreate).
- **`DIMENSION` is now sent to Bedrock embedders that support resizing.**
  Previously `DIMENSION` only shaped the fallback vector and was never sent, so
  e.g. `us.cohere.embed-v4:0` with `DIMENSION: 1024` still returned 1536-dim
  vectors (then mismatched the collection). It is now passed with the correct
  provider-specific parameter — Cohere v4 `output_dimension`, Titan v2
  `dimensions` — and only when explicitly set (a model without a resize knob, e.g.
  Titan v1, is never forced). When `DIMENSION` is unset, the real output size is
  determined by a one-time embed probe (used for the fingerprint), never guessed.
- **Stale per-session artifacts no longer accumulate on `/model`/`/params`
  restart.** Those commands re-exec the process, minting a fresh `session_id` and
  a new `chunk_cache_*.db` / `rag_store_*` while the prior run's was orphaned
  (os.execv runs no exit cleanup). `session_id` now embeds the instance id
  (`{profile}_{ts}_{instance_id}`), so each instance's artifacts are physically
  unique — this (a) lets an instance safely delete its OWN prior-session artifacts
  on restart (cleanup keyed to its own per-instance pointer), and (b) fixes a
  latent bug where two tabs started in the same second on one profile shared —
  and could clobber — a single on-disk store/cache.

## [1.5.6] — 2026-07-21

### Fixed

- **Bedrock embeddings now work for every embedding-model family, not just
  Titan.** The Bedrock embed path only ever sent the Amazon Titan request schema
  (`{"inputText": …}`), so switching to a Cohere or Nova embedding model failed
  with `ValidationException: Malformed request` and silently degraded to the
  sha256 fallback. `_embed_bedrock` now dispatches on the model id (all schemas
  verified live against Bedrock): **Cohere** (`cohere.embed-*` incl. v4) →
  batched `{"texts", "input_type", "embedding_types":["float"]}` →
  `{"embeddings":{"float":[…]}}`; **Nova Multimodal**
  (`…nova-…multimodal-embed…`) → per-text `taskType`/`singleEmbeddingParams` →
  `{"embeddings":[{"embedding":[…]}]}`; **Titan** (and any unrecognized model) →
  the existing per-text `{"inputText"}` → `{"embedding"}`.
- **A moved/locked episodic-memory database no longer crashes a turn.** After the
  answer was already produced, an episodic-store write could raise (ChromaDB
  SQLite code 1032 "readonly database moved" when the store dir was moved/replaced
  under the open connection, or a FAISS persist-dir that vanished) and surface as
  a turn error. Now: (1) the post-answer learning side effects (episodic store,
  reflection, memory auto-extraction) are each **best-effort** — a failure is
  logged, never shown as a turn error, since the user already has their answer;
  (2) the **ChromaDB** store reopens the client + collection once and retries on a
  moved-DB error; and (3) the **FAISS** store recreates its persist dir and
  retries the write once (faiss wraps file errors in `RuntimeError`, handled).

### Changed

- **A message typed while the assistant is working is now QUEUED, not steered.**
  It runs as its **own turn after** the current one ends (shown as a dim
  `> … (queued)` line), it is never folded into the running turn.
  This fixes a stranding bug where a message steered during the final,
  tool-call-free model call was never drained at turn end and leaked into
  the **next** turn (echoed `(steering →)` at submit, then answered later). The
  agent-side mid-turn steering machinery (`agent.steer`/`_drain_steering`) is
  left intact but dormant. Background sub-agent auto-delivery is unaffected — it
  uses a separate mechanism (`drain_background_completions`), not the steer queue.

## [1.5.5] — 2026-07-21

### Changed

- **A submitted paste now expands to its full text in the scrollback (dimmed),
  not just for the model.** 1.5.4 kept the `[Pasted text #N +M lines]`
  placeholder in the scrollback echo after submit; now the placeholder is
  expanded back to the real text **everywhere on submit** — the scrollback echo
  and the mid-turn-steering echo both show what was actually sent — so the
  collapsed view is only while you're composing the input. The **pasted portion
  is rendered gray** (dim) so it reads as distinct from your typed text; the
  model still receives plain text. (The input-box collapse is unchanged.)
- **Backspace deletes a paste placeholder as one token.** With the cursor right
  after a `[Pasted text #N …]` placeholder in the input, Backspace now removes
  the **whole placeholder** in a single keystroke (and forgets its stored
  content) instead of erasing it one character at a time.

## [1.5.4] — 2026-07-20

### Added

- **Large pastes collapse to a compact placeholder in the input.** Pasting a long
  transcript or file no longer floods the input box: a paste that's long by
  character (> 800) or line count (> 2 line breaks) is shown as
  `[Pasted text #N +M lines]` while the full text is stored aside, and on submit
  the placeholder is **expanded back to the real text for the model** (the
  scrollback echo stays collapsed). `M` counts line breaks (visual lines − 1).
  Placeholder-looking strings inside the pasted content are never re-expanded
  (reverse-offset splice). Short pastes insert verbatim as before; the non-TTY plain
  loop is unaffected.

### Fixed

- **Embeddings no longer fail (and spam warnings) on a long input.** A long paste
  could exceed the embedding runner's real limit — which is often FAR below the
  context length the model reports (e.g. `qwen3-embedding:0.6b` reports 32k but
  its runner EOFs at ~5.5k tokens) — so truncation trusted the wrong number, the
  over-length input was sent whole, the runner dropped the socket (a bare `EOF` /
  `400`), and the code then re-sent the identical input twice more before
  degrading to (degraded) sha256 fallback — all logged loudly at WARNING. Now:
  (1) an embed input rejected as too big (including a bare EOF/400, which is how
  llama.cpp signals it) triggers an **adaptive shrink-and-retry** that halves the
  token budget until it fits and **remembers the working limit** per (provider,
  model, endpoint) so later inputs truncate proactively — model- and
  provider-agnostic, no per-model table, so it works for every provider; (2) we
  no longer trust a model's reported generation context for embeddings; and
  (3) the self-healing retry/shrink/truncate steps log at **DEBUG** (silent at
  the default level) — only a genuine degrade-to-fallback is an **ERROR**, so a
  normal recovery is silent.
- **A mid-stream API error (e.g. a 500 `api_error` "Internal server error") no
  longer wedges the turn or makes cancel unresponsive.** On such an error the
  streaming path used to fall back to a **blocking, non-cancellable, un-retried**
  `model.invoke()` — so the turn froze and Esc/Ctrl+C couldn't preempt the
  in-flight C-level request. Two changes fix it: (1) 500 / `api_error` /
  `internal server error` / `overloaded_error` are now classified as **transient**,
  so they get the existing **abortable streaming retry** with backoff instead of
  the blocking fallback; and (2) the blocking non-streaming fallback inside `_stream_once`
  is **removed** — a stream error now re-raises so the retry wrapper (`_stream_response`)
  owns recovery, keeping the turn interruptible and idle-timeout-protected throughout.
  If a stream error survives all retries, `_call_model` ends the turn with a
  clear, non-crashing message (the conversation stays intact) rather than hanging.

## [1.5.3] — 2026-07-20

### Changed

- **`spawn_agent` now runs in the background by default.** Previously a spawned
  sub-agent ran foreground (the parent blocked on its report) unless the model
  passed `run_in_background=true`. The default is now **background**: the call
  returns immediately with an id, the main turn stays responsive, and the report
  is auto-delivered when the sub-agent finishes — so delegation no longer stalls
  the conversation. The model is instructed to pass `run_in_background=false`
  only when it needs the report to continue the same turn (its next step depends
  on the answer) or when a `general-purpose` sub-agent must edit/run un-approved
  things (a background one runs headless and would auto-skip them). The prompt
  refresh reaches existing installs via the pristine-`prompts.yaml` mechanism.

### Fixed

- **Cancelling (Esc / Ctrl+C) is now immediate, even mid-stream or mid-retry.**
  Cancellation worked only by injecting an async `KeyboardInterrupt` into the
  worker thread — but that **cannot preempt a thread parked in a C-level blocking
  wait** (verified: it can't break a `queue.get(timeout=…)` or `time.sleep`). So
  when a turn was in the stalled-stream idle wait (`STREAM_IDLE_TIMEOUT`, default
  120s) or a network-retry backoff (`time.sleep`, up to 30s), pressing cancel
  showed `(cancelling…)` but the turn kept running until that wait naturally
  returned. Now the agent carries a cooperative cancel token (a `threading.Event`,
  the mnemoai analog of an `AbortSignal`) that the UI **sets on Esc/Ctrl+C**: the
  idle-timeout stream wait polls it on a short (0.25s) interval and the retry
  backoff `.wait()`s on it instead of sleeping — so both wake **instantly** and
  the turn tears down in a fraction of a second instead of up to 120s.

## [1.5.2] — 2026-07-20

### Fixed

- **Spinner no longer dies for the rest of an orchestrator step after the first
  confirmation.** During a sequential orchestrator step (or a foreground
  sub-agent) the worker runs **quiet** — it deliberately doesn't drive the shared
  spinner, relying on the caller's. But the destructive-tool **confirmation
  prompt** stops the spinner to borrow the terminal, and nothing in the quiet
  path ever restarted it — so after the first `Proceed?` the terminal sat at a
  bare `>` while work continued, looking frozen. `_prompt_confirm` now snapshots
  the spinner's state (active + label) before prompting and **restores it
  afterward** (on accept, decline, or trust-all), so the step keeps animating
  through every subsequent tool call. The foreground path is unaffected (its
  spinner is already stopped before the tool loop, so there's nothing to restore
  — `_invoke_tool` still drives it as before).
- **Multiple concurrent instances (e.g. terminal tabs) no longer corrupt or
  delete each other's RAG/chunk-cache session data.** Each running instance and
  its MCP subprocess exchanged the active `session_id` through two **fixed-name**
  files in the shared profile dir (`rag_session_id.txt`, `chunk_session_id.txt`),
  so a second tab starting **overwrote** the first tab's pointer — its MCP
  subprocess then read the wrong session and cross-contaminated RAG/chunk
  lookups. Worse, on `/exit` (and `/clear`) the flush wildcard-swept **every**
  `rag_store_*` / `chunk_cache_*` and the shared pointer files, **wiping other
  live tabs' data**. Now: (1) the session-pointer files are **per-instance**,
  namespaced by a new `MNEMOAI_INSTANCE_ID` that the client pins before spawning
  the MCP subprocess (which inherits it), so both halves of one instance resolve
  the same pointer and different tabs never collide; (2) the exit/`clear`
  flush deletes **only this instance's own** store + cache + pointer (scoped to
  its `session_id`), never a wildcard sweep; (3) a startup housekeeping pass
  (`sweep_old_rag_artifacts`, mirroring the plan sweep) prunes only **stale**
  (default 7-day) orphans left by a crashed instance, so nothing accumulates
  while a concurrently-running instance's fresh files stay untouched.

## [1.5.1] — 2026-07-20

### Changed

- **System/sub-agent/orchestrator prompts overhauled with delegation guidance.**
  The `SYSTEM_PROMPT` now tells the model **when and how to use `spawn_agent`** —
  delegate to keep context clean or parallelize independent work; use
  `explore`/`plan` (read-only) vs `general-purpose`; and **honor explicit
  requests literally** (run agents "in parallel" → multiple `spawn_agent` calls
  in one turn; "in the background"/"don't wait" → `run_in_background=true`). It
  also gains scope-discipline ("do only what was asked"), a reversibility/blast-
  radius + confirm-before taxonomy for risky actions, prefer-dedicated-tools-over-
  bash guidance, symmetric faithful-reporting (don't over-claim _or_ over-hedge),
  and tighter conciseness/formatting rules. The `ORCHESTRATOR_PROMPT` now
  **defers to the agent** (a single passthrough subtask) when the user explicitly
  asks for sub-agents, so an explicit "run 2 background agents" reaches
  `spawn_agent` instead of being decomposed into framework workers. Sub-agent and
  aggregator prompts tightened (absolute paths, load-bearing snippets only,
  retry-alternate-spellings, lead-with-the-answer).
- **Prompt improvements now reach existing installs.** `prompts.yaml` is refreshed
  in place on upgrade **when the installed copy is still pristine** (a version we
  shipped, unmodified — hash in `_PRISTINE_BUNDLED_PROMPTS_HASHES`), so edits to
  existing prompt keys land for users who never customized it, while a
  user-customized `prompts.yaml` is left untouched. (The bundled-fallback loader
  only fills _missing_ keys, so it couldn't deliver edits to existing ones.)

### Fixed

- **Loaded conversations now render Markdown like a live turn.** On `/load`, the
  replayed assistant answers were printed **raw** — literal `**bold**`, ` ``` `
  code fences, `##` headings, and list markers — instead of going through the
  `CodeFormatter` every live turn uses. `turn_view.render_conversation` now
  renders each answer through the **same** formatter (new `render_markdown`
  helper: a fresh `CodeFormatter` with stdout captured), so a reloaded answer
  looks identical to a freshly-streamed one (Markdown, Pygments-highlighted
  code). Falls back to raw text if rendering fails, so a load can never break.
- **Orchestrator now shows a spinner during sequential steps.** A lone/sequential
  subtask ran quiet with no spinner, so the UI looked finished while a step was
  still going. Each sequential step now drives a `step N/M: …` spinner.
- **Cancelling a turn no longer leaves the question in history.** When a turn was
  cancelled (Esc) mid-flight, the user's message had already been appended to the
  conversation but no answer followed — so the **next** turn saw a dangling,
  unanswered user turn and the model addressed it out of context (e.g. cancel
  "what's the World Cup result?", then ask "who are you?" → it answered both).
  `invoke()` now snapshots the history length before the turn and, on cancel
  (`KeyboardInterrupt`), rolls the **whole** turn back — the user message plus any
  partial tool/assistant work — so a cancelled request leaves history exactly as
  it was. (Distinct from the 1.5.0 fix that cleared the mid-turn _steering_ queue
  on cancel; this covers the message stored in the agent's own history.)

## [1.5.0] — 2026-07-19

### Changed

- **Streamed answer rendering rewritten on a real Markdown parser.** The terminal
  formatter (`CodeFormatter`) no longer scans streamed deltas with a hand-rolled
  ` ``` ` state machine — it now buffers the response and **re-parses it with
  `markdown-it-py`** on each chunk, rendering completed top-level blocks once and
  re-parsing only the growing tail. This structurally eliminates a class of
  streaming glitches: a code block's language label ("python") could leak as a
  bare text line, adjacent code fences could desync the scanner, a literal `\n`
  could appear, and inline `` `code` `` after a code block could render with raw
  backticks. Fenced code is still Pygments-highlighted; headers, lists,
  blockquotes, rules, **bold**/_italic_/inline-code, and clickable links render as
  before (spaced `a * b` still isn't italicized). Adds `markdown-it-py` as a
  dependency.

### Added

- **Background sub-agents + resume.** `spawn_agent(…, run_in_background=true)` now
  launches a sub-agent on a daemon thread and **returns immediately** with an
  agent id — you keep working while it runs. When it finishes it **auto-surfaces
  its report**: if you're idle, a turn is triggered automatically (no need to
  type anything) to deliver it; if a turn is running, it's delivered at the start
  of your next turn. Either way it's folded in as a user message so the model
  addresses it. A background sub-agent has no terminal to prompt on, so it
  **auto-skips (denies) any destructive tool that isn't already approved** —
  never runs one unattended (a pre-trusted category from the session still
  proceeds); safe by construction. `resume_agent(agent_id, prompt)` continues a
  finished sub-agent with a follow-up, re-briefing it from its original task +
  prior report (no re-explaining). It **defaults to background too** (matching the
  original background run — returns immediately, report delivered on completion;
  pass `run_in_background=false` to wait inline), and works **across a restart or
  a loaded conversation**, since each run's brief + report is persisted under the
  tasks dir and resume falls back to it when the in-memory record is gone. This
  completes the sub-agent feature (foreground/parallel shipped earlier; this adds
  detached execution).
- **Mid-turn steering — talk to the assistant while it's working.** A plain
  message typed WHILE a turn is running is now **folded into that running turn**
  instead of queued as a separate one: the agent holds it on a thread-safe queue
  and drains it at the next tool-round boundary (the top of the tool node, and —
  during orchestration — into the aggregator), injecting it as a user message
  ("The user sent a new message while you were working: …") so the model
  addresses it after finishing the current step. The in-flight work isn't
  discarded, and the current tool batch runs to completion (additive steering).
  Slash commands still queue as a separate turn (they can't steer mid-run);
  **Esc still cancels** the turn — and cancelling now **discards** any message
  steered into that (aborted) turn, so it can't leak into and get answered by the
  next turn. A steered message shows a dim `> … (steering →)` echo instead of the
  `(queued)` line.

- **Sub-agents (`spawn_agent`) — model-initiated, isolated-context delegation.**
  The model can now call `spawn_agent(agent_type, prompt, description)` to hand a
  self-contained task to a fresh sub-agent that runs on its **own isolated
  context** and returns **only its final report** — the sub-agent's intermediate
  tool calls never enter the parent's window, keeping it clean during search- or
  multi-step-heavy work. It complements the existing (framework-driven)
  orchestrator. Built-in agent
  types: `general-purpose` (full toolset), `explore` and `plan` (read-only —
  search/read only, no writes or shell). Each carries its own system prompt,
  customizable in `prompts.yaml`
  (`SUBAGENT_{GENERAL,EXPLORE,PLAN}_PROMPT`). Nested spawning is blocked. Reuses
  the existing worker loop (already isolated) and the `exit_plan_mode` thin-stub +
  client-side-interception pattern.
- **Custom sub-agent types from `~/.mnemoai/agents/*.md`.** Drop a markdown file
  with YAML frontmatter (`description` required; optional `name` — defaults to the
  filename — and `tools` allowlist, list or CSV, `*`/omitted = all tools) and a
  body used as the sub-agent's system prompt, and it becomes a `spawn_agent` type.
  Tolerant scan (a bad/incomplete file is reported and skipped, never fatal); a
  custom type overrides a built-in of the same name. All available types (built-in
  - custom) are listed to the model in an `<available_subagents>` system-prompt
    block at session start, re-injected after compaction so they survive a summary.
    No `/agents` command — agents are authored as files and
    discovered automatically; the model, not the user, decides when to spawn one.
    Each type also advertises its tool scope (`(Tools: all)` or the allowlist) in
    the listing, so the model can pick by capability. A spawned sub-agent runs on
    the **same generous turn budget as the main agent loop** (`LLM.RECURSION_LIMIT`,
    default 200) rather than the orchestrator-worker default of 10 — a whole
    self-contained task (esp. search-heavy `explore`/`plan`) would otherwise starve
    and stop before delivering its report.
- **Parallel sub-agents + quiet display.** The assistant can now emit several
  `spawn_agent` calls in one turn and they run **concurrently** on a bounded pool
  (`LLM.SUBAGENT_MAX_CONCURRENCY`, default 4; 1 forces sequential; a lone spawn
  runs inline), with results collected and handed back together — a failing
  sub-agent yields an error string without aborting its siblings. Sub-agents now
  run **quiet**: their internal trace no longer floods the terminal — you see a
  live "N sub-agents running…" / "N tool calls…" status line instead, and only
  each sub-agent's final report surfaces. The model call still **streams** under
  the hood (so a sub-agent keeps the stalled-stream idle-timeout + network retry —
  it won't hang if the socket dies on sleep); only the display is suppressed, and
  no shared display state is touched, which is what makes the parallel runs safe.
  Confirmation prompts from concurrent sub-agents are serialized, and the
  nested-spawn guard is now thread-local so parallel top-level spawns don't
  interfere. Background execution + resume remain a planned follow-up.
- **Parallel orchestrator subtasks (dependency-aware).** The orchestrator (the
  framework decomposer for complex "full" queries) no longer runs its workers
  strictly one-after-another. The decomposition schema gains an optional
  `depends_on` (a list of earlier subtask indices); the orchestrator now schedules
  by that dependency graph — **independent subtasks run concurrently** on the same
  bounded pool as sub-agents (`LLM.SUBAGENT_MAX_CONCURRENCY`), while genuinely
  dependent steps wait for exactly the inputs they declared (only those results
  are threaded in, not every prior step). This lets a "do two independent things"
  request actually run them in parallel through the orchestrator. Fully backward-
  compatible: no `depends_on` → today's sequential behavior; malformed/cyclic/
  forward references are sanitized away and can never deadlock (the scheduler
  force-runs any stuck remainder). Orchestrator workers now run quiet too, so a
  parallel wave shows one "N steps running…" line instead of interleaved traces.
  Workers in a parallel wave run **headless** — an untrusted destructive tool
  auto-skips instead of stacking multiple confirmation prompts on one terminal
  (a category you already approved this session still runs); a sequential run
  (`SUBAGENT_MAX_CONCURRENCY=1`) still prompts normally.

## [1.4.5] — 2026-07-17

### Added

- **`/model` → Vision: "Use the same model as Chat?" shortcut.** When overriding
  the vision model, the first step now offers to reuse the chat model wholesale —
  choosing yes copies the chat model's provider (`TYPE`), `NAME`, connection keys
  (region, Mantle `API_PROTOCOL`, host/port, …), and `MAX_TOKENS` straight into
  `VISION_MODEL_ID`, skipping the provider/model/connection prompts. The common
  case when the chat model is already multimodal (Claude, GPT-4o/5, Qwen-VL, …).
  It's provider-agnostic (copies only the keys the vision section supports for
  that provider, so chat-only inference params like `REASONING_EFFORT` aren't
  dragged along) and prunes any stale keys from the previous vision provider.
  Choosing no falls through to the normal per-field flow.
- **New `/features` command — enable/disable app subsystems.** A checklist of the
  `ENABLE_*` toggles (RAG, episodic memory, playbook, web search, web crawl,
  routing, orchestration, curated memory, memory auto-extraction, skills) that
  reads the current config, lets you flip them (Space toggle, Enter save), and
  writes just those keys back — then restarts in place. When a feature you just
  turned **on** needs extra info, it's gathered inline: web search prompts for a
  `BRAVE_API_KEY` (if not already set), and RAG / episodic memory prompt for an
  embeddings model (reusing the `/model` embeddings flow) when none is configured.
  This is the natural sibling to `/model` (which models) and `/params` (how a
  model generates): `/features` is which subsystems are on.
- **`/model` → Embeddings is always selectable, and offers to enable RAG /
  episodic memory afterward.** Embeddings could only be picked when an
  `EMBED_MODEL_ID` block already existed — but with RAG disabled that block is
  often absent, so there was no way to set the embeddings model at all. It's now
  always offered; if its block is missing it's scaffolded under `RAG:` first.
  After configuring it, `/model` asks to turn on the features that consume it when
  they're off — first `ENABLE_RAG`, then `ENABLE_EPISODIC_MEMORY` (a prompt is
  skipped only when the toggle is explicitly on; an absent toggle counts as off).

### Fixed

- **`/model` no longer hides the embeddings option when RAG is disabled** — see
  the embeddings item above (the picker gated it on a present `EMBED_MODEL_ID`).
- **`/config` now configures the embeddings model, lets vision pick its own
  provider, and supports Back within each model section.** The reconfigure flow
  used an ad-hoc vision step (name only — no provider choice, no Back) and never
  prompted for embeddings at all. Vision and embeddings now go through the same
  composable section flow as `/model` (`_prompt_model_section`): a provider
  choice, the provider's connection steps, and `Ctrl+B` Back on each — and vision
  offers the "use the same model as Chat?" shortcut here too. (The trailing
  feature-toggle yes/no questions remain a linear sequence; Esc cancels.) The
  toggle prompts also now include the ones added since this flow was written —
  `ENABLE_MEMORY_AUTO_EXTRACTION` (asked only when persistent memory is on),
  `ENABLE_SKILLS`, and `REQUIRE_MEMORY_CONFIRMATION` — so `/config` covers the
  full set again.

## [1.4.4] — 2026-07-17

### Fixed

- **`web_crawler` now shows a "Crawling the page…" spinner instead of a dead
  gap.** It was in `_SELF_REPORTING_TOOLS` — the spinner was suppressed on the
  assumption the crawler printed its own `[INIT]/[FETCH]/[SCRAPE]` progress to the
  terminal. But the crawler runs its browser quietly (`verbose=False`) and any
  subprocess stderr is routed to the MCP log (since 1.1.2), so nothing showed
  during a crawl — the UI looked frozen. `web_crawler` now animates the spinner
  like any other slow tool (the `_SELF_REPORTING_TOOLS` carve-out is now empty).

## [1.4.3] — 2026-07-16

### Fixed

- **A dropped/stalled stream connection no longer hangs the app.** When the
  laptop sleeps (or the network drops) mid-response, the TCP socket to the model
  dies silently; the streaming read had no idle bound, so on resume it blocked the
  single worker thread forever — the conversation looked stopped AND new messages
  froze behind it. The stream is now consumed through a **per-chunk idle-timeout
  watchdog** (`_iter_stream_with_idle_timeout`, `LLM.STREAM_IDLE_TIMEOUT`, default
  120s; 0 disables): if no chunk arrives within the window the wedged read is
  abandoned so the worker is never parked. The retry wrapper (`_stream_response`)
  then classifies idle-timeout + transient network errors (reset/broken-pipe/
  timeout/5xx/overload — `_is_transient_network_error`, provider-agnostic) as
  retryable and **re-runs the turn on a fresh connection with exponential
  backoff**; if all retries fail it ends with a clear "lost the connection — send
  your message again" message instead of crashing. The partial (including partial
  reasoning) is discarded — a dropped stream can't be resumed mid-generation — but
  the conversation continues instead of crashing. This lives in the
  agent stream layer, so it applies to **every** provider, not one client.

### Changed

- **`agent.py` housekeeping:** the class-level constants that were interleaved
  between methods (streaming error/network markers, self-reporting tools,
  confirmation categories, the ephemeral-block regex) are consolidated into one
  labeled block at the top of `LangGraphAgent`. Pure reorganization — no behavior
  change.

## [1.4.2] — 2026-07-16

### Fixed

- **A turn cut off by the output-token limit now auto-continues instead of
  dead-ending.** With `REASONING_EFFORT: max` on a large context, the model can
  spend its whole `MODEL_ID.MAX_TOKENS` output budget on reasoning plus a partial
  answer/tool call and stop mid-turn (`stop_reason: max_tokens`) with no completed
  tool call — the turn just ended and the user had to type "continue". `_call_model`
  now detects this (`_was_truncated_by_tokens`) and **auto-continues** via
  `_continue_truncated_turn`: it feeds the partial turn back with a "continue where
  you left off" nudge and re-streams, accumulating text, up to
  `LLM.MAX_OUTPUT_CONTINUE_RETRIES` (default 3). It stops early when a continuation
  finishes cleanly (returns the assembled answer) or emits a tool call (returned so
  the graph runs it, partial text preserved); resumed parts glue directly so a
  split word isn't corrupted by a spurious space. Only if retries exhaust with
  nothing usable does it surface the "increase MAX_TOKENS" message. This replaces
  the earlier behavior that ended the turn with a warning. Diagnosed from a real
  session (Claude Opus via Mantle, `REASONING_EFFORT: max`, ~666k-token context).

### Changed

- Quieter compaction: dropped the green "Token limit exceeded… compacting" and
  "evicted old tool output" status lines that printed mid-task — compaction now
  runs without narrating itself.

## [1.4.1] — 2026-07-16

### Fixed

- **An approved plan could not edit files when its execution turns re-classified
  as a read-only route.** Query routing re-classifies **every** turn, and each
  route binds only its tool subset — `code`/`full` bind `file_edit`/`fs_write`/
  `execute_bash`, but `knowledge`/`research`/`simple_qa` do not. So after
  approving a plan (e.g. from a docs-review question that classified as
  `knowledge`), the execution turns could land on a read-only route with the
  mutating tools unbound — the model would truthfully report it "had no write
  tools," then find them again on a turn that happened to classify as `code`,
  flip-flopping mid-task. Plan approval now sets `agent._execute_plan_route`,
  which forces the **full** toolset (via `_effective_route`, consulted by
  `_get_route_model`/`_get_route_tools`) and routes straight to the agent (not the
  orchestrator) for the rest of the task — so an approved plan can always apply
  itself. Same plan-scoped lifetime as the pre-approved bash (cleared on `/clear`
  and when plan mode is re-entered).

## [1.4.0] — 2026-07-16

### Added

- **Background memory auto-extraction (opt-in).** After each turn, if
  `ENABLE_MEMORY_AUTO_EXTRACTION` is on (default **off**), a background pass
  distills durable facts from the exchange and writes them to `MEMORY.md`
  automatically — the auto-learning counterpart to the model calling the `memory`
  tool itself. It runs on a daemon thread (never blocks the turn), makes one
  isolated model call that doesn't touch the conversation state
  (`client._invoke_model_once`), and applies the returned JSON `add`/`replace`
  ops through `MemoryStore` (so the char cap + dedup still apply). It's gated
  behind its own toggle because, unlike the tool path, it writes **without** a
  confirmation prompt — though it can only add/consolidate entries in `MEMORY.md`.
  New `MEMORY_EXTRACTION_PROMPT` (four-kind taxonomy + "what not to save"), and
  `client.auto_extract_memory` wired at turn end. Storage is unchanged (still one
  bounded flat file).

### Changed

- **New prompts reach existing installs.** `Config._load_prompts` now layers the
  bundled package `prompts.yaml` underneath the user's file (user keys win,
  missing keys fall back to the bundle). A user's `prompts.yaml` is seeded once
  and never overwritten, so previously a prompt added in a release wouldn't
  resolve on an existing install; now it does, while any prompt the user
  customized still takes precedence. (This is what lets the new
  `MEMORY_EXTRACTION_PROMPT` work on upgrade without re-copying the file.)

## [1.3.0] — 2026-07-16

### Added

- **Tool-result eviction — a cheaper compaction layer that runs before the LLM
  summary.** When history crosses the high-water mark, the agent now first shrinks
  the bodies of **old** tool results (the grep/read/web dumps outside the recent
  window that carry most of the context but are rarely needed verbatim after the
  model has acted on them) to a short head plus an eviction marker — with **no
  model call**. If that alone brings the estimate back under budget, the expensive
  full summarization is skipped entirely. Recent turns stay verbatim and **no
  message is dropped** (only content is trimmed), so tool-call/result pairing is
  preserved and no provider rejects the next turn. New method
  `AgentConversationManager.evict_old_tool_results`, wired as the first pass in
  `client._compact_now`. Config: `LLM.TOOL_EVICTION_KEEP_RECENT` (default 8) and
  `LLM.EVICTED_TOOL_RESULT_CHARS` (default 500; 0 disables the layer). This makes
  compaction layered — cheap eviction first, full summary only when eviction isn't
  enough — the biggest structural improvement to the context handling reworked
  across 1.2.x.
- **Skills: optional `argument_hint` frontmatter + a bounded tier-1 listing.** A
  skill's `SKILL.md` frontmatter may now carry an `argument_hint` (a short phrase
  naming what the skill expects, e.g. "a PR number"); it renders in the always-on
  `<available_skills>` listing as `(expects: …)` so the model knows what to gather
  before calling `use_skill`. The whole listing is also now bounded in total size
  (`_MAX_LISTING_CHARS`, ~4000 chars): with a large skills library the overflow
  collapses into a `… (+N more — see /skills)` line, so many installed skills can't
  dominate the system prompt injected every turn. The bundled skill-creator skill
  documents the new key.
- **Plan mode: pre-approved shell commands.** `exit_plan_mode` now takes an
  optional `allowed_bash` list — the commands the plan will run during execution
  (tests, builds, installs). Approving the plan pre-approves them, so during
  execution they run **without** the per-command `Proceed?` confirmation (a command
  auto-confirms when it equals or prefix-matches an entry, e.g. `pytest` pre-approves
  `pytest tests/unit`); anything else still prompts. The pre-approvals are scoped to
  the approved plan — cleared on `/clear` and whenever plan mode is re-entered. The
  plan-mode reminder now tells the model to declare these commands.

### Changed

- **Bundled examples reach existing installs on upgrade.** The `*.example`
  reference files (`config.yaml*.example`, `mcp.json.example`) are now refreshed
  from the package when they differ (they're read-only reference, never loaded as
  config), so a newly-documented key — e.g. the eviction knobs above — appears in
  an already-installed user's examples. Bundled example skills additionally get an
  in-place `SKILL.md` refresh when the installed copy is **pristine** (byte-identical
  to a version we shipped, tracked by hash), so doc/frontmatter updates (e.g. the
  new `argument_hint`) reach existing installs — while a user-edited skill is never
  touched. The live `config.yaml` / `mcp.json` / `prompts.yaml` and user-authored
  skills are still only created when absent and never overwritten.
- **Curated memory: a four-kind tagging convention.** The `memory` tool now guides
  the model to tag each entry with its kind — `[user]` (who the user is),
  `[feedback]` (how to work, with the why), `[project]` (ongoing work/constraints,
  absolute dates), `[reference]` (external pointers) — and sharpens the "what not to
  save" rule to exclude anything the repo/git/CLAUDE.md already records. Storage is
  unchanged (still one bounded `MEMORY.md`); this is prompt guidance that yields
  more structured, higher-signal, less redundant entries.

## [1.2.2] — 2026-07-16

### Fixed

- **Context-window overflow root cause: the token counter undercounted non-OpenAI
  models.** A single tiktoken (OpenAI) encoder was used for every provider, which
  undercounts Claude ~1.5x on code/JSON history — so the compaction high-water
  mark never fired (the agent believed it was under the window when it was ~15%
  over) and the prompt overflowed the API. Two fixes:
  - **Provider-aware, never-undercount estimate** (`utils.tokenization`): an
    o200k_base basis scaled by a conservative per-provider multiplier
    (anthropic/mantle ×1.5, bedrock/litellm/sagemaker ×1.35, openai exact ×1.0;
    overridable via `LLM.TOKEN_COUNTING.<TYPE>_MULTIPLIER`). Ollama uses a
    chars/ratio (`OLLAMA_CHARS_PER_TOKEN`, default 3.0).
  - **Provider's exact count as ground truth**: the agent now captures
    `usage_metadata.input_tokens` from each response and the compaction high-water
    check + the `[Context: N]` display use `max(estimate, actual)`, so the real
    context size drives the decision (reset on `/clear` and after compaction).

### Changed

- Consolidated token counting onto the single shared `utils.tokenization.count_tokens`
  (removed the conversation manager's duplicate tiktoken encoder) and added a
  small per-message overhead for role/formatting wrappers.

## [1.2.1] — 2026-07-15

### Fixed

- **Compaction no longer overflows the summarization call on a near-full, large
  context window.** The batched summarizer (added in 1.0.2) set the per-call
  budget to 50% of the window — on a 1M-token model that meant a ~500k-token
  batch, which together with the rolling summary + prompt + reasoning output
  exceeded the window and 400'd ("could not compact"), so a full history couldn't
  be shrunk. The per-call budget is now a conservative fraction of the window
  (`_SUMMARY_CALL_FRACTION`, 0.15), the rolling summary carried between batches is
  capped, and a single message larger than the budget is truncated (head+tail) so
  it can't overflow its own batch. Verified against a real 539-message / ~750k-token
  conversation: batches drop from one 498k-token call to five ≤150k-token calls.

## [1.2.0] — 2026-07-15

### Added

- **File edits and writes render as styled diff/content blocks** (pinned UI),
  instead of a flattened `↳ file_path=… ↳ old_string=…` line:
  - `file_edit` and `fs_write` `str_replace` show an `Update(path)` header with a
    red `-` / green `+` diff of the change.
  - `fs_write` `create` shows a `Create file` header with the new content as
    numbered lines; `insert`/`append` show the added text as green `+` lines.
  - Home paths are shortened to `~/…`. Both the live turn and the `/load` replay
    use the same renderer, so restored conversations look identical.

## [1.1.6] — 2026-07-15

### Fixed

- **A large crawled page is now retrievable on the research route.** When
  `web_crawler` fetches a page over `RAG.MAX_TOKENS`, it ingests the content into
  the document store instead of returning it inline — but the `research` route
  bound no RAG retrieval tool, so the model got a "content ingested" pointer it
  couldn't follow and fell back to answering from memory. The route now also
  binds `search_in_documents` + `list_documents`, and the crawler's ingest result
  explicitly instructs the model to retrieve via `search_in_documents(query=...)`
  rather than answer from memory.
- **Approved plan files no longer accumulate.** `plans/plan_<ts>.md` (written on
  each plan approval) were never cleaned up. A startup sweep (`sweep_old_plans`)
  now prunes `plan_*.md` older than 7 days (`PLAN_MAX_AGE_DAYS`, 0 disables) and
  removes the stale `current_plan.json` left by the retired legacy plan tools.
  Recent plans are kept (they survive compaction and stay re-readable).

## [1.1.5] — 2026-07-14

### Fixed

- **Launch-banner tagline is centered under the wordmark again.** The
  "local agentic AI assistant · learns & remembers" line was centered to the
  command-box width (which widens to fit the longest command row), so it drifted
  right of the ASCII wordmark whenever the box was wider than the banner. It now
  centers to the banner's own width.

## [1.1.4] — 2026-07-13

### Fixed

- **Background-task log files no longer accumulate forever.** Each background
  task writes a `~/.mnemoai/tasks/{id}.log`; the only cleanup
  (`clear_completed_tasks`) iterated the in-memory task registry, which is empty
  on every restart — so `.log` files from prior sessions were unreachable and
  piled up indefinitely. A startup sweep (`sweep_old_task_logs`) now prunes task
  `.log` files older than 7 days (`TASK_LOG_MAX_AGE_DAYS`, 0 disables). It only
  touches `*.log` files and is best-effort (never blocks startup); recent logs
  are kept so `get_task_output` still works for active tasks.

## [1.1.3] — 2026-07-13

### Fixed

- **The MCP stderr log (`~/.mnemoai/logs/mcp.log`, added in 1.1.2) no longer
  grows without bound.** It's now size-rotated: when it reaches ~1 MB
  (`MCP_LOG_MAX_BYTES`) it's rotated to `mcp.log.1` (one backup generation,
  replaced not stacked) and a fresh log starts, bounding on-disk use to ~2 MB.
  Rotation is best-effort (never blocks startup). New `paths.open_mcp_log()`
  helper owns the rotation policy.

## [1.1.2] — 2026-07-13

### Fixed

- **MCP server subprocess stderr no longer leaks into the terminal.** External
  MCP servers (and the built-in one) are spawned via `stdio_client`, whose
  `errlog` defaulted to the parent's stderr — so a noisy server (e.g. the
  `npm notice` lines from an `npx`-launched server) wrote straight to the
  terminal, bypassing the pinned UI. Their stderr is now routed to
  `~/.mnemoai/logs/mcp.log`: the terminal stays clean while the output remains
  available for debugging a server that fails to start. New `paths.logs_dir()`
  and `paths.mcp_log_path()` helpers.

## [1.1.1] — 2026-07-13

### Fixed

- **Malformed extended-thinking blocks no longer 400 the whole request.** An
  Anthropic (Claude) request rejects wholesale with
  `messages.N.content.M.thinking.thinking: Field required` when a `thinking`
  content block reaches the API with empty/missing inner text — which can enter
  history via a cut-short stream or a mid-session model switch. The message
  sanitizer (already run before every model call) now drops **only** such
  provably-invalid thinking blocks, leaving healthy content and tool pairs
  intact. Scoped precisely to the malformed-Anthropic shape: non-Anthropic
  histories (Bedrock `reasoning_content`, OpenAI Responses `reasoning`, Ollama /
  LiteLLM string reasoning, plain text) pass through as the same object,
  untouched.

## [1.1.0] — 2026-07-10

### Added

- **`STEERING.md` — user-authored always-on instructions.** The user's
  counterpart to the agent-curated `MEMORY.md`: write build/test commands, code
  conventions, and "always do X" rules that the assistant follows every turn.
  Discovered hierarchically — a global `~/.mnemoai/STEERING.md` plus a project
  `./STEERING.md` found by walking up to the repo root, combined broadest→most
  specific. Injected as a leading instruction block **re-read from disk each
  turn** (edits apply immediately) and **never summarized by compaction** (it's
  re-injected verbatim, so long sessions never dilute it). No config toggle — the
  file's presence is the switch; no file → nothing injected.
- **`steering-creator` bundled skill.** Ask the assistant to "create a
  STEERING.md for this project" (or "document how to work in this repo") and it
  investigates the codebase and writes a well-formed file following best
  practices; "improve my STEERING.md" refines an existing one.

### Changed

- **Bundled example skills now seed per-skill (reaches existing installs on
  upgrade).** `seed_example_files()` previously seeded skills only when the
  `skills/` dir was entirely empty, so a newly-bundled skill (like
  `steering-creator`) never reached a user who already had skills. It now copies
  each bundled skill whose own directory is absent — matching how the config
  `*.example` files already seed — so new bundled content reaches existing
  installs, without ever clobbering a user's own skills. Codified as a standing
  convention in `CLAUDE.md` (new features must reach existing installs, not just
  fresh ones).

## [1.0.2] — 2026-07-10

### Fixed

- **A context-window overflow no longer abandons the in-flight task.** When a
  turn exceeded the model's context (e.g. after a large paste or file read), the
  backstop compacted history but returned a terminal "I stopped this turn — try
  again" message, dropping the work in progress (you had to re-ask). Now
  `_stream_once` raises a typed overflow, and `_call_model` compacts and
  **re-invokes on the shrunken prompt in the same turn**, so the task continues.
  A terminal message is returned only if compaction can't shrink history further
  or the prompt still overflows after compacting (no retry-storm). The worker
  loop and aggregator degrade gracefully on the same signal.
- **Summarization no longer silently loses history when it overflows.** The
  compaction summary was generated in a single model call; if the older history
  itself exceeded the context window the call 400'd and the summary degraded to
  a content-free placeholder ("Previous conversation covered multiple topics"),
  discarding hundreds of messages. Summarization is now **batched (map-reduce):**
  older messages are split into window-sized batches folded into a rolling
  summary, so the summary call never itself overflows. If every batch still
  fails, it keeps a **bounded excerpt** of the real history rather than a
  placeholder.

## [1.0.1] — 2026-07-09

### Fixed

- **Mid-turn log warnings no longer corrupt the pinned prompt.** The log handler
  captured the original `sys.stderr` at import, bypassing the pinned UI's
  `patch_stdout` (which swaps `sys.stderr` for a proxy that renders above the
  input). A warning fired during a turn (e.g. an embed retry) printed on top of
  the `(esc to cancel)` status line. The handler now resolves `sys.stderr` at
  emit time, so logs render cleanly in scrollback during a turn and to real
  stderr otherwise.
- **Embedding a very long input no longer spams retries or silently degrades.**
  Oversized text (e.g. a large paste) could exceed the embed model's context and
  fail. Fixes, all provider-agnostic (applied at the shared truncation point, so
  Ollama/OpenAI/Bedrock/SageMaker/LiteLLM all benefit):
  - a **token-aware input cap** resolved from `RAG.EMBED_MODEL_ID.MAX_INPUT_TOKENS`
    (new, optional) → else the model's own context window (Ollama probe, with a
    0.9 safety margin) → else a conservative default (8192);
  - `truncate=True` on the Ollama embed call as a runner-side backstop;
  - **deterministic context-overflow errors skip the retry loop** (retrying the
    identical input can't help) while genuinely transient EOFs still retry.

## [1.0.0] — 2026-07-09

First stable release. The public contract is now frozen under semver: breaking
changes to `config.yaml` keys, the `mcp.json` `mcpServers` schema, the CLI
slash-commands + `mnemoai` console command, and the `mnemoai-assistant` dist /
`mnemoai` import name will bump the major version.

### Changed

- **README demo is now a link instead of an inline GIF.** GitHub serves the
  10.5 MB `.gif` as `application/octet-stream`, which PyPI's image proxy refuses,
  so the demo rendered as a broken image on the PyPI project page. It now links
  to the GIF (renders inline on GitHub, resolves cleanly on PyPI).

### Fixed

- **Integration-test config gate now mirrors the real loader.** The integration
  tier keyed its skip guard off a single hardcoded `src/mnemoai/utils/config.yaml`
  path, out of sync with the app's actual resolution (`$MNEMOAI_CONFIG` →
  `~/.mnemoai/config/config.yaml` → legacy → package-relative). It now reuses
  `Config._resolve_config_path()`, so `python -m pytest -m integration` runs
  from the installed config with no env-var or file-copy workaround.

## [0.19.1] — 2026-07-08

### Changed

- Recompressed the demo GIF (`images/assistant-demo.gif`, ~19 MB → ~10 MB) so the
  README loads faster on PyPI and GitHub. No code changes.

## [0.19.0] — 2026-07-07

### Added

- **Plan mode style refactor → execute handoff.** When its
  plan is ready the model calls a new `exit_plan_mode(plan)` tool
  (`server/tools/plan_mode_exit.py`, a thin stub in `_ALWAYS_AVAILABLE_TOOLS`);
  the agent intercepts it client-side (`_handle_exit_plan_mode`) at both tool
  chokepoints and shows an in-app approval prompt (`reader.plan_approval_ui`,
  modeled on the confirmation gate): **y** approves and runs, **e** opens the
  plan in `$EDITOR` to tweak before re-review, **n** keeps planning (stays
  read-only). Approving flips plan mode off, persists the approved plan to
  `plans/plan_<ts>.md`, and hands it back so the model executes it **in the same
  turn** — no more manual `/plan`-off + "now do it" re-prompt. Non-TTY/tests
  auto-approve. The plan is shown as a **bordered, word-wrapped, markdown-aware
  block** (`turn_view.render_plan` — bold headings, cyan code spans, hanging
  list indents) above the approval prompt, instead of a flattened
  `↳ plan=…` tool-arg line.

### Fixed

- **Confirmation & plan-approval prompts cleaned up.** The pinned
  bottom-of-screen confirm line embedded the full (often multi-line) command
  detail, which prompt_toolkit clipped to its first line — rendering a confusing
  truncated `▶ Run shell command?  python3 -c "` under a duplicated options hint.
  Now the pinned line is compact (`▶ <question>   [y = yes · n = no · a = allow
all]`), the full command/plan echoes to scrollback above it just once, and the
  pinned line styles the question in the accent color with the `[y · n · a]`
  keys dimmed — so the eye separates the prompt from the actionable keys. Applies
  to both the shell/file/memory confirmation and the plan-approval prompt.

### Removed

- **Retired the legacy advisory plan-mode JSON-workflow tools**
  (`enter_plan_mode`, `add_plan_step`, `add_plan_file`, `add_plan_risk`,
  `present_plan`, `approve_plan`, `get_plan_status`, and the old
  `exit_plan_mode(cancel)`). They were bookkeeping only — never wired to
  enforcement — and the old `exit_plan_mode` collided with the new tool
  (`Tool names must be unique` → a 400 loop). The enforced `/plan` path is now
  the only plan mode.

## [0.18.3] — 2026-07-04

### Fixed

- **Force-quit now takes two Ctrl+C presses (not one), and restores the
  terminal.** The 0.18.2 escalation fired on the first Ctrl+C after an Esc
  cancel (it keyed off `_cancelled`, which Esc had already set), and `os._exit`
  left the tty in raw/no-echo mode so the shell looked frozen after exit. Now a
  dedicated per-turn flag requires a genuine second Ctrl+C (with a
  `(press Ctrl+C again to force-quit)` hint on the first), and `_force_quit`
  restores cooked+echo mode via termios (with an `stty sane` fallback) before
  exiting.

## [0.18.2] — 2026-07-04

### Fixed

- **A second Ctrl+C now force-quits when a cancel is stuck.** Pressing Esc
  cancels a turn by injecting `KeyboardInterrupt` into the worker thread, but
  that only lands at the next Python bytecode boundary — a worker blocked inside
  a long native tool call (e.g. a broad `glob_search`) stayed on `(cancelling…)`
  and Ctrl+C was a no-op (it just re-requested the already-pending cancel). A
  second Ctrl+C while cancelling now force-quits via `os._exit` (restoring the
  terminal first), bypassing the `asyncio` executor join that would otherwise
  hang on the wedged thread.

## [0.18.1] — 2026-07-04

### Changed

- **`fs_write` no longer takes `dry_run`/`confirmed` — it writes directly.** The
  old two-step dance (preview with `dry_run=True`, then re-call with
  `dry_run=False, confirmed=True`) is redundant now that the client-side
  confirmation gate (`REQUIRE_WRITE_CONFIRMATION`) asks the user before every
  write. Dropping it removes a whole extra round-trip (and the preview's tokens)
  per file operation. The server-side system-path safety floor is unchanged.

## [0.18.0] — 2026-07-04

### Added

- **Context-overflow protection (3 layers) — a runaway tool result no longer
  loops the agent on a "prompt is too long" 400.** A single oversized tool
  result (e.g. `grep_search` with a huge `max_results`) used to enter context
  whole, and every subsequent model call 400'd while the graph retried the same
  oversized prompt to the recursion limit. Now:
  1. **Tool results are capped at the source** (`tool_formatting.truncate_tool_result`)
     — each result is bounded head+tail with a truncation note at both tool-exec
     chokepoints, so one result can't alone overflow the window.
  2. **Pre-flight compaction** — before a turn, history over the high-water mark
     is compacted first (reusing the existing summarize-and-keep-recent manager).
  3. **Overflow backstop** — a context-window 400 is caught, force-compacted, and
     the turn ends with a clear message instead of re-sending the oversized prompt.
     Both size knobs **auto-derive from `MAX_CONVERSATION_TOKENS`** (the model's
     context window) so they scale per-model: `MAX_TOOL_RESULT_CHARS` → 10% of the
     window in chars, `COMPACT_HIGH_WATER_TOKENS` → 80% of the window. Either can be
     set explicitly (0 disables). Documented in the `config.yaml.*` examples and docs.

### Changed

- Removed references from code comments/docstrings and config
  example comments (retained only in `CLAUDE.md`/`ARCHITECTURE.md`).

## [0.17.5] — 2026-07-03

### Fixed

- **Ollama embeddings no longer silently degrade to deterministic fallback on a
  transient runner EOF.** The llama.cpp embedding runner can EOF intermittently
  (first call after a (re)load / under memory pressure) even for input it embeds
  fine on the next try — e.g. right after pasting a large document. `_embed_ollama`
  now retries (default 3×, `RAG.EMBEDDINGS.RETRIES` / `RETRY_DELAY`) before falling
  back, and caps per-text input as hygiene (`RAG.EMBEDDINGS.MAX_INPUT_CHARS`,
  default 200000, 0 disables) so a genuinely huge text can't OOM the runner.

## [0.17.4] — 2026-07-03

### Fixed

- **Orchestrated ("full"-route) turns now render tool calls with the styled
  block, not the old `[⚙ …]` marker.** `_run_worker_loop` printed the plain
  marker unconditionally, so a single conversation could mix the new
  `ToolName ↳ arg=value` blocks with the old style. Both tool-exec chokepoints
  now share one `_print_tool_marker` helper that honors `styled_turn_view`.

## [0.17.3] — 2026-07-03

### Fixed

- **Option+←/→ no longer cancels the running turn while typing a queued message.**
  The bare-Esc cancel binding was `eager`, so it fired on the `ESC` prefix of
  Option+←/→ (`ESC b` / `ESC f`) before the letter arrived — cancelling instead
  of moving the cursor. It's now non-eager, so word-motion works while typing
  (idle or mid-turn) and a lone Esc still cancels.
- **A cancelled turn now shows `⊘ Stopped`.** The transient `(cancelling…)` line
  previously never resolved; the cancelled response is now surfaced as a final
  stopped line.

## [0.17.2] — 2026-07-03

### Fixed

- **macOS Option+←/→ now jump by word instead of typing `b`/`f`.** Option+←/→
  send `ESC b` / `ESC f` (backward/forward-word), but the pinned input's
  `eager=True` bare-Esc binding consumed the `ESC` prefix, leaving the `b`/`f` to
  self-insert. The eager Esc-to-cancel is now gated to while a turn runs; at the
  idle prompt Esc stays a prefix so the default word-motion bindings fire.

## [0.17.1] — 2026-07-03

### Changed

- **The `/load` picker now labels each conversation with an auto-derived title**
  (the first real user message, skipping injected episodic-memory / plan-mode
  context) instead of the raw `conversation_<timestamp>.json` filename.
  No LLM call: the title is read verbatim from the saved file's first user message
  (filename shown as fallback when there's none).

## [0.17.0] — 2026-07-03

### Added

- **`/load` now replays the conversation to the screen.**.
  Previously a loaded conversation was restored into context but
  nothing was shown. It now prints the full transcript to scrollback in the
  styled view — user prompts, collapsed `Thought…` reasoning blocks,
  `ToolName ↳ arg=value` tool-call blocks, and `●`-marked answers — before the
  `[Context: N tokens]` line. New pure `turn_view.render_conversation` builder;
  tool _results_ are omitted (the call block is enough), matching the live UX.

### Changed

- **Documentation refresh.** Reorganized the docs site (`docs/`, `mkdocs.yml`)
  and moved the per-file reference `ARCHITECTURE.md` to the repo root (with the
  matching link update in `CLAUDE.md`). Docs-only — no code or behavior change.

## [0.16.1] — 2026-07-03

### Fixed

- **Spinner no longer disappears during the silent stretch before a tool batch.**
  On the Responses/Bedrock block protocols the streaming callback's `token` is the
  repr of a content-block list, so a reasoning block (`{'id':'rs_…'}`) or `[]` is a
  truthy string that `on_llm_new_token` mistook for visible answer text and stopped
  the spinner — leaving a dead pause until the next turn. The handler now inspects
  the chunk's actual blocks and stops only on a non-empty `text` block.

## [0.16.0] — 2026-07-03

### Added

- **Back navigation in the config wizards (`/config`, `/model`, `/params`).** Each
  step now shows a **Back** button (and **Ctrl+B**) that returns to the previous
  field, keeping what you already entered; Esc still cancels the whole flow. A
  small step-runner (`_run_steps`) drives the sequences and rewinds cleanly on a
  Back (restoring the pre-step text so a re-answer replaces, not stacks). Non-TTY
  runs are unchanged.

### Changed

- **Configurator no longer prints mid-flow noise on a TTY.** Section headers and
  inline notes (`-- Chat model --`, "Vision disabled", "Configuring for…", the
  inference-reset note, …) were wiped by the next full-screen dialog anyway; they
  are dropped on a TTY. Only what's useful after the in-place restart survives to
  scrollback — the final "Config written / Updated / No changes / Cancelled"
  outcome and the provider's env-based auth note — plus the unchanged non-TTY
  fallback banners.

### Fixed

- **Reasoning now surfaces on the `bedrock` provider with newer Claude (Sonnet 5,
  Opus 4.6+).** Three bugs in the Bedrock path, all only reachable on the newer
  models: (1) `_initialize_bedrock_model` gated thinking on `REASONING` alone, so
  `REASONING_EFFORT` by itself sent no thinking directive and no reasoning came
  back; (2) it hardcoded the old `thinking={"type":"enabled",…}` form, which these
  models reject with `"thinking.type.enabled" is not supported … use adaptive`
  (a hard `ValidationException`) — it now reuses the version-aware
  `_anthropic_thinking_kwargs` (adaptive + `output_config.effort`), matching the
  Mantle/Anthropic paths; (3) the reasoning extractor didn't recognize the Bedrock
  Converse `reasoning_content` block shape (`{"type":"reasoning_content",
"reasoning_content":{"text":…}}`), so even when text was returned it was dropped.

## [0.15.0] — 2026-07-02

### Added

- **Live "stream-then-collapse" reasoning in the pinned UI.** Reasoning now
  streams live inside a green-bordered **"Thinking… (Ns)"** block in the
  transient region, then commits to scrollback as the collapsed **"Thought for
  Ns…"** block once the answer begins. Implemented with a thread-safe
  `ReasoningStatus` sink (`client/ui/turn_view.py`) the agent's worker thread
  appends to and the reader renders; long lines wrap to the terminal width.

### Fixed

- **Spinner no longer freezes during a large `fs_write` (or any big tool
  argument).** Tool-call argument tokens stream as many non-empty tokens with no
  visible content; the streaming callback used to stop the spinner on them and
  never restart it, so the UI looked frozen for the whole write. It now ignores
  tool-argument tokens (detected via the chunk's `tool_call_chunks`) and stops
  only on real answer text.
- **No more "dead pause" while a reasoning model thinks.** In the styled UI
  reasoning is buffered, so the spinner must keep running through it; it was
  stopping on the first reasoning chunk and leaving the terminal blank.
- **The `●` answer marker and streamed answer no longer disappear.**
  `patch_stdout` only commits a line to scrollback on a newline; the streamed
  answer had no trailing newline and was erased by a pinned-UI repaint before
  `[Context: …]` printed. The answer line is now newline-terminated immediately,
  and the `●` marker is prepended to the first answer chunk (one committed line)
  rather than a separate write that could be erased on its own.
- **Ctrl+C on a non-empty input clears the line** instead of leaving the typed
  text; an empty line still needs a second Ctrl+C/Ctrl+D to exit.

### Changed

- Condensed docstrings and inline comments across `client/`, `models/`, and
  `utils/configurator.py` — one-line docstrings, comments only where intent
  isn't obvious; no behavior change.

## [0.14.0] — 2026-07-02

### Changed

- **Pinned-input terminal UI is now the default on a
  TTY.** The `>` input stays fixed at the bottom of the terminal while a query
  runs; the answer, a green-bordered **"Thought for Ns…"** reasoning block, and
  styled **`ToolName` + `↳ arg=value`** tool blocks stream _above_ it in native
  scrollback (wrapping, scrollback, and copy/paste preserved). Built on a
  persistent, non-full-screen prompt_toolkit `Application` with the query on a
  worker thread and output routed via `patch_stdout`. New behaviors:
  - **Esc / Ctrl+C cancels the in-flight turn** (interrupts the worker; the turn
    ends with "Operation was cancelled."); when idle, Ctrl+C/Ctrl+D twice exits.
  - **Input queue:** typing during a running turn shows a dim `> … (queued)` line
    and runs FIFO after the current turn — never a concurrent query. Each queued
    line is echoed to scrollback (paired with its own answer) only when it starts.
  - **Confirmations** (`Proceed?` for bash/writes) are answered with an in-app
    `y`/`n`/`a` keypress.
  - **Dialog commands** (`/load`, `/config`, `/model`, `/params`, `/memory clear`)
    briefly drop the pinned app to show their normal full-screen dialogs, then
    relaunch — and no longer print the redundant banner / "Current setup"
    overview to scrollback (only the final outcome line remains; `/config`'s
    overwrite caveat now rides in the confirmation prompt).
  - The animated status shows `⠙ Thinking… (esc to cancel)` with cycling dots.
  - **Non-TTY sessions** (pipes / CI / tests) keep the plain `input()` loop.

### Removed

- The previous **app-per-prompt** input reader (`PromptReader`) and its `Ctrl+O`
  tool-call panel — the styled tool blocks now show full arguments inline, so the
  panel is redundant. Also removed the associated `VDISCARD`/termios handling and
  the now-unused `last_turn_tool_calls` capture.

## [0.13.1] — 2026-07-01

### Fixed

- **`/model` and `/params` now show the "Current setup" overview inside the
  selection dialog.** The chat/vision/embeddings summary was printed to
  scrollback and then hidden by the full-screen model-picker dialog, so it was
  invisible exactly when you needed it. It's now rendered in the dialog body
  above the choices (and still printed to scrollback for non-TTY runs).
- **Spinner no longer looks frozen while the model streams a large tool-call
  argument.** A tool call (e.g. `fs_write` with a big document as `file_text`)
  streams its arguments as many content-less chunks; after reasoning had already
  stopped the "Thinking" spinner, that long silent stretch presented a frozen
  terminal. The stream loop now re-raises a "Preparing tool call" spinner while
  tool-call-argument chunks are flowing (stopped as soon as the tool marker
  prints), so a large write never looks stuck.

## [0.13.0] — 2026-07-01

### Added

- **Server-side safety policies (`server/tools/safety/`).** The catastrophic-command
  and system-path limits the tool docstrings advertise are now _enforced inside the
  MCP server_, not only by the client's confirmation gate — so they hold even if the
  server is driven directly.
  - `bash_policy.classify_shell_command` blocks a small, curated set of
    irreversible, system-destroying commands: recursive force-deletes of a
    root/home target (`rm -rf /`, `rm -rf ~`, `rm -rf /*`), filesystem creation
    (`mkfs*`), raw-device overwrite (`dd of=/dev/…`, `shred`/`wipefs` on a device),
    power-state changes (`shutdown`/`reboot`/`halt`/`poweroff`, `init 0/6`), and the
    classic fork bomb. Enforced in **`execute_bash`** and **`start_background_task`**
    (one shared policy). Ordinary scoped mutations (`rm -rf build/`,
    `git reset --hard`) are intentionally **not** blocked here — they remain gated
    by the client confirmation prompt.
  - `path_policy.classify_write_path` blocks writes into critical system
    directories (`/`, `/etc`, `/bin`, `/usr`, `/boot`, `/dev`, `/System`,
    `/Library`, …; `..` escapes are normalized first). Enforced in **`fs_write`**
    and **`file_edit`**. Home, project, temp (incl. the macOS `/private/var/folders`
    temp dir), and app paths remain writable.
  - Both are pure, config-independent functions with direct unit tests
    (`tests/unit/test_safety_policies.py`).

### Changed

- **Token counting lives in `utils/tokenization.py`.** `count_tokens` moved to a
  neutral module that both the client and the server import, removing the
  `client.py → server.tools` import that crossed the MCP layer boundary.
  `ToolManager.count_tokens` now delegates to it (single implementation). The
  encoder is created lazily, so importing the module has no side effects.

### Fixed

- **Duplicate RAG/chunk-cache flush** in `LangGraphClient.clear_context()` — the
  flush block was executed twice; removed the redundant copy.

### Internal

- **`agent.py` decomposition (pure helpers).** Extracted the stateless logic out of
  `client/agent/agent.py` (2120 → ~1685 lines) into four focused, independently
  testable modules, with `LangGraphAgent` keeping thin delegating methods so its
  public surface is unchanged:
  - `client/agent/message_codec.py` — Strands↔LangChain message conversion.
  - `client/agent/message_sanitizer.py` — orphaned tool-call/result pair repair.
  - `client/agent/plan_policy.py` — plan-mode block decision, read-only-bash
    heuristic, and the associated data tables.
  - `client/agent/tool_formatting.py` — tool-call marker rendering, ctrl+o capture,
    arg-normalization, and tool-error translation.
    No behavior change; all existing unit tests pass unchanged. The heavier
    streaming / tool-execution / orchestration paths intentionally stay in `agent.py`
    until they have unit coverage.
- `requirements.txt`: collapsed the redundant `mcp` / `mcp[cli]` pins into a single
  `mcp[cli]>=1.26.0`, and dropped `uv` from runtime deps (it's an install/dev tool,
  not imported at runtime).

## [0.12.0] — 2026-07-01

### Added

- **Richer terminal UI (prompt_toolkit).** The input prompt is now a
  prompt_toolkit reader with slash-command completion, history, a plan-mode tag,
  and **`Ctrl+O`** — a toggle panel that shows the last turn's tool calls with
  their **full, un-elided arguments** (the inline `[⚙ …]` marker still elides
  long commands/paths to keep the stream readable; `Ctrl+O` reveals them in
  full). The panel is ephemeral (press again to hide) and leaves nothing in
  scrollback. The design is **app-per-prompt**: prompt_toolkit runs only while
  reading a line, and the answer then streams with ordinary `print()` — so
  markdown, the `●` marker, line-wrapping, native scrollback, and copy/paste all
  work exactly as before. Non-TTY runs (pipes / CI) transparently fall back to
  plain `input()`.
- **Dialog-driven configuration.** `/load` now offers an arrow-key picker of
  saved conversations, and `/config`, `/model`, `/params` present their prompts
  as full-screen dialogs (text, yes/no, and single-choice lists). Consistent
  keys throughout: **Enter confirms, Esc cancels** (cancelling aborts the flow
  with nothing written). Text fields prefill the current value when you're
  editing an existing setting; a fresh model name is offered as a _suggestion_
  (empty field) rather than a prefilled default; **mandatory fields won't advance
  while empty**. Everything degrades to the previous `input()` prompts when not a
  TTY.

### Changed

- **`DOC_MAX_TOKENS` is now derived as 25% of `MAX_CONVERSATION_TOKENS`**
  automatically in `/config`, `/model`, and `/params`, so the document read cap
  scales with the context window instead of being tuned by hand.

## [0.11.3] — 2026-06-30

### Fixed

- **Ollama embeddings now honor the configured `HOST`/`PORT`** instead of
  silently following `$OLLAMA_HOST`. `_embed_ollama` called the bare
  `ollama.embed()`, whose host comes from the `OLLAMA_HOST` env var (or the lib
  default) — so a stray `OLLAMA_HOST` pointing at another server (e.g. a
  `llama-swap` port left over from a local-engine experiment) hijacked the embed
  request, which 400'd and degraded _silently_ to the sha256 fallback vectors
  ("semantic search will be DEGRADED"). It now builds an explicit
  `ollama.Client(host=…)` from `RAG.EMBED_MODEL_ID.HOST`/`PORT` (default
  `localhost:11434`), matching how the LLM and vision controllers resolve their
  Ollama base URL. The host is resolved inside the Ollama path only, so a
  non-Ollama embeddings provider (Bedrock/OpenAI/SageMaker/LiteLLM) is
  unaffected. Episodic-memory and RAG retrieval are unchanged in shape — they
  just stop falling back to degraded embeddings when an unrelated `OLLAMA_HOST`
  is set.

## [0.11.2] — 2026-06-30

### Fixed

- **Plan mode no longer "sticks" to a saved-then-reloaded conversation.** When
  plan mode is active, the client prepends an ephemeral `<plan-mode-active>`
  reminder to the turn's prompt. That banner was being baked into the stored
  `HumanMessage`, so saving the conversation and reloading it (with `/plan` since
  toggled off) re-fed the banner to the model, which then behaved as if still in
  plan mode. The banner is now stripped before the turn is stored (the model
  still receives it that turn, so enforcement is unchanged) and stripped again
  from any banner-bearing messages on load, so reloaded chats start clean.
- **Clearer tool arg-validation errors.** When the model called a tool with a
  required argument missing (notably `file_edit` without `new_string`, which a
  capable model still did ~80× in one session while trying to DELETE text), the
  raw pydantic "Field required" text was handed back and retried verbatim in a
  loop. Such failures are now translated into plain guidance naming the missing
  argument(s) (e.g. _"the call to `file_edit` is missing required argument(s):
  new_string … pass \"\" to delete text"_), and `file_edit`'s description now
  states all three args are required and documents `new_string=""` for deletion
  (with an example), steering wholesale rewrites toward `fs_write`.

## [0.11.1] — 2026-06-30

### Fixed

- **`fs_read` (and RAG/memory/compaction token counting) no longer crash on
  files containing special-token text** like `<|endoftext|>`. tiktoken's
  `encode()` raises on such text by default; the token-counting helpers now pass
  `disallowed_special=()` so the text is COUNTED as ordinary content (these are
  length estimates, not encodings sent to the model). Fixed across all four call
  sites: `ToolManager.count_tokens` (fs_read's size check), the RAG chunking
  helper, `EpisodicMemoryManager`, and `AgentConversationManager`. Reading a file
  that mentions `<|endoftext|>` (e.g. a config `STOP` list or a prompts file) now
  works instead of erroring with "disallowed special token".

## [0.11.0] — 2026-06-30

### Added

- **`TYPE: openai` can point at any OpenAI-compatible server** via `API_BASE`
  (alias `ENDPOINT_URL`) + optional `API_KEY` — making a local
  [`llama-server`](https://github.com/ggml-org/llama.cpp) (llama.cpp), LM Studio,
  vLLM, or `llama-swap` a drop-in alternative to Ollama with no extra provider.
  Local servers usually ignore auth, so a placeholder key is sent automatically
  when `API_BASE` is set and no `API_KEY` is given. Applies to **all three model
  roles** — `MODEL_ID` (chat), `VISION_MODEL_ID`, and `RAG.EMBED_MODEL_ID`
  (embeddings) — so a single local endpoint (e.g. `llama-swap` hot-swapping
  several `llama-server` instances) can serve everything. The openai embeddings
  path previously **ignored** `API_BASE`/`API_KEY` (always hit api.openai.com);
  it now honors them, matching the chat/vision controllers. Ollama
  (`TYPE: ollama`) remains fully supported — this is an alternative, not a
  replacement. See the new "Local OpenAI-compatible servers" recipe in
  `docs/configuration.md`.
- **Configurable embedding dimension** via `RAG.EMBED_MODEL_ID.DIMENSION`. The
  vector size was hardcoded (name lookup → 1024 default); it's used for the
  SHA256/zeros fallback and empty-result shape (real embeddings pass through at
  the provider's native size). Set it to your embedder's real dimension so the
  fallback stays index-consistent if the provider ever flaps. Settable via
  **`/params`** (the embeddings model is now a `/params` target) and **`/model`**
  (an "Embedding dimension" prompt), or by hand.
- **`/model` and `/params` now expose the local-server keys.** `/model`'s openai
  connection flow prompts for an optional OpenAI-compatible base URL + key (chat,
  vision, embeddings), and the embeddings flow prompts for `DIMENSION` — so the
  whole local-engine switch is UI-driven, no hand-editing required.

## [0.10.5] — 2026-06-29

### Fixed

- **Newer Claude models (Opus 4.7+) no longer 400 on the Anthropic protocol with
  "thinking.type.enabled is not supported for this model".** The factory and the
  direct-Anthropic controller always sent the legacy
  `thinking={"type":"enabled","budget_tokens":…}` form, which Opus 4.7+ rejects —
  they require `thinking={"type":"adaptive","display":"summarized"}` plus
  `output_config={"effort":…}`. The thinking request form is now chosen by model
  version (parsed from the id): **4.7+** → adaptive + summarized display,
  **4.6** → adaptive (no display), **≤4.5 / 3.x** → enabled + budget (these
  reject adaptive). An unknown version assumes the current adaptive API;
  `EXTRA_PARAMS` (e.g. your own `thinking`/`output_config`) overrides if version
  detection is wrong. Verified live against a Mantle Opus 4.8 endpoint — both
  the no-crash path and a returned `thinking` summary block. Applies to both the
  Mantle `anthropic` protocol and direct `TYPE: anthropic`.
- **Thinking on the Mantle `anthropic` protocol is now opt-in**, matching direct
  `TYPE: anthropic`. It was always-requested (a default `thinking_tokens` made
  the gate always true), so a Claude model that doesn't support extended
  thinking would 400 on every call with no way to disable it. Thinking is now
  enabled only when `REASONING_EFFORT` or `REASONING: true` is set; a
  non-reasoning Claude on Mantle works again. (Note: the OpenAI Responses
  reasoning _summary_ is requested correctly on Mantle GPT-5, but the Mantle
  gateway returns an empty summary — the reasoning happens and is billed, the
  text just isn't forwarded; direct `TYPE: openai` on the responses protocol
  does return it. This is upstream, not a client issue.)

## [0.10.4] — 2026-06-29

### Fixed

- **Regression from 0.10.3: every query failed on the Responses protocol with
  "Responses.create() got an unexpected keyword argument 'reasoning_effort'".**
  0.10.3 builds responses-protocol models with a `reasoning` OBJECT
  (`{"effort": …, "summary": "auto"}`), but `disable_reasoning` (used for the
  auxiliary classify/decompose calls) still set the scalar `model.reasoning_effort
= "none"`. The model then carried BOTH, and the Responses API rejects them
  together. `disable_reasoning`/`restore_reasoning` now operate on the
  `reasoning` object when present (setting its effort to `"none"` in place,
  preserving the summary), and only fall back to the scalar `reasoning_effort`
  for the legacy shape. Also fixed the Ollama-branch guard so a `reasoning`
  _dict_ is no longer mistaken for the Ollama boolean toggle. Verified at the
  request-payload level (no `reasoning_effort` reaches the Responses API).

## [0.10.3] — 2026-06-29

### Fixed

- **Reasoning is now visible for OpenAI-style reasoning models, not just
  Anthropic.** Two halves were missing:
  - **Request side:** the OpenAI Responses API only returns a reasoning
    _summary_ if you ask for it (`reasoning={"effort": …, "summary": "auto"}`);
    effort alone reasons invisibly. mnemoai sent only the bare effort, so
    Mantle GPT-5/Grok (and direct OpenAI) produced no visible reasoning. Now,
    on the `responses` protocol, `REASONING_EFFORT` is sent as a `reasoning`
    object that requests the summary. Direct `TYPE: openai` gains an
    `API_PROTOCOL` key (`chat_completions` default | `responses`) to opt into
    the Responses API and get the same treatment.
  - **Extraction side:** the agent's reasoning extractor now recognizes the
    OpenAI Responses summary block shape
    (`{"type":"reasoning","summary":[{"type":"summary_text","text":…}]}`)
    alongside the existing Bedrock `thinking` blocks, `reasoning_content`, and
    `<think>` tags. The summary streams as gray reasoning and never leaks into
    the visible answer.
- **Anthropic thinking enables on `REASONING_EFFORT` alone.** Direct
  `TYPE: anthropic` previously needed `REASONING: true` to turn on extended
  thinking; `REASONING_EFFORT` alone now enables it too (matching the OpenAI
  behavior). Anthropic thinking was already extracted/displayed once enabled —
  this just removes the extra flag.
- `EXTRA_PARAMS` still wins everywhere: set your own `reasoning`/
  `reasoning_effort` (OpenAI/responses) or `thinking` (Anthropic) to override
  the derived defaults — e.g. `reasoning: {effort: high, summary: detailed}` or
  `summary: none`.

## [0.10.2] — 2026-06-29

### Fixed

- **Shell-escaped / quoted file paths now resolve in tools.** A path the user
  copied from a terminal — `/Users/me/Screenshot\ 2026.png` (backslash-escaped
  spaces) or `"…/My File.png"` (quoted) — was passed verbatim to tools like
  `describe_image`, `fs_read`, `file_edit`, `glob_search`/`grep_search`, which
  aren't shells, so the literal backslashes/quotes made the file "not found"
  (often sending a smaller model into a retry/guess spiral). New
  `utils/path_utils.normalize_path` resolves these without breaking legitimate
  paths: it prefers the path exactly as given and only falls back to a
  de-escaped/de-quoted variant when the literal one doesn't exist on disk. Write
  targets (which may not exist yet) use `clean_path_syntax`, a syntactic-only
  cleanup. Applied across the file/image/search tools.

## [0.10.1] — 2026-06-29

### Fixed

- **`/save` now writes back to the open conversation file.** After `/load`-ing a
  conversation (or saving one earlier in the session), a bare `/save` overwrites
  that same file instead of creating a new `conversation_<timestamp>.json`. The
  open file is tracked in `current_conversation_path`, set on load and on first
  save; `/clear` resets it (a fresh conversation saves to a new file), and
  `/save <path>` still targets an explicit file/dir (and becomes the new open
  file).

### Changed

- **Removed the model name from the input prompt.** The status indicator added
  in 0.10.0 crowded the prompt line with long model names (e.g.
  `brnpistone/Qwen3.5-4B-AgentCoder-q6-k:latest`). The prompt is back to a clean
  `>`, keeping only the compact `🔒 plan` tag while plan mode is active. Use
  `/model` to see or change the current model.

## [0.10.0] — 2026-06-29

### Added

- **Lightweight Markdown rendering in streamed answers.** The model's Markdown is
  now rendered in the terminal instead of printed as literal syntax: headers
  (`##`) show as bold without the hashes, `-`/`*` bullets become `•`, numbered
  lists are kept, and `**bold**`/`*italic*` are styled. Inline `code` (bold cyan)
  and fenced code blocks (Pygments syntax highlighting) are unchanged, and
  clickable URLs still work. Implemented in `CodeFormatter` with a line-buffered
  pass — **no new dependency** (no `rich`). Emphasis is conservative: a spaced
  expression like `a * b * c` (e.g. `cost = instance_count * price_per_hour`) is
  never mis-italicized, and `**x**` inside backticks stays literal.
- **"Allow for this session" at confirmation prompts.** The destructive-tool gate
  now offers `Proceed? (y/N/a)` — answering `a` trusts that whole category
  (shell / file-write / memory) for the rest of the session, so a multi-step task
  no longer re-prompts for every command. Default-deny is unchanged; trust is
  per-category and resets on restart.
- **Status indicators on the input prompt.** The prompt line now shows a
  `🔒 plan` tag while plan mode is active and a dim chat-model name, so it's
  always clear which model is answering and whether mutations are blocked.

### Changed

- **Tool-call markers keep both ends of long values.** `[⚙ execute_bash(command=…)]`
  now middle-elides (`head…tail`) instead of cutting the tail, so a command's
  meaningful end (subcommand, flags) and trailing args (e.g. `timeout=30`) stay
  visible.

## [0.9.3] — 2026-06-27

### Fixed

- **Web-crawler progress no longer collides with the loading spinner.** The
  per-tool progress spinner added in 0.8.19 animated (carriage-return redraws)
  _during_ a `web_crawler` call, while crawl4ai writes its own live
  `[INIT]/[FETCH]/[SCRAPE]/[COMPLETE]` progress to the terminal — the two
  overwrote each other on the same lines (e.g. `⠏ Running web_crawler…[FETCH]…`).
  Tools that report their own progress are now listed in
  `_SELF_REPORTING_TOOLS`, and `_invoke_tool` keeps the spinner **stopped** for
  them so their output shows cleanly (as it did before 0.8.19), while every other
  tool still gets the progress spinner. Also fixed a stale progress-label key
  (`web_crawl` → the real tool name is `web_crawler`).

## [0.9.2] — 2026-06-26

### Fixed

- **Orphaned tool call/result pairs no longer wedge the conversation.** A turn
  cut short (recursion limit, stream error, interrupt) or a history slice could
  leave an assistant `tool_call` with no matching tool result (or vice-versa)
  persisted in the conversation. Strict providers — the OpenAI/Mantle Responses
  API — then rejected **every** subsequent turn with "No tool output found for
  function call …" (or the mirror, "No tool call found for function call
  output …"), with retries and the non-streaming fallback all re-sending the
  same broken history and failing identically. Added `_sanitize_tool_pairs`: an
  id-based repair that drops orphaned tool calls (keeping any visible text on
  that turn) and orphaned tool results before each model call, and on the kept
  window during compaction so the persisted history is repaired too. Extends the
  existing `_safe_tool_boundary` guard (which only covered the orphaned-_result_
  case at the compaction split) to both orphan directions, anywhere in history.

## [0.9.1] — 2026-06-26

### Fixed

- **Trivial queries no longer produce a blank answer.** A short conversational
  prompt ("can you do it?", "please do it") could be classified as `full` and
  sent through the orchestrator, which decomposed it into a single subtask run by
  the worker loop. That loop — unlike the normal `call_model` path — had **no
  empty-turn salvage**, so when a reasoning model streamed only hidden thinking
  (no visible text), nothing was printed and the turn ended silently after the
  `[Step 1/1]` / `[Context: …]` lines. Two fixes:
  - **Worker-loop salvage:** `_run_worker_loop` now mirrors `_call_model`'s
    guarantee — when a worker finishes with no visible content, it retries once
    with reasoning disabled (streamed, so the answer prints) and falls back to a
    visible message if still empty. The orchestrator path can no longer surface a
    blank answer.
  - **Smarter orchestrator gating:** a trivial, signal-free query classified as
    `full` now goes to the normal `call_model` path (which binds the same tools
    and already has the empty-turn safety net) instead of being decomposed. Only
    substantive tasks are orchestrated (`router.is_trivial_query`).

## [0.9.0] — 2026-06-26

### Added

- **Agent Skills** — authored, on-demand instruction packs.
  A skill is a directory `~/.mnemoai/skills/<name>/SKILL.md`
  with YAML frontmatter (`name` + `description`) and a markdown body, optionally
  bundling `reference.md` / `scripts/`. Skills fill the gap between always-on
  context (system prompt, `MEMORY.md`) and learned tactics (the ACE playbook):
  _authored procedures the model follows when a task matches_.
  - **Three-tier progressive disclosure:** (1) only each skill's name+description
    is injected into the system prompt at session start (cheap, always-on — the
    `<available_skills>` block); (2) the full body is loaded into context **only
    when the model calls the new `use_skill` tool** (its return value becomes the
    body); (3) bundled `reference.md`/`scripts/` are read/run on demand via the
    existing `fs_read`/`execute_bash` tools. Installing many skills stays cheap.
  - `use_skill` is bound on **every** route (it's in `_ALWAYS_AVAILABLE_TOOLS`),
    so a skill-matching request that classifies as `simple_qa` (e.g. "write a
    commit message") can still trigger it.
  - The metadata block is **re-injected after conversation compaction** (which
    rebuilds the system prompt fresh), so skills don't vanish mid-session.
  - **`/skills`** lists installed skills; a malformed skill is shown under
    _Skipped_ with the reason (missing `description`, bad YAML, over-long
    description) instead of being silently ignored — the authoring-feedback loop.
    `/skills <name>` previews a skill's full body.
  - Two skills are **seeded on first run**: a `commit-message` example, and a
    **`skill-creator`** meta-skill — ask the assistant to "create a skill for X"
    and it loads that guidance and authors a well-formed `SKILL.md` for you.
  - New module `client/memory/skill_store.py` (`SkillStore`, shared by the
    server tool and the client like `MemoryStore`), `server/tools/skill_tool.py`,
    `utils/paths.py:skills_dir()` + seeding. Gated by `ENABLE_SKILLS` (default
    true). See the Skills section in `docs/usage.md`.

## [0.8.21] — 2026-06-25

### Fixed

- **External MCP tools are now reachable on the `simple_qa` route.** A short
  factual question ("what time is in Singapore?") classifies as `simple_qa` —
  the no-tools route, which bound only the built-in meta tools (`memory`,
  `describe_image`, `fs_read`). External (`mcp.json`) tools were appended only to
  _non-empty_ routes, so a configured server like `time` was invisible on the
  very route such questions land in, and the model answered from its own
  knowledge ("I don't have access to the live clock") instead of calling the
  tool. External tools are user-configured capabilities, so they're now bound on
  **every** route, including `simple_qa`. (Built-in tool pruning per route is
  unchanged — only the handful of explicitly configured external tools ride
  along everywhere.)

## [0.8.20] — 2026-06-25

### Fixed

- **Closed the last "stuck"-looking gap: the final answer after the last tool.**
  0.8.19 added a spinner _during_ tool execution, but the model call that turns
  the tool results into the final reply (`_call_model`) relied on the spinner
  being left running by the preceding tool node — which no longer held once each
  tool call stopped its own spinner on completion. The result was a blank pause
  between the last tool and the model's answer. `_call_model` now starts the
  spinner at entry itself (idempotent; stopped as soon as visible text/reasoning
  streams or the next tool starts), mirroring `_aggregate`, so the wait for the
  model's first token always shows progress.

## [0.8.19] — 2026-06-25

### Fixed

- **No more "stuck"-looking terminal while a tool runs.** Previously the spinner
  stopped when a tool was about to execute and only restarted _after_ all tools
  finished, so a slow `tool.invoke()` (executing Python, a long shell command, a
  web fetch, a large file write) — especially right after the user confirmed it
  — showed a frozen, blank terminal with no sign of progress. The agent now
  animates a spinner with a per-tool label (e.g. `Running: python run.py`,
  `Searching the web`, `Writing /path/to/file`) for the full duration of each
  tool call, at both execution chokepoints (main loop and orchestrator worker
  loop). The spinner is always stopped afterward, even if the tool errors.

## [0.8.18] — 2026-06-25

### Changed

- **Plan mode is now "read-only except…" rather than a blanket block.** While `/plan` is active:
  - **Read-only shell commands run.** `execute_bash` is allowed when the command
    is read-only (leading program in an allowlist — `ls`, `cat`, `grep`, `rg`,
    `find`, `git status/log/diff/show`, etc. — with no redirection/chaining
    operators, and `git` limited to read-only subcommands). Mutating commands stay
    blocked. This lets the agent investigate properly while planning.
  - **The plan can be written to disk.** `fs_write`/`file_edit` are allowed only
    for a Markdown (`.md`) file under the plans directory (`paths.plans_dir()`),
    so the model can draft its plan incrementally. All other file writes remain
    blocked.
  - **The per-turn reminder is firmer.** ("Plan mode is active… you MUST NOT make any
    edits… this supersedes any other instructions"), points the model at the
    single writable plan file, and tells it to ask clarifying questions rather
    than guess.
  - Blocked-tool feedback is tailored per tool (read-only-bash hint; plan-file
    path hint) so the model pivots cleanly instead of just erroring.

## [0.8.17] — 2026-06-25

### Changed

- **Prompts moved out of `config.yaml` into a dedicated `prompts.yaml`.** All
  model-facing prompts — `SYSTEM_PROMPT`, `ROUTING_PROMPT`, `ORCHESTRATOR_PROMPT`,
  `AGGREGATOR_PROMPT`, and the compaction prompts (`SUMMARY_SYSTEM_PROMPT`,
  `SUMMARY_TASK_PROMPT`, previously hardcoded in Python) — now live in a
  `prompts.yaml` sibling of `config.yaml` (same `config/` dir, same resolution +
  first-run seeding; `$MNEMOAI_PROMPTS` overrides). `config.yaml` now holds
  configuration only (~165 lines, down from ~400). Access is via
  `Config().prompt("KEY")`.

### Breaking

- Prompts are read **only** from `prompts.yaml` — never from `config.yaml`, and
  there are no in-code prompt fallbacks. Prompt keys left in `config.yaml` are
  ignored (a one-time migration warning is logged); move customizations to
  `prompts.yaml`. The app **fails fast at startup** (`PromptError`) if a required
  prompt is missing: the mandatory prompts (`SYSTEM_PROMPT`,
  `SUMMARY_SYSTEM_PROMPT`, `SUMMARY_TASK_PROMPT`) are always required, and a
  feature's prompt is required when that feature is enabled (`ROUTING_PROMPT`
  with `ENABLE_ROUTING`; `ORCHESTRATOR_PROMPT`/`AGGREGATOR_PROMPT` with
  `ENABLE_ORCHESTRATION`). A bundled `prompts.yaml` seeds new installs and
  provides the defaults, so out-of-the-box runs are unaffected.

## [0.8.16] — 2026-06-24

### Fixed

- Image and file questions no longer fail by routing to a tool subset that
  can't handle them. `describe_image` and `fs_read` are now always-available
  meta tools (bound on every route, like `memory`), so "what's in this image?"
  or "what's in config.yaml?" — which classify as `simple_qa`/`knowledge` — can
  always reach the vision/read tools instead of falling back to reading bytes
  as text.
- `fs_read` on a binary/image file now returns a clear message steering the
  model to `describe_image` (for images) instead of dumping a raw
  `UnicodeDecodeError` stack trace. Binary files are detected up front
  (extension + content sniff) and logged calmly.
- **Route table audit & repair.** The `knowledge` route named four reader tools
  that don't exist (`read_csv/json/pdf/docx` — `fs_read` handles all formats via
  its `mode`), and six tools (`list_background_tasks`, `cancel_background_task`,
  `clear_completed_tasks`, `get_plan_status`, `todo_clear`, `clear_documents`)
  were reachable only via `full`. Routes are now re-derived from the real tool
  surface: `code` binds the complete todo/plan/background-task suites; `knowledge`
  is scoped to the RAG index (`list_documents`/`search_in_documents`/
  `clear_documents`). A regression test guards against future orphans/stale refs.

### Changed

- The query router gained a deterministic heuristic fast-path: obvious queries
  (greetings → `simple_qa`, a lone file path → `code`, a URL → `research`, a doc
  extension → `knowledge`) route instantly with **no LLM classification call**,
  while multi-signal or ambiguous queries fall back to `full`/the LLM. This cuts
  a per-query round-trip, sidesteps the empty/blank-route issues on reasoning
  models, and — by always preferring `full` on mixed signals — never under-binds
  tools. The `knowledge` category in `ROUTING_PROMPT` is re-scoped to RAG
  document search (reading a file by path works from any category).

## [0.8.15] — 2026-06-24

### Fixed

- The "Thinking" spinner no longer stops too early on models that stream hidden
  reasoning. Some providers (e.g. Anthropic via Bedrock) send redacted/secret
  reasoning chunks before the answer; the spinner used to stop on the first such
  chunk, leaving a dead pause (spinner gone, nothing printed) until the visible
  answer arrived. The spinner now stops only when something is actually about to
  be displayed — visible answer text, or reasoning shown in verbose mode — and
  keeps spinning through hidden reasoning. The streaming callback likewise
  ignores empty/whitespace-only tokens.

## [0.8.14] — 2026-06-24

### Fixed

- The interactive configurator (`/config`, `/model`, `/params`, first-run setup)
  now **re-asks on invalid input** instead of silently proceeding. Previously a
  bad menu choice kept the current value (or cancelled), and a non-numeric entry
  for `MAX_TOKENS` / `MAX_CONVERSATION_TOKENS` / the Mantle protocol was accepted
  or defaulted. Now menu prompts re-ask until a listed option is chosen, numeric
  prompts re-ask until the value parses as an int/float (`none` still clears
  optional `MAX_TOKENS`), and the `/model` and `/params` model pickers re-ask
  within the configured sections rather than cancelling on a wrong number.
- Streamed output no longer drops code at end-of-stream. The `CodeFormatter`'s
  `flush()` (now actually called when a stream ends) emits a response that ended
  inside an unclosed ` ``` ` fence, a held-back trailing backtick, and resets the
  terminal color after an unbalanced inline backtick — previously those were
  silently lost or left the prompt stuck in cyan. Bare `except:` clauses in the
  highlighter were also narrowed to `except Exception`.

### Changed

- Inline code / identifiers in streamed output are now **bold cyan**,
  instead of plain cyan, for a crisper distinction from surrounding prose.
  Fenced code blocks keep Pygments/monokai highlighting.

### Added

- A visible cancel affordance in the configurator: every interactive flow shows
  "Press Ctrl+C or Ctrl+D at any prompt to cancel — nothing is saved", and
  EOF/interrupt at any prompt now aborts cleanly (config left untouched) instead
  of half-applying an entry.

## [0.8.13] — 2026-06-24

### Fixed

- `/save` now writes to the profile's `conversations/` directory instead of the
  profile root (`~/.mnemoai/<profile>/`), where saved chats were cluttering the
  top level. New `paths.conversations_dir()` helper; existing files in the root
  are not moved (load them with an explicit `/load <path>`).

### Added

- `/save [path]` accepts an optional destination — a directory (saved there with
  the default `conversation_<ts>.json` name) or a full file path (`.json` added
  if missing). With no argument it saves to `conversations/` as before.
- `/load` with no argument now lists saved conversations (newest first, with
  relative times) and lets you pick one by number, instead of requiring a typed
  path. `/load <path>` still loads a specific file directly.

### Changed

- Compaction now shows phased progress on the spinner instead of a static
  "Generating summary …" line that looked frozen during the (single, long) LLM
  summary call. The spinner animates through `Summarizing N older messages` →
  `Applying summary` → the green "Compacted: …" result. (A true % bar isn't
  possible — one LLM generation has no measurable total — so this surfaces the
  discrete stages honestly rather than a fake bar.) The `Spinner` gained an
  optional `start(label=…)` argument and a `set_label()` method; the default
  label stays "Thinking" for the normal agent loop.

## [0.8.12] — 2026-06-24

### Changed

- The agent loop no longer hard-stops at 50 steps. That cap cut off legitimate
  long tasks mid-work with "Agent hit recursion limit". Where context compaction,
  not a step count, is the real limiter — the default `LLM.RECURSION_LIMIT`
  is now **200** (still configurable). It remains a runaway guard
  (LangGraph requires a finite bound), so hitting it now signals a
  likely stuck loop and the message says so and points at the config knob.

### Fixed

- Log lines (warnings/errors/info) no longer print inline with streamed answer
  text. The chat UI streams chunks to stdout without a trailing newline, so a
  log written to stderr afterwards landed on the same visual line. A cursor
  tracker now records whether stdout is mid-line, and the log handler prepends a
  newline when needed — so logs always start on their own line (no-op on
  piped/non-TTY output).

## [0.8.11] — 2026-06-23

### Fixed

- Compaction no longer corrupts the conversation by splitting a tool
  call/result pair. Previously the kept-verbatim window could start with an
  orphaned tool result (its originating assistant tool-call turn summarized
  away), which the OpenAI Responses API rejects on the very next turn with a
  deterministic 400 — "No tool call found for function call output with call_id
  …" — looping until the query fails. The split point is now tool-pair-safe
  (`_safe_tool_boundary`): it moves earlier as needed so a tool call and its
  result are always kept (or summarized) together.

### Changed

- Conversation compaction: a "summarizing conversations" system framing plus
  the verbatim structured task prompt (an `<analysis>` pass, then nine fixed
  sections — Primary Request, Key Technical Concepts, Files and Code Sections,
  Errors and fixes, Problem Solving, All user messages, Pending Tasks, Current Work,
  Optional Next Step). `/compact <focus>` is injected under a `## Compact Instructions`
  header; the `<analysis>` scratchpad is stripped from the result; and the injected summary
  block carries the verbatim continuation instruction so the model resumes the
  work seamlessly instead of recapping.

## [0.8.10] — 2026-06-23

### Added

- `REASONING_EFFORT` is now a first-class, `/params`-tunable knob for **every**
  provider that supports reasoning — `openai`, `anthropic`, `bedrock`, `mantle`,
  and `litellm` — translated to each provider's mechanism: forwarded as
  `reasoning_effort` on OpenAI and Mantle's `responses` protocol; mapped to a
  `thinking` token budget on Anthropic, standard Bedrock, and Mantle's
  `anthropic` protocol; passed through LiteLLM (which translates per backend).
  This gives Bedrock Mantle a real reasoning path it previously lacked. When
  thinking is enabled this way, `temperature`/`top_p`/`top_k` are dropped (the
  providers reject them).
- `EXTRA_PARAMS`: a generic per-model passthrough for the long tail. Any
  `MODEL_ID` / `VISION_MODEL_ID` may include an `EXTRA_PARAMS` dict whose
  contents are forwarded verbatim to the underlying model's request body (using
  the provider's own parameter names), so provider-specific knobs the curated
  registry doesn't model need no code change. Works for every provider (OpenAI,
  Anthropic, Bedrock, Ollama, SageMaker, LiteLLM, and all three Mantle
  protocols). `reasoning_effort` is lifted to a first-class arg on OpenAI-family
  clients; everything else merges into `model_kwargs`. A non-dict value is
  ignored. It is **config.yaml-only** — supported everywhere (never pruned by
  `/model`) but not prompted by `/model` or `/params`. `EXTRA_PARAMS` overrides
  the first-class `REASONING_EFFORT` if both set the same key.

### Changed

- `/params` only offers (and only writes) params the chosen provider actually
  supports — now covered by a regression test (e.g. Anthropic is never prompted
  for `PRESENCE_PENALTY`/`FREQUENCY_PENALTY`, but is for `REASONING_EFFORT`).

## [0.8.9] — 2026-06-23

### Changed

- The curated `MEMORY.md` now separates entries with a Markdown `---` rule
  instead of a `§` section sign, so the file renders as clean prose with
  dividers rather than showing a stray symbol. Existing `§`-delimited files are
  still read correctly and migrate to `---` automatically on the next memory
  write.

## [0.8.8] — 2026-06-22

### Changed

- `/clear` now wipes the terminal screen and scrollback and re-shows the welcome
  banner, for a true fresh start, instead of appending "Context cleared!" below
  the old conversation. No-op when stdout isn't a TTY (piped/redirected output
  stays clean).

## [0.8.7] — 2026-06-22

### Removed

- Reverted the streaming repetition-loop guard added in 0.8.6
  (`_is_degenerate_repetition`). Investigation confirmed the `<unused6226>`
  degeneration is a **non-deterministic serving-side issue** with
  `google.gemma-4-31b` on Bedrock Mantle (the identical request produces a clean
  answer on most calls and a repetition loop on a minority), not a client
  problem — so it belongs upstream, not as a heuristic in the stream loop. The
  guard added complexity and a regex scan on every chunk for a vendor bug that
  is being reported to AWS; removing it keeps the streaming path lean.

## [0.8.6] — 2026-06-22

### Fixed

- A runaway repetition loop no longer hangs the UI or burns the whole token
  budget. Some Bedrock Mantle-served models (observed with `google.gemma-4-31b`)
  intermittently degenerate into emitting a single reserved token
  (`<unused6226>`) until `MAX_TOKENS`, flooding the screen. The streaming loop
  now detects a special token repeated many times at the tail, aborts the
  stream early, and shows a clear message suggesting a different model or
  `API_PROTOCOL`. Conservative thresholds avoid tripping on legitimate
  repetition (lists, code, prose).

## [0.8.5] — 2026-06-22

### Fixed

- Arrow keys and backspace now work while typing answers in the `/config` and
  `/model` (and first-run) configurator prompts. They previously leaked raw
  escape sequences (e.g. `^[[D`) into the value because the `input()` prompts
  had no line editing; importing the stdlib `readline` module enables it.

## [0.8.4] — 2026-06-22

### Changed

- The answer marker (`●`) now precedes the answer on the **same line**
  (`● answer…`) instead of sitting on its own line above it, and it is also
  shown when reasoning is visible — printed after the gray reasoning block, on
  the answer line — so every assistant reply carries the marker.

## [0.8.3] — 2026-06-22

### Added

- A subtle cyan `●` marker is printed before a streamed answer when the model
  shows no reasoning, so the reply is visually distinct from the user's prompt
  instead of butting directly against it. Shown only on user-facing answer
  turns (main reply, retried answer, aggregated result) — worker streams already
  carry a `[Step N/N]` header.

## [0.8.2] — 2026-06-22

### Fixed

- The query classifier no longer crashes when no `ROUTING_PROMPT` is configured
  (e.g. a stripped config): `get_classifier_prompt()` now returns a built-in
  default instead of `None`, which previously raised a pydantic
  `ValidationError` while building the classification `SystemMessage`. This also
  fixes the unit suite running without a `config.yaml` (CI).

## [0.8.1] — 2026-06-22

### Changed

- `/model` now resets a section's inference parameters (temperature, top_p,
  penalties, reasoning, stop, stream — everything except the separately-prompted
  `MAX_TOKENS`) whenever the model is changed. These are model-specific, so a
  value tuned for one model is no longer silently carried into another that may
  reject it (e.g. newer Claude/GPT reject `temperature`); re-tune via `/params`.

### Fixed

- Reasoning models on the OpenAI Responses API (e.g. Bedrock Mantle Grok /
  GPT-5) no longer spam `Router returned unknown route '', falling back to
'full'` on every turn. The query classifier now disables reasoning on these
  models (`reasoning_effort="none"`) so the one-word route lands in the
  response instead of being eaten by reasoning, retries once on a transient
  empty response, and falls back to `full` quietly (debug, not a warning) when
  classification genuinely yields nothing.
- No more silent empty turn when a reasoning model is truncated by the
  output-token limit: if a turn ends with no answer and the response reports a
  token-limit cutoff (`status: incomplete` / `finish_reason: length` /
  `stop_reason: max_tokens`), the agent now surfaces a clear "increase
  `MAX_TOKENS`" message instead of an empty reply. Reasoning models spend output
  tokens reasoning before answering, so a low `MAX_TOKENS` could consume the
  whole budget before any answer was produced.
- Transient empty model responses are now retried. Some endpoints (notably
  Bedrock Mantle reasoning models on the Responses API) intermittently return a
  completely empty response — no content, reasoning, or tool call — for the same
  prompt that succeeds on a retry. Every model call (the main loop, orchestrator
  workers, and the aggregator) now retries an empty turn up to `LLM.MAX_RETRIES`
  times. This fixes blank `[Step N/N: …]` turns seen under orchestration.

## [0.8.0] — 2026-06-22

### Removed

- The `/good` command and its conversation "quality marker" plumbing. The
  markers were written into saved-conversation JSON but never consumed (no
  training/export pipeline existed), so the feature over-promised. `/save` now
  writes a plain conversation file. (Removing a command pre-1.0; noted here so
  it isn't a surprise.)

### Added

- CI: a `tests` GitHub Actions workflow runs the unit suite (and an import-sort
  check) on every push/PR across Python 3.11-3.13.
- This `CHANGELOG.md`, plus a documented Stability & Versioning contract and a
  release checklist (see `docs/development.md`).
- Expanded the integration tier: plan-mode enforcement (deterministic — asserts
  the file write is blocked) and the bash-timeout regression run end-to-end.

### Fixed

- No more silent empty turns: when the model ends a turn with no visible text
  right after a tool ran (e.g. a bash timeout) and produced no reasoning either,
  the agent now salvages the last tool result (or a fallback message) instead of
  returning an empty string.
- Streaming errors no longer lose the turn: on a mid-stream failure the agent
  now retries once non-streaming and prefers that complete result, instead of
  keeping a truncated partial chunk.
- Integration tests are provider-aware: they no longer falsely skip with
  "Ollama not reachable" when the configured model is a cloud provider
  (Bedrock/Mantle/OpenAI/Anthropic/SageMaker/LiteLLM). Verified end-to-end
  against Bedrock (Claude Sonnet).
- Welcome box auto-sizes to its widest command row instead of a hardcoded
  width, so longer command descriptions (e.g. `/plan`) no longer overflow the
  border.

## [0.7.0] — 2026-06-19

### Added

- **Enforced plan mode (`/plan`).** A user-toggled, read-only mode that
  hard-blocks mutating/exec tools (`execute_bash`, `fs_write`, `file_edit`,
  git writes, background tasks) client-side until you exit — so "analyze the
  repo" can never turn into edits, regardless of the model. Read-only tools and
  the memory notebook stay allowed. Distinct from the advisory `plan_mode.py`
  bookkeeping tools.

## [0.6.1] — 2026-06-19

### Changed

- Refreshed the PyPI project description after the README was split into a
  MkDocs site; re-exported the demo GIF smaller.

## [0.6.0] — 2026-06-19

### Added

- **Curated persistent memory (`MEMORY.md`).** A small, bounded, profile-scoped
  markdown file the agent maintains itself via a `memory` tool
  (`add`/`replace`/`remove`), injected whole into the system prompt at session
  start. Char-capped (forces consolidation); complements episodic memory and the
  ACE playbook. `ENABLE_MEMORY` / `REQUIRE_MEMORY_CONFIRMATION` toggles; `/memory`
  command. The `memory` tool is bound on every route (incl. `simple_qa`).
- Documentation split into a MkDocs Material site published to GitHub Pages.

## [0.5.2] — 2026-06-19

### Fixed

- Logger output is colored by level (errors red, warnings yellow) on a TTY;
  redirected/piped logs stay plain.

## [0.5.1] — 2026-06-19

### Fixed

- Repair a common malformed tool-args shape from smaller models
  (`{'query="…"': ''}` → `{'query': '…'}`) before validation, so the call
  succeeds instead of failing the turn.

## [0.5.0] — 2026-06-19

### Added

- **Hard confirmation gate** for destructive tools: `execute_bash`
  (`REQUIRE_BASH_CONFIRMATION`) and `fs_write`/`file_edit`
  (`REQUIRE_WRITE_CONFIRMATION`), both default on. Enforced client-side; the
  prompt always fires regardless of the model. Non-interactive runs auto-proceed.

## [0.4.0] — 2026-06-19

### Added

- **External MCP servers** via `~/.mnemoai/mcp/mcp.json`.
  Tools from external servers are merged with the built-in
  server; colliding names are namespaced `servername__tool`. `/mcp` lists them.
- Orchestrator awareness of external tools (routes subtasks needing them to the
  `full` category).
- App-home reorganized into `config/` and `mcp/` subfolders, each seeded with
  example templates on first run (legacy flat paths still read).

## [0.3.0] — 2026-06-19

### Added

- **Direct Anthropic API provider** (`TYPE: anthropic`) for chat and vision via
  `langchain-anthropic` — distinct from the Bedrock Mantle `anthropic` protocol.

## [0.2.1] — 2026-06-19

### Added

- A gray `[⚙ tool(args)]` marker between reasoning blocks so tool calls are
  visible after MCP request logs were silenced.

## [0.2.0] — 2026-06-19

### Added

- `/params` command to tune a model's inference parameters interactively.
- Numbered provider-type menu in `/model` (Mantle shown as `bedrock-mantle`).

## [0.1.0] — 2026-06

### Added

- Initial public release on PyPI as `mnemoai-assistant`.
- Local agentic assistant on LangGraph + MCP: StateGraph agent (classify →
  route/orchestrate → call model ↔ execute tools), multi-provider LLM support
  (Ollama, Amazon Bedrock, Bedrock Mantle, OpenAI, SageMaker, LiteLLM), episodic
  memory, ACE playbook, user-profile learning, RAG, web search/crawl, vision,
  and a `prompt_toolkit` chat UI with `/config` / `/model` configurators.

[Unreleased]: https://github.com/brunopistone/mnemoai/compare/v0.11.1...HEAD
[0.11.1]: https://github.com/brunopistone/mnemoai/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/brunopistone/mnemoai/compare/v0.10.5...v0.11.0
[0.10.5]: https://github.com/brunopistone/mnemoai/compare/v0.10.4...v0.10.5
[0.10.4]: https://github.com/brunopistone/mnemoai/compare/v0.10.3...v0.10.4
[0.10.3]: https://github.com/brunopistone/mnemoai/compare/v0.10.2...v0.10.3
[0.10.2]: https://github.com/brunopistone/mnemoai/compare/v0.10.1...v0.10.2
[0.10.1]: https://github.com/brunopistone/mnemoai/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/brunopistone/mnemoai/compare/v0.9.3...v0.10.0
[0.9.3]: https://github.com/brunopistone/mnemoai/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/brunopistone/mnemoai/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/brunopistone/mnemoai/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/brunopistone/mnemoai/compare/v0.8.21...v0.9.0
[0.8.21]: https://github.com/brunopistone/mnemoai/compare/v0.8.20...v0.8.21
[0.8.20]: https://github.com/brunopistone/mnemoai/compare/v0.8.19...v0.8.20
[0.8.19]: https://github.com/brunopistone/mnemoai/compare/v0.8.18...v0.8.19
[0.8.18]: https://github.com/brunopistone/mnemoai/compare/v0.8.17...v0.8.18
[0.8.17]: https://github.com/brunopistone/mnemoai/compare/v0.8.16...v0.8.17
[0.8.16]: https://github.com/brunopistone/mnemoai/compare/v0.8.15...v0.8.16
[0.8.15]: https://github.com/brunopistone/mnemoai/compare/v0.8.14...v0.8.15
[0.8.14]: https://github.com/brunopistone/mnemoai/compare/v0.8.13...v0.8.14
[0.8.13]: https://github.com/brunopistone/mnemoai/compare/v0.8.12...v0.8.13
[0.8.12]: https://github.com/brunopistone/mnemoai/compare/v0.8.11...v0.8.12
[0.8.11]: https://github.com/brunopistone/mnemoai/compare/v0.8.10...v0.8.11
[0.8.10]: https://github.com/brunopistone/mnemoai/compare/v0.8.9...v0.8.10
[0.8.9]: https://github.com/brunopistone/mnemoai/compare/v0.8.8...v0.8.9
[0.8.8]: https://github.com/brunopistone/mnemoai/compare/v0.8.7...v0.8.8
[0.8.7]: https://github.com/brunopistone/mnemoai/compare/v0.8.6...v0.8.7
[0.8.6]: https://github.com/brunopistone/mnemoai/compare/v0.8.5...v0.8.6
[0.8.5]: https://github.com/brunopistone/mnemoai/compare/v0.8.4...v0.8.5
[0.8.4]: https://github.com/brunopistone/mnemoai/compare/v0.8.3...v0.8.4
[0.8.3]: https://github.com/brunopistone/mnemoai/compare/v0.8.2...v0.8.3
[0.8.2]: https://github.com/brunopistone/mnemoai/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/brunopistone/mnemoai/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/brunopistone/mnemoai/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/brunopistone/mnemoai/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/brunopistone/mnemoai/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/brunopistone/mnemoai/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/brunopistone/mnemoai/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/brunopistone/mnemoai/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/brunopistone/mnemoai/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/brunopistone/mnemoai/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/brunopistone/mnemoai/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/brunopistone/mnemoai/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/brunopistone/mnemoai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/brunopistone/mnemoai/releases/tag/v0.1.0
