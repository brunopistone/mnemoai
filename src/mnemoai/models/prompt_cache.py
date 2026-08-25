"""Provider prompt-cache breakpoints (the stable prefix of every request).

A cached prefix is read at a fraction of the input price AND skips prefill
entirely — the same cost measured as ~123s to first byte on a very large turn
(see ``llm_controller._boto_config``). Caching doesn't shorten the first turn's
prefill; it removes it from every subsequent call in the session, and an agentic
turn makes many calls (one per tool round) whose prompts share everything but the
newest messages.

**This prompt is already cache-shaped**, which is why one breakpoint is enough:
the system prompt is assembled ONCE at agent construction (playbook block
included), and every per-turn injection (``<steering>``, the episodic-memory
block, the plan-mode reminder) rides the newest user message and is stripped
before storage — so stored history is append-only and the prefix is stable.

**One kwarg, two expansions.** Both client libraries take the same
``cache_control`` call kwarg and place the markers themselves:

* ``langchain-aws`` (``bedrock``) turns it into Converse ``cachePoint`` blocks
  after the system blocks, after the tools array, and after the last message.
* ``langchain-anthropic`` (``anthropic``, and ``mantle`` on the ``anthropic``
  protocol) forwards it as the Messages API's top-level ``cache_control``, which
  marks the last cacheable block of the request.

The dict carries ``type`` AND ``ttl`` because each side reads a different half:
Anthropic requires ``type: "ephemeral"``, Bedrock ignores ``type`` and reads only
a non-default ``ttl``.

**Why the system prompt is ALSO marked block-level on the Anthropic transports**
(:func:`system_message`): the top-level marker lands at the tail, and between two
turns the tail changes more than by appending — the previous turn's injections
were stripped before storage — so the newest cached entry is no longer a prefix
of the next request. A breakpoint on the system prompt is unaffected by that
churn and keeps the system+tools prefix a hit on the FIRST call of every turn.
Measured against a live endpoint: with only the tail marker a changed tail read 0
tokens; with the system prompt marked as well the same request read the whole
system prefix. Bedrock needs no equivalent — its expansion already marks the
system blocks, and it would reject an Anthropic-style ``cache_control`` key
inside a Converse content block.

**Provider-gated, deliberately.** ``ChatOpenAI`` would forward the unknown kwarg
into the request body (a 400), and OpenAI-compatible servers cache automatically
anyway; Ollama has no such concept. So only the three transports above opt in, and
only for model families that support caching. A prompt below the model's own
minimum (1024 tokens on most Claude models) is silently not cached rather than
rejected, which is why there is no client-side size gate — the system prompt alone
clears it in every real configuration.

Pure logic: takes the ``MODEL_ID`` mapping, returns a policy. No config import, so
it is unit-testable and can't fail at import time.
"""

from typing import Any, Dict, Mapping, NamedTuple, Optional

from langchain_core.messages import SystemMessage

# Provider TYPEs whose client library understands a `cache_control` kwarg.
CACHEABLE_TYPES = ("anthropic", "bedrock", "mantle")

# Mantle serves three protocols; only the anthropic one is a ChatAnthropic (the
# OpenAI-shaped ones would put the kwarg in the request body).
_ANTHROPIC_TRANSPORTS = ("anthropic",)

# Model families that support prompt caching. Substring-matched so Bedrock
# inference-profile prefixes (`us.`/`eu.`/`global.`) and provider prefixes are
# covered. A family that isn't here gets no marker, since Bedrock REJECTS a
# cachePoint for a model that can't cache.
_CACHEABLE_FAMILIES = ("claude", "anthropic.", "amazon.nova")

VALID_TTLS = ("5m", "1h")
DEFAULT_TTL = "5m"

# Model attributes holding the model id, in probe order (ChatBedrockConverse,
# ChatAnthropic, ChatOpenAI). NOT `name`, which is the Runnable's display label.
_NAME_ATTRS = ("model_id", "model", "model_name")


class CachePolicy(NamedTuple):
    """What to send for this model: the kwarg, and whether to mark the system prompt.

    ``control`` empty means caching is off — the single flag every caller checks.
    """

    control: Dict[str, str]
    mark_system: bool

    @property
    def enabled(self) -> bool:
        return bool(self.control)


OFF = CachePolicy({}, False)


def is_cacheable_model(name: str) -> bool:
    """Whether the model id names a family that supports prompt caching."""
    lowered = (name or "").lower()
    return any(family in lowered for family in _CACHEABLE_FAMILIES)


def ttl(model_id: Mapping[str, Any]) -> str:
    """Configured cache TTL, falling back to the 5-minute default.

    ``1h`` costs more to write, so an unrecognized value must land on the cheap
    default rather than the expensive one.
    """
    value = str((model_id or {}).get("PROMPT_CACHE_TTL", "") or "").strip().lower()
    return value if value in VALID_TTLS else DEFAULT_TTL


def _opted_out(value: Any) -> bool:
    """Whether ``PROMPT_CACHE`` is set to something meaning "off".

    Hand-edited YAML, so a quoted ``"false"`` must disable it too; an absent key
    (None) is the default-on case.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("false", "no", "off", "0")
    return not value


def policy(model_id: Optional[Mapping[str, Any]]) -> CachePolicy:
    """The cache policy for a ``MODEL_ID`` section (:data:`OFF` when it can't cache).

    On by default where it applies: the win is per-turn and needs no tuning, and
    an existing install must get it without editing ``config.yaml``.
    ``PROMPT_CACHE: false`` opts out; ``true`` cannot force it onto a provider or
    model family that would reject the marker.
    """
    section = model_id or {}
    if _opted_out(section.get("PROMPT_CACHE")):
        return OFF
    provider = str(section.get("TYPE", "") or "").strip().lower()
    if provider not in CACHEABLE_TYPES:
        return OFF
    protocol = str(section.get("API_PROTOCOL", "") or "").strip().lower()
    if provider == "mantle" and protocol not in _ANTHROPIC_TRANSPORTS:
        return OFF
    if not is_cacheable_model(str(section.get("NAME", "") or "")):
        return OFF
    return CachePolicy(
        control={"type": "ephemeral", "ttl": ttl(section)},
        # Bedrock's expansion already marks the system blocks; only the
        # Anthropic transports need the extra block-level breakpoint.
        mark_system=provider != "bedrock",
    )


def _model_name(model: Any) -> str:
    """The model id read off a chat model (or a binding wrapping one), or ``""``."""
    for attr in _NAME_ATTRS:
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def bind(model: Any, cache_policy: CachePolicy = OFF) -> Any:
    """``model`` with the cache kwarg attached, or unchanged when caching is off.

    Re-checks the family against the model's OWN id so a custom sub-agent type
    that overrides the model name (same provider, different family) isn't sent a
    marker it would reject; an id we can't read falls back to trusting the policy.
    Never raises — losing the marker costs money, losing the model breaks the turn.
    """
    if not cache_policy.enabled:
        return model
    name = _model_name(model)
    if name and not is_cacheable_model(name):
        return model
    try:
        return model.bind(cache_control=dict(cache_policy.control))
    except Exception:  # noqa: BLE001 — an exotic model without .bind stays uncached
        return model


def system_message(text: str, cache_policy: CachePolicy = OFF) -> SystemMessage:
    """The system prompt as a message, carrying a breakpoint where it belongs.

    Marked form is a single text block with ``cache_control`` — the shape the
    Messages API has always accepted. Callers build the system message through
    this so the breakpoint can't be forgotten on one of the paths that assembles
    a prompt (turn start, worker loop, post-compaction rebuild).
    """
    if not cache_policy.mark_system or not text:
        return SystemMessage(content=text)
    return SystemMessage(
        content=[
            {
                "type": "text",
                "text": text,
                "cache_control": dict(cache_policy.control),
            }
        ]
    )
