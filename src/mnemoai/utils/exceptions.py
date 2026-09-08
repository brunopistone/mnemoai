"""What actually failed inside an exception group (pure logic).

``str(ExceptionGroup)`` is the WRAPPER's own text — ``unhandled errors in a
TaskGroup (1 sub-exception)`` — never the failure it carries. So every reader of
an exception sees the plumbing instead of the cause: the message on screen names
a TaskGroup, the retry classifier matches no phrasing (a dropped socket inside a
group reads as deterministic and is retried ZERO times), and the recovery advice
has nothing to go on. Observed as ``✗ MCP server 'aws' failed to start;
skipping. (unhandled errors in a TaskGroup (1 sub-exception))`` — a line that
tells the user only that the failure was wrapped.

**Unwrapping once is not enough:** a group's sub-exception is itself a group (the
MCP client stack nests anyio task groups), so the real failure sat two levels
down and ``.exceptions[0]`` still yielded the same opaque text. Hence the
recursion, and hence one shared definition of "what failed" rather than a copy
per call site — this is read by the screen text, the log record, the transient
retry policy and the turn-failure classifier alike.

Deliberately dependency-free (stdlib only, no logger, no config): the retry
policy that consumes it is itself pure, and a helper that answers "what went
wrong" must not be able to fail for a reason of its own.
"""

from typing import List

# A group can hold a group. Bounded because the recursion is driven by data we
# don't own — a pathological (or self-referential) nesting must not spin.
_MAX_UNWRAP_DEPTH = 8


def exception_leaves(exc: BaseException, _depth: int = 0) -> List[BaseException]:
    """The real failures inside ``exc``, flattened; ``[exc]`` when it isn't a group.

    Order is the group's own, so ``[0]`` is the first thing that went wrong.
    """
    if isinstance(exc, BaseExceptionGroup) and _depth < _MAX_UNWRAP_DEPTH:
        leaves: List[BaseException] = []
        for sub in exc.exceptions:
            leaves.extend(exception_leaves(sub, _depth + 1))
        if leaves:
            return leaves
    return [exc]


def leaf_exception(exc: BaseException) -> BaseException:
    """The first real failure inside ``exc`` — what to name, classify and report on."""
    return exception_leaves(exc)[0]


def exception_text(exc: BaseException) -> str:
    """The text to MATCH phrasings against: every leaf's message, joined.

    A drop-in for ``str(exc)`` (identical for anything that isn't a group), so a
    classifier keeps matching exactly what it did before and additionally sees
    through the wrapper.
    """
    return "; ".join(str(leaf) for leaf in exception_leaves(exc))


def exception_signature(exc: BaseException) -> str:
    """Every leaf as ``Type: message``, joined — for a classifier that also reads
    the exception's CLASS (a provider names a condition after it)."""
    return "; ".join(f"{type(leaf).__name__}: {leaf}" for leaf in exception_leaves(exc))
