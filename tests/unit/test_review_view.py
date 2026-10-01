"""Review chrome is compact, grey and separate from the chat model's answer."""

import re

import pytest
from prompt_toolkit.utils import get_cwidth

from mnemoai.client import review
from mnemoai.client.ui import review_view


def report(verdict="revise"):
    return {
        "verdict": verdict, "reviewer": "some-provider/long-model-name",
        "summary": "A long explanation. " * 50,
        "findings": [{"issue": "DETAIL-ONLY finding", "verification": "Run the required check",
                      "evidence_ids": ["file-1"]}],
        "coverage_gaps": ["Some evidence was truncated."],
    }


def visible(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.mark.parametrize("width", [20, 40, 80])
def test_automatic_feedback_is_at_most_two_grey_lines(width):
    original = report()
    detail = review.render(original)
    display = review_view.snapshot((1, 1), "complete", original, details=detail)
    output = review_view.compact(display, width)
    assert len(output.splitlines()) == 2
    assert all(line.startswith(review_view.GRAY) and line.endswith(review_view.RESET)
               for line in output.splitlines())
    assert all(get_cwidth(line) <= width for line in visible(output).splitlines())
    assert "DETAIL-ONLY" not in output
    assert "some-provider" not in output
    assert "DETAIL-ONLY" in display.details and "file-1" in display.details
    assert original == report(), "presentation must not change the verdict/evidence"


def test_wide_unicode_and_terminal_controls_do_not_break_compact_layout():
    display = review_view.snapshot(
        (1, 1), "complete",
        {**report(), "summary": "\x1b[2J" + "界" * 100},
    )
    output = review_view.compact(display, 30)
    assert "\x1b[2J" not in output
    assert all(get_cwidth(line) <= 30 for line in visible(output).splitlines())


@pytest.mark.parametrize("phase,label", [
    ("waiting", "waiting for chat model"),
    ("reviewing", "checking answer"),
    ("correcting", "chat model revising · round 2"),
])
def test_progress_is_not_a_verdict(phase, label):
    display = review_view.snapshot((1, 1), phase, revision=2)
    assert label in display.label
    assert "pass" not in display.label


@pytest.mark.parametrize("verdict,header", [
    ("pass", "Final answer · chat model"),
    ("revise", "Chat model answer · unresolved review findings"),
    ("inconclusive", "Chat model answer · review incomplete"),
])
def test_final_actor_answer_is_distinct_and_never_replaced_by_reviewer(verdict, header):
    output = review_view.final_answer("ACTOR-ANSWER", report(verdict))
    assert header in visible(output)
    assert "ACTOR-ANSWER" in output
    assert "DETAIL-ONLY" not in output and "A long explanation" not in output
    if verdict != "pass":
        assert "Final answer" not in output and "verified completion" in output


def test_reset_clears_the_clickable_snapshot():
    reviewer = review.Reviewer(enabled=True)
    reviewer.view = review_view.snapshot((1, 1), "complete", report())
    reviewer.reset()
    assert reviewer.view is None


def test_a_late_snapshot_cannot_reappear_after_context_reset():
    reviewer = review.Reviewer(enabled=True)
    old = review_view.snapshot((id(reviewer), reviewer._generation), "complete", report())
    reviewer.view = old
    assert review_view.current_display(reviewer) is old
    reviewer.reset()
    reviewer.view = old  # even a late UI-only publisher must stay out of the new task
    assert review_view.current_display(reviewer) is None
