"""Run the chat model's corrections in the original turn and permission context."""

import threading
from contextlib import contextmanager

from mnemoai.client.agent import mid_turn
from mnemoai.utils.review_protocol import ReviewStopped


def budget_for(agent):
    return getattr(getattr(agent, "_supervision_local", None), "budget", None)


@contextmanager
def budget_scope(agent, budget):
    local = getattr(agent, "_supervision_local", None)
    if local is None:
        local = agent._supervision_local = threading.local()
    previous = getattr(local, "budget", None)
    local.budget = budget
    try:
        yield
    finally:
        local.budget = previous


def continue_work(agent, state, feedback, budget):
    """Use the normal foreground model/tool methods, without opening a new turn."""
    budget.check()
    state["messages"].append(feedback)
    previously_displayed = agent._answer_displayed
    agent._answer_displayed = False
    completed = False
    mid_turn.open_window(agent, reset=False)

    def apply(update):
        state["messages"].extend(update.get("messages", []))
        state.update({k: v for k, v in update.items() if k != "messages"})

    with budget_scope(agent, budget):
        try:
            while True:
                budget.take_step()
                apply(agent._call_model(state))
                budget.check()
                while True:
                    decision = agent._should_continue(state)
                    if decision == "end":
                        budget.check()
                        answer = agent._last_visible_from(state["messages"])
                        if not answer:
                            raise ReviewStopped("The chat model produced no correction or counterevidence")
                        agent._emit_answer(answer)
                        completed = True
                        return answer
                    budget.take_step()
                    if decision == "continue":
                        apply(agent._execute_tools(state))
                        break
                    apply(agent._deliver_mid_turn(state))
                    break
        finally:
            mid_turn.close(agent)
            if not completed and previously_displayed:
                agent._answer_displayed = True  # don't re-emit an old claim after an unresolved review
