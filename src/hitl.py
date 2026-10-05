"""Pure functions for applying explicit human review decisions."""

from typing import Any, Literal

from src.state import SupportState


HumanDecision = Literal["approve", "edit_approve", "reject_custom"]


class ReviewDecisionError(ValueError):
    """Raised when a review action is invalid or its response is blank."""


def apply_human_decision(
    state: SupportState,
    decision: HumanDecision,
    human_response: str = "",
) -> SupportState:
    """Resolve an escalated graph state with a human-selected action.

    The function never chooses an action itself. The Streamlit reviewer passes
    one of the three explicit actions after clicking its matching button.
    """
    if not state.get("needs_escalation"):
        raise ReviewDecisionError("This question does not need human review.")
    if state.get("escalation_status") != "pending_review":
        raise ReviewDecisionError("This review is no longer pending.")

    generated_answer = (
        state.get("generated_answer") or state.get("answer", "")
    ).strip()

    if decision == "approve":
        if not generated_answer:
            raise ReviewDecisionError("There is no generated answer to approve.")
        final_answer = generated_answer
        response = generated_answer
        decision_name = "approved"
    elif decision == "edit_approve":
        response = human_response.strip()
        if not response:
            raise ReviewDecisionError("Enter the edited answer before approving it.")
        final_answer = response
        decision_name = "edited_and_approved"
    elif decision == "reject_custom":
        response = human_response.strip()
        if not response:
            raise ReviewDecisionError("Enter a custom response before submitting it.")
        final_answer = response
        decision_name = "rejected_with_custom_response"
    else:
        raise ReviewDecisionError("Choose a valid human review action.")

    resolved_state: dict[str, Any] = dict(state)
    resolved_state.update(
        {
            "human_decision": decision_name,
            "human_response": response,
            "final_answer": final_answer,
            "escalation_status": "resolved",
        }
    )
    return resolved_state  # type: ignore[return-value]
