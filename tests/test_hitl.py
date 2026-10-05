"""Tests for explicit human approval and response editing."""

import unittest

from src.hitl import ReviewDecisionError, apply_human_decision


def pending_review_state() -> dict:
    """Return a minimal state as if LangGraph requested human review."""
    return {
        "question": "Can you make an exception to the return policy?",
        "answer": "Returns are accepted within 7 days.",
        "generated_answer": "Returns are accepted within 7 days.",
        "sources": [{"source": "return_policy.md", "text": "Returns: 7 days."}],
        "evaluation": "Escalation required: policy exception.",
        "needs_escalation": True,
        "escalation_reason": "The customer is asking for a policy exception.",
        "escalation_status": "pending_review",
        "human_decision": "",
        "human_response": "",
        "final_answer": "",
    }


class HumanReviewTests(unittest.TestCase):
    def test_normal_question_does_not_require_human_review(self) -> None:
        normal_state = pending_review_state()
        normal_state["needs_escalation"] = False
        normal_state["escalation_status"] = "not_required"

        with self.assertRaises(ReviewDecisionError):
            apply_human_decision(normal_state, "approve")

    def test_escalated_question_starts_in_pending_review_state(self) -> None:
        state = pending_review_state()

        self.assertTrue(state["needs_escalation"])
        self.assertEqual(state["escalation_status"], "pending_review")

    def test_approve_uses_the_generated_answer_as_final(self) -> None:
        state = apply_human_decision(pending_review_state(), "approve")

        self.assertEqual(state["final_answer"], "Returns are accepted within 7 days.")
        self.assertEqual(state["human_response"], state["generated_answer"])
        self.assertEqual(state["human_decision"], "approved")
        self.assertEqual(state["escalation_status"], "resolved")

    def test_edit_and_approve_uses_the_edited_answer(self) -> None:
        edited = "I reviewed your request. Please contact the store for help."
        state = apply_human_decision(
            pending_review_state(), "edit_approve", f"  {edited}  "
        )

        self.assertEqual(state["final_answer"], edited)
        self.assertEqual(state["human_decision"], "edited_and_approved")

    def test_reject_and_custom_response_uses_the_human_written_answer(self) -> None:
        custom = "We are unable to make an exception, but can explain the policy."
        state = apply_human_decision(
            pending_review_state(), "reject_custom", custom
        )

        self.assertEqual(state["final_answer"], custom)
        self.assertEqual(state["human_decision"], "rejected_with_custom_response")

    def test_edit_and_reject_require_non_empty_human_text(self) -> None:
        with self.assertRaises(ReviewDecisionError):
            apply_human_decision(pending_review_state(), "edit_approve", "  ")
        with self.assertRaises(ReviewDecisionError):
            apply_human_decision(pending_review_state(), "reject_custom", "")


if __name__ == "__main__":
    unittest.main()
