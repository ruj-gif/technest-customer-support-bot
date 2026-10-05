"""Tests for the LangGraph workflow using stubbed retrieval and LLM calls."""

import importlib.util
import unittest
from unittest.mock import patch


LANGGRAPH_AVAILABLE = importlib.util.find_spec("langgraph") is not None

if LANGGRAPH_AVAILABLE:
    from src.graph import route_after_evaluation, run_support_graph


@unittest.skipUnless(LANGGRAPH_AVAILABLE, "LangGraph is not installed in this environment")
class SupportGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        # Graph tests should never write to the project's real SQLite file.
        self.database_patch = patch("src.graph.save_conversation", return_value=1)
        self.database_patch.start()

    def tearDown(self) -> None:
        self.database_patch.stop()

    def test_faq_question_reaches_final_answer(self) -> None:
        context = [
            {
                "source": "faq.md",
                "text": "Store hours: Monday through Saturday, 9:00 AM to 7:00 PM.",
                "score": 0.8,
            }
        ]
        answer = "The store is open Monday through Saturday, 9:00 AM to 7:00 PM."

        with patch("src.graph.retrieve", return_value=context), patch(
            "src.graph.generate_answer", return_value=answer
        ):
            result = run_support_graph("What are your store hours?")

        self.assertEqual(result["answer"], answer)
        self.assertFalse(result["needs_escalation"])
        self.assertEqual(result["sources"], context)

    def test_product_question_reaches_final_answer(self) -> None:
        context = [
            {
                "source": "products.csv",
                "text": "Product Name: TechNest Studio Headphones; Price Inr: 4990; Stock Status: Low stock",
                "score": 0.76,
            }
        ]
        answer = "TechNest Studio Headphones cost INR 4,990 and are low stock."

        with patch("src.graph.retrieve", return_value=context), patch(
            "src.graph.generate_answer", return_value=answer
        ):
            result = run_support_graph("What is the price of the headphones?")

        self.assertEqual(result["answer"], answer)
        self.assertFalse(result["needs_escalation"])
        self.assertEqual(result["sources"], context)

    def test_insufficient_context_triggers_escalation(self) -> None:
        with patch("src.graph.retrieve", return_value=[]), patch(
            "src.graph.generate_answer",
            return_value="I couldn't find enough information in the TechNest knowledge base to answer this. Please contact our support team.",
        ):
            result = run_support_graph("Can you repair my television?")

        self.assertTrue(result["needs_escalation"])
        self.assertIn("No relevant information", result["escalation_reason"])
        self.assertEqual(result["escalation_status"], "pending_review")
        self.assertEqual(result["generated_answer"], result["answer"])

    def test_conditional_router_selects_expected_paths(self) -> None:
        self.assertEqual(
            route_after_evaluation({"needs_escalation": False}), "__end__"
        )
        self.assertEqual(
            route_after_evaluation({"needs_escalation": True}), "escalation"
        )

    def test_graph_result_contains_required_state_fields(self) -> None:
        context = [
            {
                "source": "faq.md",
                "text": "TechNest accepts major debit cards and credit cards.",
                "score": 0.7,
            }
        ]
        answer = "TechNest accepts major debit cards and credit cards."

        with patch("src.graph.retrieve", return_value=context), patch(
            "src.graph.generate_answer", return_value=answer
        ):
            result = run_support_graph("Which cards can I use?")

        expected_fields = {
            "question",
            "retrieved_context",
            "answer",
            "sources",
            "evaluation",
            "needs_escalation",
            "escalation_reason",
        }
        self.assertTrue(expected_fields.issubset(result))
        self.assertEqual(result["question"], "Which cards can I use?")
        self.assertEqual(result["retrieved_context"], context)
        self.assertTrue(result["evaluation"])

    def test_policy_exception_triggers_escalation(self) -> None:
        context = [
            {"source": "return_policy.md", "text": "Returns are accepted within 7 days."}
        ]

        with patch("src.graph.retrieve", return_value=context), patch(
            "src.graph.generate_answer", return_value="Returns are accepted within 7 days."
        ):
            result = run_support_graph("Can you make an exception to the return policy?")

        self.assertTrue(result["needs_escalation"])
        self.assertIn("policy exception", result["escalation_reason"])

    def test_recent_context_is_passed_to_existing_answer_service(self) -> None:
        context = [{"source": "faq.md", "text": "TechNest sells headphones."}]
        history = [
            {"role": "user", "content": "Do you sell headphones?"},
            {"role": "assistant", "content": "Yes, TechNest sells headphones."},
        ]
        with patch("src.graph.retrieve", return_value=context), patch(
            "src.graph.generate_answer", return_value="Yes, TechNest sells headphones."
        ) as answer_service:
            result = run_support_graph(
                "Do they have Bluetooth?", history, session_id="session-test"
            )

        self.assertFalse(result["needs_escalation"])
        self.assertEqual(answer_service.call_args.kwargs["conversation_history"], history)
        self.assertEqual(result["session_id"], "session-test")

    def test_follow_up_reference_adds_recent_context_to_retrieval_query(self) -> None:
        history = [
            {"role": "user", "content": "Do you have headphones?"},
            {"role": "assistant", "content": "We have Studio Headphones."},
        ]
        with patch("src.graph.retrieve", return_value=[]) as retrieval, patch(
            "src.graph.generate_answer",
            return_value="I couldn't find enough information in the TechNest knowledge base to answer this. Please contact our support team.",
        ):
            run_support_graph("How much are they?", history, "session-test")

        retrieval_query = retrieval.call_args.args[0]
        self.assertIn("Studio Headphones", retrieval_query)
        self.assertIn("How much are they?", retrieval_query)

    def test_unsupported_answer_triggers_escalation(self) -> None:
        context = [
            {
                "source": "faq.md",
                "text": "The store is open Monday through Saturday from 9:00 AM to 7:00 PM.",
            }
        ]
        unsupported_answer = "TechNest is open 24 hours every day with free delivery."

        with patch("src.graph.retrieve", return_value=context), patch(
            "src.graph.generate_answer", return_value=unsupported_answer
        ):
            result = run_support_graph("Can I get free 24-hour delivery?")

        self.assertTrue(result["needs_escalation"])
        self.assertIn("supported", result["escalation_reason"])


if __name__ == "__main__":
    unittest.main()
