"""Tests for database-driven dashboard analytics and weekly summaries."""

import os
import unittest
from datetime import date, timedelta

from src.summarizer import (
    build_dashboard_analytics,
    build_weekly_summary,
    classify_topic,
)


TODAY = date(2026, 10, 6)


def conversation(day_offset, question, status="not_required", reason=""):
    day = TODAY - timedelta(days=day_offset)
    return {
        "timestamp": f"{day.isoformat()}T12:00:00+00:00",
        "user_question": question,
        "escalation_status": status,
        "escalation_reason": reason,
        "assistant_response": "Example response",
    }


SAMPLE_ROWS = [
    conversation(0, "What is the price of the keyboard?"),
    conversation(1, "How do I return a damaged item?", "pending_review", "Policy exception requested"),
    conversation(2, "When will my shipment arrive?"),
    conversation(3, "Can I get a warranty repair?", "resolved_approved", "Unsupported answer"),
    conversation(12, "Where is your store located?"),
]


class DashboardAnalyticsTests(unittest.TestCase):
    def test_dashboard_totals_rate_topics_reasons_and_daily_trends(self):
        analytics = build_dashboard_analytics(SAMPLE_ROWS, today=TODAY)

        self.assertEqual(analytics["total_conversations"], 5)
        self.assertEqual(analytics["escalated_conversations"], 2)
        self.assertEqual(analytics["non_escalated_conversations"], 3)
        self.assertEqual(analytics["escalation_rate"], 40.0)
        self.assertEqual(analytics["topic_counts"]["Products"], 1)
        self.assertEqual(analytics["topic_counts"]["Returns / Refunds"], 1)
        self.assertEqual(analytics["topic_counts"]["Shipping"], 1)
        self.assertEqual(analytics["topic_counts"]["Warranty"], 1)
        self.assertEqual(analytics["escalation_reasons"]["Policy exception requested"], 1)
        self.assertEqual(len(analytics["daily_trends"]), 7)
        self.assertEqual(analytics["daily_trends"][-1]["conversations"], 1)

    def test_topic_classifier_uses_question_text(self):
        self.assertEqual(classify_topic("Please refund my order"), "Returns / Refunds")
        self.assertEqual(classify_topic("Can I track my delivery?"), "Shipping")
        self.assertEqual(classify_topic("Tell me about your hours"), "Store Information")
        self.assertEqual(classify_topic("Something unrelated"), "Other")


class WeeklySummaryTests(unittest.TestCase):
    def test_weekly_period_metrics_and_out_of_period_exclusion(self):
        summary = build_weekly_summary(SAMPLE_ROWS, today=TODAY)

        self.assertEqual(summary["reporting_start"], "2026-09-30")
        self.assertEqual(summary["reporting_end"], "2026-10-06")
        self.assertEqual(summary["reporting_period"], "2026-09-30 to 2026-10-06")
        self.assertEqual(summary["total_conversations"], 4)
        self.assertEqual(summary["escalated_conversations"], 2)
        self.assertEqual(summary["escalation_rate"], 50.0)
        self.assertFalse(summary["insufficient_data"])
        self.assertEqual(summary["top_topics"][0], {"topic": "Products", "count": 1})

    def test_empty_data_has_zero_metrics_and_explicit_insufficient_data(self):
        summary = build_weekly_summary([], today=TODAY)

        self.assertEqual(summary["total_conversations"], 0)
        self.assertEqual(summary["escalation_rate"], 0.0)
        self.assertTrue(summary["insufficient_data"])
        self.assertIn("Insufficient conversation data", summary["observations"][0])

    def test_small_sample_explicitly_avoids_reliable_trend_claim(self):
        summary = build_weekly_summary(SAMPLE_ROWS[:2], today=TODAY)

        self.assertTrue(summary["insufficient_data"])
        self.assertEqual(
            summary["observations"],
            ["Insufficient conversation data to identify a reliable weekly trend."],
        )

    def test_export_contains_sections_and_never_includes_api_key(self):
        summary = build_weekly_summary(SAMPLE_ROWS, today=TODAY)
        report = summary["report"]
        required_sections = (
            "Reporting period", "Key metrics", "Topic breakdown",
            "Escalation breakdown", "Observations", "Recommendations",
        )
        for section in required_sections:
            self.assertIn(section, report)

        previous_key = os.environ.get("GROQ_API_KEY")
        try:
            os.environ["GROQ_API_KEY"] = "test-secret-must-not-appear"
            self.assertNotIn(os.environ["GROQ_API_KEY"], report)
        finally:
            if previous_key is None:
                os.environ.pop("GROQ_API_KEY", None)
            else:
                os.environ["GROQ_API_KEY"] = previous_key


if __name__ == "__main__":
    unittest.main()
