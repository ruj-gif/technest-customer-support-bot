"""SQLite persistence tests using isolated temporary databases."""

import tempfile
import unittest
from pathlib import Path

from src.database import (
    clear_session,
    get_conversation_history,
    get_all_conversations,
    get_recent_conversations,
    initialize_database,
    save_conversation,
    update_conversation,
)


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialize_save_and_load_turn(self):
        self.assertTrue(initialize_database(self.db_path))
        row_id = save_conversation(
            "session-a", "Store hours?", "Open weekdays.",
            [{"source": "faq.md", "score": 0.8}], db_path=self.db_path,
        )
        rows = get_conversation_history("session-a", self.db_path)
        self.assertIsNotNone(row_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["assistant_response"], "Open weekdays.")
        self.assertEqual(rows[0]["retrieved_sources"][0]["source"], "faq.md")

    def test_pending_escalation_can_be_updated_after_human_review(self):
        row_id = save_conversation(
            "session-a", "Exception?", "", [], "pending_review", "Needs review",
            self.db_path,
        )
        self.assertTrue(update_conversation(
            row_id, "session-a", "Reviewed reply", "resolved_approved",
            "Needs review", self.db_path,
        ))
        row = get_conversation_history("session-a", self.db_path)[0]
        self.assertEqual(row["assistant_response"], "Reviewed reply")
        self.assertEqual(row["escalation_status"], "resolved_approved")

    def test_new_session_is_separate_and_new_chat_does_not_clear_history(self):
        save_conversation("old-session", "Old question", "Old answer", db_path=self.db_path)
        save_conversation("new-session", "New question", "New answer", db_path=self.db_path)
        self.assertEqual(len(get_conversation_history("old-session", self.db_path)), 1)
        self.assertEqual(len(get_conversation_history("new-session", self.db_path)), 1)
        self.assertEqual(len(get_recent_conversations(20, self.db_path)), 2)

    def test_get_all_conversations_returns_rows_for_analytics(self):
        save_conversation("one", "Question one", "Answer one", db_path=self.db_path)
        save_conversation("two", "Question two", "Answer two", db_path=self.db_path)
        self.assertEqual(len(get_all_conversations(self.db_path)), 2)

    def test_clear_session_only_removes_selected_session(self):
        save_conversation("one", "Q", "A", db_path=self.db_path)
        save_conversation("two", "Q", "A", db_path=self.db_path)
        self.assertTrue(clear_session("one", self.db_path))
        self.assertEqual(get_conversation_history("one", self.db_path), [])
        self.assertEqual(len(get_conversation_history("two", self.db_path)), 1)


if __name__ == "__main__":
    unittest.main()
