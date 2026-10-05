"""Tests for bounded, session-local chat context."""

import unittest

from src.memory import add_message, clear_memory, create_session_id, get_recent_context, initialize_session


class ConversationMemoryTests(unittest.TestCase):
    def test_session_initialization_preserves_existing_values(self):
        state = {"messages": [{"role": "user", "content": "hello"}], "session_id": "fixed"}
        initialize_session(state)
        self.assertEqual(state["session_id"], "fixed")
        self.assertEqual(len(state["messages"]), 1)
        self.assertIsNone(state["pending_review"])

    def test_context_keeps_recent_messages_in_order_without_ui_metadata(self):
        messages = []
        add_message(messages, "user", "Question", sources=[{"source": "faq.md"}])
        add_message(messages, "assistant", "Answer")
        self.assertEqual(get_recent_context(messages), [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ])

    def test_context_obeys_message_and_character_limits(self):
        messages = [{"role": "user", "content": f"turn-{index}"} for index in range(5)]
        recent = get_recent_context(messages, max_messages=2, max_characters=20)
        self.assertEqual(len(recent), 2)
        self.assertLessEqual(sum(len(item["content"]) for item in recent), 20)

    def test_new_chat_memory_is_empty_and_session_ids_are_distinct(self):
        self.assertEqual(clear_memory(), [])
        self.assertNotEqual(create_session_id(), create_session_id())


if __name__ == "__main__":
    unittest.main()
