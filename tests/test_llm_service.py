"""Mocked tests for prompt construction and safe Groq service behavior."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.llm_service import (
    GROQ_MODEL,
    INSUFFICIENT_CONTEXT_RESPONSE,
    LLMConnectionError,
    LLMServiceError,
    MissingAPIKeyError,
    RateLimitError,
    build_messages,
    generate_answer,
)


SAMPLE_CHUNKS = [
    {
        "source": "products.csv",
        "text": "Product Name: TechNest Compact Keyboard; Price Inr: 1790",
        "score": 0.82,
    }
]


class FakeClient:
    """A tiny fake with the same method shape as the Groq SDK client."""

    def __init__(self, content: str = "The keyboard costs INR 1,790.", error=None):
        self.content = content
        self.error = error
        self.call_arguments = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create_completion)
        )

    def create_completion(self, **kwargs):
        self.call_arguments = kwargs
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class LLMServiceTests(unittest.TestCase):
    def test_messages_include_grounding_rules_question_and_sources(self) -> None:
        messages = build_messages("What is the keyboard price?", SAMPLE_CHUNKS)

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("using only the supplied TechNest context", messages[0]["content"])
        self.assertIn("Never invent", messages[0]["content"])
        self.assertIn("What is the keyboard price?", messages[1]["content"])
        self.assertIn("products.csv", messages[1]["content"])
        self.assertIn("Price Inr: 1790", messages[1]["content"])

    def test_empty_context_returns_safe_insufficient_information_message(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = generate_answer("What is the price?", [], client=FakeClient())

        self.assertEqual(result, INSUFFICIENT_CONTEXT_RESPONSE)

    def test_generate_answer_uses_configured_model_and_messages(self) -> None:
        client = FakeClient()
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            result = generate_answer("What is the keyboard price?", SAMPLE_CHUNKS, client=client)

        self.assertEqual(result, "The keyboard costs INR 1,790.")
        self.assertEqual(client.call_arguments["model"], GROQ_MODEL)
        self.assertEqual(client.call_arguments["messages"][1]["role"], "user")

    def test_messages_include_bounded_conversation_before_current_question(self) -> None:
        history = [
            {"role": "user", "content": "I am looking for a keyboard."},
            {"role": "assistant", "content": "The compact keyboard is available."},
        ]
        messages = build_messages("What was the model?", SAMPLE_CHUNKS, history)

        self.assertEqual(messages[1:], [
            *history,
            messages[-1],
        ])
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("What was the model?", messages[-1]["content"])

    def test_missing_api_key_raises_safe_configuration_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(MissingAPIKeyError) as context:
                generate_answer("Question", SAMPLE_CHUNKS, client=FakeClient())

        self.assertNotIn("test-key", str(context.exception))

    def test_rate_limit_error_is_translated_without_raw_error_details(self) -> None:
        client = FakeClient(error=type("RateLimitError", (Exception,), {})("secret detail"))
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with self.assertRaises(RateLimitError) as context:
                generate_answer("Question", SAMPLE_CHUNKS, client=client)

        self.assertNotIn("secret detail", str(context.exception))

    def test_network_error_is_translated_without_raw_error_details(self) -> None:
        client = FakeClient(
            error=type("APIConnectionError", (Exception,), {})("secret detail")
        )
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with self.assertRaises(LLMConnectionError) as context:
                generate_answer("Question", SAMPLE_CHUNKS, client=client)

        self.assertNotIn("secret detail", str(context.exception))

    def test_other_api_error_is_translated_without_raw_error_details(self) -> None:
        client = FakeClient(error=RuntimeError("private API response"))
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with self.assertRaises(LLMServiceError) as context:
                generate_answer("Question", SAMPLE_CHUNKS, client=client)

        self.assertNotIn("private API response", str(context.exception))


if __name__ == "__main__":
    unittest.main()
