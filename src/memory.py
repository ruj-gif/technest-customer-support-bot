"""Small in-session conversation memory helpers for the TechNest chatbot."""

from collections.abc import MutableMapping, MutableSequence
from typing import Any
from uuid import uuid4


MAX_CONTEXT_MESSAGES = 8
MAX_CONTEXT_CHARACTERS = 4000
VALID_ROLES = {"user", "assistant"}


def create_session_id() -> str:
    """Create a random ID for one chat; no personal data is embedded in it."""
    return uuid4().hex


def initialize_session(session_state: MutableMapping[str, Any]) -> None:
    """Set up chat memory and a session ID without replacing existing values."""
    session_state.setdefault("messages", [])
    session_state.setdefault("session_id", create_session_id())
    session_state.setdefault("pending_review", None)


def add_message(
    messages: MutableSequence[dict[str, Any]],
    role: str,
    content: str,
    **metadata: Any,
) -> MutableSequence[dict[str, Any]]:
    """Append one customer-visible message, with optional UI metadata."""
    if role not in VALID_ROLES:
        raise ValueError("Message role must be 'user' or 'assistant'.")
    if not isinstance(content, str):
        raise TypeError("Message content must be text.")

    messages.append({"role": role, "content": content, **metadata})
    return messages


def get_recent_context(
    messages: list[dict[str, Any]],
    max_messages: int = MAX_CONTEXT_MESSAGES,
    max_characters: int = MAX_CONTEXT_CHARACTERS,
) -> list[dict[str, str]]:
    """Return recent role/content turns within both message and size limits.

    Messages are selected from newest to oldest, then returned in their
    original order. Source widgets and other UI metadata are not sent to Groq.
    """
    if max_messages <= 0 or max_characters <= 0:
        return []

    valid_messages = [
        message
        for message in messages
        if message.get("role") in VALID_ROLES
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ][-max_messages:]

    selected: list[dict[str, str]] = []
    characters_left = max_characters
    for message in reversed(valid_messages):
        content = message["content"].strip()
        if characters_left <= 0:
            break
        if len(content) > characters_left:
            # Keep the most recent end of an unusually long message.
            content = "…" + content[-max(0, characters_left - 1) :]
        selected.append({"role": message["role"], "content": content})
        characters_left -= len(content)

    return list(reversed(selected))


def clear_memory() -> list[dict[str, Any]]:
    """Return a fresh empty message list for the New Chat action."""
    return []
