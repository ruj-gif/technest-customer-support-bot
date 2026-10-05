"""Typed state shared by the TechNest LangGraph workflow nodes."""

from typing import Any, TypedDict


class SupportState(TypedDict, total=False):
    """Values passed between retrieval, answer, evaluation, and escalation."""

    # The only value the caller must provide to start the graph.
    question: str

    # Retrieved chunks retain their original source and similarity metadata.
    retrieved_context: list[dict[str, Any]]
    sources: list[dict[str, Any]]

    # Answer and evaluation outputs are filled by later workflow nodes.
    answer: str
    generated_answer: str
    final_answer: str
    evaluation: str
    needs_escalation: bool
    escalation_reason: str

    # Human review is explicit and only used when escalation is required.
    human_decision: str
    human_response: str
    escalation_status: str

    # Safe internal markers let evaluation route service failures for review.
    retrieval_error: str
    generation_error: str

    # Bounded conversational context and SQLite persistence metadata.
    conversation_history: list[dict[str, str]]
    conversation_context: list[dict[str, str]]
    session_id: str
    conversation_id: int
    database_saved: bool
    database_error: str
