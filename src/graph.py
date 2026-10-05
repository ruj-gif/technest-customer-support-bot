"""LangGraph orchestration for TechNest customer support questions."""

import re
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from src.llm_service import (
    LLMConnectionError,
    LLMServiceError,
    MissingAPIKeyError,
    RateLimitError,
    generate_answer,
)
from src.retrieval import retrieve
from src.state import SupportState
from src.memory import create_session_id, get_recent_context
from src.database import save_conversation


INSUFFICIENT_ANSWER_MARKERS = (
    "couldn't find enough information",
    "could not find enough information",
    "not enough information",
    "insufficient information",
    "i don't know",
    "i do not know",
)

POLICY_EXCEPTION_PATTERNS = (
    r"\bmake an exception\b",
    r"\bpolicy exception\b",
    r"\bwaive\b",
    r"\boverride\b",
    r"\bspecial approval\b",
    r"\bmanager approval\b",
    r"\bspeak to (a )?(person|human|agent|representative)\b",
    r"\btalk to (a )?(person|human|agent|representative|someone)\b",
)

STOP_WORDS = {
    "about", "after", "also", "and", "are", "can", "could", "does", "for",
    "from", "have", "into", "just", "our", "please", "that", "the", "their",
    "them", "there", "this", "through", "was", "were", "what", "when", "where",
    "which", "while", "will", "with", "would", "you", "your",
}
FOLLOW_UP_REFERENCES = re.compile(
    r"\b(it|they|them|their|this|that|these|those|there|one|ones)\b|"
    r"\b(how much|what price|how many)\b",
    flags=re.IGNORECASE,
)


def _retrieval_query(question: str, history: list[dict[str, str]]) -> str:
    """Add recent context only when a question refers back to prior turns."""
    if not history or not FOLLOW_UP_REFERENCES.search(question):
        return question

    recent_text = " ".join(
        item.get("content", "") for item in history[-4:]
        if item.get("role") in {"user", "assistant"}
    )
    return f"{recent_text} {question}".strip()


def retrieve_node(state: SupportState) -> dict[str, Any]:
    """Retrieve the three most relevant knowledge chunks for the question."""
    question = state.get("question", "").strip()
    try:
        query = _retrieval_query(
            question, state.get("conversation_context", [])
        )
        chunks = retrieve(query, top_k=3) if question else []
        return {
            "retrieved_context": chunks,
            "sources": chunks,
            "retrieval_error": "",
        }
    except Exception:
        # Keep retrieval failures in state so the graph can route them safely.
        return {
            "retrieved_context": [],
            "sources": [],
            "retrieval_error": "The knowledge base could not be searched.",
        }


def prepare_context_node(state: SupportState) -> dict[str, Any]:
    """Bound recent chat history before it is passed to the answer service."""
    return {
        "conversation_context": get_recent_context(
            state.get("conversation_history", [])
        )
    }


def generate_answer_node(state: SupportState) -> dict[str, Any]:
    """Ask the existing Groq service to answer using the retrieved chunks."""
    question = state.get("question", "")
    context = state.get("retrieved_context", [])

    try:
        answer = generate_answer(
            question, context, conversation_history=state.get("conversation_context", [])
        )
        return {
            "answer": answer,
            "generated_answer": answer,
            "generation_error": "",
        }
    except MissingAPIKeyError:
        return {
            "answer": "The TechNest AI assistant is not configured yet. Please contact our support team.",
            "generated_answer": "The TechNest AI assistant is not configured yet. Please contact our support team.",
            "generation_error": "The Groq API key is not configured.",
        }
    except RateLimitError:
        return {
            "answer": "TechNest support is busy right now. Please try again shortly or contact our support team.",
            "generated_answer": "TechNest support is busy right now. Please try again shortly or contact our support team.",
            "generation_error": "The Groq service rate limit was reached.",
        }
    except LLMConnectionError:
        return {
            "answer": "TechNest support could not connect to the answer service. Please try again shortly.",
            "generated_answer": "TechNest support could not connect to the answer service. Please try again shortly.",
            "generation_error": "The Groq service could not be reached.",
        }
    except LLMServiceError:
        return {
            "answer": "The TechNest answer service is temporarily unavailable. Please contact our support team.",
            "generated_answer": "The TechNest answer service is temporarily unavailable. Please contact our support team.",
            "generation_error": "The Groq answer service returned an error.",
        }
    except Exception:
        # Do not surface exception details, which could contain request data.
        return {
            "answer": "The TechNest answer service encountered an unexpected error. Please contact our support team.",
            "generated_answer": "The TechNest answer service encountered an unexpected error. Please contact our support team.",
            "generation_error": "An unexpected answer service error occurred.",
        }


def _requires_human_decision(question: str) -> bool:
    """Catch requests that need an exception or a person to make a decision."""
    return any(
        re.search(pattern, question, flags=re.IGNORECASE)
        for pattern in POLICY_EXCEPTION_PATTERNS
    )


def _answer_is_supported(answer: str, context: list[dict[str, Any]]) -> bool:
    """Apply a small deterministic grounding check to the generated answer."""
    normalized_answer = answer.lower()
    if any(marker in normalized_answer for marker in INSUFFICIENT_ANSWER_MARKERS):
        return False

    context_text = " ".join(str(chunk.get("text", "")) for chunk in context).lower()

    # Any number in an answer should also appear in the supplied context. This
    # helps catch invented prices, phone numbers, delivery times, or dates.
    def numbers(text: str) -> set[str]:
        return {value.replace(",", "") for value in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}

    if not numbers(normalized_answer).issubset(numbers(context_text)):
        return False

    def terms(text: str) -> set[str]:
        words = re.findall(r"[a-z][a-z'-]+", text.lower())
        normalized = {word[:-1] if word.endswith("s") and len(word) > 4 else word for word in words}
        return {word for word in normalized if len(word) > 3 and word not in STOP_WORDS}

    answer_terms = terms(answer)
    if len(answer_terms) < 2:
        return False

    context_terms = terms(context_text)
    # Require a clear majority of the answer's factual terms to be represented
    # in source context. The LLM can still paraphrase ordinary connecting words.
    return len(answer_terms & context_terms) / len(answer_terms) >= 0.5


def evaluate_answer_node(state: SupportState) -> dict[str, Any]:
    """Decide whether the answer is supported or should go to escalation."""
    question = state.get("question", "")
    context = state.get("retrieved_context", [])
    answer = state.get("answer", "")

    if state.get("retrieval_error"):
        reason = state["retrieval_error"]
    elif state.get("generation_error"):
        reason = state["generation_error"]
    elif _requires_human_decision(question):
        reason = "The customer is asking for a policy exception or a human decision."
    elif not context:
        reason = "No relevant information was found in the TechNest knowledge base."
    elif not _answer_is_supported(answer, context):
        reason = "The answer could not be fully supported by the retrieved TechNest information."
    else:
        return {
            "evaluation": "The answer appears adequately supported by the retrieved context.",
            "needs_escalation": False,
            "escalation_reason": "",
        }

    return {
        "evaluation": f"Escalation required: {reason}",
        "needs_escalation": True,
        "escalation_reason": reason,
    }


def escalation_node(state: SupportState) -> dict[str, Any]:
    """Keep the answer pending until a human explicitly reviews it."""
    return {
        "generated_answer": state.get("generated_answer", state.get("answer", "")),
        "escalation_status": "pending_review",
        "human_decision": "",
        "human_response": "",
        "final_answer": "",
    }


def save_conversation_node(state: SupportState) -> dict[str, Any]:
    """Persist this turn without saving any API secrets or internal memory."""
    pending = state.get("needs_escalation", False)
    conversation_id = save_conversation(
        session_id=state.get("session_id", ""),
        user_question=state.get("question", ""),
        # An escalated draft is not yet a customer-facing response.
        assistant_response="" if pending else state.get("answer", ""),
        retrieved_sources=state.get("sources", []),
        escalation_status=state.get("escalation_status", "not_required"),
        escalation_reason=state.get("escalation_reason", ""),
    )
    return {
        "conversation_id": conversation_id or 0,
        "database_saved": conversation_id is not None,
        "database_error": "" if conversation_id is not None else "Conversation history could not be saved.",
    }


def route_after_evaluation(state: SupportState) -> Literal["escalation", "__end__"]:
    """Choose the final-answer path or the escalation node."""
    if state.get("needs_escalation", False):
        return "escalation"
    return END


def _build_graph():
    """Connect the support steps and compile an executable StateGraph."""
    workflow = StateGraph(SupportState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("prepare_context", prepare_context_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("evaluate_answer", evaluate_answer_node)
    workflow.add_node("escalation", escalation_node)
    workflow.add_node("save_conversation", save_conversation_node)

    workflow.add_edge(START, "prepare_context")
    workflow.add_edge("prepare_context", "retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", "evaluate_answer")
    workflow.add_conditional_edges(
        "evaluate_answer",
        route_after_evaluation,
        {"escalation": "escalation", END: "save_conversation"},
    )
    workflow.add_edge("escalation", "save_conversation")
    workflow.add_edge("save_conversation", END)
    return workflow.compile()


support_graph = _build_graph()


def run_support_graph(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
    session_id: str | None = None,
) -> SupportState:
    """Run the complete TechNest retrieval, answer, and evaluation workflow."""
    initial_state: SupportState = {
        "question": question,
        "conversation_history": conversation_history or [],
        "session_id": session_id or create_session_id(),
    }
    return support_graph.invoke(initial_state)
