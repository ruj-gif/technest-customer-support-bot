"""A small Streamlit customer support chat demo for TechNest."""

import streamlit as st
from src.database import get_all_conversations, initialize_database, update_conversation
from src.graph import run_support_graph
from src.hitl import ReviewDecisionError, apply_human_decision
from src.memory import add_message, clear_memory, create_session_id, get_recent_context, initialize_session
from src.summarizer import build_dashboard_analytics, build_weekly_summary


STORE_NAME = "TechNest"


def start_new_chat() -> None:
    """Clear the current conversation and return to the welcome screen."""
    st.session_state.messages = clear_memory()
    st.session_state.session_id = create_session_id()
    st.session_state.pending_review = None


def show_sidebar() -> str:
    """Display store details and a simple app-section navigation control."""
    with st.sidebar:
        st.title(f"🛍️ {STORE_NAME}")
        st.caption("Your neighborhood electronics store")
        st.divider()

        st.subheader("Store information")
        st.write("📍 24 Market Street, Springfield")
        st.write("🕒 Mon–Sat, 9:00 AM–7:00 PM")
        st.write("☎️ (555) 010-2020")
        st.write("✉️ hello@technest.example")

        st.divider()
        selected_page = st.radio(
            "Navigate",
            ["Customer Support", "Support Dashboard"],
            label_visibility="collapsed",
        )
        if selected_page == "Customer Support":
            st.button(
                "＋  New chat",
                width="stretch",
                on_click=start_new_chat,
            )
    return selected_page


def show_customer_support() -> None:
    """Render the existing customer chat and simulated review interface."""
    st.title("Customer Support")
    if not st.session_state.messages:
        show_welcome()

    show_messages()
    if st.session_state.pending_review:
        show_review_panel()
    else:
        handle_chat_input()


def show_support_dashboard() -> None:
    """Display aggregate support analytics and a downloadable weekly report."""
    st.title("Support Dashboard")
    st.caption("Analytics are calculated from TechNest's local SQLite conversation records.")
    rows = get_all_conversations()
    analytics = build_dashboard_analytics(rows)
    weekly = build_weekly_summary(rows)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Total conversations", analytics["total_conversations"])
    metric_columns[1].metric("Escalated", analytics["escalated_conversations"])
    metric_columns[2].metric("Not escalated", analytics["non_escalated_conversations"])
    metric_columns[3].metric("Escalation rate", f"{analytics['escalation_rate']:.1f}%")

    st.subheader("Conversation trends")
    st.caption(f"Daily counts for {analytics['trend_period']}")
    trend = analytics["daily_trends"]
    st.bar_chart(
        {
            "date": [item["date"] for item in trend],
            "conversations": [item["conversations"] for item in trend],
            "escalations": [item["escalations"] for item in trend],
        },
        x="date",
        y=["conversations", "escalations"],
    )

    topic_column, reason_column = st.columns(2)
    with topic_column:
        st.subheader("Customer query topics")
        topic_rows = [
            {"Topic": topic, "Conversations": count}
            for topic, count in analytics["topic_counts"].items()
        ]
        st.dataframe(topic_rows, hide_index=True)
        st.bar_chart(
            {
                "topic": [row["Topic"] for row in topic_rows],
                "conversations": [row["Conversations"] for row in topic_rows],
            },
            x="topic",
            y="conversations",
        )

    with reason_column:
        st.subheader("Common escalation reasons")
        reasons = analytics["escalation_reasons"]
        if reasons:
            reason_rows = [
                {"Reason": reason, "Cases": count}
                for reason, count in reasons.items()
            ]
            st.dataframe(reason_rows, hide_index=True)
            st.bar_chart(
                {
                    "reason": [row["Reason"] for row in reason_rows],
                    "cases": [row["Cases"] for row in reason_rows],
                },
                x="reason",
                y="cases",
            )
        else:
            st.info("No escalations are recorded yet.")

    st.divider()
    st.subheader("Weekly support summary")
    st.markdown(weekly["report"])
    st.download_button(
        "Export Weekly Summary",
        data=weekly["report"],
        file_name=f"technest-weekly-summary-{weekly['reporting_start']}.md",
        mime="text/markdown",
    )


def show_welcome() -> None:
    """Show the introductory content and example questions."""
    st.markdown("## 👋 Welcome to TechNest support")
    st.write(
        "Hi! I’m the TechNest support assistant. Ask me about our products, "
        "store hours, or how to get in touch with our team."
    )
    st.markdown("**Here are a few things you can ask:**")

    examples = [
        "What are your store hours?",
        "Do you offer a warranty on laptops?",
        "Where is your store located?",
    ]
    columns = st.columns(len(examples))
    for column, question in zip(columns, examples):
        with column:
            st.info(question)


def show_messages() -> None:
    """Render the conversation stored for this browser session."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Assistant messages can include retrieved knowledge passages.
            sources = message.get("sources", [])
            if sources:
                with st.expander("Sources used"):
                    for number, source in enumerate(sources, start=1):
                        score = source.get("score")
                        score_text = (
                            f" · Similarity: {score:.3f}"
                            if isinstance(score, (int, float))
                            else ""
                        )
                        st.markdown(
                            f"**{number}. {source.get('source', 'Unknown source')}"
                            f"{score_text}**"
                        )
                        st.write(source.get("text", ""))


def get_assistant_response(
    prompt: str,
    conversation_history: list[dict[str, str]] | None = None,
    session_id: str | None = None,
) -> dict:
    """Run the LangGraph workflow and return its complete result state."""
    try:
        return run_support_graph(prompt, conversation_history, session_id)
    except Exception:
        # Graph failures should not take down the Streamlit conversation.
        return {
            "question": prompt,
            "answer": (
                "We're having trouble preparing an answer right now. "
                "Please try again or contact our support team."
            ),
            "sources": [],
            "needs_escalation": False,
        }


def finish_human_review(decision: str, human_response: str = "") -> None:
    """Apply a reviewer choice and publish only the approved final response."""
    review_state = st.session_state.pending_review
    try:
        resolved_state = apply_human_decision(
            review_state,
            decision,  # type: ignore[arg-type]
            human_response,
        )
    except ReviewDecisionError as error:
        st.error(str(error))
        return

    # Customers see a professional confirmation and final reply, not the
    # evaluator's internal notes or the escalation reason.
    customer_message = (
        "Your request has been reviewed by our support team.\n\n"
        f"{resolved_state['final_answer']}"
    )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": customer_message,
            "sources": resolved_state.get("sources", []),
        }
    )
    conversation_id = review_state.get("conversation_id")
    if conversation_id:
        saved = update_conversation(
            conversation_id=conversation_id,
            session_id=st.session_state.session_id,
            assistant_response=customer_message,
            escalation_status=f"resolved_{resolved_state.get('human_decision', 'reviewed')}",
            escalation_reason=review_state.get("escalation_reason", ""),
        )
        if not saved:
            st.session_state.database_warning = True
    st.session_state.pending_review = None
    st.rerun()


def show_review_panel() -> None:
    """Show a local, simulated support-agent review for pending escalations."""
    review_state = st.session_state.pending_review
    if not review_state:
        return

    with st.container(border=True):
        st.error("Human Support Review Required")
        st.caption("Simulated support-agent review for this project demo")

        st.markdown("**Customer question**")
        st.write(review_state.get("question", ""))

        st.markdown("**Escalation reason**")
        st.write(review_state.get("escalation_reason", "Additional review was requested."))

        st.markdown("**Current evaluation**")
        st.write(review_state.get("evaluation", "No evaluation details are available."))

        st.markdown("**Retrieved sources and context**")
        sources = review_state.get("sources", [])
        if sources:
            for number, source in enumerate(sources, start=1):
                score = source.get("score")
                score_text = (
                    f" · Similarity: {score:.3f}"
                    if isinstance(score, (int, float))
                    else ""
                )
                with st.expander(
                    f"{number}. {source.get('source', 'Unknown source')}{score_text}"
                ):
                    st.write(source.get("text", ""))
        else:
            st.write("No source chunks were retrieved.")

        generated_answer = review_state.get("generated_answer", "").strip()
        st.markdown("**Generated answer**")
        if generated_answer:
            st.write(generated_answer)
        else:
            st.caption("No generated answer is available for approval.")

        # Each submit button represents an explicit human choice. Text fields
        # are saved in this Streamlit session only; there is no database.
        with st.form("human_support_review"):
            edited_answer = st.text_area(
                "Edit & approve response",
                key="human_review_edited_answer",
                help="Review the generated response and make any needed changes.",
            )
            custom_answer = st.text_area(
                "Reject & write a new response",
                key="human_review_custom_answer",
                help="Use this when the generated response should be replaced completely.",
            )

            approve_column, edit_column, reject_column = st.columns(3)
            approve_clicked = approve_column.form_submit_button("Approve answer")
            edit_clicked = edit_column.form_submit_button("Edit & approve")
            reject_clicked = reject_column.form_submit_button("Reject & write response")

        if approve_clicked:
            finish_human_review("approve")
        elif edit_clicked:
            finish_human_review("edit_approve", edited_answer)
        elif reject_clicked:
            finish_human_review("reject_custom", custom_answer)


def handle_chat_input() -> None:
    """Add the customer's question and a knowledge-base response."""
    prompt = st.chat_input("Ask TechNest a question...")
    if prompt:
        history = get_recent_context(st.session_state.messages)
        add_message(st.session_state.messages, "user", prompt)
        result = get_assistant_response(
            prompt, history, st.session_state.session_id
        )
        if result.get("database_saved") is False:
            st.session_state.database_warning = True
        if result.get("needs_escalation"):
            st.session_state.pending_review = result
            st.session_state.human_review_edited_answer = result.get(
                "generated_answer", result.get("answer", "")
            )
            st.session_state.human_review_custom_answer = ""
        else:
            answer = result.get("answer", "")
            if not isinstance(answer, str) or not answer.strip():
                answer = "Please contact our support team for help with this request."
            add_message(
                st.session_state.messages,
                "assistant",
                answer,
                sources=result.get("sources", []),
            )
        st.rerun()


def main() -> None:
    """Configure the page and assemble the chat interface."""
    st.set_page_config(
        page_title=f"{STORE_NAME} Support",
        page_icon="💬",
        layout="wide",
    )

    # Session state keeps the conversation while this browser session is open.
    initialize_session(st.session_state)
    # Persistence is local; graph/database helpers provide graceful fallbacks.
    if "database_initialized" not in st.session_state:
        st.session_state.database_initialized = initialize_database()
    if not st.session_state.database_initialized:
        st.session_state.database_warning = True

    selected_page = show_sidebar()
    if st.session_state.get("database_warning", False):
        st.info(
            "Chat is available, but local conversation history could not be saved."
        )

    if selected_page == "Customer Support":
        show_customer_support()
    else:
        show_support_dashboard()


if __name__ == "__main__":
    main()
