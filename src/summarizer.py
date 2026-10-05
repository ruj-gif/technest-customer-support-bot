"""Deterministic analytics and weekly reports from SQLite conversation rows.

All counts and recommendations are calculated locally. This module does not
call an LLM or read environment variables, so reports remain available offline.
"""

from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Any


TOPIC_CATEGORIES = (
    "Products",
    "Returns / Refunds",
    "Shipping",
    "Warranty",
    "Store Information",
    "Other",
)
MINIMUM_TREND_CONVERSATIONS = 3


def classify_topic(question: str) -> str:
    """Assign one lightweight category using words in the customer's question."""
    text = question.casefold()
    rules = (
        ("Returns / Refunds", ("return", "refund", "exchange", "damaged", "defective")),
        ("Shipping", ("shipping", "delivery", "deliver", "track", "tracking", "shipment")),
        ("Warranty", ("warranty", "guarantee", "repair")),
        ("Store Information", ("store", "hours", "open", "close", "address", "location", "contact", "phone", "email")),
        ("Products", ("product", "price", "cost", "headphone", "keyboard", "laptop", "mouse", "monitor", "charger", "speaker", "tablet", "camera", "stock", "available", "sell")),
    )
    for category, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return category
    return "Other"


def _is_escalated(row: dict[str, Any]) -> bool:
    """Treat pending and resolved HITL states as escalated conversations."""
    status = str(row.get("escalation_status", "not_required")).strip().casefold()
    return status not in {"", "not_required", "none", "false", "0"}


def _conversation_date(row: dict[str, Any]) -> date | None:
    """Parse the stored ISO timestamp into its calendar date."""
    try:
        timestamp = datetime.fromisoformat(str(row.get("timestamp", "")))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.date()
    except (TypeError, ValueError):
        return None


def _metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    escalated = sum(_is_escalated(row) for row in rows)
    topic_counts = Counter(classify_topic(str(row.get("user_question", ""))) for row in rows)
    reason_counts = Counter(
        str(row.get("escalation_reason") or "Unspecified").strip() or "Unspecified"
        for row in rows
        if _is_escalated(row)
    )
    return {
        "total_conversations": total,
        "escalated_conversations": escalated,
        "non_escalated_conversations": total - escalated,
        "escalation_rate": (escalated / total * 100) if total else 0.0,
        "topic_counts": {topic: topic_counts.get(topic, 0) for topic in TOPIC_CATEGORIES},
        "escalation_reasons": dict(reason_counts.most_common()),
    }


def build_dashboard_analytics(
    rows: list[dict[str, Any]], today: date | None = None
) -> dict[str, Any]:
    """Build all-time overview plus a seven-calendar-day conversation trend."""
    end_day = today or date.today()
    start_day = end_day - timedelta(days=6)
    metrics = _metrics_for_rows(rows)
    daily_rows = [
        row for row in rows
        if (row_date := _conversation_date(row)) is not None
        and start_day <= row_date <= end_day
    ]
    daily_counts = {start_day + timedelta(days=offset): {"conversations": 0, "escalations": 0}
                    for offset in range(7)}
    for row in daily_rows:
        row_day = _conversation_date(row)
        if row_day in daily_counts:
            daily_counts[row_day]["conversations"] += 1
            daily_counts[row_day]["escalations"] += int(_is_escalated(row))

    metrics["daily_trends"] = [
        {
            "date": day.isoformat(),
            "conversations": values["conversations"],
            "escalations": values["escalations"],
        }
        for day, values in daily_counts.items()
    ]
    metrics["trend_period"] = f"{start_day.isoformat()} to {end_day.isoformat()}"
    return metrics


def build_weekly_summary(
    rows: list[dict[str, Any]], today: date | None = None
) -> dict[str, Any]:
    """Calculate a last-seven-days summary from stored conversation rows."""
    end_day = today or date.today()
    start_day = end_day - timedelta(days=6)
    week_rows = [
        row for row in rows
        if (row_date := _conversation_date(row)) is not None
        and start_day <= row_date <= end_day
    ]
    summary = _metrics_for_rows(week_rows)
    insufficient = summary["total_conversations"] < MINIMUM_TREND_CONVERSATIONS

    if insufficient:
        observations = [
            "Insufficient conversation data to identify a reliable weekly trend."
        ]
        recommendations = [
            "Continue collecting support conversations before drawing conclusions about recurring issues."
        ]
    else:
        observations = _observations(summary)
        recommendations = _recommendations(summary)

    summary.update(
        {
            "reporting_start": start_day.isoformat(),
            "reporting_end": end_day.isoformat(),
            "reporting_period": f"{start_day.isoformat()} to {end_day.isoformat()}",
            "insufficient_data": insufficient,
            "observations": observations,
            "recommendations": recommendations,
            "top_topics": [
                {"topic": topic, "count": count}
                for topic, count in sorted(
                    summary["topic_counts"].items(),
                    key=lambda item: (-item[1], item[0]),
                )
                if count
            ],
        }
    )
    summary["report"] = format_weekly_report(summary)
    return summary


def _observations(summary: dict[str, Any]) -> list[str]:
    """Describe only patterns directly visible in computed counts."""
    observations: list[str] = []
    topics = summary["topic_counts"]
    top_topic = max(topics, key=topics.get) if topics else "Other"
    if topics.get(top_topic, 0):
        observations.append(
            f"{top_topic} was the most common topic ({topics[top_topic]} conversations)."
        )
    if summary["escalated_conversations"]:
        top_reason, count = next(iter(summary["escalation_reasons"].items()))
        observations.append(f"Most frequent escalation reason: {top_reason} ({count}).")
    else:
        observations.append("No conversations were escalated during this reporting period.")
    return observations


def _recommendations(summary: dict[str, Any]) -> list[str]:
    """Offer simple actions tied to high observed topic/escalation counts."""
    recommendations: list[str] = []
    topics = summary["topic_counts"]
    top_topic = max(topics, key=topics.get) if topics else "Other"
    if topics.get(top_topic, 0):
        recommendations.append(
            f"Review the TechNest knowledge-base content for {top_topic.lower()} questions."
        )
    if summary["escalated_conversations"]:
        recommendations.append(
            "Review the listed escalation reasons to see whether clearer support guidance could resolve common cases."
        )
    if summary["escalation_rate"] >= 25:
        recommendations.append(
            "Check whether the escalation rate can be reduced by improving answers for the observed support gaps."
        )
    return recommendations or ["Continue monitoring support topics and escalation patterns."]


def format_weekly_report(summary: dict[str, Any]) -> str:
    """Format an aggregate-only Markdown report suitable for download."""
    top_topics = summary.get("top_topics") or [
        {"topic": topic, "count": count}
        for topic, count in summary["topic_counts"].items()
    ]
    topic_lines = [
        f"- {item['topic']}: {item['count']}"
        for item in top_topics
    ]
    if summary["escalation_reasons"]:
        reason_lines = [f"- {reason}: {count}" for reason, count in summary["escalation_reasons"].items()]
    else:
        reason_lines = ["- No escalations recorded"]
    observations = "\n".join(f"- {item}" for item in summary["observations"])
    recommendations = "\n".join(f"- {item}" for item in summary["recommendations"])
    return "\n".join(
        [
            "# TechNest Weekly Support Summary",
            f"\n## Reporting period\n{summary['reporting_period']}",
            "\n## Key metrics",
            f"- Total conversations: {summary['total_conversations']}",
            f"- Escalated conversations: {summary['escalated_conversations']}",
            f"- Non-escalated conversations: {summary['non_escalated_conversations']}",
            f"- Escalation rate: {summary['escalation_rate']:.1f}%",
            "\n## Topic breakdown",
            *topic_lines,
            "\n## Escalation breakdown",
            *reason_lines,
            "\n## Observations",
            observations,
            "\n## Recommendations",
            recommendations,
        ]
    )
