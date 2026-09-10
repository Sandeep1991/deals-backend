from __future__ import annotations

from app.orchestrator.state import ChatTurn


MAX_HISTORY_TURNS = 12
MAX_TURN_CHARS = 600


def normalize_history(
    messages: list[dict] | list[ChatTurn] | None,
    *,
    current_query: str = "",
) -> list[ChatTurn]:
    """Cap and clean browser-sent chat history for prompts."""
    if not messages:
        return []

    turns: list[ChatTurn] = []
    for raw in messages:
        if isinstance(raw, ChatTurn):
            role = raw.role
            content = (raw.content or "").strip()
        elif isinstance(raw, dict):
            role = str(raw.get("role") or "").strip().lower()
            content = str(raw.get("content") or "").strip()
        else:
            continue
        if role not in {"user", "assistant", "system"} or not content:
            continue
        if len(content) > MAX_TURN_CHARS:
            content = content[: MAX_TURN_CHARS - 1] + "…"
        turns.append(ChatTurn(role=role, content=content))  # type: ignore[arg-type]

    # Drop a trailing duplicate of the current query (client may include it)
    if current_query and turns:
        last = turns[-1]
        if last.role == "user" and last.content.strip() == current_query.strip():
            turns = turns[:-1]

    return turns[-MAX_HISTORY_TURNS:]


def format_history_block(history: list[ChatTurn]) -> str:
    if not history:
        return ""
    lines = ["Prior conversation:"]
    for turn in history:
        label = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{label}: {turn.content}")
    return "\n".join(lines)
