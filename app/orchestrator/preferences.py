"""LangChain-style rolling preference summary for a chat session.

Uses langchain_core chat messages + an LLM summarizer (ConversationSummaryMemory
pattern) instead of hardcoded dietary keyword lists. The browser stores the
summary per chat_id and sends it back on each turn.
"""

from __future__ import annotations

from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm_client import LLMNotConfiguredError, complete_json, is_decompose_configured
from app.orchestrator.state import ChatTurn, CompositeItem

PREFERENCE_SUMMARY_SYSTEM = """You maintain a rolling USER PREFERENCE SUMMARY for a shopping assistant
(LangChain ConversationSummary-style memory).

Given the prior summary (may be empty), recent chat messages, and the latest user message,
return JSON only:
{
  "summary": "2-4 sentence rolling summary of stable user preferences and active shopping intent",
  "preferences": ["organic", "vegan", "..."],
  "is_list_rewrite": true,
  "prior_grocery_items": ["bottled water", "trail mix"],
  "rewrite_guidance": "short instruction for rewriting the grocery list search terms"
}

Rules:
- preferences: free-form tags inferred from the user (diet, brand, budget, store, organic, etc.).
  Also capture household/trip composition when known as tags like
  "household:2 adults", "household:2 children ages 5-8", "household:1 dog".
  Do NOT invent preferences or household counts they did not imply. Empty list if none.
- summary: when the user defines family/group size, record adults, children (ages if given),
  and pets explicitly so later turns can justify assumptions instead of re-asking.
- is_list_rewrite: true when the latest message is refining/replacing an EXISTING grocery list
  (e.g. asking if items are organic, make it vegan/gluten-free, swap ingredients) rather than
  starting a brand-new unrelated shopping trip.
- prior_grocery_items: concrete products from the prior shopping list when is_list_rewrite,
  otherwise []. Never put the user's question sentence in this list.
- rewrite_guidance: how to adjust search_terms (empty string if not a rewrite).
- Merge with prior_summary: keep older prefs unless the user clearly changed them.
- summary must stay concise and useful for downstream grocery and trip planning agents.
"""


class PreferenceSummary(BaseModel):
    """Session preference memory (ConversationSummary-style buffer)."""

    summary: str = ""
    preferences: list[str] = Field(default_factory=list)
    is_list_rewrite: bool = False
    prior_grocery_items: list[str] = Field(default_factory=list)
    rewrite_guidance: str = ""

    def label(self) -> str:
        if self.preferences:
            return ", ".join(self.preferences[:6])
        if self.summary:
            return self.summary[:80]
        return "preferences"


def history_to_lc_messages(history: list[ChatTurn] | None) -> list[BaseMessage]:
    """Convert DealFinder turns into LangChain chat messages."""
    out: list[BaseMessage] = []
    for turn in history or []:
        content = (turn.content or "").strip()
        if not content:
            continue
        if turn.role == "user":
            out.append(HumanMessage(content=content))
        elif turn.role == "assistant":
            out.append(AIMessage(content=content))
        else:
            out.append(SystemMessage(content=content))
    return out


def build_session_history(
    history: list[ChatTurn] | None,
    *,
    session_id: str = "",
) -> InMemoryChatMessageHistory:
    """LangChain InMemoryChatMessageHistory for this chat_id (ephemeral per request)."""
    store = InMemoryChatMessageHistory()
    for msg in history_to_lc_messages(history):
        store.add_message(msg)
    # session_id reserved for future Redis/Cosmos-backed ChatMessageHistory
    _ = session_id
    return store


def _format_lc_transcript(messages: list[BaseMessage], limit: int = 10) -> str:
    lines: list[str] = []
    for msg in messages[-limit:]:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content}")
        else:
            lines.append(f"System: {msg.content}")
    return "\n".join(lines)


async def summarize_preferences(
    *,
    query: str,
    history: list[ChatTurn] | None,
    prior: PreferenceSummary | None = None,
    chat_id: str = "",
    extracted_items: list[str] | None = None,
) -> PreferenceSummary:
    """Update rolling preference summary (ConversationSummaryMemory pattern)."""
    prior = prior or PreferenceSummary()
    chat_history = build_session_history(history, session_id=chat_id)
    transcript = _format_lc_transcript(list(chat_history.messages))

    if not is_decompose_configured():
        return _heuristic_preference_update(query, history, prior, extracted_items or [])

    user_prompt = (
        f"chat_id: {chat_id or '(none)'}\n\n"
        f"Prior preference summary JSON:\n{prior.model_dump_json(indent=2)}\n\n"
        f"Recent conversation (LangChain messages):\n{transcript or '(empty)'}\n\n"
        f"Latest user message:\n{query.strip()}\n\n"
    )
    if extracted_items:
        user_prompt += (
            "Candidate grocery items extracted from prior assistant list "
            f"(may help prior_grocery_items): {', '.join(extracted_items)}\n\n"
        )
    user_prompt += "Return the updated preference summary JSON now."

    try:
        data = await complete_json(PREFERENCE_SUMMARY_SYSTEM, user_prompt, max_tokens=800)
        summary = PreferenceSummary.model_validate(data)
    except (LLMNotConfiguredError, Exception):
        return _heuristic_preference_update(query, history, prior, extracted_items or [])

    # Prefer structured extraction when LLM omits items but rewrite is intended
    if summary.is_list_rewrite and not summary.prior_grocery_items and extracted_items:
        summary.prior_grocery_items = list(extracted_items)

    # Bare clarification replies ("2.", "ready-made") are NOT grocery list rewrites.
    from app.orchestrator.clarify import looks_like_option_answer

    if looks_like_option_answer(query):
        summary.is_list_rewrite = False
        summary.prior_grocery_items = []
        summary.rewrite_guidance = ""
    return summary


def _heuristic_preference_update(
    query: str,
    history: list[ChatTurn] | None,
    prior: PreferenceSummary,
    extracted_items: list[str],
) -> PreferenceSummary:
    """Minimal fallback when LLM is unavailable — still no hardcoded diet taxonomy."""
    from app.orchestrator.clarify import looks_like_option_answer

    q = (query or "").strip()
    if looks_like_option_answer(q):
        return PreferenceSummary(
            summary=prior.summary,
            preferences=list(prior.preferences),
            is_list_rewrite=False,
            prior_grocery_items=[],
            rewrite_guidance="",
        )
    has_history = bool(history)
    looks_follow_up = has_history and (
        q.endswith("?")
        or any(
            w in q.lower()
            for w in ("instead", "prefer", "make it", "make them", "considering", "only ", "swap")
        )
    )
    summary = PreferenceSummary(
        summary=prior.summary
        or (f"User follow-up on prior shopping list: {q[:160]}" if looks_follow_up else prior.summary),
        preferences=list(prior.preferences),
        is_list_rewrite=bool(looks_follow_up and extracted_items),
        prior_grocery_items=extracted_items if looks_follow_up else [],
        rewrite_guidance=(
            f"Apply the user's latest preference request to the prior grocery list: {q}"
            if looks_follow_up
            else prior.rewrite_guidance
        ),
    )
    return summary


def items_from_preference_summary(pref: PreferenceSummary) -> list[CompositeItem]:
    """Build grocery CompositeItems from summarized preferences + prior items."""
    from app.orchestrator.dietary import items_from_prior_list

    if not pref.prior_grocery_items:
        return []
    return items_from_prior_list(pref.prior_grocery_items, pref.preferences)


def preference_rewrite_prompt(query: str, pref: PreferenceSummary, history_block: str = "") -> str:
    prefs = ", ".join(pref.preferences) if pref.preferences else "(see summary)"
    prior_txt = ", ".join(pref.prior_grocery_items) if pref.prior_grocery_items else "(see conversation)"
    guidance = pref.rewrite_guidance or "Rewrite food items to match user preferences."
    return (
        f"{history_block}\n\n".lstrip()
        + f"Preference summary: {pref.summary}\n"
        f"Active preferences: {prefs}\n"
        f"Rewrite guidance: {guidance}\n"
        f"Current user request: {query.strip()}\n\n"
        "Rewrite the EXISTING grocery shopping list to satisfy the preference summary.\n"
        f"Prior items to replace/keep: {prior_txt}.\n\n"
        "Rules:\n"
        "- search_terms MUST reflect the preferences where they apply to food/drinks.\n"
        "- Keep household non-food items (trash bags, paper towels) unless preferences say otherwise.\n"
        "- Choose closest common Kroger/Walmart substitutes when exact matches are unlikely.\n"
        "- Return a FULL required_items list for a fresh Kroger vs Walmart price compare.\n"
        "- event_summary should mention the preference rewrite.\n"
        "- Never use the user's question sentence as an item name.\n"
    )


def preference_summary_from_raw(raw: Any) -> PreferenceSummary | None:
    if raw is None:
        return None
    if isinstance(raw, PreferenceSummary):
        return raw
    if isinstance(raw, dict):
        try:
            return PreferenceSummary.model_validate(raw)
        except Exception:
            return None
    return None
