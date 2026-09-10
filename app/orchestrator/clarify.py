"""Generic ask-back clarification for ambiguous shopping intents.

LangGraph runs this after split. If the request still has multiple materially
different shopping paths, we pause and ask the user. After they answer, the
same node resolves the choice and attaches planning_guidance for downstream
agents — no product-specific hardcoding.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.llm_client import LLMNotConfiguredError, complete_json, is_decompose_configured
from app.orchestrator.memory import format_history_block
from app.orchestrator.preferences import preference_summary_from_raw
from app.orchestrator.state import ChatTurn, CompositeItem, OrchestratorState

CLARIFY_SYSTEM = """You are DealFinder's clarification gate (LangGraph node).
Decide whether the shopper's request is clear enough to build a priced shopping list,
or whether you must ask them to choose among a few materially different paths.

Return JSON only:
{
  "needs_clarification": true,
  "question": "one clear question for the user",
  "options": ["option A", "option B"],
  "resolved_choice": "",
  "planning_guidance": "",
  "reason": "short internal reason"
}

When to set needs_clarification=true:
- Multiple valid fulfillment paths would produce VERY DIFFERENT carts
  (examples of patterns, not an exhaustive list: ready-made vs make-at-home;
  rent/buy vs ingredients; one meal approach vs another; unclear size/count that
  changes the cart; unclear which product family they want).
- The latest user message is vague relative to prior context and you cannot
  safely price without guessing.

When needs_clarification=false:
- The request is specific enough, OR
- Prior conversation already answered the ambiguity, OR
- The latest user message is answering a prior clarification question.

If the user is answering a prior clarification:
- needs_clarification=false
- resolved_choice = their chosen option (short label)
- planning_guidance = concrete instructions for the shopping planner
  (what to include AND what to exclude)

Rules:
- options: 2-4 short, mutually exclusive choices. No prices.
- Do NOT invent brands.
- Do NOT clarify routine preference tweaks that preference memory already handles
  (organic/vegan follow-ups) unless the cart structure itself is still ambiguous.
- Do NOT clarify just to be polite when one path is obvious.
- question should be plain and direct.
- planning_guidance must be actionable for Kroger/Walmart grocery planning.
"""


class ClarificationDecision(BaseModel):
    needs_clarification: bool = False
    question: str = ""
    options: list[str] = Field(default_factory=list)
    resolved_choice: str = ""
    planning_guidance: str = ""
    reason: str = ""

    def reply_markdown(self) -> str:
        lines = [self.question.strip() or "Which option should I price for you?"]
        opts = [o.strip() for o in self.options if (o or "").strip()]
        if opts:
            lines.append("")
            for i, opt in enumerate(opts, start=1):
                lines.append(f"{i}. **{opt}**")
            lines.append("")
            lines.append("Reply with the option number or name and I’ll compare store prices.")
        return "\n".join(lines).strip()

    def model_dump_state(self) -> dict[str, Any]:
        return self.model_dump()


def clarification_from_raw(raw: Any) -> ClarificationDecision | None:
    if raw is None:
        return None
    if isinstance(raw, ClarificationDecision):
        return raw
    if isinstance(raw, dict):
        try:
            return ClarificationDecision.model_validate(raw)
        except Exception:
            return None
    return None


def _items_blurb(items: list[CompositeItem]) -> str:
    if not items:
        return "(none yet)"
    parts = []
    for item in items[:12]:
        parts.append(f"{item.name} [{item.category}]")
    return ", ".join(parts)


async def decide_clarification(
    *,
    query: str,
    history: list[ChatTurn] | None,
    items: list[CompositeItem] | None = None,
    preference_summary: dict | None = None,
    prior_clarification: dict | None = None,
) -> ClarificationDecision:
    """LLM gate: ask options when ambiguous; resolve when user already chose."""
    if not is_decompose_configured():
        return ClarificationDecision(
            needs_clarification=False,
            reason="LLM unavailable — proceed without clarification gate",
        )

    history_block = format_history_block(history or [])
    pref = preference_summary_from_raw(preference_summary)
    pref_bits = ""
    if pref and (pref.summary or pref.preferences):
        pref_bits = f"Preference summary: {pref.summary}\n"
        if pref.preferences:
            pref_bits += f"Active preferences: {', '.join(pref.preferences)}\n"

    prior_bits = ""
    if prior_clarification:
        prior_bits = f"Prior clarification JSON:\n{prior_clarification}\n\n"

    user_prompt = (
        f"{history_block}\n\n".lstrip()
        + prior_bits
        + pref_bits
        + f"Split items so far: {_items_blurb(list(items or []))}\n\n"
        f"Latest user message:\n{query.strip()}\n\n"
        "Decide whether clarification is required. Return JSON now."
    )

    try:
        data = await complete_json(CLARIFY_SYSTEM, user_prompt, max_tokens=700)
        decision = ClarificationDecision.model_validate(data)
    except (LLMNotConfiguredError, Exception):
        return ClarificationDecision(needs_clarification=False, reason="clarify LLM failed")

    # Normalize empty options
    decision.options = [o.strip() for o in decision.options if (o or "").strip()][:4]
    if decision.needs_clarification and len(decision.options) < 2:
        # Can't ask meaningfully — proceed rather than block
        decision.needs_clarification = False
        decision.reason = (decision.reason or "") + " (insufficient options)"
    if decision.needs_clarification:
        decision.resolved_choice = ""
        decision.planning_guidance = ""
    return decision


async def clarify_intent_node(state: OrchestratorState) -> dict:
    """LangGraph node: pause for options or attach resolved planning guidance."""
    pref_raw = state.get("preference_summary")
    pref = preference_summary_from_raw(pref_raw if isinstance(pref_raw, dict) else None)
    # Dietary/list rewrites already have a clear cart to adjust — don't re-ask.
    if pref and pref.is_list_rewrite:
        return {
            "clarification": ClarificationDecision(
                needs_clarification=False,
                reason="preference list rewrite",
                planning_guidance=pref.rewrite_guidance or "",
            ).model_dump_state()
        }

    decision = await decide_clarification(
        query=state["query"],
        history=list(state.get("history") or []),
        items=list(state.get("items") or []),
        preference_summary=pref_raw if isinstance(pref_raw, dict) else None,
        prior_clarification=state.get("clarification")
        if isinstance(state.get("clarification"), dict)
        else None,
    )

    out: dict[str, Any] = {"clarification": decision.model_dump_state()}

    if decision.needs_clarification:
        from app.orchestrator.state import CategoryResult

        out["category_results"] = [
            CategoryResult(
                category="clarify",
                notes=["Awaiting user clarification before pricing."],
                reply_fragment=decision.reply_markdown(),
            )
        ]
        # Avoid fan-out pricing on ambiguous turns
        out["items"] = []
    elif decision.planning_guidance and not list(state.get("items") or []):
        # User answered briefly ("option 2") — seed grocery so decompose can run.
        out["items"] = [
            CompositeItem(
                name="event shopping list",
                search_terms=["groceries"],
                category="grocery",
            )
        ]
        if not (state.get("event_summary") or "").strip():
            out["event_summary"] = decision.resolved_choice or "Shopping list"
    return out


def route_after_clarify(state: OrchestratorState) -> str:
    """Conditional edge target name."""
    decision = clarification_from_raw(state.get("clarification"))
    if decision and decision.needs_clarification:
        return "merge_results"
    return "fan_out"
