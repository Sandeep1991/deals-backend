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

Priority order — ask the HIGHEST-impact ambiguity first:
1. Fulfillment PATH that changes cart structure (most important):
   ready-made / store-bought / bakery finished goods
   vs make-at-home / DIY ingredients / from-scratch components.
   Also: buy a finished kit vs assemble from parts; takeout-style vs cook-from-ingredients.
2. Fundamentally different product families or meal approaches that yield different carts.
3. Only then: count/size that materially changes packages — and ONLY if the path is already clear.

When to set needs_clarification=true:
- The user message allows multiple of those high-impact paths and they have not chosen yet.
- Judge primarily from the USER MESSAGE + history. Split items are ONLY hints and may be a
  premature guess (e.g. mix/liners/frosting). Do NOT treat split DIY items as the user's choice.
- If both "buy finished [food]" and "buy ingredients to make [food]" are plausible for an
  event/school/party/kids bring-along request, you MUST clarify that path. Do not assume bake-at-home.

When needs_clarification=false:
- The request (or prior answer) already picks the fulfillment path, OR
- The latest user message answers a prior clarification, OR
- One path is clearly stated (e.g. "cupcake mix", "ready-made cupcakes", "bakery cookies").

If the user is answering a prior clarification:
- needs_clarification=false
- resolved_choice = their chosen option (short label)
- planning_guidance = concrete instructions for the shopping planner
  (what to include AND what to exclude). Example: "READY-MADE only: bakery/store cupcake packs
  for ~10 kids. Exclude mix, liners, frosting, sprinkles." or the DIY inverse.

Rules:
- options: 2-4 short, mutually exclusive choices. No prices. Prefer path choices over tiny
  quantity variants ("exactly 10" vs "a few more") when a path ambiguity exists.
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


def _looks_quantity_only_options(options: list[str]) -> bool:
    blob = " ".join(options).lower()
    path_words = (
        "ready",
        "store",
        "bakery",
        "homemade",
        "home-made",
        "home made",
        "bake",
        "ingredient",
        "make at",
        "from scratch",
        "diy",
        "premade",
        "pre-made",
    )
    qty_words = (
        "exactly",
        "more than",
        "fewer",
        "less than",
        "about ",
        "around ",
        "how many",
        "enough for",
        "dozen",
    )
    has_path = any(w in blob for w in path_words)
    has_qty = any(w in blob for w in qty_words)
    return has_qty and not has_path


def _user_already_chose_path(query: str) -> bool:
    q = (query or "").lower()
    return any(
        w in q
        for w in (
            "ready-made",
            "ready made",
            "store-bought",
            "store bought",
            "bakery",
            "homemade",
            "home-made",
            "home made",
            "from scratch",
            "mix and",
            "cake mix",
            "cupcake mix",
            "ingredients to",
            "bake at",
            "make at home",
            "make them",
            "buy them ready",
        )
    )


def _prefer_path_options(decision: ClarificationDecision, query: str) -> ClarificationDecision:
    """If the model asked a weak quantity question, upgrade to fulfillment-path options."""
    if not decision.needs_clarification:
        return decision
    if _user_already_chose_path(query):
        return decision

    blob = f"{decision.question} {' '.join(decision.options)}".lower()
    has_path = any(
        w in blob
        for w in (
            "ready",
            "store",
            "bakery",
            "homemade",
            "home-made",
            "home made",
            "bake",
            "ingredient",
            "make at",
            "from scratch",
            "diy",
        )
    )
    has_qty = any(
        w in blob
        for w in ("exactly", "more than", "fewer", "less than", "enough", "how many", "about ", "around ")
    )
    if has_path and not _looks_quantity_only_options(decision.options):
        return decision
    if not has_qty and has_path:
        return decision
    if not has_qty and not _looks_quantity_only_options(decision.options):
        return decision

    decision.question = (
        "Do you want to buy this ready-made / store-bought, or buy ingredients to make it at home?"
    )
    decision.options = [
        "Ready-made / store-bought",
        "Ingredients to make at home",
    ]
    decision.reason = (decision.reason or "") + " (upgraded quantity ask → fulfillment path)"
    return decision


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
        + f"Split items so far (may be a premature guess — do not treat as user intent): "
        f"{_items_blurb(list(items or []))}\n\n"
        f"Latest user message:\n{query.strip()}\n\n"
        "Decide whether clarification is required. Prefer fulfillment-path options "
        "(ready-made vs make-at-home) over quantity tweaks when both apply. Return JSON now."
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
    decision = _prefer_path_options(decision, query)
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
