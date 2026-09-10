"""Generic ask-back clarification for ambiguous shopping intents.

LangGraph runs this after split. If the request still has multiple materially
different shopping paths, we pause and ask the user. After they answer, the
same node resolves the choice and attaches planning_guidance for downstream
agents — no product-specific hardcoding.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from app.llm_client import LLMNotConfiguredError, complete_json, is_decompose_configured
from app.orchestrator.memory import format_history_block
from app.orchestrator.preferences import preference_summary_from_raw
from app.orchestrator.state import ChatTurn, CompositeItem, OrchestratorState

_OPTION_LINE_RE = re.compile(
    r"^\s*(?:[-*]|\d+[.)])\s+\*{0,2}(.+?)\*{0,2}\s*$",
    re.M,
)
_BARE_OPTION_RE = re.compile(
    r"^\s*(?:option\s*)?(\d+)\s*[.)]?\s*$",
    re.I,
)
_OPTION_WITH_TEXT_RE = re.compile(
    r"^\s*(?:option\s*)?(\d+)\s*[.):]?\s+(.+)$",
    re.I,
)


def looks_like_option_answer(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    if _BARE_OPTION_RE.match(q):
        return True
    if len(q) <= 48 and _OPTION_WITH_TEXT_RE.match(q):
        return True
    ql = q.lower()
    return any(
        w in ql
        for w in (
            "ready-made",
            "ready made",
            "pre-made",
            "premade",
            "store-bought",
            "homemade",
            "home made",
            "bake at home",
            "make at home",
            "ingredients",
            "buy pre",
            "buy ready",
        )
    )


def extract_options_from_assistant(content: str) -> list[str]:
    """Pull numbered/bulleted choices from a clarification reply."""
    text = content or ""
    opts: list[str] = []
    for match in _OPTION_LINE_RE.finditer(text):
        label = re.sub(r"\*\*", "", match.group(1)).strip(" -–—:")
        if not label or len(label) > 120:
            continue
        low = label.lower()
        if low.startswith("reply with"):
            continue
        if "option number" in low:
            continue
        opts.append(label)
    # Dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for opt in opts:
        key = opt.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(opt)
    return out[:4]


def last_clarification_options(history: list[ChatTurn] | None) -> tuple[list[str], str]:
    """Return (options, assistant_question_text) from the latest clarification ask."""
    for turn in reversed(history or []):
        if turn.role != "assistant":
            continue
        content = turn.content or ""
        opts = extract_options_from_assistant(content)
        if len(opts) >= 2:
            return opts, content
        break
    return [], ""


def original_user_ask(history: list[ChatTurn] | None, fallback: str = "") -> str:
    """Most recent substantive user ask (skip bare option answers)."""
    found = ""
    for turn in history or []:
        if turn.role != "user":
            continue
        content = (turn.content or "").strip()
        if not content or looks_like_option_answer(content):
            continue
        found = content
    return found or (fallback or "").strip()


def resolve_option_choice(query: str, options: list[str]) -> str | None:
    """Map '2' / '2.' / 'option 2' / text overlap to an option label."""
    q = (query or "").strip()
    if not q or not options:
        return None

    bare = _BARE_OPTION_RE.match(q)
    if bare:
        idx = int(bare.group(1)) - 1
        if 0 <= idx < len(options):
            return options[idx]

    with_text = _OPTION_WITH_TEXT_RE.match(q)
    if with_text:
        idx = int(with_text.group(1)) - 1
        if 0 <= idx < len(options):
            return options[idx]

    ql = q.lower()
    # Exact / substring match against option labels
    for opt in options:
        ol = opt.lower()
        if ql == ol or ql in ol or ol in ql:
            return opt

    # Soft keyword mapping
    ready_hints = ("ready", "pre-made", "premade", "store", "bakery", "buy pre", "buy ready", "purchased")
    home_hints = ("home", "bake", "ingredient", "scratch", "diy", "make")
    if any(h in ql for h in ready_hints):
        for opt in options:
            ol = opt.lower()
            if any(h in ol for h in ("ready", "pre-made", "premade", "store", "buy", "bakery")):
                return opt
    if any(h in ql for h in home_hints):
        for opt in options:
            ol = opt.lower()
            if any(h in ol for h in ("home", "bake", "ingredient", "make", "scratch", "diy")):
                return opt
    return None


def _is_ready_made_choice(choice: str) -> bool:
    c = (choice or "").lower()
    if any(w in c for w in ("ingredient", "scratch", "diy", "mix", "homemade", "home-made", "home made", "bake")):
        # "Buy pre-made" should still win over homemade keywords if buy/ready present
        if any(w in c for w in ("ready", "pre-made", "premade", "store-bought", "bakery")) and "ingredient" not in c:
            return True
        if any(w in c for w in ("make", "homemade", "home made", "bake", "ingredient")):
            return False
    return any(w in c for w in ("ready", "pre-made", "premade", "store", "bakery", "buy pre", "buy ready"))


def planning_guidance_for_choice(choice: str, original_ask: str) -> str:
    ask = original_ask.strip() or "the user's event request"
    if _is_ready_made_choice(choice):
        return (
            f"User chose: {choice}. "
            f"Original request: {ask}. "
            "Plan READY-MADE / store-bought / bakery finished products only "
            "(packs/dozen as needed for the guest count). "
            "EXCLUDE baking mix, liners, frosting, sprinkles, flour, and other DIY ingredients "
            "unless the user also asked for those."
        )
    return (
        f"User chose: {choice}. "
        f"Original request: {ask}. "
        "Plan BAKE-AT-HOME ingredients only (mix or base ingredients, liners, frosting/icing, "
        "sprinkles/decorations as relevant). "
        "EXCLUDE ready-made bakery packs of the finished item."
    )


def seed_items_for_choice(choice: str, original_ask: str) -> list[CompositeItem]:
    """Concrete grocery seeds so decompose does not fall back to word-salad."""
    ask = (original_ask or "").lower()
    # Generic finished-food token from the ask; fall back to "cupcakes" only if present
    product = "party dessert"
    for token in (
        "cupcakes",
        "cupcake",
        "cookies",
        "cookie",
        "muffins",
        "muffin",
        "brownies",
        "brownie",
        "cake",
        "donuts",
        "donut",
        "pizza",
    ):
        if token in ask or token.replace("cupcake", "cup cake") in ask.replace("-", " "):
            product = "cupcakes" if token.startswith("cupcake") else (
                "cookies" if token.startswith("cookie") else (
                    "muffins" if token.startswith("muffin") else (
                        "brownies" if token.startswith("brownie") else (
                            "donuts" if token.startswith("donut") else token
                        )
                    )
                )
            )
            break
    # Normalize cup cakes
    if "cup cake" in ask.replace("-", " "):
        product = "cupcakes"

    if _is_ready_made_choice(choice):
        return [
            CompositeItem(
                name=f"ready-made {product}",
                search_terms=[f"ready made {product}", f"bakery {product}", product],
                category="grocery",
                quantity=1.0,
            )
        ]
    return [
        CompositeItem(
            name=f"{product} mix",
            search_terms=[f"{product} mix", f"{product} baking mix"],
            category="grocery",
            quantity=1.0,
        ),
        CompositeItem(
            name="cupcake liners" if product == "cupcakes" else "baking cups",
            search_terms=["cupcake liners", "baking cups"] if product == "cupcakes" else ["baking cups"],
            category="grocery",
            quantity=1.0,
        ),
        CompositeItem(
            name="frosting",
            search_terms=["frosting", "icing"],
            category="grocery",
            quantity=1.0,
        ),
    ]


def try_resolve_from_history(
    query: str,
    history: list[ChatTurn] | None,
) -> ClarificationDecision | None:
    """Deterministically resolve '2' / 'ready-made' against the prior clarification ask."""
    if not looks_like_option_answer(query):
        return None
    options, _asst = last_clarification_options(history)
    if len(options) < 2:
        return None
    choice = resolve_option_choice(query, options)
    if not choice:
        return None
    ask = original_user_ask(history, fallback=query)
    return ClarificationDecision(
        needs_clarification=False,
        question="",
        options=options,
        resolved_choice=choice,
        planning_guidance=planning_guidance_for_choice(choice, ask),
        reason="resolved option answer from prior clarification",
    )

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
    resolved = try_resolve_from_history(query, history)
    if resolved:
        return resolved

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
    query = state["query"]
    history = list(state.get("history") or [])
    pref_raw = state.get("preference_summary")
    pref = preference_summary_from_raw(pref_raw if isinstance(pref_raw, dict) else None)

    # Short answers like "2." must resolve against the prior ask — never treat as diet rewrite.
    resolved = try_resolve_from_history(query, history)
    if resolved:
        ask = original_user_ask(history, fallback=query)
        out: dict[str, Any] = {
            "clarification": resolved.model_dump_state(),
            "items": seed_items_for_choice(resolved.resolved_choice, ask),
            "event_summary": resolved.resolved_choice or ask[:120] or "Shopping list",
        }
        return out

    # Dietary/list rewrites already have a clear cart to adjust — don't re-ask.
    if pref and pref.is_list_rewrite and not looks_like_option_answer(query):
        return {
            "clarification": ClarificationDecision(
                needs_clarification=False,
                reason="preference list rewrite",
                planning_guidance=pref.rewrite_guidance or "",
            ).model_dump_state()
        }

    decision = await decide_clarification(
        query=query,
        history=history,
        items=list(state.get("items") or []),
        preference_summary=pref_raw if isinstance(pref_raw, dict) else None,
        prior_clarification=state.get("clarification")
        if isinstance(state.get("clarification"), dict)
        else None,
    )

    out = {"clarification": decision.model_dump_state()}

    if decision.needs_clarification:
        from app.orchestrator.state import CategoryResult

        out["category_results"] = [
            CategoryResult(
                category="clarify",
                notes=["Awaiting user clarification before pricing."],
                reply_fragment=decision.reply_markdown(),
            )
        ]
        out["items"] = []
    elif decision.resolved_choice and decision.planning_guidance:
        ask = original_user_ask(history, fallback=query)
        out["items"] = seed_items_for_choice(decision.resolved_choice, ask)
        out["event_summary"] = decision.resolved_choice or ask[:120] or "Shopping list"
    elif decision.planning_guidance and not list(state.get("items") or []):
        ask = original_user_ask(history, fallback=query)
        out["items"] = seed_items_for_choice(decision.resolved_choice or decision.planning_guidance, ask)
        out["event_summary"] = decision.resolved_choice or ask[:120] or "Shopping list"
    return out


def route_after_clarify(state: OrchestratorState) -> str:
    """Conditional edge target name."""
    decision = clarification_from_raw(state.get("clarification"))
    if decision and decision.needs_clarification:
        return "merge_results"
    return "fan_out"
