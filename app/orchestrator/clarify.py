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
    # "2. Ready-made" style — short, and the text should look like a menu label,
    # not a household description like "2 adults, 2 kids ages 5 and 8".
    if len(q) <= 48:
        with_text = _OPTION_WITH_TEXT_RE.match(q)
        if with_text:
            rest = (with_text.group(2) or "").strip().lower()
            householdish = any(
                w in rest
                for w in ("adult", "child", "kid", "pet", "dog", "cat", "year", "age", "infant", "toddler")
            )
            if not householdish:
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


def last_clarification_ask(history: list[ChatTurn] | None) -> tuple[list[str], str]:
    """Return (options, assistant_question_text) from the latest clarification ask.

    Options may be empty for open-ended trip/family questions.
    """
    for turn in reversed(history or []):
        if turn.role != "assistant":
            continue
        content = turn.content or ""
        low = content.lower()
        opts = extract_options_from_assistant(content)
        looks_clarify = (
            len(opts) >= 2
            or "reply with the option" in low
            or "reply with a short description" in low
            or "i’ll continue planning" in low
            or "i'll continue planning" in low
            or ("awaiting" in low and "clarif" in low)
        )
        if looks_clarify:
            return opts, content
        break
    return [], ""


def last_clarification_options(history: list[ChatTurn] | None) -> tuple[list[str], str]:
    """Return (options, assistant_question_text) from the latest clarification ask."""
    return last_clarification_ask(history)


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


def is_trip_intent(query: str, history: list[ChatTurn] | None = None) -> bool:
    """True when the active ask (or recent history) is camping / road-trip packing."""
    blobs = [query or ""]
    for turn in (history or [])[-6:]:
        if turn.role == "user":
            blobs.append(turn.content or "")
    text = " ".join(blobs).lower()
    return any(
        t in text
        for t in (
            "camp",
            "camping",
            "campsite",
            "road trip",
            "roadtrip",
            "weekend trip",
            "pack for",
            "packing list",
            "pack essentials",
            "rv trip",
            "glamping",
        )
    )


def is_fulfillment_path_choice(choice: str, original_ask: str = "") -> bool:
    """Whether resolved choice is make-vs-buy bakery/party path (not trip household answers)."""
    blob = f"{choice} {original_ask}".lower()
    if is_trip_intent(original_ask) and not any(
        w in blob for w in ("ready", "bakery", "homemade", "ingredient", "bake", "mix")
    ):
        return False
    return _is_ready_made_choice(choice) or any(
        w in blob
        for w in (
            "ready-made",
            "ready made",
            "store-bought",
            "homemade",
            "home made",
            "bake at",
            "ingredients to",
            "from scratch",
            "diy",
        )
    )


def planning_guidance_for_choice(choice: str, original_ask: str) -> str:
    ask = original_ask.strip() or "the user's event request"
    if is_trip_intent(ask) and not is_fulfillment_path_choice(choice, ask):
        return (
            f"User answered clarification with: {choice}. "
            f"Original request: {ask}. "
            "Use this answer for trip planning (household size, kids food, care items, "
            "and/or location/season/weather gear). Do not invent missing household counts."
        )
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

    diet_prefix = ""
    for diet in ("vegan", "organic", "gluten-free", "gluten free", "dairy-free", "dairy free"):
        if diet in ask:
            diet_prefix = "vegan" if diet.startswith("vegan") else diet
            break

    label = f"{diet_prefix} {product}".strip() if diet_prefix else product

    if _is_ready_made_choice(choice) or any(
        w in ask for w in ("store bought", "store-bought", "ready made", "ready-made", "premade", "pre-made")
    ):
        terms = [
            f"{label}",
            f"bakery {label}",
            f"ready made {label}",
        ]
        if diet_prefix:
            terms.append(f"{diet_prefix} bakery {product}")
        return [
            CompositeItem(
                name=f"ready-made {label}",
                search_terms=terms,
                category="grocery",
                quantity=1.0,
            )
        ]
    mix_name = f"{label} mix" if diet_prefix else f"{product} mix"
    return [
        CompositeItem(
            name=mix_name,
            search_terms=[mix_name, f"{product} baking mix"],
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
            name=f"{diet_prefix} frosting".strip() if diet_prefix else "frosting",
            search_terms=[f"{diet_prefix} frosting".strip(), "frosting", "icing"]
            if diet_prefix
            else ["frosting", "icing"],
            category="grocery",
            quantity=1.0,
        ),
    ]


def try_resolve_from_history(
    query: str,
    history: list[ChatTurn] | None,
) -> ClarificationDecision | None:
    """Deterministically resolve bare '2' / 'ready-made' against a prior option menu.

    Free-text trip/family answers go through the LLM gate so it can ask the next
    high-impact follow-up (kids food, care items, weather) when still needed.
    """
    if not looks_like_option_answer(query):
        return None
    options, asst = last_clarification_ask(history)
    if not asst or len(options) < 2:
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

def _looks_trip_clarify_blob(blob: str) -> bool:
    b = (blob or "").lower()
    return any(
        w in b
        for w in (
            "adult",
            "child",
            "children",
            "kids",
            "pet",
            "family",
            "diaper",
            "medicine",
            "medication",
            "weather",
            "jacket",
            "tent",
            "blanket",
            "location",
            "camping",
            "where",
            "season",
            "warm",
            "cold",
            "rain",
        )
    )


def _prefer_path_options(decision: ClarificationDecision, query: str) -> ClarificationDecision:
    """If the model asked a weak quantity question, upgrade to fulfillment-path options."""
    if not decision.needs_clarification:
        return decision
    if _user_already_chose_path(query):
        return decision
    # Never overwrite trip/family/weather clarifications with bakery path options.
    if is_trip_intent(query) or _looks_trip_clarify_blob(
        f"{decision.question} {' '.join(decision.options)} {decision.reason}"
    ):
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
    ask = original_user_ask(history, fallback=query)
    trip = is_trip_intent(query, history) or is_trip_intent(ask, history)

    # Bare option picks for bakery/party paths can resolve immediately.
    # Trip option picks still go through the LLM so it can ask the next unset
    # household/food/care/weather question (or finalize planning_guidance).
    if resolved and not (
        trip and not is_fulfillment_path_choice(resolved.resolved_choice, ask)
    ):
        return resolved

    if not is_decompose_configured():
        if resolved:
            return resolved
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
    merged_prior = prior_clarification
    if resolved:
        merged_prior = {
            **(prior_clarification or {}),
            "just_answered_option": resolved.resolved_choice,
            "interim_planning_guidance": resolved.planning_guidance,
            "reason": resolved.reason,
        }
    if merged_prior:
        prior_bits = f"Prior clarification JSON:\n{merged_prior}\n\n"

    trip_bits = ""
    if trip:
        trip_bits = (
            "Context: this looks like a camping/road-trip packing request. "
            "Prioritize family composition, kids vs adult food, care items "
            "(diapers/meds), and location/season/weather gear — one question at a time. "
            "If the latest user message (or just_answered_option) answered one of those, "
            "fold it into planning_guidance and only ask the NEXT unset high-impact item. "
            "If preference summary or history already defines household composition, "
            "state those assumed adult/child/pet counts in planning_guidance instead of re-asking.\n\n"
        )

    user_prompt = (
        f"{history_block}\n\n".lstrip()
        + prior_bits
        + pref_bits
        + trip_bits
        + f"Split items so far (may be a premature guess — do not treat as user intent): "
        f"{_items_blurb(list(items or []))}\n\n"
        f"Latest user message:\n{query.strip()}\n\n"
        "Decide whether clarification is required. For trips, ask the highest unset "
        "household/food/care/weather ambiguity first. For parties, prefer fulfillment-path "
        "options (ready-made vs make-at-home) over quantity tweaks. Return JSON now."
    )

    try:
        data = await complete_json(CLARIFY_SYSTEM, user_prompt, max_tokens=700)
        decision = ClarificationDecision.model_validate(data)
    except (LLMNotConfiguredError, Exception):
        if resolved:
            return resolved
        return ClarificationDecision(needs_clarification=False, reason="clarify LLM failed")

    # Normalize options (may be empty for open-ended trip questions)
    decision.options = [o.strip() for o in decision.options if (o or "").strip()][:4]
    if decision.needs_clarification:
        has_question = bool((decision.question or "").strip())
        if not has_question:
            decision.needs_clarification = False
            decision.reason = (decision.reason or "") + " (missing question)"
        elif len(decision.options) == 1:
            # A single option is not a real choice — treat as open question
            decision.options = []
        decision.resolved_choice = ""
        decision.planning_guidance = ""
    elif resolved and not (decision.planning_guidance or "").strip():
        # LLM proceeded but forgot guidance — keep deterministic trip/path guidance
        decision.resolved_choice = decision.resolved_choice or resolved.resolved_choice
        decision.planning_guidance = resolved.planning_guidance
    decision = _prefer_path_options(decision, query)
    return decision

def _apply_resolved_seeds(
    *,
    choice: str,
    ask: str,
    existing_items: list[CompositeItem] | None = None,
) -> list[CompositeItem]:
    """Seed bakery path items only; trip answers keep split items / trip planner."""
    if is_fulfillment_path_choice(choice, ask):
        return seed_items_for_choice(choice, ask)
    return list(existing_items or [])


async def clarify_intent_node(state: OrchestratorState) -> dict:
    """LangGraph node: pause for options or attach resolved planning guidance."""
    query = state["query"]
    history = list(state.get("history") or [])
    pref_raw = state.get("preference_summary")
    pref = preference_summary_from_raw(pref_raw if isinstance(pref_raw, dict) else None)
    existing_items = list(state.get("items") or [])

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
        items=existing_items,
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
        out["items"] = []
    elif decision.resolved_choice and decision.planning_guidance:
        ask = original_user_ask(history, fallback=query)
        seeded = _apply_resolved_seeds(
            choice=decision.resolved_choice,
            ask=ask,
            existing_items=existing_items,
        )
        if seeded:
            out["items"] = seeded
        out["event_summary"] = decision.resolved_choice or ask[:120] or "Shopping list"
    elif decision.planning_guidance and not existing_items:
        ask = original_user_ask(history, fallback=query)
        if is_fulfillment_path_choice(decision.resolved_choice or decision.planning_guidance, ask):
            out["items"] = seed_items_for_choice(
                decision.resolved_choice or decision.planning_guidance, ask
            )
        out["event_summary"] = decision.resolved_choice or ask[:120] or "Shopping list"
    return out


def route_after_clarify(state: OrchestratorState) -> str:
    """Conditional edge target name."""
    decision = clarification_from_raw(state.get("clarification"))
    if decision and decision.needs_clarification:
        return "merge_results"
    return "plan_trip"

CLARIFY_SYSTEM = """You are DealFinder's clarification gate (LangGraph node).
Decide whether the shopper's request is clear enough to build a priced shopping list,
or whether you must ask them one high-impact clarifying question first.

Return JSON only:
{
  "needs_clarification": true,
  "question": "one clear question for the user",
  "options": ["option A", "option B"],
  "resolved_choice": "",
  "planning_guidance": "",
  "reason": "short internal reason"
}

Priority order — ask the HIGHEST-impact ambiguity first (ONE question per turn):

0. TRIP / CAMPING / ROAD-TRIP context (when the ask is packing, camping, road trip, weekend trip):
   a) FAMILY / PARTY COMPOSITION — If the user said "family" (or similar) and session history /
      preference summary does NOT already define who is going, you MUST ask.
      Never silently invent a household size.
      Prefer options like:
        "Adults only (say how many)",
        "Adults + children (say counts/ages)",
        "Adults + children + pets",
        "I'll describe the group in my reply"
      OR an open question with options=[] so they can answer in free text
      (e.g. "Who is coming — how many adults, children (ages), and pets?").
      If history/preferences ALREADY state composition, do NOT re-ask; instead put it in
      planning_guidance explicitly, e.g. "Assuming 2 adults, 2 children (ages 5–8), 1 dog
      from prior conversation."
   b) KIDS vs ADULT FOOD — If children are (or may be) present and kids' food needs are unknown,
      clarify whether kids eat the same meals/snacks as adults or need kid-specific food
      (toddler foods, kid snacks, formula/milk, etc.).
   c) SPECIAL CARE ITEMS — If kids, infants, elders, or medical needs are plausible and unset,
      ask whether anyone needs diapers, wipes, medications, first-aid extras, or other care items
      (options can include "None of these" / "I'll list what we need").
   d) LOCATION / SEASON / WEATHER GEAR — If destination or timing is unclear, ask where/when
      (or warm vs cold/rainy) so the trip planner can decide tents, jackets, warming blankets,
      rain gear, etc. Offer a "groceries/consumables only — skip weather gear" option.

1. Fulfillment PATH that changes cart structure (parties/bakery/events):
   ready-made / store-bought / bakery finished goods
   vs make-at-home / DIY ingredients / from-scratch components.
   Also: buy a finished kit vs assemble from parts; takeout-style vs cook-from-ingredients.
2. Fundamentally different product families or meal approaches that yield different carts.
3. Only then: count/size that materially changes packages — and ONLY if the path is already clear.

When to set needs_clarification=true:
- Trip packing with undefined "family", unknown kids-food needs, unknown care items, or
  unknown location/season/weather gear path — ask the highest unset item above.
- The user message allows multiple high-impact fulfillment paths and they have not chosen yet.
- Judge primarily from the USER MESSAGE + history + preference summary. Split items are ONLY
  hints and may be a premature guess. Do NOT treat split DIY items as the user's choice.
- If both "buy finished [food]" and "buy ingredients to make [food]" are plausible for an
  event/school/party/kids bring-along request, you MUST clarify that path. Do not assume bake-at-home.

When needs_clarification=false:
- The request (or prior answer) already settles the highest-impact ambiguity, OR
- The latest user message answers a prior clarification, OR
- One path is clearly stated (e.g. "cupcake mix", "ready-made cupcakes", "bakery cookies"), OR
- For trips: household composition is known from this message or history, and remaining
  unknowns are minor enough to proceed with stated assumptions in planning_guidance.

If the user is answering a prior clarification:
- needs_clarification=false
- resolved_choice = their chosen option or a short paraphrase of their free-text answer
- planning_guidance = concrete instructions for trip/grocery planners (include AND exclude).
  Examples:
  - "Household: 2 adults, 1 child age 4, no pets. Kids need toddler snacks + different from adult meals. Include diapers size 4 + wipes. No Rx meds. Camping near Asheville this weekend — cool nights, include jackets/blankets; tent if not already owned unknown so ask only if still needed otherwise include basic shelter consumables only."
  - "READY-MADE only: bakery cupcake packs for ~10 kids. Exclude mix, liners, frosting."

Rules:
- options: 2-4 short mutually exclusive choices when helpful; use [] for a single open question
  that needs free-text (family counts, location/dates). No prices.
- Ask ONE ambiguity per turn. Do not stack family + weather + meds in one question.
- Do NOT invent brands.
- Do NOT invent household size. If you proceed using history, you MUST state the assumed
  adults/children/pets counts in planning_guidance.
- Do NOT clarify routine preference tweaks that preference memory already handles
  (organic/vegan follow-ups) unless the cart structure itself is still ambiguous.
- Do NOT clarify just to be polite when one path is obvious.
- question should be plain and direct.
- planning_guidance must be actionable for Kroger/Walmart + trip category agents
  (grocery, clothing, electronics, other gear).
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
            lines.append(
                "Reply with the option number or name "
                "(you can add counts/ages/location details) and I’ll continue planning."
            )
        else:
            lines.append("")
            lines.append("Reply with a short description and I’ll continue planning.")
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
