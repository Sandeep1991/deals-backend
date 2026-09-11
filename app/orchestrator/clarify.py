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
            or "reply with answers per letter" in low
            or "before i plan" in low
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
    """Most recent substantive user ask (skip bare option / lettered clarify answers)."""
    found = ""
    for turn in history or []:
        if turn.role != "user":
            continue
        content = (turn.content or "").strip()
        if not content or looks_like_option_answer(content):
            continue
        # Skip lettered multi-clarify replies so we keep the original shopping ask.
        if re.search(r"(?im)^\s*[A-D]\s*[:.)\-]", content) or re.search(
            r"(?i)\b[A-D]\s*:\s*\S+", content
        ):
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


def _is_power_electronics_blob(text: str) -> bool:
    t = (text or "").lower()
    return any(
        w in t
        for w in (
            "power",
            "solar",
            "charger",
            "battery",
            "electronic",
            "electronics",
            "inverter",
            "generator",
            "power station",
            "power bank",
            "accessories",
            "component",
            "components",
        )
    )


def mentions_family_or_group(query: str, history: list[ChatTurn] | None = None) -> bool:
    ask = original_user_ask(history, fallback=query)
    text = f"{query} {ask}".lower()
    return any(w in text for w in ("family", "families", "kids", "children", "my wife", "my husband", "our kids"))


def household_composition_known(
    query: str,
    history: list[ChatTurn] | None = None,
    preference_summary: dict | None = None,
) -> bool:
    """True when adults/children/pets counts are already stated (not just the word 'family')."""
    pref = preference_summary_from_raw(preference_summary)
    parts = [query or "", original_user_ask(history, fallback="")]
    if pref:
        parts.append(pref.summary or "")
        parts.extend(pref.preferences or [])
    for turn in history or []:
        parts.append(turn.content or "")
    blob = " ".join(parts).lower()
    if "household:" in blob:
        return True
    if re.search(r"\d+\s*adults?", blob):
        return True
    if re.search(r"\d+\s*(kids?|children|child)\b", blob):
        return True
    if re.search(r"(adults?\s*(and|&|/)\s*(kids?|children)|kids?\s*(and|&|/)\s*adults?)", blob):
        return True
    if re.search(r"\d+\s*(dogs?|cats?|pets?)\b", blob) and re.search(r"\d+\s*adults?", blob):
        return True
    return False


def needs_family_clarify(
    query: str,
    history: list[ChatTurn] | None = None,
    preference_summary: dict | None = None,
) -> bool:
    if not is_trip_intent(query, history):
        return False
    if not mentions_family_or_group(query, history):
        return False
    return not household_composition_known(query, history, preference_summary)


def family_clarify_decision() -> "ClarificationDecision":
    question = "Who is coming on this camping trip?"
    options = [
        "Adults only (say how many)",
        "Adults + children (say counts/ages)",
        "Adults + children + pets",
        "I'll describe the group in my reply",
    ]
    return ClarificationDecision(
        needs_clarification=True,
        question=question,
        options=options,
        questions=[
            {
                "id": "trip.family",
                "intent": "trip",
                "question": question,
                "options": options,
                "reason": "trip mentions family without defined household composition",
            }
        ],
        reason="trip mentions family without defined household composition",
    )


def is_fulfillment_path_choice(choice: str, original_ask: str = "") -> bool:
    """Whether resolved choice is make-vs-buy bakery/party path (not trip/power answers)."""
    blob = f"{choice} {original_ask}".lower()
    # Portable power ready-made vs DIY is NOT the bakery fulfillment path.
    if _is_power_electronics_blob(choice):
        return False
    if is_trip_intent(original_ask) and not any(
        w in blob for w in ("bakery", "homemade", "ingredient", "bake", "cupcake", "cookie", "frosting")
    ):
        # Trip answers that say "ready-made power" still aren't bakery.
        if _is_power_electronics_blob(blob):
            return False
        if not any(w in blob for w in ("ready-made", "ready made", "store-bought", "homemade", "bake")):
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
    ) and not _is_power_electronics_blob(choice)


def planning_guidance_for_choice(choice: str, original_ask: str) -> str:
    ask = original_ask.strip() or "the user's event request"
    if _is_power_electronics_blob(choice):
        if _is_ready_made_choice(choice) or "ready" in (choice or "").lower():
            return (
                f"User chose ready-made portable power for: {ask}. "
                "Include a portable power station (and a portable solar panel if daytime charging helps). "
                "Exclude DIY wire/inverter/component kits. "
                "Also plan camping groceries/consumables for the trip."
            )
        return (
            f"User chose DIY power components for: {ask}. "
            "Plan discrete power components only if commonly sold; still include camping consumables."
        )
    if is_trip_intent(ask) and not is_fulfillment_path_choice(choice, ask):
        return (
            f"User answered clarification with: {choice}. "
            f"Original request: {ask}. "
            "Use this answer for trip planning (household size, kids food, care items, "
            "and/or location/season/weather gear). Do not invent missing household counts. "
            "If the trip mentions lots of electronics/devices, also include ready-made "
            "portable power station (+ panel if useful)."
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
    choice_l = (choice or "").lower()

    # Never turn electronics/power path answers into bakery desserts.
    if _is_power_electronics_blob(choice) or _is_power_electronics_blob(ask):
        return [
            CompositeItem(
                name="portable power station",
                search_terms=["portable power station", "C1000", "portable solar generator"],
                category="electronics",
                quantity=1.0,
            )
        ]

    # Only seed bakery/DIY desserts when the ask is actually about those foods.
    product = ""
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
    if not product:
        # No bakery product in the ask — do not invent "party dessert".
        return []

    diet_prefix = ""
    for diet in ("vegan", "organic", "gluten-free", "gluten free", "dairy-free", "dairy free"):
        if diet in ask:
            diet_prefix = "vegan" if diet.startswith("vegan") else diet
            break

    label = f"{diet_prefix} {product}".strip() if diet_prefix else product

    if _is_ready_made_choice(choice) or any(
        w in ask for w in ("store bought", "store-bought", "ready made", "ready-made", "premade", "pre-made")
    ) or "ready" in choice_l:
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
    family_needed = needs_family_clarify(query, history, preference_summary)

    # If the user just answered a mistaken power DIY ask, keep that preference but
    # still ask family composition when camping "family" is undefined.
    if resolved and trip and family_needed and _is_power_electronics_blob(resolved.resolved_choice):
        forced = family_clarify_decision()
        forced.reason = (
            f"{forced.reason}; absorbed power choice={resolved.resolved_choice!r}"
        )
        return forced

    # Bare option picks for bakery/party paths can resolve immediately.
    # Trip option picks still go through the LLM so it can ask the next unset
    # household/food/care/weather question (or finalize planning_guidance).
    if resolved and not (
        trip and not is_fulfillment_path_choice(resolved.resolved_choice, ask)
    ):
        return resolved

    if not is_decompose_configured():
        if trip and family_needed:
            return family_clarify_decision()
        if resolved:
            return _normalize_trip_clarification(
                resolved,
                query=query,
                history=history,
                preference_summary=preference_summary,
            )
        if trip and mentions_family_or_group(query, history):
            return family_clarify_decision()
        return ClarificationDecision(
            needs_clarification=False,
            reason="LLM unavailable — proceed without clarification gate",
        )

    # Deterministic short-circuit before LLM when family is clearly required.
    if trip and family_needed and not looks_like_option_answer(query):
        # Still allow LLM if user is mid-answer to a family question (free text).
        opts, asst = last_clarification_ask(history)
        if not asst or not _looks_family_clarify_blob(asst):
            return family_clarify_decision()

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
            "FIRST priority is family/household composition when 'family' is mentioned "
            "and adults/children/pets counts are unknown — ask that before anything else. "
            "Do NOT ask ready-made vs DIY for portable power/chargers/electronics; "
            "assume ready-made portable power stations when devices are mentioned. "
            "Then kids vs adult food, care items (diapers/meds), and location/season/weather — "
            "one question at a time. "
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
        "Decide whether clarification is required. For trips, ask household composition "
        "before power or bakery paths. Never ask DIY vs ready-made for electronics/power. "
        "For parties only, prefer fulfillment-path options (ready-made vs make-at-home) "
        "over quantity tweaks. Return JSON now."
    )

    try:
        data = await complete_json(CLARIFY_SYSTEM, user_prompt, max_tokens=700)
        decision = ClarificationDecision.model_validate(data)
    except (LLMNotConfiguredError, Exception):
        if trip and family_needed:
            return family_clarify_decision()
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
    decision = _normalize_trip_clarification(
        decision,
        query=query,
        history=history,
        preference_summary=preference_summary,
        interim_power_guidance=(resolved.planning_guidance if resolved else ""),
    )
    return decision

def _apply_resolved_seeds(
    *,
    choice: str,
    ask: str,
    existing_items: list[CompositeItem] | None = None,
) -> list[CompositeItem]:
    """Seed bakery path or power-station items; other trip answers keep/expand via trip planner."""
    if is_fulfillment_path_choice(choice, ask):
        return seed_items_for_choice(choice, ask)
    if _is_power_electronics_blob(choice) or (
        is_trip_intent(ask) and _is_power_electronics_blob(f"{choice} {ask}")
    ):
        power = CompositeItem(
            name="portable power station",
            search_terms=["portable power station", "C1000", "portable solar generator"],
            category="electronics",
            quantity=1.0,
        )
        items = list(existing_items or [])
        if not any(i.category == "electronics" for i in items):
            items.append(power)
        return items
    return list(existing_items or [])


def _looks_family_clarify_blob(blob: str) -> bool:
    b = (blob or "").lower()
    return any(
        w in b
        for w in (
            "who is coming",
            "who is going",
            "adults only",
            "adults + children",
            "how many adults",
            "household",
            "pets",
            "describe the group",
        )
    )


def _looks_power_diy_clarify(blob: str) -> bool:
    b = (blob or "").lower()
    # Any ready-made vs DIY ask about electronics/power/accessories/components.
    if any(w in b for w in ("diy", "ready-made", "ready made", "components", "assemble", "accessories")):
        if _is_power_electronics_blob(b) or "electronic" in b:
            return True
    return False


def _looks_food_path_clarify(blob: str) -> bool:
    b = (blob or "").lower()
    return any(
        w in b
        for w in (
            "bakery",
            "cupcake",
            "cookie",
            "homemade",
            "home made",
            "bake",
            "ingredient",
            "store-bought",
            "store bought",
            "ready-made / store",
            "make at home",
        )
    )


def _normalize_trip_clarification(
    decision: ClarificationDecision,
    *,
    query: str,
    history: list[ChatTurn] | None,
    preference_summary: dict | None,
    interim_power_guidance: str = "",
) -> ClarificationDecision:
    """Force family asks for camping; never ask ready-made vs DIY for power gear."""
    ask = original_user_ask(history, fallback=query)
    trip = is_trip_intent(query, history) or is_trip_intent(ask, history)
    blob = f"{decision.question} {' '.join(decision.options)} {decision.reason}"
    family_needed = trip and needs_family_clarify(query, history, preference_summary)

    # On trips, any non-family clarify while household is unknown → force family ask.
    if family_needed and decision.needs_clarification and not _looks_family_clarify_blob(blob):
        return family_clarify_decision()

    # Power/electronics DIY vs ready-made is the wrong question — especially on trips.
    if decision.needs_clarification and _looks_power_diy_clarify(blob) and not _looks_food_path_clarify(blob):
        if family_needed:
            return family_clarify_decision()
        # Family already known: skip ask, assume ready-made portable power.
        decision.needs_clarification = False
        decision.question = ""
        decision.options = []
        decision.resolved_choice = decision.resolved_choice or "Ready-made portable power solutions"
        decision.planning_guidance = (
            interim_power_guidance
            or planning_guidance_for_choice("Ready-made portable power solutions", ask)
        )
        decision.reason = (decision.reason or "") + " (skipped power DIY clarify; assume ready-made)"
        return decision

    # Trip + "family" without composition: always ask who is coming first.
    if family_needed and not decision.needs_clarification:
        return family_clarify_decision()

    return decision


async def clarify_intent_node(state: OrchestratorState) -> dict:
    """LangGraph node: collect per-intent planner questions, or attach guidance."""
    from app.llm_client import LLMNotConfiguredError, complete_json, is_decompose_configured
    from app.orchestrator.planner_clarify import (
        MULTI_ANSWER_SYSTEM,
        collect_planner_clarifications,
        format_multi_clarify_reply,
        looks_like_multi_clarify_answer,
    )

    query = state["query"]
    history = list(state.get("history") or [])
    pref_raw = state.get("preference_summary")
    pref = preference_summary_from_raw(pref_raw if isinstance(pref_raw, dict) else None)
    existing_items = list(state.get("items") or [])
    pref_dict = pref_raw if isinstance(pref_raw, dict) else None
    prior = state.get("clarification") if isinstance(state.get("clarification"), dict) else None

    def _clarify_pause(decision: ClarificationDecision) -> dict[str, Any]:
        from app.orchestrator.state import CategoryResult

        return {
            "clarification": decision.model_dump_state(),
            "category_results": [
                CategoryResult(
                    category="clarify",
                    notes=["Awaiting user clarification before pricing."],
                    reply_fragment=decision.reply_markdown(),
                )
            ],
            "items": [],
        }

    # Dietary/list rewrites already have a clear cart to adjust — don't re-ask.
    if pref and pref.is_list_rewrite and not looks_like_option_answer(query):
        if not looks_like_multi_clarify_answer(query, history):
            return {
                "clarification": ClarificationDecision(
                    needs_clarification=False,
                    reason="preference list rewrite",
                    planning_guidance=pref.rewrite_guidance or "",
                ).model_dump_state()
            }

    # User answering a prior multi-intent clarify batch.
    # Frontend may not echo clarification JSON — recover questions from history / re-collect.
    prior_questions = list((prior or {}).get("questions") or [])
    answering_multi = looks_like_multi_clarify_answer(query, history)

    if answering_multi:
        ask = original_user_ask(history, fallback=query)
        if not prior_questions:
            # Reconstruct what planners asked on the original request (ignore this answer text).
            hist_for_recollect = [
                t for t in history
                if not (
                    t.role == "assistant"
                    and "before i plan" in (t.content or "").lower()
                )
            ]
            prior_questions = [
                n.model_dump()
                for n in collect_planner_clarifications(
                    query=ask,
                    history=hist_for_recollect,
                    preference_summary=pref_dict,
                    items=existing_items,
                )
            ]
        questions_blurb = ""
        if prior_questions:
            for i, q in enumerate(prior_questions):
                letter = chr(ord("A") + i)
                questions_blurb += (
                    f"{letter}. [{q.get('intent')}] {q.get('question')} "
                    f"options={q.get('options')}\n"
                )
        else:
            questions_blurb = "(see prior assistant multi-clarify message in history)\n"

        guidance = (
            f"User answered multi-intent clarification for: {ask}. "
            f"Their reply: {query.strip()}"
        )
        resolved = ClarificationDecision(
            needs_clarification=False,
            resolved_choice=query.strip()[:160],
            planning_guidance=guidance,
            questions=prior_questions,
            reason="multi-clarify answers received",
        )
        if is_decompose_configured():
            try:
                data = await complete_json(
                    MULTI_ANSWER_SYSTEM,
                    (
                        f"Original request:\n{ask}\n\n"
                        f"Questions asked:\n{questions_blurb}\n"
                        f"User answers:\n{query.strip()}\n\n"
                        "Return JSON now."
                    ),
                    max_tokens=900,
                )
                if data.get("planning_guidance"):
                    resolved.planning_guidance = str(data["planning_guidance"])
                if data.get("resolved_choice"):
                    resolved.resolved_choice = str(data["resolved_choice"])[:200]
                # If model says still unclear, re-collect remaining planner needs.
                if data.get("needs_clarification"):
                    needs = collect_planner_clarifications(
                        query=f"{ask}\n{query}",
                        history=history,
                        preference_summary=pref_dict,
                        items=existing_items,
                    )
                    if needs:
                        return _clarify_pause(
                            ClarificationDecision(
                                needs_clarification=True,
                                question=needs[0].question,
                                options=list(needs[0].options),
                                questions=[n.model_dump() for n in needs],
                                reason="multi-clarify still incomplete",
                            )
                        )
            except (LLMNotConfiguredError, Exception):
                pass

        # After answers, check if planners still need more (e.g. partial reply).
        # Fold answers into a synthetic history blob via preference-less re-collect
        # using query+answer text so known facts suppress questions.
        still = collect_planner_clarifications(
            query=f"{ask}. User clarification answers: {query}",
            history=history,
            preference_summary=pref_dict,
            items=existing_items,
        )
        # Drop needs already addressed by keywords in the answer when possible.
        if still and len(still) < len(prior_questions or still):
            # Some remain — ask only remaining
            return _clarify_pause(
                ClarificationDecision(
                    needs_clarification=True,
                    question=still[0].question,
                    options=list(still[0].options),
                    questions=[n.model_dump() for n in still],
                    reason="remaining planner questions after partial answers",
                    planning_guidance=resolved.planning_guidance,
                )
            )

        out: dict[str, Any] = {"clarification": resolved.model_dump_state()}
        seeded = _apply_resolved_seeds(
            choice=resolved.resolved_choice or resolved.planning_guidance,
            ask=ask,
            existing_items=existing_items,
        )
        if seeded:
            out["items"] = seeded
        out["event_summary"] = resolved.resolved_choice or ask[:120] or "Shopping list"
        return out

    # Fresh turn: ask EVERY relevant planner what it still needs (multi-intent).
    needs = collect_planner_clarifications(
        query=query,
        history=history,
        preference_summary=pref_dict,
        items=existing_items,
    )
    if needs:
        decision = ClarificationDecision(
            needs_clarification=True,
            question=needs[0].question,
            options=list(needs[0].options),
            questions=[n.model_dump() for n in needs],
            reason="aggregated planner clarification needs: "
            + ", ".join(n.id for n in needs),
        )
        # Ensure reply uses multi formatter
        _ = format_multi_clarify_reply(needs)
        return _clarify_pause(decision)

    # No planner needs — fall back to legacy single-path LLM gate (bakery etc.).
    decision = await decide_clarification(
        query=query,
        history=history,
        items=existing_items,
        preference_summary=pref_dict,
        prior_clarification=prior,
    )
    decision = _normalize_trip_clarification(
        decision,
        query=query,
        history=history,
        preference_summary=pref_dict,
    )

    out = {"clarification": decision.model_dump_state()}
    if decision.needs_clarification:
        return _clarify_pause(decision)
    if decision.resolved_choice and decision.planning_guidance:
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
      preference summary does NOT already define who is going, you MUST ask THIS FIRST —
      before power, weather, or food-path questions.
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
   e) ELECTRONICS / POWER — If the user mentions lots of devices/electronics, do NOT ask
      ready-made vs DIY power. Assume ready-made portable power stations (+ panel if useful)
      and include that in planning_guidance / trip items. Only ask a power question if they
      explicitly mention building a custom solar kit from components.

1. Fulfillment PATH that changes cart structure (parties/bakery/FOOD events ONLY — never electronics):
   ready-made / store-bought / bakery finished goods
   vs make-at-home / DIY ingredients / from-scratch components.
   Also: buy a finished kit vs assemble from parts; takeout-style vs cook-from-ingredients.
2. Fundamentally different product families or meal approaches that yield different carts.
3. Only then: count/size that materially changes packages — and ONLY if the path is already clear.

When to set needs_clarification=true:
- Trip packing with undefined "family", unknown kids-food needs, unknown care items, or
  unknown location/season/weather gear path — ask the highest unset item above.
- NEVER set needs_clarification for portable power ready-made vs DIY on camping trips.
- The user message allows multiple high-impact FOOD fulfillment paths and they have not chosen yet.
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
    # Multi-intent planner questions (serialized ClarificationNeed dicts).
    questions: list[dict[str, Any]] = Field(default_factory=list)
    resolved_choice: str = ""
    planning_guidance: str = ""
    reason: str = ""

    def reply_markdown(self) -> str:
        if self.questions:
            from app.orchestrator.planner_clarify import ClarificationNeed, format_multi_clarify_reply

            needs = []
            for raw in self.questions:
                try:
                    needs.append(ClarificationNeed.model_validate(raw))
                except Exception:
                    continue
            if needs:
                return format_multi_clarify_reply(needs)
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
