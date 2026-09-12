"""Per-intent planner clarification needs.

Each planner inspects the user ask + history + prefs and returns zero or more
ClarificationNeed items. The clarify gate aggregates them and asks the user
about all unresolved intents in one turn before pricing/planning continues.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.orchestrator.state import ChatTurn, CompositeItem

IntentName = Literal["trip", "grocery", "electronics", "clothing", "other"]


class ClarificationNeed(BaseModel):
    """One question a planner needs answered before it can plan well."""

    id: str
    intent: IntentName
    question: str
    options: list[str] = Field(default_factory=list)
    reason: str = ""
    # Cross-intent dedupe key (party_size, kids_food, power_capacity, ...).
    similarity_key: str = ""

    def model_dump_state(self) -> dict[str, Any]:
        return self.model_dump()


def _clarify_helpers():
    """Lazy import to avoid circular dependency with clarify.py."""
    from app.orchestrator import clarify as c

    return c


def _blob(
    query: str,
    history: list[ChatTurn] | None,
    preference_summary: dict | None,
) -> str:
    from app.orchestrator.preferences import preference_summary_from_raw

    c = _clarify_helpers()
    pref = preference_summary_from_raw(preference_summary)
    parts = [query or "", c.original_user_ask(history, fallback="")]
    if pref:
        parts.append(pref.summary or "")
        parts.extend(pref.preferences or [])
    for turn in history or []:
        parts.append(turn.content or "")
    return " ".join(parts).lower()


def active_intents(
    query: str,
    history: list[ChatTurn] | None,
    items: list[CompositeItem] | None,
) -> set[IntentName]:
    c = _clarify_helpers()
    intents: set[IntentName] = set()
    ask = c.original_user_ask(history, fallback=query)
    text = f"{query} {ask}".lower()
    cats = {i.category for i in (items or [])}

    if c.is_trip_intent(query, history) or c.is_trip_intent(ask, history):
        intents.add("trip")
    if "grocery" in cats or any(
        t in text for t in ("food", "snack", "grocery", "meal", "party", "cupcake", "water", "pack")
    ):
        intents.add("grocery")
    if "electronics" in cats or c._is_power_electronics_blob(text) or any(
        t in text for t in ("device", "laptop", "phone", "charger", "solar")
    ):
        intents.add("electronics")
    if "clothing" in cats or any(t in text for t in ("jacket", "coat", "clothing", "rain gear")):
        intents.add("clothing")
    if not intents:
        for cat in cats:
            if cat in {"grocery", "electronics", "clothing"}:
                intents.add(cat)  # type: ignore[arg-type]
            elif cat in {"stationery", "other"}:
                intents.add("other")
    return intents


# Back-compat alias used by older call sites / tests.
_active_intents = active_intents


def trip_planner_needs(
    *,
    query: str,
    history: list[ChatTurn] | None,
    preference_summary: dict | None,
) -> list[ClarificationNeed]:
    c = _clarify_helpers()
    ask = c.original_user_ask(history, fallback=query)
    if not c.is_trip_intent(query, history) and not c.is_trip_intent(ask, history):
        return []

    needs: list[ClarificationNeed] = []
    blob = _blob(query, history, preference_summary)

    if c.mentions_family_or_group(query, history) and not c.household_composition_known(
        query, history, preference_summary
    ):
        needs.append(
            ClarificationNeed(
                similarity_key="party_size",
                id="trip.family",
                intent="trip",
                question="Who is coming on this camping / road trip?",
                options=[
                    "Adults only (say how many)",
                    "Adults + children (say counts/ages)",
                    "Adults + children + pets",
                    "I'll describe the group in my reply",
                ],
                reason="family mentioned without adults/children/pets counts",
            )
        )

    kids_mentioned = any(w in blob for w in ("kid", "child", "children", "toddler", "infant", "baby"))
    kids_food_known = any(
        w in blob
        for w in (
            "same food",
            "same meals",
            "kids eat",
            "toddler food",
            "kid snack",
            "formula",
            "different food",
            "no kids",
        )
    )
    if kids_mentioned and not kids_food_known:
        needs.append(
            ClarificationNeed(
                similarity_key="kids_food",
                id="trip.kids_food",
                intent="trip",
                question="Do kids need different food than adults (toddler meals, kid snacks, formula)?",
                options=[
                    "Same meals/snacks for everyone",
                    "Kids need different food (I'll note details)",
                    "No kids on this trip",
                ],
                reason="kids food requirements unknown",
            )
        )
    elif c.mentions_family_or_group(query, history) and not kids_food_known and not kids_mentioned:
        needs.append(
            ClarificationNeed(
                similarity_key="kids_food",
                id="trip.kids_food",
                intent="trip",
                question="If kids are coming, do they need different food than adults?",
                options=[
                    "No kids / same food for everyone",
                    "Kids need different food (I'll note details)",
                ],
                reason="family trip — kids food unknown",
            )
        )

    care_known = any(
        w in blob
        for w in (
            "diaper",
            "wipe",
            "medicine",
            "medication",
            "no meds",
            "no diapers",
            "first aid",
            "none of these",
        )
    )
    if (kids_mentioned or c.mentions_family_or_group(query, history)) and not care_known:
        needs.append(
            ClarificationNeed(
                similarity_key="care_items",
                id="trip.care",
                intent="trip",
                question="Anyone need care items like diapers, wipes, medicines, or extra first-aid?",
                options=[
                    "None of these",
                    "Diapers/wipes (say size if you can)",
                    "Medicines / first-aid extras (I'll list)",
                    "I'll list what we need",
                ],
                reason="care-item requirements unknown",
            )
        )

    weather_known = any(
        w in blob
        for w in (
            "warm",
            "hot",
            "cold",
            "cool night",
            "rain",
            "jacket",
            "blanket",
            "tent",
            "consumables only",
            "skip gear",
            "skip weather",
            "forecast",
        )
    ) or bool(re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", blob))
    if not weather_known:
        needs.append(
            ClarificationNeed(
                similarity_key="weather_gear",
                id="trip.weather",
                intent="trip",
                question="Where/when are you camping (or warm vs cold/rainy) so we can plan gear?",
                options=[
                    "I'll share location + dates",
                    "Warm / summer — minimal weather gear",
                    "Cool nights or rain — jackets/blankets",
                    "Groceries/consumables only — skip tents/jackets",
                ],
                reason="location/season/weather gear path unknown",
            )
        )

    return needs


def _bakery_fulfillment_question(text: str) -> str:
    """Product-specific ready-made vs bake wording when we can detect the treat."""
    t = (text or "").lower()
    for label, needles in (
        ("cupcakes", ("cupcake", "cup cake")),
        ("cookies", ("cookie",)),
        ("muffins", ("muffin",)),
        ("brownies", ("brownie",)),
        ("cake", ("cake",)),
        ("donuts", ("donut", "doughnut")),
        ("pizza", ("pizza",)),
    ):
        if any(n in t for n in needles):
            return (
                f"Do you want ready-made / store-bought {label}, "
                f"or ingredients to make {label} at home?"
            )
    return (
        "For the food/treats, do you want ready-made / store-bought, "
        "or ingredients to make at home?"
    )


def grocery_planner_needs(
    *,
    query: str,
    history: list[ChatTurn] | None,
    preference_summary: dict | None,
    items: list[CompositeItem] | None,
) -> list[ClarificationNeed]:
    _ = items
    c = _clarify_helpers()
    ask = c.original_user_ask(history, fallback=query)
    text = f"{query} {ask}".lower()
    needs: list[ClarificationNeed] = []

    # Shared party-size need for trip/grocery quantity planning (deduped with trip.family).
    if c.is_trip_intent(query, history) or c.is_trip_intent(ask, history):
        if c.mentions_family_or_group(query, history) and not c.household_composition_known(
            query, history, preference_summary
        ):
            needs.append(
                ClarificationNeed(
                    similarity_key="party_size",
                    id="grocery.party_size",
                    intent="grocery",
                    question="How many people should we shop food for (adults, children, pets)?",
                    options=[
                        "Adults only (say how many)",
                        "Adults + children (say counts/ages)",
                        "Adults + children + pets",
                        "I'll describe the group in my reply",
                    ],
                    reason="grocery quantities need household size",
                )
            )
        blob = _blob(query, history, preference_summary)
        meal_known = bool(
            re.search(
                r"\b("
                r"\d+\s*meals?|breakfast|lunch|dinner|one night|single meal|just dinner|"
                r"one gathering|full camping weekend|"
                r"fri(day)?|sat(urday)?|sun(day)?"
                r")\b",
                blob,
            )
        )
        if not meal_known:
            needs.append(
                ClarificationNeed(
                    similarity_key="meal_count",
                    id="grocery.meal_count",
                    intent="grocery",
                    question="Are you shopping for one meal, or multiple meals across the trip (e.g. Fri dinner–Sun lunch)?",
                    options=[
                        "One meal / one gathering",
                        "Full camping weekend (~6 meals)",
                        "I'll list which meals",
                    ],
                    reason="package quantities depend on one meal vs multi-meal trip",
                )
            )

    looks_bakery = any(
        t in text
        for t in ("cupcake", "cookie", "muffin", "brownie", "cake", "donut", "pizza", "bakery")
    ) or ("party" in text and any(t in text for t in ("dessert", "snack", "treat")))
    if looks_bakery:
        if not (
            c.is_trip_intent(query, history)
            and not any(t in text for t in ("cupcake", "cookie", "cake", "bakery", "bake"))
        ):
            if not (
                c._user_already_chose_path(query)
                or c._user_already_chose_path(ask)
                or c.is_fulfillment_path_choice(query, ask)
                or any(
                    t in text
                    for t in ("ready-made", "ready made", "store-bought", "homemade", "bake at", "mix")
                )
            ):
                needs.append(
                    ClarificationNeed(
                        similarity_key="fulfillment_path",
                        id="grocery.fulfillment",
                        intent="grocery",
                        question=_bakery_fulfillment_question(text),
                        options=[
                            "Ready-made / store-bought",
                            "Ingredients to make at home",
                        ],
                        reason="bakery/party fulfillment path ambiguous",
                    )
                )
    return needs


def electronics_planner_needs(
    *,
    query: str,
    history: list[ChatTurn] | None,
    preference_summary: dict | None,
    items: list[CompositeItem] | None,
) -> list[ClarificationNeed]:
    c = _clarify_helpers()
    ask = c.original_user_ask(history, fallback=query)
    text = f"{query} {ask}".lower()
    cats = {i.category for i in (items or [])}
    wants_power = "electronics" in cats or c._is_power_electronics_blob(text) or any(
        t in text for t in ("device", "laptop", "phone", "lots of electronic", "lot of electronic")
    )
    if not wants_power:
        return []

    blob = _blob(query, history, preference_summary)
    capacity_known = any(
        w in blob
        for w in (
            "power station",
            "power bank",
            "c1000",
            "weekend power",
            "rv overnight",
            "day hike",
            "phone only",
            "skip power",
            "no power gear",
            "light charging",
            "higher capacity",
        )
    )
    if capacity_known:
        return []

    return [
        ClarificationNeed(
            similarity_key="power_capacity",
            id="electronics.power",
            intent="electronics",
            question="What ready-made power setup do you need for your devices?",
            options=[
                "Light charging (power banks for phones)",
                "Weekend portable power station",
                "Higher capacity / RV overnight",
                "Skip power gear for now",
            ],
            reason="electronics/devices mentioned without power capacity",
        )
    ]


def clothing_planner_needs(
    *,
    query: str,
    history: list[ChatTurn] | None,
    preference_summary: dict | None,
    items: list[CompositeItem] | None,
) -> list[ClarificationNeed]:
    c = _clarify_helpers()
    if c.is_trip_intent(query, history):
        return []
    cats = {i.category for i in (items or [])}
    ask = c.original_user_ask(history, fallback=query)
    text = f"{query} {ask}".lower()
    if "clothing" not in cats and not any(t in text for t in ("jacket", "coat", "rain")):
        return []
    blob = _blob(query, history, preference_summary)
    if any(w in blob for w in ("warm", "cold", "rain", "size", "mens", "womens", "kids jacket")):
        return []
    return [
        ClarificationNeed(
            similarity_key="weather_gear",
            id="clothing.use",
            intent="clothing",
            question="What clothing/gear conditions should we plan for?",
            options=[
                "Warm weather",
                "Cool / rain — jacket or shell",
                "I'll specify sizes/items",
            ],
            reason="clothing intent without weather/use detail",
        )
    ]


def collect_planner_clarifications(
    *,
    query: str,
    history: list[ChatTurn] | None = None,
    preference_summary: dict | None = None,
    items: list[CompositeItem] | None = None,
    max_questions: int = 5,
) -> list[ClarificationNeed]:
    """Ask each relevant planner what it still needs; dedupe and cap."""
    intents = active_intents(query, history, items)
    collected: list[ClarificationNeed] = []

    if "trip" in intents:
        collected.extend(
            trip_planner_needs(
                query=query, history=history, preference_summary=preference_summary
            )
        )
    if "grocery" in intents:
        collected.extend(
            grocery_planner_needs(
                query=query,
                history=history,
                preference_summary=preference_summary,
                items=items,
            )
        )
    if "electronics" in intents:
        collected.extend(
            electronics_planner_needs(
                query=query,
                history=history,
                preference_summary=preference_summary,
                items=items,
            )
        )
    if "clothing" in intents:
        collected.extend(
            clothing_planner_needs(
                query=query,
                history=history,
                preference_summary=preference_summary,
                items=items,
            )
        )

    seen: set[str] = set()
    out: list[ClarificationNeed] = []
    for need in collected:
        if need.id in seen:
            continue
        seen.add(need.id)
        out.append(need)
        if len(out) >= max_questions:
            break
    return out


def format_multi_clarify_reply(
    needs: list[ClarificationNeed],
    *,
    used_facts: list | None = None,
) -> str:
    """Markdown ask-back covering every planner question."""
    if not needs and not used_facts:
        return ""
    from app.orchestrator.session_facts import SessionFact, format_used_facts_markdown

    facts: list[SessionFact] = []
    for raw in used_facts or []:
        if isinstance(raw, SessionFact):
            facts.append(raw)
            continue
        try:
            facts.append(SessionFact.model_validate(raw))
        except Exception:
            continue

    intent_labels = {
        "trip": "Trip / camping",
        "grocery": "Grocery / food",
        "electronics": "Electronics / power",
        "clothing": "Clothing / gear",
        "other": "Other",
    }
    lines: list[str] = []
    facts_md = format_used_facts_markdown(facts)
    if facts_md:
        lines.extend([facts_md, ""])

    if not needs:
        lines.append("I have enough from this chat to continue planning with the facts above.")
        return "\n".join(lines).strip()

    intro_bits = sorted({intent_labels.get(n.intent, n.intent) for n in needs})
    if facts:
        intro = (
            f"I'll use the details above. Before I finish planning "
            f"**{' + '.join(intro_bits)}**, I still need:"
        )
    elif len(intro_bits) > 1:
        intro = f"Before I plan **{' + '.join(intro_bits)}**, I need a few details:"
    else:
        intro = f"Before I plan **{intro_bits[0]}**, I need a few details:"

    lines.extend([intro, ""])
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, need in enumerate(needs):
        label = letters[i] if i < len(letters) else str(i + 1)
        intent = intent_labels.get(need.intent, need.intent)
        lines.append(f"**{label}. {intent}** — {need.question}")
        if need.options:
            for j, opt in enumerate(need.options, start=1):
                lines.append(f"   {j}. {opt}")
        lines.append("")
    lines.append(
        "Tap an option below, or reply with answers per letter "
        "(e.g. `A: Ready-made / store-bought`)."
    )
    return "\n".join(lines).strip()


def clarification_prompt_payload(needs: list[ClarificationNeed]) -> dict:
    """Structured clarify payload for clickable UI clients."""
    intent_labels = {
        "trip": "Trip / camping",
        "grocery": "Grocery / food",
        "electronics": "Electronics / power",
        "clothing": "Clothing / gear",
        "other": "Other",
    }
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    questions = []
    for i, need in enumerate(needs):
        letter = letters[i] if i < len(letters) else str(i + 1)
        questions.append(
            {
                "id": need.id,
                "letter": letter,
                "intent": need.intent,
                "intent_label": intent_labels.get(need.intent, need.intent),
                "question": need.question,
                "options": list(need.options or []),
                "similarity_key": need.similarity_key,
            }
        )
    intro_bits = sorted({q["intent_label"] for q in questions})
    if len(intro_bits) > 1:
        intro = f"Before I plan {' + '.join(intro_bits)}, I need a few details:"
    elif intro_bits:
        intro = f"Before I plan {intro_bits[0]}, I need a few details:"
    else:
        intro = "I need a few details before planning:"
    return {
        "needs_clarification": True,
        "intro": intro,
        "questions": questions,
    }


def looks_like_multi_clarify_answer(query: str, history: list[ChatTurn] | None) -> bool:
    """True when the user is answering a prior multi-question clarify ask."""
    q = (query or "").strip()
    if not q:
        return False
    for turn in reversed(history or []):
        if turn.role != "assistant":
            continue
        low = (turn.content or "").lower()
        if "before i plan" in low and ("a few details" in low or "**a." in low):
            return True
        if "reply with answers per letter" in low:
            return True
        break
    if re.search(r"(?im)^\s*[A-D]\s*[:.)\-]", q) or re.search(r"(?i)\b[A-D]\s*:\s*\S+", q):
        return True
    return False


MULTI_ANSWER_SYSTEM = """You consolidate the shopper's answers to multiple clarification questions
into planning guidance for DealFinder planners (trip, grocery, electronics, clothing).

Return JSON only:
{
  "needs_clarification": false,
  "planning_guidance": "concrete include/exclude instructions covering EVERY intent answered",
  "resolved_choice": "short summary of answers",
  "unanswered_ids": [],
  "reason": "short"
}

Rules:
- Map lettered answers (A/B/C…) to the listed questions.
- If a critical question is clearly unanswered, set needs_clarification=true and list unanswered_ids.
- Prefer proceeding when the user gave a reasonable free-text answer covering the themes.
- Never invent household counts. State assumptions only when the user provided them.
- For electronics: prefer ready-made portable power; never push DIY component kits.
- planning_guidance must be actionable for downstream grocery + trip + electronics agents.
"""
