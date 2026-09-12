"""Extract known clarification facts from the current chat session.

Used so ReAct clarify skips questions already answered in history/preferences
and so replies can show which specific session facts drove the decision.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.orchestrator.state import ChatTurn

FactSource = Literal["this_message", "chat_history", "preferences"]


class SessionFact(BaseModel):
    key: str
    label: str
    value: str
    source: FactSource = "chat_history"

    def model_dump_state(self) -> dict[str, Any]:
        return self.model_dump()

    def display_line(self) -> str:
        src = {
            "this_message": "this message",
            "chat_history": "earlier in this chat",
            "preferences": "saved preferences",
        }.get(self.source, self.source)
        return f"- **{self.label}:** {self.value} (from {src})"


_LABELS = {
    "party_size": "Household / group size",
    "ages": "Ages",
    "pets": "Pets",
    "meal_count": "Meals covered",
    "kids_food": "Kids vs adult food",
    "care_items": "Care items",
    "weather_gear": "Location / weather / gear",
    "power_capacity": "Power / electronics capacity",
    "fulfillment_path": "Ready-made vs make-at-home",
}


def _pref_blob(preference_summary: dict | None) -> str:
    from app.orchestrator.preferences import preference_summary_from_raw

    pref = preference_summary_from_raw(preference_summary)
    if not pref:
        return ""
    parts = [pref.summary or ""]
    parts.extend(pref.preferences or [])
    return " ".join(parts).lower()


def _history_user_blob(history: list[ChatTurn] | None) -> str:
    return " ".join(
        (t.content or "") for t in (history or []) if t.role == "user"
    ).lower()


def _locate_source(
    *,
    needle_re: str,
    query: str,
    history: list[ChatTurn] | None,
    preference_summary: dict | None,
) -> FactSource | None:
    q = (query or "").lower()
    if re.search(needle_re, q, re.I):
        return "this_message"
    for turn in reversed(history or []):
        if turn.role != "user":
            continue
        if re.search(needle_re, turn.content or "", re.I):
            return "chat_history"
    if re.search(needle_re, _pref_blob(preference_summary), re.I):
        return "preferences"
    return None


def _party_size_value(blob: str) -> str | None:
    adults = re.search(r"(\d+)\s*adults?", blob)
    kids = re.search(r"(\d+)\s*(?:kids?|children|child)\b", blob)
    pets = re.search(r"(\d+)\s*(?:dogs?|cats?|pets?)\b", blob)
    ages = re.search(r"ages?\s*([0-9,\-\sand]+)", blob)
    household_tag = re.search(r"household:\s*([^;.|]+)", blob)
    if household_tag and not (adults or kids):
        return household_tag.group(1).strip()[:120]
    bits: list[str] = []
    if adults:
        bits.append(f"{adults.group(1)} adult(s)")
    if kids:
        bit = f"{kids.group(1)} child(ren)"
        if ages:
            bit += f" (ages {ages.group(1).strip()})"
        bits.append(bit)
    if pets:
        bits.append(f"{pets.group(1)} pet(s)")
    if bits:
        return ", ".join(bits)
    if re.search(r"(adults?\s*(and|&|/)\s*(kids?|children)|kids?\s*(and|&|/)\s*adults?)", blob):
        return "adults and children (counts stated in chat)"
    return None


def extract_session_facts(
    *,
    query: str,
    history: list[ChatTurn] | None = None,
    preference_summary: dict | None = None,
) -> list[SessionFact]:
    """Pull clarification-relevant facts already present in the session."""
    q = (query or "").lower()
    hist = _history_user_blob(history)
    pref = _pref_blob(preference_summary)
    combined = f"{q} {hist} {pref}"
    facts: list[SessionFact] = []

    party_val = _party_size_value(combined)
    if party_val:
        src = (
            _locate_source(
                needle_re=r"\d+\s*adults?|\d+\s*(kids?|children)|household:",
                query=query,
                history=history,
                preference_summary=preference_summary,
            )
            or "chat_history"
        )
        facts.append(
            SessionFact(key="party_size", label=_LABELS["party_size"], value=party_val, source=src)
        )

    kids_food_patterns = [
        (r"same (meals?|food|snacks?)|same for everyone", "Same meals/snacks for everyone"),
        (r"kids need different|different food|toddler (meals?|snacks?)|formula", "Kids need different food"),
        (r"no kids( on this trip)?", "No kids on this trip"),
    ]
    for pat, value in kids_food_patterns:
        src = _locate_source(
            needle_re=pat, query=query, history=history, preference_summary=preference_summary
        )
        if src:
            facts.append(
                SessionFact(key="kids_food", label=_LABELS["kids_food"], value=value, source=src)
            )
            break

    care_patterns = [
        (r"\bnone of these\b|no (diapers|meds|medicines|care items)", "No special care items"),
        (r"diapers?|wipes", "Diapers/wipes needed"),
        (r"medicines?|medication|first[- ]?aid", "Medicines / first-aid extras"),
    ]
    for pat, value in care_patterns:
        src = _locate_source(
            needle_re=pat, query=query, history=history, preference_summary=preference_summary
        )
        if src:
            facts.append(
                SessionFact(key="care_items", label=_LABELS["care_items"], value=value, source=src)
            )
            break

    weather_patterns = [
        (r"consumables only|skip (gear|weather|tents?|jackets?)", "Consumables only — skip tents/jackets"),
        (r"cool nights?|cold|rain|jacket|blanket", "Cool nights / rain — include jackets/blankets"),
        (r"warm|summer|hot", "Warm / summer — minimal weather gear"),
        (r"camping (near|in|at)|this weekend|next weekend|near [a-z]", None),
    ]
    for pat, value in weather_patterns:
        src = _locate_source(
            needle_re=pat, query=query, history=history, preference_summary=preference_summary
        )
        if not src:
            continue
        if value is None:
            loc = re.search(
                r"(camping (?:near|in|at) [^.;\n]+|near [^.;\n]{3,40}|this weekend|next weekend)",
                combined,
                re.I,
            )
            value = loc.group(0).strip() if loc else "Location/season noted in chat"
        facts.append(
            SessionFact(key="weather_gear", label=_LABELS["weather_gear"], value=value, source=src)
        )
        break

    power_patterns = [
        (r"skip power|no power gear", "Skip power gear for now"),
        (r"weekend (portable )?power|power station|c1000", "Weekend portable power station"),
        (r"higher capacity|rv overnight", "Higher capacity / RV overnight"),
        (r"light charging|power banks? for phones|phone only", "Light charging (power banks)"),
    ]
    for pat, value in power_patterns:
        src = _locate_source(
            needle_re=pat, query=query, history=history, preference_summary=preference_summary
        )
        if src:
            facts.append(
                SessionFact(
                    key="power_capacity",
                    label=_LABELS["power_capacity"],
                    value=value,
                    source=src,
                )
            )
            break

    path_patterns = [
        (r"ready[- ]made|store[- ]bought|buy pre", "Ready-made / store-bought"),
        (r"homemade|make at home|bake at home|ingredients", "Ingredients to make at home"),
    ]
    for pat, value in path_patterns:
        src = _locate_source(
            needle_re=pat, query=query, history=history, preference_summary=preference_summary
        )
        if src:
            facts.append(
                SessionFact(
                    key="fulfillment_path",
                    label=_LABELS["fulfillment_path"],
                    value=value,
                    source=src,
                )
            )
            break

    meal_patterns = [
        (r"\b(\d+)\s*meals?\b", None),
        (r"full camping weekend|fri.*sun|~?\s*6 meals", "Full camping weekend (~6 meals)"),
        (r"one meal|single meal|just dinner|one gathering", "One meal / one gathering"),
        (r"breakfast|lunch|dinner", "Specific meals noted in chat"),
    ]
    for pat, value in meal_patterns:
        src = _locate_source(
            needle_re=pat, query=query, history=history, preference_summary=preference_summary
        )
        if not src:
            continue
        if value is None:
            m = re.search(r"\b(\d+)\s*meals?\b", combined, re.I)
            value = f"{m.group(1)} meals" if m else "Meal count noted in chat"
        facts.append(
            SessionFact(key="meal_count", label=_LABELS["meal_count"], value=value, source=src)
        )
        break

    # De-dupe by key (first wins).
    seen: set[str] = set()
    out: list[SessionFact] = []
    for fact in facts:
        if fact.key in seen:
            continue
        seen.add(fact.key)
        out.append(fact)
    return out


def facts_for_intent(facts: list[SessionFact], intent: str) -> list[SessionFact]:
    """Filter facts that matter for a given shopping intent."""
    relevant = {
        "trip": {
            "party_size",
            "ages",
            "pets",
            "meal_count",
            "kids_food",
            "care_items",
            "weather_gear",
            "power_capacity",
        },
        "grocery": {"party_size", "meal_count", "kids_food", "care_items", "fulfillment_path"},
        "electronics": {"power_capacity", "party_size"},
        "clothing": {"weather_gear", "party_size"},
        "other": {"party_size", "weather_gear"},
    }.get(intent, {"party_size"})
    return [f for f in facts if f.key in relevant]


def format_used_facts_markdown(facts: list[SessionFact]) -> str:
    if not facts:
        return ""
    lines = ["**Using from this chat:**", ""]
    lines.extend(f.display_line() for f in facts)
    return "\n".join(lines).strip()


def guidance_from_facts(facts: list[SessionFact], *, intent: str = "") -> str:
    if not facts:
        return ""
    bits = [f"{f.label}={f.value} (from {f.source})" for f in facts]
    prefix = f"Using session facts for {intent}: " if intent else "Using session facts: "
    return prefix + "; ".join(bits)


def merge_unique_facts(blocks: list[dict[str, Any]]) -> list[SessionFact]:
    """Collect used_facts from parallel intent clarify blocks."""
    seen: set[str] = set()
    out: list[SessionFact] = []
    for block in blocks:
        for raw in block.get("used_facts") or []:
            try:
                fact = SessionFact.model_validate(raw)
            except Exception:
                continue
            if fact.key in seen:
                continue
            seen.add(fact.key)
            out.append(fact)
    return out
