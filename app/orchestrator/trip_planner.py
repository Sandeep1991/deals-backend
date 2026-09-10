"""Dedicated camping / road-trip planner.

Runs after clarification. Breaks a trip into smaller shopping steps across
categories, then the orchestrator fans those items out to specialized agents
(grocery, clothing, electronics, other).
"""

from __future__ import annotations

from typing import Any

from app.llm_client import LLMNotConfiguredError, complete_json, is_decompose_configured
from app.orchestrator.clarify import clarification_from_raw, is_trip_intent, original_user_ask
from app.orchestrator.memory import format_history_block
from app.orchestrator.preferences import preference_summary_from_raw
from app.orchestrator.state import CompositeItem, OrchestratorState

TRIP_PLANNER_SYSTEM = """You are DealFinder's dedicated camping / road-trip planner.
Break the trip into concrete buyable items across categories. Downstream agents will
price grocery, clothing, electronics, and other gear separately.

Return JSON only:
{
  "event_summary": "short trip summary including assumed household (adults/children/pets)",
  "planning_steps": [
    "1. groceries/consumables for N people",
    "2. kid/adult food differences if any",
    "3. care items (diapers/meds) if any",
    "4. weather gear based on location/season",
    "5. power/electronics if devices mentioned"
  ],
  "items": [
    {
      "name": "bottled water",
      "search_terms": ["bottled water"],
      "category": "grocery|electronics|clothing|stationery|other",
      "quantity": 1
    }
  ]
}

Category guide:
- grocery: food, drinks, snacks (adult AND kid-specific), trash bags, paper towels,
  sunscreen, bug spray, diapers, wipes, OTC first-aid, pet food if pets are going
- clothing: jackets, rain shells, warm layers, gloves (when weather/season warrants)
- electronics: portable power station, power banks, chargers, batteries when devices/RV/night power
- other: tent, sleeping bags, warming blankets, lanterns, camp chairs — only when
  planning guidance or weather path says gear is needed (not "consumables only")
- stationery: rarely; skip unless explicitly needed

Rules:
- Honor planning_guidance strictly (household counts, kids vs adult food, care items,
  location/season/weather, consumables-only vs include gear).
- If guidance states assumed adults/children/pets, repeat those counts in event_summary.
- Never invent household size; if unknown, keep quantities conservative and note it.
- Kids food: when kids differ from adults, emit BOTH adult staples and kid-specific items.
- Care items: only include diapers/meds/wipes when guidance says they are needed.
- Weather: cold/rainy → jackets and/or warming blankets (clothing/other); warm → sunscreen etc.
- Emit CONCRETE products, not blobs like "Food" or "Camping essentials".
- quantity = packages/units to buy, not headcount.
- Prefer items sold at Kroger/Walmart/supercenters when possible.
- Do not invent brands unless the user named them.
"""


def _heuristic_trip_items(
    query: str,
    *,
    planning_guidance: str = "",
    existing: list[CompositeItem] | None = None,
) -> list[CompositeItem]:
    """Fallback staples when the trip LLM is unavailable."""
    items: list[CompositeItem] = []
    seen: set[str] = set()

    def add(name: str, terms: list[str], category: str, qty: float = 1.0) -> None:
        key = name.strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        items.append(
            CompositeItem(name=name, search_terms=terms, category=category, quantity=qty)  # type: ignore[arg-type]
        )

    for item in existing or []:
        add(item.name, item.search_terms or [item.name], item.category, item.quantity or 1.0)

    guidance = (planning_guidance or "").lower()
    q = f"{query} {guidance}".lower()

    for name, terms in (
        ("bottled water", ["bottled water", "water bottles"]),
        ("trail mix", ["trail mix", "snacks"]),
        ("trash bags", ["trash bags"]),
        ("paper towels", ["paper towels"]),
        ("sunscreen", ["sunscreen"]),
        ("insect repellent", ["insect repellent", "bug spray"]),
    ):
        add(name, terms, "grocery")

    if any(w in q for w in ("kid", "child", "toddler", "infant", "baby")):
        add("kid snacks", ["kids snacks", "toddler snacks"], "grocery")
        if any(w in q for w in ("different", "toddler", "formula", "baby food")):
            add("toddler pouches", ["toddler food pouches", "baby food"], "grocery")

    if any(w in q for w in ("diaper", "wipe")):
        add("diapers", ["diapers"], "grocery")
        add("baby wipes", ["baby wipes"], "grocery")
    if any(w in q for w in ("medicine", "medication", "first aid", "first-aid")):
        add("first aid kit", ["first aid kit"], "grocery")

    if any(w in q for w in ("pet", "dog", "cat")):
        add("pet food", ["dog food", "pet food"], "grocery")

    consumables_only = any(
        w in guidance for w in ("consumables only", "skip weather", "skip gear", "groceries only")
    )
    if not consumables_only:
        if any(w in q for w in ("cold", "cool night", "jacket", "rain", "blanket")):
            add("packable jacket", ["packable jacket", "rain jacket"], "clothing")
            add("warming blanket", ["fleece blanket", "warming blanket"], "other")
        if any(w in q for w in ("tent", "shelter", "camp")):
            add("camping tent", ["camping tent"], "other")

    if any(w in q for w in ("power", "device", "electronic", "phone", "laptop", "rv", "solar")):
        add("portable power station", ["portable power station"], "electronics")

    return items


async def plan_trip_node(state: OrchestratorState) -> dict:
    """Expand camping/road-trip asks into multi-category items for agent fan-out."""
    query = state.get("query") or ""
    history = list(state.get("history") or [])
    ask = original_user_ask(history, fallback=query)
    existing = list(state.get("items") or [])

    if not is_trip_intent(query, history) and not is_trip_intent(ask, history):
        return {}

    decision = clarification_from_raw(state.get("clarification"))
    if decision and decision.needs_clarification:
        return {}

    planning_guidance = (decision.planning_guidance if decision else "") or ""
    pref = preference_summary_from_raw(state.get("preference_summary"))
    pref_bits = ""
    if pref and (pref.summary or pref.preferences):
        pref_bits = f"Preference summary: {pref.summary}\n"
        if pref.preferences:
            pref_bits += f"Active preferences: {', '.join(pref.preferences)}\n"

    history_block = format_history_block(history)
    seeds = ", ".join(f"{i.name} [{i.category}]" for i in existing[:12]) or "(none)"

    if not is_decompose_configured():
        items = _heuristic_trip_items(ask or query, planning_guidance=planning_guidance, existing=existing)
        return {
            "event_summary": (state.get("event_summary") or ask or query)[:160],
            "items": items,
        }

    user_prompt = (
        f"{history_block}\n\n".lstrip()
        + pref_bits
        + (f"Planning guidance (follow strictly):\n{planning_guidance}\n\n" if planning_guidance else "")
        + f"Existing split seeds (refine/expand, do not ignore guidance): {seeds}\n\n"
        + f"Original trip request:\n{(ask or query).strip()}\n\n"
        + f"Latest user message:\n{query.strip()}\n\n"
        + "Return the trip plan JSON now."
    )

    try:
        data = await complete_json(TRIP_PLANNER_SYSTEM, user_prompt, max_tokens=1400)
    except (LLMNotConfiguredError, Exception):
        items = _heuristic_trip_items(ask or query, planning_guidance=planning_guidance, existing=existing)
        return {
            "event_summary": (state.get("event_summary") or ask or query)[:160],
            "items": items,
        }

    raw_items = data.get("items") or []
    items: list[CompositeItem] = []
    for raw in raw_items:
        try:
            item = CompositeItem.model_validate(raw)
            if item.category not in {"grocery", "electronics", "clothing", "stationery", "other"}:
                item.category = "other"
            if "?" in item.name:
                continue
            items.append(item)
        except Exception:
            continue

    if not items:
        items = _heuristic_trip_items(ask or query, planning_guidance=planning_guidance, existing=existing)

    steps = data.get("planning_steps") or []
    summary = str(data.get("event_summary") or state.get("event_summary") or ask or query).strip()
    if isinstance(steps, list) and steps:
        # Keep summary readable; steps are for the planner/agents via guidance merge
        step_txt = "; ".join(str(s) for s in steps[:6] if s)
        if decision and (decision.planning_guidance or step_txt):
            merged = (decision.planning_guidance or "").strip()
            if step_txt:
                merged = f"{merged}\nTrip steps: {step_txt}".strip()
            out_clar: dict[str, Any] = decision.model_dump()
            out_clar["planning_guidance"] = merged
            return {
                "event_summary": summary[:200],
                "items": items,
                "clarification": out_clar,
            }

    return {"event_summary": summary[:200], "items": items}
