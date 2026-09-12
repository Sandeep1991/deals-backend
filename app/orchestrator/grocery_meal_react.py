"""Grocery meal-planning ReAct loop.

Breaks the ask into meals, scales package quantities for adults/children/pets,
and records a short justification for each ingredient before pricing.
"""

from __future__ import annotations

import math
import re
from typing import Any

from pydantic import BaseModel, Field

from app.llm_client import LLMNotConfiguredError, complete_json, is_decompose_configured
from app.orchestrator.memory import format_history_block
from app.orchestrator.preferences import preference_summary_from_raw
from app.orchestrator.session_facts import extract_session_facts
from app.orchestrator.state import ChatTurn, CompositeItem
from app.party_planner.quantities import normalize_plan_quantities
from app.party_planner.state import ShoppingItem, ShoppingPlan

MAX_MEAL_REACT_STEPS = 8

MEAL_REACT_SYSTEM = """You are DealFinder's grocery meal planner using a ReAct loop.
Break the shopping ask into meals, then choose concrete supermarket products with
package quantities justified by household size and meal count.

Return JSON only for ONE action per step:
{
  "thought": "brief reasoning",
  "tool": "get_known_context" | "set_meal_scope" | "add_item" | "finalize_plan",
  "set_meal_scope": {
    "adults": 2,
    "children": 1,
    "pets": 0,
    "meal_count": 6,
    "meal_labels": ["Fri dinner", "Sat breakfast", "Sat lunch", "Sat dinner", "Sun breakfast", "Sun lunch"],
    "assumptions": ["Assumed Fri–Sun camping = 6 meals because days were not listed"]
  },
  "add_item": {
    "name": "bottled water",
    "search_terms": ["bottled water"],
    "quantity": 2,
    "for_whom": "all|adults|children|pets",
    "meals": ["all"] or ["Sat dinner"],
    "justification": "2 cases ≈ drinking water for 3 people across 6 meals + cooking"
  },
  "finalize_plan": {
    "event_summary": "short summary including household + meal count"
  }
}

Rules:
- quantity = store PACKAGES/UNITS to buy (boxes, bags, bottles), NOT headcount.
- Scale food/water for adults + children across meal_count (camping weekend ≠ one dinner).
- If kids eat differently, add kid-specific items with for_whom=children.
- Include care items only when guidance/session facts say they are needed.
- Prefer Kroger/Walmart staples; skip tents/electronics (other agents handle those).
- Do NOT invent household counts — use session facts / guidance; if unknown, assume
  2 adults for a family camping ask and state that assumption.
- Call set_meal_scope once early, then add_item repeatedly, then finalize_plan.
- Cap at ~12–18 grocery items; focus on high-impact staples + meal ingredients.
"""


class MealScope(BaseModel):
    adults: int = 0
    children: int = 0
    pets: int = 0
    meal_count: int = 1
    meal_labels: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    @property
    def people_count(self) -> int:
        return max(0, int(self.adults) + int(self.children))


class PlannedGroceryItem(BaseModel):
    name: str
    search_terms: list[str] = Field(default_factory=list)
    quantity: float = 1.0
    for_whom: str = "all"
    meals: list[str] = Field(default_factory=list)
    justification: str = ""


class MealPlanResult(BaseModel):
    plan: ShoppingPlan
    scope: MealScope = Field(default_factory=MealScope)
    items: list[PlannedGroceryItem] = Field(default_factory=list)
    justification_markdown: str = ""
    used_react: bool = False


def needs_meal_react(
    *,
    query: str,
    planning_guidance: str = "",
    history: list[ChatTurn] | None = None,
    preference_summary: dict | None = None,
    items: list[CompositeItem] | None = None,
) -> bool:
    """True when meal breakdown + household scaling is valuable."""
    blob = f"{query} {planning_guidance}".lower()
    for turn in history or []:
        if turn.role == "user":
            blob += " " + (turn.content or "").lower()
    if any(
        t in blob
        for t in (
            "camp",
            "camping",
            "weekend",
            "road trip",
            "family",
            "meal",
            "meals",
            "breakfast",
            "lunch",
            "dinner",
            "party",
            "guests",
            "pack for",
        )
    ):
        return True
    facts = extract_session_facts(
        query=query, history=history, preference_summary=preference_summary
    )
    if any(f.key in {"party_size", "kids_food"} for f in facts):
        return True
    if items and len(items) <= 3:
        return True
    return False


def _parse_household_from_text(blob: str) -> tuple[int, int, int]:
    adults = 0
    children = 0
    pets = 0
    m = re.search(r"(\d+)\s*adults?", blob, re.I)
    if m:
        adults = int(m.group(1))
    m = re.search(r"(\d+)\s*(?:kids?|children|child)\b", blob, re.I)
    if m:
        children = int(m.group(1))
    m = re.search(r"(\d+)\s*(?:dogs?|cats?|pets?)\b", blob, re.I)
    if m:
        pets = int(m.group(1))
    # "household:2 adults" style tags
    if not adults:
        m = re.search(r"household:\s*(\d+)\s*adults?", blob, re.I)
        if m:
            adults = int(m.group(1))
    return adults, children, pets


def _infer_meal_scope(
    *,
    query: str,
    planning_guidance: str,
    history: list[ChatTurn] | None,
    preference_summary: dict | None,
) -> MealScope:
    facts = extract_session_facts(
        query=query, history=history, preference_summary=preference_summary
    )
    pref = preference_summary_from_raw(preference_summary)
    blob_parts = [query or "", planning_guidance or ""]
    if pref:
        blob_parts.append(pref.summary or "")
        blob_parts.extend(pref.preferences or [])
    for turn in history or []:
        blob_parts.append(turn.content or "")
    for fact in facts:
        blob_parts.append(f"{fact.key} {fact.value}")
    blob = " ".join(blob_parts).lower()

    adults, children, pets = _parse_household_from_text(blob)
    assumptions: list[str] = []

    if adults == 0 and children == 0:
        if any(w in blob for w in ("family", "kids", "children")):
            adults, children = 2, 1
            assumptions.append(
                "Household size was not fully numeric; assuming 2 adults + 1 child for scaling."
            )
        elif re.search(r"\b(\d+)\s+(?:people|guests|persons)\b", blob):
            m = re.search(r"\b(\d+)\s+(?:people|guests|persons)\b", blob)
            adults = int(m.group(1)) if m else 2
        else:
            adults = 2
            assumptions.append("Guest count unclear; assuming 2 adults for package scaling.")

    meal_count = 1
    meal_labels: list[str] = ["single meal / gathering"]
    if re.search(r"\b(\d+)\s*meals?\b", blob):
        m = re.search(r"\b(\d+)\s*meals?\b", blob)
        meal_count = max(1, int(m.group(1))) if m else 1
        meal_labels = [f"meal {i + 1}" for i in range(meal_count)]
    elif any(w in blob for w in ("camping", "camp", "weekend trip", "road trip", "weekend")):
        # Fri dinner → Sun lunch style default for a camping weekend
        meal_count = 6
        meal_labels = [
            "Fri dinner",
            "Sat breakfast",
            "Sat lunch",
            "Sat dinner",
            "Sun breakfast",
            "Sun lunch",
        ]
        assumptions.append(
            "Treating this as a camping/weekend trip covering ~6 meals "
            "(Fri dinner through Sun lunch) because meal count was not specified."
        )
    elif any(w in blob for w in ("breakfast and lunch", "lunch and dinner", "two meals", "2 meals")):
        meal_count = 2
        meal_labels = ["meal 1", "meal 2"]
    elif "party" in blob or "taco" in blob or "bbq" in blob:
        meal_count = 1
        meal_labels = ["party / one meal"]

    # Cite known session party_size in assumptions when present
    for fact in facts:
        if fact.key == "party_size":
            assumptions.insert(
                0,
                f"Using household from session: {fact.value} (from {fact.source}).",
            )
            break

    return MealScope(
        adults=adults,
        children=children,
        pets=pets,
        meal_count=meal_count,
        meal_labels=meal_labels,
        assumptions=assumptions,
    )


def _heuristic_items(scope: MealScope, seeds: list[CompositeItem] | None) -> list[PlannedGroceryItem]:
    people = max(1, scope.people_count)
    meals = max(1, scope.meal_count)
    water_cases = max(1, math.ceil(people * meals / 12))  # ~1 bottle/person/meal, 12/case
    snack_bags = max(1, math.ceil(people * meals / 8))
    items: list[PlannedGroceryItem] = [
        PlannedGroceryItem(
            name="bottled water",
            search_terms=["bottled water", "water bottles"],
            quantity=float(water_cases),
            for_whom="all",
            meals=["all"],
            justification=(
                f"{water_cases} case(s) for {people} people across {meals} meals "
                f"({scope.adults} adults"
                + (f", {scope.children} children" if scope.children else "")
                + ")."
            ),
        ),
        PlannedGroceryItem(
            name="trail mix",
            search_terms=["trail mix", "snacks"],
            quantity=float(snack_bags),
            for_whom="all",
            meals=["all"],
            justification=f"{snack_bags} bag(s) of snacks shared across {meals} meals for {people} people.",
        ),
        PlannedGroceryItem(
            name="trash bags",
            search_terms=["trash bags"],
            quantity=1.0,
            for_whom="all",
            meals=["all"],
            justification="One box of trash bags covers the trip cleanup.",
        ),
        PlannedGroceryItem(
            name="paper towels",
            search_terms=["paper towels"],
            quantity=1.0,
            for_whom="all",
            meals=["all"],
            justification="One pack of paper towels for meal cleanup across the trip.",
        ),
    ]
    if scope.children:
        kid_snacks = max(1, math.ceil(scope.children * meals / 6))
        items.append(
            PlannedGroceryItem(
                name="kid snacks",
                search_terms=["kids snacks", "applesauce pouches", "granola bars"],
                quantity=float(kid_snacks),
                for_whom="children",
                meals=["all"],
                justification=(
                    f"{kid_snacks} pack(s) of kid-friendly snacks for {scope.children} "
                    f"child(ren) across {meals} meals."
                ),
            )
        )
    # Keep concrete seed SKUs with light scaling
    seen = {i.name.lower() for i in items}
    for seed in seeds or []:
        key = (seed.name or "").strip().lower()
        if not key or key in seen:
            continue
        qty = float(seed.quantity or 1.0)
        if any(w in key for w in ("water", "snack", "juice")):
            qty = max(qty, float(max(1, math.ceil(people * meals / 12))))
        items.append(
            PlannedGroceryItem(
                name=seed.name,
                search_terms=seed.search_terms or [seed.name],
                quantity=qty,
                for_whom="all",
                meals=["all"],
                justification=f"Kept from trip/grocery seeds; qty {qty:g} for {people} people × {meals} meals.",
            )
        )
        seen.add(key)
    return items


def _to_shopping_plan(
    *,
    items: list[PlannedGroceryItem],
    scope: MealScope,
    event_summary: str,
    query: str,
) -> ShoppingPlan:
    plan = ShoppingPlan(
        event_summary=event_summary,
        people_count=scope.people_count or None,
        required_items=[
            ShoppingItem(
                name=i.name,
                search_terms=i.search_terms or [i.name],
                quantity=max(1.0, float(i.quantity or 1.0)),
            )
            for i in items
            if (i.name or "").strip()
        ],
    )
    return normalize_plan_quantities(plan, query)


def format_meal_justification(
    scope: MealScope,
    items: list[PlannedGroceryItem],
) -> str:
    lines = ["**Meal plan & quantities:**", ""]
    who = f"{scope.adults} adult(s)"
    if scope.children:
        who += f", {scope.children} child(ren)"
    if scope.pets:
        who += f", {scope.pets} pet(s)"
    labels = ", ".join(scope.meal_labels[:8]) if scope.meal_labels else f"{scope.meal_count} meal(s)"
    lines.append(f"- Feeding **{who}** across **{scope.meal_count} meal(s)** ({labels}).")
    for assumption in scope.assumptions[:4]:
        lines.append(f"- _{assumption}_")
    lines.append("")
    lines.append("**Why these package counts:**")
    for item in items[:16]:
        whom = f" · {item.for_whom}" if item.for_whom and item.for_whom != "all" else ""
        just = item.justification or "Scaled to household and meal count."
        lines.append(f"- **{item.name}** × {item.quantity:g}{whom}: {just}")
    return "\n".join(lines).strip()


def _context_blob(
    *,
    query: str,
    planning_guidance: str,
    history: list[ChatTurn] | None,
    preference_summary: dict | None,
    seeds: list[CompositeItem] | None,
    scope: MealScope | None,
    items: list[PlannedGroceryItem],
) -> str:
    facts = extract_session_facts(
        query=query, history=history, preference_summary=preference_summary
    )
    fact_bits = "\n".join(f"- {f.key}: {f.value} ({f.source})" for f in facts) or "(none)"
    seed_bits = ", ".join(i.name for i in (seeds or [])[:12]) or "(none)"
    item_bits = "\n".join(
        f"- {i.name} qty={i.quantity:g} ({i.for_whom}): {i.justification}" for i in items
    ) or "(none yet)"
    scope_bits = "(not set)"
    if scope:
        scope_bits = (
            f"adults={scope.adults}, children={scope.children}, pets={scope.pets}, "
            f"meals={scope.meal_count}, labels={scope.meal_labels}, "
            f"assumptions={scope.assumptions}"
        )
    hist = format_history_block(history or [])
    return (
        f"User ask / grocery focus:\n{query}\n\n"
        f"Planning guidance:\n{planning_guidance or '(none)'}\n\n"
        f"Session facts:\n{fact_bits}\n\n"
        f"Seed grocery items: {seed_bits}\n"
        f"Current meal scope: {scope_bits}\n"
        f"Items so far:\n{item_bits}\n\n"
        f"{hist}"
    )


async def run_grocery_meal_react(
    *,
    query: str,
    planning_guidance: str = "",
    history: list[ChatTurn] | None = None,
    preference_summary: dict | None = None,
    seeds: list[CompositeItem] | None = None,
) -> MealPlanResult:
    """Run meal-breakdown ReAct (or deterministic fallback) into a ShoppingPlan."""
    inferred = _infer_meal_scope(
        query=query,
        planning_guidance=planning_guidance,
        history=history,
        preference_summary=preference_summary,
    )
    scope: MealScope | None = None
    planned: list[PlannedGroceryItem] = []
    summary = ""

    if not is_decompose_configured():
        planned = _heuristic_items(inferred, seeds)
        scope = inferred
        summary = (
            f"Groceries for {scope.people_count} people across {scope.meal_count} meals"
        )
        plan = _to_shopping_plan(
            items=planned, scope=scope, event_summary=summary, query=query
        )
        return MealPlanResult(
            plan=plan,
            scope=scope,
            items=planned,
            justification_markdown=format_meal_justification(scope, planned),
            used_react=False,
        )

    observations: list[str] = []
    for step in range(MAX_MEAL_REACT_STEPS):
        obs = _context_blob(
            query=query,
            planning_guidance=planning_guidance,
            history=history,
            preference_summary=preference_summary,
            seeds=seeds,
            scope=scope or inferred,
            items=planned,
        )
        if observations:
            obs += "\nPrior observations:\n" + "\n".join(observations[-5:])
        obs += (
            f"\nStep {step + 1}/{MAX_MEAL_REACT_STEPS}. "
            "Choose ONE tool. If scope is set and you have enough staples, finalize_plan."
        )
        try:
            data = await complete_json(MEAL_REACT_SYSTEM, obs, max_tokens=900)
        except (LLMNotConfiguredError, Exception) as exc:
            observations.append(f"LLM failed: {exc}")
            break

        tool = str(data.get("tool") or "").strip()
        thought = str(data.get("thought") or "").strip()
        if thought:
            observations.append(f"thought: {thought}")

        if tool == "get_known_context":
            observations.append(
                f"observation: inferred scope adults={inferred.adults} "
                f"children={inferred.children} meals={inferred.meal_count}"
            )
            continue

        if tool == "set_meal_scope":
            raw = data.get("set_meal_scope") if isinstance(data.get("set_meal_scope"), dict) else {}
            try:
                scope = MealScope(
                    adults=int(raw.get("adults") or inferred.adults or 0),
                    children=int(raw.get("children") or inferred.children or 0),
                    pets=int(raw.get("pets") or inferred.pets or 0),
                    meal_count=max(1, int(raw.get("meal_count") or inferred.meal_count or 1)),
                    meal_labels=[str(x) for x in (raw.get("meal_labels") or inferred.meal_labels)][:10],
                    assumptions=[str(x) for x in (raw.get("assumptions") or [])][:6]
                    or list(inferred.assumptions),
                )
            except Exception:
                scope = inferred
            if scope.people_count <= 0:
                scope.adults = max(1, inferred.adults or 2)
            observations.append(
                f"observation: meal scope set people={scope.people_count} meals={scope.meal_count}"
            )
            continue

        if tool == "add_item":
            raw = data.get("add_item") if isinstance(data.get("add_item"), dict) else {}
            name = str(raw.get("name") or "").strip()
            if not name:
                observations.append("observation: add_item missing name")
                continue
            qty = float(raw.get("quantity") or 1.0)
            if qty < 1:
                qty = 1.0
            if qty > 24:
                qty = 24.0
            terms = [str(t) for t in (raw.get("search_terms") or [name]) if str(t).strip()]
            item = PlannedGroceryItem(
                name=name,
                search_terms=terms or [name],
                quantity=qty,
                for_whom=str(raw.get("for_whom") or "all"),
                meals=[str(m) for m in (raw.get("meals") or ["all"])][:6],
                justification=str(raw.get("justification") or "").strip()
                or f"Package qty {qty:g} for household across meals.",
            )
            # Upsert by name
            planned = [p for p in planned if p.name.lower() != name.lower()] + [item]
            observations.append(f"observation: added {name} qty={qty:g}")
            if len(planned) >= 18:
                observations.append("observation: item cap reached — finalize next")
            continue

        if tool == "finalize_plan":
            raw = data.get("finalize_plan") if isinstance(data.get("finalize_plan"), dict) else {}
            summary = str(raw.get("event_summary") or "").strip()
            break

        observations.append(f"observation: unknown tool {tool!r}")

    if scope is None:
        scope = inferred
    if not planned:
        planned = _heuristic_items(scope, seeds)
    if not summary:
        summary = (
            f"Groceries for {scope.adults} adult(s)"
            + (f", {scope.children} child(ren)" if scope.children else "")
            + f" across {scope.meal_count} meal(s)"
        )

    plan = _to_shopping_plan(
        items=planned, scope=scope, event_summary=summary, query=query
    )
    return MealPlanResult(
        plan=plan,
        scope=scope,
        items=planned,
        justification_markdown=format_meal_justification(scope, planned),
        used_react=True,
    )
