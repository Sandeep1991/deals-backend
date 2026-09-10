from __future__ import annotations

from app.orchestrator.dietary import (
    detect_dietary_constraints,
    dietary_rewrite_prompt,
    extract_prior_grocery_names,
    is_dietary_follow_up,
    items_from_prior_list,
)
from app.orchestrator.memory import format_history_block
from app.orchestrator.state import CategoryResult, CompositeItem, OrchestratorState
from app.party_planner.nodes import compare_node, decompose_node, fetch_prices_node
from app.party_planner.state import ShoppingItem, ShoppingPlan
from app.party_planner.quantities import normalize_plan_quantities
from app.search import SearchService

# Split often emits category blobs; these must be expanded before pricing.
_COARSE_GROCERY_NAMES = {
    "snacks",
    "snack",
    "food",
    "foods",
    "groceries",
    "grocery",
    "essentials",
    "camping essentials",
    "supplies",
    "household",
    "consumables",
    "drinks",
    "beverages",
}


def _is_coarse(name: str) -> bool:
    n = (name or "").strip().lower()
    if not n:
        return True
    if n in _COARSE_GROCERY_NAMES:
        return True
    if len(n.split()) <= 2 and any(n == c or n.endswith(f" {c}") for c in _COARSE_GROCERY_NAMES):
        return True
    return False


def _looks_like_question_sku(name: str, query: str) -> bool:
    n = (name or "").strip().lower()
    q = (query or "").strip().lower()
    if not n:
        return True
    if "?" in n:
        return True
    if n == q:
        return True
    if len(n.split()) >= 6 and any(
        w in n for w in ("considering", "organic", "vegan", "gluten", "ingredients", "these")
    ):
        return True
    return False


def _needs_decompose(query: str, items: list[CompositeItem]) -> bool:
    """Expand when split only handed coarse labels or trip/meal packing intent."""
    if not items:
        return True
    if any(_is_coarse(i.name) for i in items):
        return True
    if any(_looks_like_question_sku(i.name, query) for i in items):
        return True
    if len(items) <= 2:
        q = query.lower()
        if any(
            t in q
            for t in (
                "camp",
                "camping",
                "weekend",
                "pack",
                "family",
                "party",
                "meal",
                "recipe",
                "friends",
                "guests",
            )
        ):
            return True
    return False


def _items_to_plan(query: str, summary: str, items: list[CompositeItem]) -> ShoppingPlan:
    plan = ShoppingPlan(
        event_summary=summary or query,
        required_items=[
            ShoppingItem(
                name=item.name,
                search_terms=item.search_terms or [item.name],
                quantity=item.quantity or 1.0,
            )
            for item in items
            if not _looks_like_question_sku(item.name, query)
        ],
    )
    return normalize_plan_quantities(plan, query)


def _grocery_decompose_query(query: str, items: list[CompositeItem]) -> str:
    seeds = ", ".join(dict.fromkeys(i.name for i in items if i.name)) or "camping/household staples"
    return (
        f"{query.strip()}\n\n"
        "Plan ONLY grocery and household consumables sold at Kroger or Walmart. "
        f"Expand these into concrete buyable products (do not leave category labels): {seeds}. "
        "For camping/weekend/family trips include water, snacks, trash bags, paper towels, "
        "and other staples as needed. "
        "Skip tents, sleeping bags, specialty outdoor gear, and electronics/power stations."
    )


def _plan_to_items(plan: ShoppingPlan) -> list[CompositeItem]:
    return [
        CompositeItem(
            name=si.name,
            search_terms=si.search_terms or [si.name],
            category="grocery",
            quantity=si.quantity or 1.0,
        )
        for si in plan.required_items
    ]


async def _rewrite_for_dietary(
    query: str,
    state: OrchestratorState,
    items: list[CompositeItem],
    summary: str,
) -> tuple[ShoppingPlan, list[CompositeItem]]:
    history = list(state.get("history") or [])
    constraints = detect_dietary_constraints(query)
    prior_names = extract_prior_grocery_names(history)
    if not prior_names:
        prior_names = [
            i.name for i in items if not _looks_like_question_sku(i.name, query) and not _is_coarse(i.name)
        ]

    history_block = format_history_block(history)
    prompt = dietary_rewrite_prompt(query, prior_names, constraints, history_block=history_block)
    decomposed = await decompose_node(
        {
            "query": prompt,
            "plan": None,
            "quotes": [],
            "comparison": None,
            "reply": "",
            "ads": [],
        }  # type: ignore[arg-type]
    )
    plan = decomposed.get("plan")
    if plan and (plan.required_items or plan.alternative_options):
        if summary and not (plan.event_summary or "").strip():
            plan.event_summary = summary
        return plan, _plan_to_items(plan)

    # Heuristic fallback: prefix prior names with constraints and price those
    fallback_items = items_from_prior_list(prior_names, constraints) if prior_names else items
    fallback_items = [i for i in fallback_items if not _looks_like_question_sku(i.name, query)]
    label = ", ".join(constraints) if constraints else "dietary"
    plan = _items_to_plan(query, summary or f"Revised list ({label})", fallback_items)
    plan.event_summary = plan.event_summary or f"Revised grocery list ({label})"
    return plan, fallback_items


async def grocery_agent_node(state: OrchestratorState, search_service: SearchService) -> dict:
    query = state["query"]
    summary = state.get("event_summary") or query
    history = list(state.get("history") or [])
    items = [i for i in (state.get("items") or []) if i.category == "grocery"]

    dietary = is_dietary_follow_up(query, history)

    if not items and dietary:
        prior = extract_prior_grocery_names(history)
        constraints = detect_dietary_constraints(query)
        items = items_from_prior_list(prior, constraints)

    if not items:
        return {
            "category_results": [
                CategoryResult(category="grocery", notes=["No grocery items in this request."])
            ]
        }

    if dietary:
        plan, items = await _rewrite_for_dietary(query, state, items, summary)
    elif _needs_decompose(query, items):
        decomposed = await decompose_node(
            {
                "query": _grocery_decompose_query(query, items),
                "plan": None,
                "quotes": [],
                "comparison": None,
                "reply": "",
                "ads": [],
            }  # type: ignore[arg-type]
        )
        plan = decomposed.get("plan")
        if not plan or (not plan.required_items and not plan.alternative_options):
            plan = _items_to_plan(query, summary, items)
        else:
            if summary and not (plan.event_summary or "").strip():
                plan.event_summary = summary
            items = _plan_to_items(plan)
    else:
        plan = _items_to_plan(query, summary, items)

    if not plan.required_items and not plan.alternative_options:
        return {
            "category_results": [
                CategoryResult(
                    category="grocery",
                    notes=["Could not build a grocery list from that dietary follow-up."],
                )
            ]
        }

    fetch_state = {"query": query, "plan": plan, "quotes": []}
    priced = await fetch_prices_node(fetch_state, search_service)  # type: ignore[arg-type]
    quotes = priced.get("quotes") or []
    compare_state = {
        "query": query,
        "plan": plan,
        "quotes": quotes,
        "comparison": None,
        "reply": "",
        "ads": [],
    }
    compared = compare_node(compare_state)  # type: ignore[arg-type]
    comparison = compared.get("comparison")
    ads = compared.get("ads") or []
    notes: list[str] = []
    constraints = detect_dietary_constraints(query) if dietary else []
    if comparison is None:
        notes.append("Could not build a grocery store comparison.")
    if dietary and constraints:
        notes.append(f"Repriced list with preferences: {', '.join(constraints)}.")

    reply_fragment = comparison.reply if comparison else ""
    if dietary and comparison and constraints:
        preface = (
            f"Updated the prior grocery list for **{' / '.join(constraints)}** options "
            "and re-compared Kroger vs Walmart.\n\n"
        )
        reply_fragment = preface + (reply_fragment or "")

    return {
        "category_results": [
            CategoryResult(
                category="grocery",
                items=items,
                quotes=quotes,
                ads=ads,
                comparison=comparison,
                notes=notes,
                reply_fragment=reply_fragment,
            )
        ]
    }
