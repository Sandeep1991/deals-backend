from __future__ import annotations

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


def _needs_decompose(query: str, items: list[CompositeItem]) -> bool:
    """Expand when split only handed coarse labels or trip/meal packing intent."""
    if not items:
        return True
    if any(_is_coarse(i.name) for i in items):
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


async def grocery_agent_node(state: OrchestratorState, search_service: SearchService) -> dict:
    query = state["query"]
    summary = state.get("event_summary") or query
    items = [i for i in (state.get("items") or []) if i.category == "grocery"]
    if not items:
        return {
            "category_results": [
                CategoryResult(category="grocery", notes=["No grocery items in this request."])
            ]
        }

    if _needs_decompose(query, items):
        decomposed = await decompose_node(
            {"query": _grocery_decompose_query(query, items), "plan": None, "quotes": [], "comparison": None, "reply": "", "ads": []}  # type: ignore[arg-type]
        )
        plan = decomposed.get("plan")
        if not plan or (not plan.required_items and not plan.alternative_options):
            plan = _items_to_plan(query, summary, items)
        else:
            # Keep routing summary when decompose summary is generic.
            if summary and not (plan.event_summary or "").strip():
                plan.event_summary = summary
            items = [
                CompositeItem(
                    name=si.name,
                    search_terms=si.search_terms or [si.name],
                    category="grocery",
                    quantity=si.quantity or 1.0,
                )
                for si in plan.required_items
            ]
    else:
        plan = _items_to_plan(query, summary, items)

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
    if comparison is None:
        notes.append("Could not build a grocery store comparison.")

    return {
        "category_results": [
            CategoryResult(
                category="grocery",
                items=items,
                quotes=quotes,
                ads=ads,
                comparison=comparison,
                notes=notes,
                reply_fragment=(comparison.reply if comparison else ""),
            )
        ]
    }
