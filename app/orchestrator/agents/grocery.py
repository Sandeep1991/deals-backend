from __future__ import annotations

from app.orchestrator.state import CategoryResult, CompositeItem, OrchestratorState
from app.party_planner.nodes import compare_node, fetch_prices_node
from app.party_planner.state import ShoppingItem, ShoppingPlan
from app.party_planner.quantities import normalize_plan_quantities
from app.search import SearchService


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
