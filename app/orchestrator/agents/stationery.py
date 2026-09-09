from __future__ import annotations

from app.advisory import build_advisory_plan, format_advisory_reply
from app.orchestrator.state import CategoryResult, CompositeItem, OrchestratorState
from app.party_planner.state import ShoppingItem, ShoppingPlan
from app.search import SearchService


async def stationery_agent_node(state: OrchestratorState, search_service: SearchService) -> dict:
    query = state["query"]
    items = [i for i in (state.get("items") or []) if i.category in {"stationery", "other"}]
    limit = int(state.get("limit") or 5)

    # "other" alone with no stationery: still try search then advisory
    if not items:
        return {
            "category_results": [
                CategoryResult(category="stationery", notes=["No stationery items in this request."])
            ]
        }

    ads = []
    for item in items:
        term = item.search_terms[0] if item.search_terms else item.name
        hits = search_service.search(term, limit=limit)
        ads.extend(hit.ad for hit in hits)

    seen = set()
    unique = []
    for ad in ads:
        if ad.id in seen:
            continue
        seen.add(ad.id)
        unique.append(ad)

    notes: list[str] = []
    fragment = ""
    if unique:
        top = unique[0]
        fragment = f"Closest catalog matches include {top.title} at {top.price}."
    else:
        # Fall back to advisory shopping list
        try:
            if any(i.category == "stationery" for i in items):
                plan = await build_advisory_plan(query)
            else:
                plan = ShoppingPlan(
                    event_summary=query[:100],
                    required_items=[
                        ShoppingItem(name=i.name, search_terms=i.search_terms or [i.name])
                        for i in items
                    ],
                )
            fragment = format_advisory_reply(plan, in_catalog_scope=True)
            notes.append("No current catalog deals for these items.")
        except Exception:
            notes.append("No stationery/other deals found in the catalog.")
            fragment = "I couldn't find catalog deals for those items."

    category = "stationery" if any(i.category == "stationery" for i in items) else "other"
    return {
        "category_results": [
            CategoryResult(
                category=category,
                items=items,
                ads=unique[:limit],
                notes=notes,
                reply_fragment=fragment,
            )
        ]
    }


async def advisory_agent_node(state: OrchestratorState, search_service: SearchService) -> dict:
    """Handles empty splits or leftover 'other' when no other agent ran."""
    _ = search_service
    query = state["query"]
    items = list(state.get("items") or [])
    try:
        plan = await build_advisory_plan(query)
        fragment = format_advisory_reply(plan, in_catalog_scope=False)
    except Exception:
        fragment = f'I could not find deals for "{query}".'
        plan_items: list[CompositeItem] = items
    else:
        plan_items = [
            CompositeItem(name=i.name, search_terms=i.search_terms, category="other", quantity=i.quantity)
            for i in plan.required_items
        ] or items

    return {
        "category_results": [
            CategoryResult(
                category="other",
                items=plan_items,
                notes=["Advisory list — limited catalog coverage."],
                reply_fragment=fragment,
            )
        ]
    }
