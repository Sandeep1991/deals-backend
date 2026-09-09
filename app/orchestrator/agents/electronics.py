from __future__ import annotations

from app.catalog_pick import llm_catalog_search
from app.orchestrator.state import CategoryResult, OrchestratorState
from app.search import SearchService


async def electronics_agent_node(state: OrchestratorState, search_service: SearchService) -> dict:
    query = state["query"]
    items = [i for i in (state.get("items") or []) if i.category == "electronics"]
    limit = int(state.get("limit") or 5)

    if not items:
        return {
            "category_results": [
                CategoryResult(category="electronics", notes=["No electronics items in this request."])
            ]
        }

    # Use the full user query so use-case (camping/RV/home) survives routing.
    # Seed terms only help when the original ask is vague ("electronic devices").
    terms: list[str] = []
    for item in items:
        terms.extend(item.search_terms or [item.name])
    focused = " ".join(dict.fromkeys(terms))[:200]
    search_query = query
    if focused and focused.lower() not in query.lower():
        search_query = f"{query}\nProducts of interest: {focused}"

    results, reply = await llm_catalog_search(search_query, search_service, limit=limit)
    ads = [r.ad for r in results]
    notes: list[str] = []
    if not ads:
        notes.append("No electronics deals matched in the catalog.")

    return {
        "category_results": [
            CategoryResult(
                category="electronics",
                items=items,
                ads=ads,
                notes=notes,
                reply_fragment=reply or "",
            )
        ]
    }
