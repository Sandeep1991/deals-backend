from __future__ import annotations

from app.catalog_pick import llm_catalog_search
from app.orchestrator.clarify import looks_like_option_answer, original_user_ask
from app.orchestrator.state import CategoryResult, OrchestratorState
from app.search import SearchService


async def electronics_agent_node(state: OrchestratorState, search_service: SearchService) -> dict:
    query = state["query"]
    history = list(state.get("history") or [])
    items = [i for i in (state.get("items") or []) if i.category == "electronics"]
    limit = int(state.get("limit") or 5)

    if not items:
        return {
            "category_results": [
                CategoryResult(category="electronics", notes=["No electronics items in this request."])
            ]
        }

    # After clarification answers like "1", search the original camping/electronics ask.
    ask = original_user_ask(history, fallback=query)
    base_query = ask if (looks_like_option_answer(query) or len((query or "").strip()) <= 12) else query
    if ask and ask.strip().lower() != (base_query or "").strip().lower():
        # Keep both so use-case (camping/RV) + any follow-up detail survive.
        base_query = f"{ask}\n{query}".strip() if query and not looks_like_option_answer(query) else ask

    # Seed terms only help when the original ask is vague ("electronic devices").
    terms: list[str] = []
    for item in items:
        terms.extend(item.search_terms or [item.name])
    focused = " ".join(dict.fromkeys(terms))[:200]
    search_query = base_query
    if focused and focused.lower() not in (base_query or "").lower():
        search_query = f"{base_query}\nProducts of interest: {focused}"

    from app.orchestrator.clarify import clarification_from_raw

    decision = clarification_from_raw(state.get("clarification"))
    guidance = (decision.planning_guidance if decision else "") or ""
    if guidance and "power" in guidance.lower():
        search_query = f"{search_query}\nPlanning guidance: {guidance[:400]}"

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
