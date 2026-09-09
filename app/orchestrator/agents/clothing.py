from __future__ import annotations

from app.orchestrator.state import CategoryResult, OrchestratorState
from app.search import SearchService


async def clothing_agent_node(state: OrchestratorState, search_service: SearchService) -> dict:
    items = [i for i in (state.get("items") or []) if i.category == "clothing"]
    limit = int(state.get("limit") or 5)
    if not items:
        return {
            "category_results": [
                CategoryResult(category="clothing", notes=["No clothing items in this request."])
            ]
        }

    ads = []
    notes: list[str] = []
    for item in items:
        term = item.search_terms[0] if item.search_terms else item.name
        hits = search_service.search(term, limit=limit)
        for hit in hits:
            blob = f"{hit.ad.title} {hit.ad.keywords} {hit.ad.category}".lower()
            if any(tok in blob for tok in ("cloth", "apparel", "shoe", "jacket", "shirt", "pant")):
                ads.append(hit.ad)
        if not hits:
            notes.append(f"No catalog deals for {item.name}.")

    if not ads:
        notes.append("Clothing catalog coverage is limited — no matching deals right now.")

    # Dedupe
    seen = set()
    unique = []
    for ad in ads:
        if ad.id in seen:
            continue
        seen.add(ad.id)
        unique.append(ad)

    fragment = ""
    if unique:
        top = unique[0]
        fragment = f"For clothing, {top.title} at {top.price} is the closest catalog match."
    elif notes:
        fragment = " ".join(notes)

    return {
        "category_results": [
            CategoryResult(
                category="clothing",
                items=items,
                ads=unique[:limit],
                notes=notes,
                reply_fragment=fragment,
            )
        ]
    }
