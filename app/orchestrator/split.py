from __future__ import annotations

from app.llm_client import LLMNotConfiguredError, complete_json, is_decompose_configured
from app.orchestrator.state import CompositeItem, OrchestratorState
from app.routing import PRODUCT_SEARCH_HINTS, should_compare

SPLIT_SYSTEM = """You split a shopper request into buyable items tagged by category.
Return JSON only:
{
  "event_summary": "short summary",
  "items": [
    {
      "name": "item name",
      "search_terms": ["term1"],
      "category": "grocery|electronics|clothing|stationery|other",
      "quantity": 1
    }
  ]
}

Category guide:
- grocery: food, drinks, household consumables, snacks, paper towels, trash bags
- electronics: solar, power stations, chargers, batteries, Anker/Solix, gadgets, electronic devices
- clothing: apparel, shoes, jackets
- stationery: notebooks, pencils, school/office supplies
- other: anything else (tents, sleeping bags, specialty gear we cannot price at grocery)

Rules:
- quantity = packages/units to buy, NOT guest count.
- Emit CONCRETE products, never category blobs like "Snacks", "Food", "Essentials", or "Electronics".
- For night RV/camping power / lots of devices → electronics (portable power station + panel if useful).
- For camping/weekend/family trips → also emit several grocery consumables:
  bottled water, trail mix or snacks, trash bags, paper towels (and sunscreen/bug spray if relevant).
  Do NOT put tents/sleeping bags under grocery.
- Mixed queries must emit items in multiple categories.
- search_terms: 1-3 short supermarket/catalog phrases.
- Never invent brands unless the user named them.
- If prior conversation is provided, treat follow-ups relative to that context
  (e.g. "cheaper one", "add drinks", "what about Walmart")."""


def _history_prompt(state: OrchestratorState, query: str) -> str:
    from app.orchestrator.memory import format_history_block

    history = list(state.get("history") or [])
    block = format_history_block(history)
    if block:
        return f"{block}\n\nCurrent user request: {query}"
    return f"User request: {query}"


def _heuristic_split(query: str) -> tuple[str, list[CompositeItem]]:
    q = query.lower()
    items: list[CompositeItem] = []
    summary = query.strip()[:120] or "Shopping request"

    if PRODUCT_SEARCH_HINTS.search(query):
        if any(t in q for t in ("solar", "panel")):
            items.append(
                CompositeItem(
                    name="portable solar panel",
                    search_terms=["portable solar panel", "PS100"],
                    category="electronics",
                )
            )
        if any(t in q for t in ("power station", "battery", "charger", "night", "rv", "camping")):
            items.append(
                CompositeItem(
                    name="portable power station",
                    search_terms=["portable power station", "C1000"],
                    category="electronics",
                )
            )
        if not items:
            items.append(
                CompositeItem(
                    name=query.strip()[:80] or "product",
                    search_terms=[query.strip()[:80]],
                    category="electronics",
                )
            )

    grocery_tokens = (
        "snack",
        "trail mix",
        "food",
        "water",
        "grocery",
        "chai",
        "latte",
        "taco",
        "party",
        "camp",
        "camping",
        "weekend",
        "family",
        "pack",
    )
    if any(t in q for t in grocery_tokens):
        if "trail mix" in q or "snack" in q:
            items.append(
                CompositeItem(name="trail mix", search_terms=["trail mix", "snacks"], category="grocery")
            )
        if "water" in q:
            items.append(
                CompositeItem(name="bottled water", search_terms=["bottled water"], category="grocery")
            )
        # Camping/weekend packing → seed a real grocery list (agent may expand further)
        if any(t in q for t in ("camp", "camping", "weekend trip", "pack essentials", "family")):
            for name, terms in (
                ("bottled water", ["bottled water", "water bottles"]),
                ("trail mix", ["trail mix", "snacks"]),
                ("trash bags", ["trash bags"]),
                ("paper towels", ["paper towels"]),
            ):
                if not any(i.name == name and i.category == "grocery" for i in items):
                    items.append(CompositeItem(name=name, search_terms=terms, category="grocery"))
        if should_compare(query, "auto") and not any(i.category == "grocery" for i in items):
            # Fall back to treating the whole query as grocery planning
            items.append(
                CompositeItem(name=query.strip()[:80], search_terms=[query.strip()[:80]], category="grocery")
            )

    if any(t in q for t in ("notebook", "pencil", "stationery", "school supply")):
        items.append(
            CompositeItem(name="notebooks", search_terms=["notebooks"], category="stationery")
        )

    if any(t in q for t in ("jacket", "shoes", "clothing", "shirt")):
        items.append(
            CompositeItem(name="clothing", search_terms=[query.strip()[:60]], category="clothing")
        )

    if not items:
        # Default: grocery-style plan for longer queries, else other
        cat = "grocery" if should_compare(query, "auto") else "other"
        items.append(
            CompositeItem(name=query.strip()[:80] or "items", search_terms=[query.strip()[:80]], category=cat)
        )

    return summary, items


async def split_query_node(state: OrchestratorState) -> dict:
    query = state["query"]
    if is_decompose_configured():
        try:
            data = await complete_json(
                SPLIT_SYSTEM,
                _history_prompt(state, query),
                max_tokens=1200,
            )
            raw_items = data.get("items") or []
            items: list[CompositeItem] = []
            for raw in raw_items:
                try:
                    item = CompositeItem.model_validate(raw)
                    if item.category not in {"grocery", "electronics", "clothing", "stationery", "other"}:
                        item.category = "other"
                    items.append(item)
                except Exception:
                    continue
            summary = str(data.get("event_summary") or query).strip() or query
            if items:
                return {"event_summary": summary, "items": items}
        except (LLMNotConfiguredError, Exception):
            pass

    summary, items = _heuristic_split(query)
    return {"event_summary": summary, "items": items}
