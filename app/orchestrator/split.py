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
- Ambiguous finished foods for events (bring cupcakes/cookies/cake/pizza/etc. without saying
  ready-made vs bake): emit ONE coarse grocery item for the finished food name only
  (e.g. "cupcakes"). Do NOT expand into mix, liners, frosting, or sprinkles yet —
  a later clarification step will choose the path.
- For night RV/camping power / lots of devices → electronics (portable power station + panel if useful).
- For camping/weekend/family trips → also emit several grocery consumables:
  bottled water, trail mix or snacks, trash bags, paper towels (and sunscreen/bug spray if relevant).
  Do NOT put tents/sleeping bags under grocery.
- Mixed queries must emit items in multiple categories.
- search_terms: 1-3 short supermarket/catalog phrases.
- Never invent brands unless the user named them.
- If prior conversation / preference summary is provided, treat follow-ups relative to that context
  (e.g. "cheaper one", "add drinks", dietary rewrites like organic/vegan).
- Preference / list-rewrite follow-ups:
  do NOT emit the question as an item name.
  Re-emit PRIOR grocery items with search_terms that reflect the preference summary.
  Keep trash bags/paper towels unless the user asked to change them.
  category stays grocery."""


def _history_prompt(state: OrchestratorState, query: str) -> str:
    from app.orchestrator.memory import format_history_block
    from app.orchestrator.preferences import preference_summary_from_raw

    history = list(state.get("history") or [])
    block = format_history_block(history)
    pref = preference_summary_from_raw(state.get("preference_summary"))
    parts: list[str] = []
    if pref and (pref.summary or pref.preferences):
        parts.append(f"Preference summary: {pref.summary}")
        if pref.preferences:
            parts.append(f"Active preferences: {', '.join(pref.preferences)}")
    if block:
        parts.append(block)
    parts.append(f"Current user request: {query}")
    return "\n\n".join(parts)


def _preference_follow_up_items(state: OrchestratorState) -> list[CompositeItem] | None:
    from app.orchestrator.preferences import (
        items_from_preference_summary,
        preference_summary_from_raw,
    )

    pref = preference_summary_from_raw(state.get("preference_summary"))
    if not pref or not pref.is_list_rewrite:
        return None
    items = items_from_preference_summary(pref)
    return items or None


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
    history = list(state.get("history") or [])

    # LangChain-style preference summary (rolling ConversationSummary memory)
    from app.orchestrator.dietary import extract_prior_grocery_names
    from app.orchestrator.preferences import (
        preference_summary_from_raw,
        summarize_preferences,
    )

    extracted = extract_prior_grocery_names(history)
    prior_pref = preference_summary_from_raw(state.get("preference_summary"))
    preference_summary = await summarize_preferences(
        query=query,
        history=history,
        prior=prior_pref,
        chat_id=str(state.get("chat_id") or ""),
        extracted_items=extracted,
    )
    # Stash on state for follow-up helpers in this node
    state_with_pref = {**state, "preference_summary": preference_summary.model_dump()}

    preference_items = _preference_follow_up_items(state_with_pref)  # type: ignore[arg-type]
    from app.orchestrator.clarify import looks_like_option_answer

    if preference_items and not looks_like_option_answer(query):
        return {
            "event_summary": f"Revised grocery list ({preference_summary.label()})",
            "items": preference_items,
            "preference_summary": preference_summary.model_dump(),
        }

    # Bare clarification answers ("2.") — leave items empty; clarify_intent will seed.
    if looks_like_option_answer(query):
        return {
            "event_summary": query.strip()[:120] or "Clarification follow-up",
            "items": [],
            "preference_summary": preference_summary.model_dump(),
        }

    if is_decompose_configured():
        try:
            data = await complete_json(
                SPLIT_SYSTEM,
                _history_prompt(state_with_pref, query),  # type: ignore[arg-type]
                max_tokens=1200,
            )
            raw_items = data.get("items") or []
            items: list[CompositeItem] = []
            for raw in raw_items:
                try:
                    item = CompositeItem.model_validate(raw)
                    if item.category not in {"grocery", "electronics", "clothing", "stationery", "other"}:
                        item.category = "other"
                    if item.category == "grocery" and item.name.strip().lower() == query.strip().lower():
                        continue
                    if item.category == "grocery" and "?" in item.name:
                        continue
                    items.append(item)
                except Exception:
                    continue
            summary = str(data.get("event_summary") or query).strip() or query
            if items:
                return {
                    "event_summary": summary,
                    "items": items,
                    "preference_summary": preference_summary.model_dump(),
                }
        except (LLMNotConfiguredError, Exception):
            pass

    summary, items = _heuristic_split(query)
    return {
        "event_summary": summary,
        "items": items,
        "preference_summary": preference_summary.model_dump(),
    }
