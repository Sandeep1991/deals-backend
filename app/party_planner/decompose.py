from __future__ import annotations

import re

from app.party_planner.state import AlternativeOption, ShoppingItem, ShoppingPlan

DECOMPOSE_SYSTEM = """You are a grocery shopping planner. Break the user's request into grocery items to buy at a store.

Return JSON with this exact shape:
{
  "event_summary": "short summary of the event or meal",
  "required_items": [
    {"name": "item name", "search_terms": ["term1", "term2"], "quantity": 1}
  ],
  "alternative_options": [
    {
      "label": "optional group name",
      "items": [
        {"name": "item name", "search_terms": ["term1"], "quantity": 1}
      ]
    }
  ]
}

Rules:
- required_items: every ingredient or product needed for the request.
- alternative_options: OR choices when multiple approaches work (same label = pick cheapest per store).
  Example: milkshake from ice cream OR from milk + chocolate syrup — two entries with label "milkshake".
- search_terms: 1-3 simple grocery terms that would match supermarket ads (brand names OK).
- quantity: estimate servings/items needed (default 1).
- Think step by step: meals → proteins, carbs, produce, dairy, condiments, drinks, etc.
- Only include grocery/pantry items, not utensils or decorations unless explicitly requested.

Examples:
- "PBJ party" → bread, peanut butter, jelly
- "taco night for 4" → ground beef or chicken, taco shells, shredded cheese, salsa, sour cream, lettuce
- "BBQ for 6" → hot dogs or burgers, buns, condiments, chips, soda
- "chocolate milkshake" as add-on → alternative: chocolate ice cream OR milk + Hershey/chocolate syrup"""


def heuristic_decompose(query: str) -> ShoppingPlan:
    """Legacy fallback when LLM is unavailable or fails. Limited to known patterns."""
    q = query.lower()
    required: list[ShoppingItem] = []
    alternatives: list[AlternativeOption] = []

    if any(token in q for token in ("pbj", "peanut butter", "pb&j", "pb j")):
        required.extend(
            [
                ShoppingItem(name="bread", search_terms=["bread", "white bread"]),
                ShoppingItem(name="peanut butter", search_terms=["peanut butter"]),
                ShoppingItem(name="jelly", search_terms=["jelly", "grape jelly", "strawberry jelly"]),
            ]
        )

    if any(token in q for token in ("taco", "tacos", "taco night")):
        required.extend(
            [
                ShoppingItem(name="ground beef", search_terms=["ground beef", "taco meat"]),
                ShoppingItem(name="taco shells", search_terms=["taco shells", "tortillas"]),
                ShoppingItem(name="shredded cheese", search_terms=["shredded cheese", "cheddar"]),
                ShoppingItem(name="salsa", search_terms=["salsa"]),
                ShoppingItem(name="sour cream", search_terms=["sour cream"]),
            ]
        )

    if any(token in q for token in ("bbq", "barbecue", "cookout", "grill out")):
        required.extend(
            [
                ShoppingItem(name="hot dogs", search_terms=["hot dogs", "burgers"]),
                ShoppingItem(name="buns", search_terms=["hot dog buns", "hamburger buns"]),
                ShoppingItem(name="ketchup", search_terms=["ketchup", "mustard"]),
                ShoppingItem(name="chips", search_terms=["potato chips"]),
                ShoppingItem(name="soda", search_terms=["soda", "cola"]),
            ]
        )

    if any(token in q for token in ("milkshake", "milk shake", "chocolate milk")):
        alternatives.extend(
            [
                AlternativeOption(
                    label="milkshake",
                    items=[ShoppingItem(name="chocolate ice cream", search_terms=["chocolate ice cream"])],
                ),
                AlternativeOption(
                    label="milkshake",
                    items=[
                        ShoppingItem(name="milk", search_terms=["milk", "whole milk"]),
                        ShoppingItem(name="chocolate syrup", search_terms=["hershey syrup", "chocolate syrup"]),
                    ],
                ),
            ]
        )

    if not required and not alternatives:
        words = re.findall(r"[a-zA-Z]+", q)
        stop = {
            "i", "want", "to", "throw", "a", "for", "my", "kids", "and", "maybe", "some", "the",
            "which", "store", "has", "great", "deals", "party", "with", "or", "should", "be",
            "made", "from", "can", "it", "possible", "need", "what", "host", "hosting",
        }
        terms = [w for w in words if w not in stop and len(w) > 2]
        for term in terms[:6]:
            required.append(ShoppingItem(name=term, search_terms=[term]))

    summary = query.strip()[:120] or "Shopping list"
    return ShoppingPlan(
        event_summary=summary,
        required_items=required,
        alternative_options=alternatives,
    )
