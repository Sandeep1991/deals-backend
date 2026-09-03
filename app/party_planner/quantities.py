from __future__ import annotations

import math
import re
from typing import Optional

from app.party_planner.state import ShoppingItem, ShoppingPlan

PEOPLE_RE = re.compile(
    r"\b(?:for|feed|serve|serves)\s+(\d+)\s+(?:people|guests|persons|kids|adults)\b"
    r"|\b(\d+)\s+(?:people|guests|persons)\b",
    re.IGNORECASE,
)

PACK_COUNT_RE = re.compile(
    r"\b(\d+)\s*(?:-?\s*)?(?:ct|count|counts|pk|pack|packs)\b",
    re.IGNORECASE,
)

BAG_UNIT_RE = re.compile(r"\b(?:\d+\s*(?:lb|oz))?\s*bag\b|\bbunch\b", re.IGNORECASE)

# Items typically bought as one package for a small group meal
SHAREABLE_RE = re.compile(
    r"\b("
    r"taco shells?|shells?|tortillas?|cheese|lettuce|beans|corn|sour cream|salsa|"
    r"seasoning|onions?|pepper|tomatoes|zucchini|squash|bread|peanut butter|jelly|"
    r"ice cream|milk|syrup|ground beef|chicken|rice|pasta"
    r")\b",
    re.IGNORECASE,
)

# Items where each guest often gets their own unit
INDIVIDUAL_RE = re.compile(
    r"\b("
    r"jarritos|soda|beverage|drink|drinks|water bottle|juice box|beer|seltzer|"
    r"smoothie|shake"
    r")\b",
    re.IGNORECASE,
)

# Produce sold by weight — quantity means pounds, not package count
PER_POUND_RE = re.compile(
    r"\b(bell peppers?|peppers?|tomatoes?|onions?|zucchini|squash|grapes?|apples?)\b",
    re.IGNORECASE,
)


def infer_people_count(query: str, plan: ShoppingPlan) -> Optional[int]:
    if plan.people_count and plan.people_count > 0:
        return plan.people_count
    match = PEOPLE_RE.search(query) or PEOPLE_RE.search(plan.event_summary)
    if match:
        return int(next(g for g in match.groups() if g))
    return None


def parse_pack_count(text: str) -> Optional[int]:
    match = PACK_COUNT_RE.search(text)
    if match:
        return int(match.group(1))
    return None


def is_individual_item(name: str) -> bool:
    return bool(INDIVIDUAL_RE.search(name))


def is_shareable_item(name: str) -> bool:
    return bool(SHAREABLE_RE.search(name))


def is_per_pound_item(name: str) -> bool:
    return bool(PER_POUND_RE.search(name))


def is_sold_by_weight(ad_price: str, ad_title: str) -> bool:
    if "/lb" in ad_price.lower() or "per lb" in ad_price.lower():
        return True
    if BAG_UNIT_RE.search(ad_title):
        return False
    return is_per_pound_item(ad_title)


def default_pounds_for_people(name: str, people: int) -> float:
    """Rough produce weight for a meal serving `people` guests."""
    if people <= 4:
        return 1.0
    if people <= 6:
        return 1.5
    if people <= 10:
        return 2.0
    return math.ceil(people / 4)


def packages_for_ad(item: ShoppingItem, ad_title: str, ad_price: str, people: Optional[int]) -> float:
    """
    How many store units (boxes, bags, cans) to buy.
    Uses pack size from ad title (e.g. 12 ct) and guest count when available.
    """
    name = item.name
    title = ad_title
    pack = parse_pack_count(title)

    # Individual drinks: one per guest
    if people and is_individual_item(name):
        return float(max(1, people))

    # Taco shells / buns: use pack count (12 ct box feeds ~6 at 2 tacos each)
    if people and pack and re.search(r"shell|tortilla|bun", name, re.I):
        per_guest = 2 if "shell" in name.lower() or "tortilla" in name.lower() else 1
        return float(max(1, math.ceil(people * per_guest / pack)))

    # Per-pound produce sold loose
    if is_sold_by_weight(ad_price, title):
        if people:
            return default_pounds_for_people(name, people)
        return min(item.quantity, 3.0) if item.quantity > 1 else 1.0

    # Pre-packaged bag (e.g. 3 lb onion bag) — one unit covers the meal
    if BAG_UNIT_RE.search(title):
        return 2.0 if people and people > 8 else 1.0

    # Shareable pantry/produce: one package is usually enough for ≤8 guests
    if is_shareable_item(name):
        if people and people > 8:
            return 2.0
        return 1.0

    # Fallback: trust LLM but never assume 1 unit per guest for shareables
    if people and item.quantity == people and is_shareable_item(name):
        return 1.0

    return max(1.0, min(item.quantity, 4.0))


def normalize_plan_quantities(plan: ShoppingPlan, query: str) -> ShoppingPlan:
    """Fix LLM setting quantity = guest count for shareable multi-packs."""
    people = infer_people_count(query, plan)
    plan.people_count = people

    def fix_item(item: ShoppingItem) -> ShoppingItem:
        if people is None:
            if item.quantity > 4:
                item.quantity = 1.0
            return item

        if is_individual_item(item.name):
            item.quantity = float(people)
            return item

        # Onions are often sold as a bag; default to one package unless clearly by weight
        if is_per_pound_item(item.name) and not re.search(r"\bonions?\b", item.name, re.I):
            item.quantity = default_pounds_for_people(item.name, people)
            return item

        if re.search(r"\bonions?\b", item.name, re.I):
            item.quantity = 1.0 if people <= 8 else 2.0
            return item

        if is_shareable_item(item.name) and item.quantity >= people:
            # LLM often sets quantity = guest count; one 12ct box ≠ 6 boxes
            item.quantity = 1.0 if people <= 8 else 2.0
            return item

        if item.quantity > 4:
            item.quantity = 1.0
        return item

    plan.required_items = [fix_item(i) for i in plan.required_items]
    for option in plan.alternative_options:
        option.items = [fix_item(i) for i in option.items]

    return plan
