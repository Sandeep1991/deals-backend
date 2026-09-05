from __future__ import annotations

import re

# Meal / party / packing lists → LLM decompose + Kroger vs Walmart compare.
GROCERY_COMPARE_HINTS = re.compile(
    r"\b("
    r"which store|best store|where should i shop|where to buy|where can i buy|"
    r"compare|cheaper|cheapest|"
    r"party|meal|dinner|breakfast|lunch|brunch|recipe|recipes|ingredients|"
    r"bbq|barbecue|cookout|grill|"
    r"taco|tacos|pizza night|pasta night|latte|chai|smoothie|milkshake|"
    r"pbj|pb&j|peanut butter|"
    r"birthday|celebration|gathering|get together|"
    r"grocery list|shopping list|what do i need|"
    r"throw a|host(?:ing)?(?:\s+a)?|planning to host|feed \d+|serve \d+|"
    r"for \d+\s+(?:people|guests|friends|kids|persons)|"
    r"for \d+\s+of\s+my\s+(?:friends|guests|kids)|"
    r"for my (?:kids|friends|guests)|"
    r"pack essentials|weekend trip|"
    # Grocery packing only when not about gear/solar (see PRODUCT_SEARCH_HINTS priority)
    r"camp(?:ing)?\s+(?:food|groceries|snacks|essentials)|"
    r"kroger|walmart"
    r")\b",
    re.IGNORECASE,
)

# Electronics / affiliate catalog products → open AI Search (all merchants).
PRODUCT_SEARCH_HINTS = re.compile(
    r"\b("
    r"solar|charger|power\s*station|anker|solix|portable\s+power|"
    r"generator|battery\s+bank|inverter|power\s*bank|"
    r"rv\b|camper\b|van\s*life|"
    r"electronics|laptop|headphones|earbuds|smartwatch|"
    r"rakuten|affiliate"
    r")\b",
    re.IGNORECASE,
)


def should_compare(query: str, mode: str = "auto") -> bool:
    """Choose store-compare vs open catalog search.

    Compare only searches Kroger/Walmart. Affiliate ads (e.g. Anker Solix via
    Rakuten) live under other merchants, so product queries must use search.

    Product intent wins when both product and grocery hints appear
    (e.g. "portable solar for night camping in rvs").
    """
    mode = (mode or "auto").lower().strip()
    if mode == "search":
        return False
    if mode == "compare":
        return True

    q = query or ""
    product = bool(PRODUCT_SEARCH_HINTS.search(q))
    grocery = bool(GROCERY_COMPARE_HINTS.search(q))

    if product:
        return False
    if grocery:
        return True

    # Short product-style queries → full-catalog search; longer plans → compare.
    return len(q.split()) > 6
