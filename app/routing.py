from __future__ import annotations

import re

# Route to multi-item store comparison when the query sounds like meal/event planning.
COMPARE_HINTS = re.compile(
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
    r"for my (?:kids|friends|guests)"
    r")\b",
    re.IGNORECASE,
)


def should_compare(query: str, mode: str = "auto") -> bool:
    mode = (mode or "auto").lower().strip()
    if mode == "compare":
        return True
    if mode == "search":
        return False
    return bool(COMPARE_HINTS.search(query))
