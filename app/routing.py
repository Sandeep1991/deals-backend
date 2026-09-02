from __future__ import annotations

import re

# Route to multi-item store comparison when the query sounds like meal/event planning.
COMPARE_HINTS = re.compile(
    r"\b("
    r"which store|best store|where should i shop|compare|cheaper|cheapest|"
    r"party|meal|dinner|breakfast|lunch|brunch|"
    r"bbq|barbecue|cookout|grill|"
    r"taco|tacos|pizza night|pasta night|"
    r"pbj|pb&j|peanut butter|"
    r"milkshake|smoothie|"
    r"birthday|celebration|gathering|get together|"
    r"grocery list|shopping list|what do i need|ingredients|"
    r"throw a|host a|hosting|feed \d+|serve \d+|for \d+ people|for my kids"
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
