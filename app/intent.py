from __future__ import annotations

import re

# Queries about non-grocery categories — we should not search the food deals index.
NON_GROCERY_HINTS = re.compile(
    r"\b("
    r"stationery|stationary|school supplies|classroom supplies|"
    r"notebook|notebooks|pencil|pencils|crayon|crayons|"
    r"backpack|back pack|binder|binders|eraser|erasers|"
    r"marker|markers|textbook|textbooks|homework|"
    r"back to school|returning to school|returning back to school|"
    r"first grade|1st grade|second grade|2nd grade|"
    r"office supplies|printer paper|calculator|"
    r"clothing|clothes|shoes|uniform"
    r")\b",
    re.IGNORECASE,
)

GROCERY_HINTS = re.compile(
    r"\b("
    r"grocery|groceries|food|snack|snacks|lunch|lunchbox|"
    r"breakfast|cereal|milk|bread|meat|produce|"
    r"kroger|walmart|deal|deals|discount|sale|"
    r"party|dinner|taco|pbj|peanut butter|"
    r"soap|coffee|tea|household|paper towel|toilet paper"
    r")\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "a", "an", "the", "i", "my", "me", "we", "our", "you", "your",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "what", "which", "who", "whom", "where", "when", "why", "how",
    "to", "of", "in", "on", "at", "for", "with", "about", "from",
    "and", "or", "but", "not", "no", "yes",
    "need", "want", "get", "buy", "purchase", "find", "help",
    "items", "item", "list", "should", "could", "would", "can",
    "kid", "kids", "child", "children", "returning", "back",
}


def is_non_grocery_query(query: str) -> bool:
    """True when the user is clearly asking about something outside our grocery catalog."""
    if not NON_GROCERY_HINTS.search(query):
        return False
    # Allow mixed queries like "back to school lunch snacks"
    if GROCERY_HINTS.search(query):
        return False
    return True


def significant_tokens(query: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", query.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def results_match_query(query: str, title: str, keywords: str, description: str, category: str) -> bool:
    """Require overlap between query and ad text to avoid semantic false positives."""
    tokens = significant_tokens(query)
    if not tokens:
        return True
    haystack = f"{title} {keywords} {description} {category}".lower()
    return any(token in haystack for token in tokens)


def out_of_scope_reply(query: str) -> str:
    if NON_GROCERY_HINTS.search(query):
        return (
            "DealFinder specializes in **Kroger and Walmart grocery deals** — we don't have "
            "school supplies or stationery in our catalog.\n\n"
            "For **1st grade**, you'll typically need pencils, crayons, glue sticks, safety scissors, "
            "wide-rule notebooks, folders, and erasers. Check Walmart, Target, or office supply stores for those.\n\n"
            "I *can* compare grocery deals for **lunchbox snacks**, **after-school treats**, or **packed lunches**. "
            'Try: "Which store has cheaper lunchbox snacks?" or "PBJ party deals."'
        )

    return (
        f"I couldn't find grocery deals that match \"{query}\". "
        "DealFinder covers **food and household items** at Kroger and Walmart. "
        "Try a specific product like bread, milk, cereal, or soap."
    )
