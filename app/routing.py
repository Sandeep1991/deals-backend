from __future__ import annotations


def should_compare(query: str, mode: str = "auto") -> bool:
    """Route chat through LLM decompose → AI Search (store compare).

    Auto/compare always use the planner so natural-language requests (camping,
    parties, recipes) never dump the raw sentence into keyword search.
    Explicit mode=search keeps the legacy single-query search path.
    """
    _ = query  # reserved for future intent overrides
    mode = (mode or "auto").lower().strip()
    if mode == "search":
        return False
    return True
