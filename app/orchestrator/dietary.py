from __future__ import annotations

"""Helpers for grocery list extraction / search-term shaping.

Preference detection itself lives in preferences.py (LangChain summary memory).
"""

import re

from app.orchestrator.state import ChatTurn, CompositeItem

_LIST_LINE_RE = re.compile(
    r"^\s*(?:[-*]|\d+[.)])\s+(.+?)(?:\s*[—\-–:]\s*.*)?$",
    re.M,
)

_NON_FOOD = {
    "trash bags",
    "paper towels",
    "batteries",
    "matches",
    "lighter",
    "sunscreen",
    "insect repellent",
    "bug spray",
}


def _clean_list_item(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip(" -–—:")
    text = re.split(r"\s+[—\-–]\s+|:\s*\$", text, maxsplit=1)[0].strip()
    if len(text) < 2 or len(text) > 80:
        return ""
    if text.lower().startswith(("kroger", "walmart", "estimated", "recommendation", "shopping list")):
        return ""
    return text


def extract_prior_grocery_names(history: list[ChatTurn] | None) -> list[str]:
    """Pull concrete item names from prior assistant shopping-list bullets."""
    if not history:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for turn in reversed(history):
        if turn.role != "assistant":
            continue
        content = turn.content or ""
        section = content
        m = re.search(
            r"shopping list[:\*\s]*([\s\S]+?)(?=\n\s*\n\s*\*\*store|\n\s*store comparison|\n\s*###|\Z)",
            content,
            re.I,
        )
        if m:
            section = m.group(1)
        for line in section.splitlines():
            match = _LIST_LINE_RE.match(line.strip()) if line.strip() else None
            if not match:
                match = re.match(r"^(?:[-*]|\d+[.)])\s+(.+)$", line.strip())
            if not match:
                continue
            name = _clean_list_item(match.group(1))
            key = name.lower()
            if not name or key in seen:
                continue
            if len(name.split()) > 8:
                continue
            seen.add(key)
            names.append(name)
        if names:
            break
    return names


def constrain_search_terms(name: str, preferences: list[str]) -> list[str]:
    """Build search phrases. When preferences exist, do NOT fall back to bare
    unconstrained terms — that caused 'organic peanut butter' to price as regular Jif.
    """
    base = name.strip()
    if not base:
        return []
    lower = base.lower()
    if lower in _NON_FOOD:
        return [base]

    prefs = [p.strip() for p in preferences if (p or "").strip()]
    # Strip preference prefixes already on the name
    clean = base
    for pref in prefs:
        if clean.lower().startswith(pref.lower() + " "):
            clean = clean[len(pref) :].strip()

    if not prefs:
        return [clean]

    terms: list[str] = []
    for pref in prefs[:2]:
        if pref.lower() in clean.lower():
            candidate = clean
        else:
            candidate = f"{pref} {clean}"
        if candidate.lower() not in {t.lower() for t in terms}:
            terms.append(candidate)
    return terms[:3]


def preference_tokens_from_item(name: str, preferences: list[str] | None = None) -> list[str]:
    """Tokens that a priced ad must include when the shopper asked for preferences."""
    tokens: list[str] = []
    for pref in preferences or []:
        p = (pref or "").strip().lower()
        if p:
            tokens.append(p)
    # Also detect preference words already baked into the item name
    lower = (name or "").lower()
    for hint in (
        "organic",
        "vegan",
        "gluten-free",
        "gluten free",
        "dairy-free",
        "dairy free",
        "keto",
        "sugar-free",
        "halal",
        "kosher",
        "non-gmo",
    ):
        if hint in lower and hint not in tokens:
            tokens.append(hint)
    return tokens


def items_from_prior_list(
    names: list[str],
    preferences: list[str],
) -> list[CompositeItem]:
    items: list[CompositeItem] = []
    for name in names:
        clean = name
        for pref in preferences:
            p = (pref or "").strip()
            if p and clean.lower().startswith(p.lower() + " "):
                clean = clean[len(p) :].strip()
        terms = constrain_search_terms(clean, preferences)
        label = clean
        if preferences and clean.lower() not in _NON_FOOD:
            label = f"{preferences[0]} {clean}"
        items.append(
            CompositeItem(
                name=label,
                search_terms=terms or [clean],
                category="grocery",
                quantity=1.0,
            )
        )
    return items
