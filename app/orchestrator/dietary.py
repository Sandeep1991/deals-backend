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
    """Prefix food items with free-form preference tags from the summary."""
    base = name.strip()
    if not base:
        return []
    lower = base.lower()
    if lower in _NON_FOOD:
        return [base]
    terms = [base]
    for pref in preferences:
        tag = (pref or "").strip()
        if not tag:
            continue
        # Avoid duplicating if name already contains the preference
        if tag.lower() in lower:
            continue
        prefixed = f"{tag} {base}"
        if prefixed.lower() not in {t.lower() for t in terms}:
            terms.insert(0, prefixed)
    return terms[:3]


def items_from_prior_list(
    names: list[str],
    preferences: list[str],
) -> list[CompositeItem]:
    items: list[CompositeItem] = []
    for name in names:
        # Strip previous preference prefixes when re-applying
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
