from __future__ import annotations

import re

from app.orchestrator.state import ChatTurn, CompositeItem

# Follow-ups that rewrite an existing grocery list rather than naming a new SKU.
DIETARY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("organic", re.compile(r"\borganic\b", re.I)),
    ("vegan", re.compile(r"\bvegan\b|\bplant[- ]based\b", re.I)),
    ("gluten-free", re.compile(r"\bgluten[- ]free\b|\bglutenfree\b|\bno gluten\b", re.I)),
    ("dairy-free", re.compile(r"\bdairy[- ]free\b|\blactose[- ]free\b|\bnon[- ]dairy\b", re.I)),
    ("keto", re.compile(r"\bketo\b|\bketogenic\b|\blow[- ]carb\b", re.I)),
    ("sugar-free", re.compile(r"\bsugar[- ]free\b|\bno sugar\b", re.I)),
    ("non-GMO", re.compile(r"\bnon[- ]gmo\b|\bno gmo\b", re.I)),
    ("halal", re.compile(r"\bhalal\b", re.I)),
    ("kosher", re.compile(r"\bkosher\b", re.I)),
]

_FOLLOW_UP_RE = re.compile(
    r"\b("
    r"is (?:this|these|it|that)|are (?:these|they|those)|"
    r"considering|instead|swap|replace|make (?:it|them)|"
    r"can (?:we|i|you)|do you have|what about|prefer|only\b"
    r")",
    re.I,
)

_LIST_LINE_RE = re.compile(
    r"^\s*(?:[-*]|\d+[.)])\s+(.+?)(?:\s*[—\-–:]\s*.*)?$",
    re.M,
)

# Household / non-food — usually keep unless user asked for eco variants.
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


def detect_dietary_constraints(text: str) -> list[str]:
    found: list[str] = []
    for label, pattern in DIETARY_PATTERNS:
        if pattern.search(text or ""):
            found.append(label)
    return found


def is_dietary_follow_up(query: str, history: list[ChatTurn] | None) -> bool:
    """True when the user is refining a prior grocery list with diet constraints."""
    constraints = detect_dietary_constraints(query)
    if not constraints:
        return False
    if not history:
        # Still treat pure constraint asks as follow-ups if phrased as questions
        return bool(_FOLLOW_UP_RE.search(query or "")) or query.strip().endswith("?")
    # Any prior user/assistant grocery-ish context
    blob = " ".join(t.content for t in history).lower()
    grocery_hints = (
        "shopping list",
        "kroger",
        "walmart",
        "trail mix",
        "bottled water",
        "grocery",
        "camping",
        "taco",
        "party",
        "ingredients",
        "est. total",
        "compare",
    )
    return any(h in blob for h in grocery_hints) or bool(_FOLLOW_UP_RE.search(query or ""))


def _clean_list_item(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip(" -–—:")
    # Drop store/price suffixes
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
                # Also accept bare "- item" without re.M anchors issues
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


def constrain_search_terms(name: str, constraints: list[str]) -> list[str]:
    base = name.strip()
    if not base:
        return []
    lower = base.lower()
    # Household non-food: don't force "organic trash bags" etc.
    if lower in _NON_FOOD:
        return [base]
    terms = [base]
    for c in constraints:
        prefixed = f"{c} {base}"
        if prefixed.lower() not in {t.lower() for t in terms}:
            terms.insert(0, prefixed)
    return terms[:3]


def items_from_prior_list(
    names: list[str],
    constraints: list[str],
) -> list[CompositeItem]:
    items: list[CompositeItem] = []
    for name in names:
        terms = constrain_search_terms(name, constraints)
        label = name
        if constraints and name.lower() not in _NON_FOOD:
            label = f"{constraints[0]} {name}"
        items.append(
            CompositeItem(
                name=label,
                search_terms=terms or [name],
                category="grocery",
                quantity=1.0,
            )
        )
    return items


def dietary_rewrite_prompt(
    query: str,
    prior_names: list[str],
    constraints: list[str],
    history_block: str = "",
) -> str:
    constraint_txt = ", ".join(constraints) if constraints else "dietary preference"
    prior_txt = ", ".join(prior_names) if prior_names else "(see prior conversation)"
    return (
        f"{history_block}\n\n".lstrip()
        + f"Current user request: {query.strip()}\n\n"
        "Rewrite the EXISTING grocery shopping list to satisfy dietary constraints. "
        f"Constraints: {constraint_txt}.\n"
        f"Prior items to replace/keep: {prior_txt}.\n\n"
        "Rules:\n"
        f"- Prefer {constraint_txt} versions of food/drink items sold at Kroger or Walmart.\n"
        "- search_terms MUST include the constraint where it applies "
        '(example: "organic trail mix", "gluten-free bread", "vegan cheese").\n'
        "- Keep household non-food items (trash bags, paper towels) unless the user asked to change them.\n"
        "- If a perfect match is unlikely, choose the closest common supermarket substitute "
        "and keep the item on the list (do not drop the need).\n"
        "- Return a FULL required_items list for a fresh Kroger vs Walmart price compare.\n"
        "- event_summary should mention the dietary rewrite (e.g. 'Camping staples — organic options').\n"
        "- Never use the user's question sentence as an item name.\n"
    )
