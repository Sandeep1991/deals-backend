from __future__ import annotations

import re

from app.config import Settings, get_settings
from app.llm_client import LLMNotConfiguredError, complete_json, complete_text, is_decompose_configured
from app.models import SearchResultItem
from app.pricing import parse_price
from app.replies import build_template_reply, generate_reply
from app.search import SearchService

PLAN_SYSTEM = """You turn a shopper request into catalog search queries for affiliate/product ads.
Return JSON only:
{
  "intent": "one sentence of what they want",
  "use_case": "home_backup|portable_outdoor|solar_panels|ev_charging|rv_camping|general",
  "size_hint": "phone_powerbank|day_hike|weekend_camping|rv_overnight|apartment|family_home|whole_home",
  "search_terms": ["term1", "term2", "term3"],
  "avoid_terms": ["optional words that indicate wrong products"]
}

Rules for search_terms (1-4 short phrases that match product titles):
- portable / hike / backpacking / camping / RV overnight → "portable power station", "C1000", "PS100"
  (for night use prioritize power station; panels alone do not help at night)
  (NOT whole-home F3800 kits, NOT "4x 400W", NOT EV adapters)
- family / home backup / house solar → "solar panel", "home backup", "power station solar panel"
- solar panels only → "solar panel", "portable solar panel"
- Do not invent brands unless the user named them.
- Prefer concrete product-class terms over repeating the full user sentence."""

PICK_SYSTEM = """You pick deals that match THIS user's use case.
Return JSON only:
{
  "selected_ids": ["id1", "id2"]
}

Hard rules:
- selected_ids: best first, max 5, ONLY from the candidate list. No reply field.
- Obey use_case and size_hint strictly:
  - portable_outdoor / day_hike / weekend_camping / RV: prefer portable power stations + portable panels
    (C1000, C2000, PS100/PS200). For night use, a power station matters more than a panel alone.
    REJECT whole-home kits (F3000/F3800, multi-kWh expansion + 4× rigid panels) unless nothing smaller exists.
  - family_home / whole_home / home_backup: larger home backup + panel kits are OK.
- Prefer real positive prices; skip $0/blank when priced alternatives exist.
- Never invent ids."""

WRITE_REPLY_SYSTEM = """You are DealFinder — a sharp shopping advisor, not a product brochure.
Write 3-5 sentences that actually answer the user's question using ONLY the selected deals.

Rules:
- Lead with advice for THEIR scenario, then name products and prices as supporting evidence.
- Example: for "solar for night camping in an RV", explain that panels charge by day and a portable
  power station covers night loads — then recommend a concrete station (+ panel if useful).
- Sound like a knowledgeable friend. Vary wording. Never use: "I found N deals", "best match is",
  "click any deal card", "compact and efficient choice", "outdoor adventures", "maximize energy".
- Do not invent products, prices, or specs. Do not include markdown links or URLs.
- If the selected gear is a poor fit, say so and suggest the best available option from the list."""

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BARE_AFFILIATE_RE = re.compile(r"https?://click\.linksynergy\.com/\S+")


def _attach_tracking_links(reply: str, selected: list[SearchResultItem]) -> str:
    """Weave one trusted tracking link into the LLM reply without a canned 'Shop:' footer."""
    text = (reply or "").strip()
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _BARE_AFFILIATE_RE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text).strip()

    if not selected:
        return text

    best = selected[0].ad
    linked = f"[{best.title}]({best.url})"
    if best.title in text:
        return text.replace(best.title, linked, 1)
    # Fallback: link the first priced title that appears in the prose
    for item in selected:
        if item.ad.title in text:
            return text.replace(item.ad.title, f"[{item.ad.title}]({item.ad.url})", 1)
    return f"{text} See {linked}."

# Title cues used to demote/promote before the LLM sees the list.
_HOME_SCALE = (
    "f3800", "f3000", "whole home", "home backup", "expansion battery",
    "4×", "4x", "400w solar", "smart home power",
)
_PORTABLE_SCALE = (
    "c1000", "c2000", "c800", "c300", "ps100", "ps200", "ps400",
    "portable", "briefcase", "backpack",
)


def _title_blob(item: SearchResultItem) -> str:
    ad = item.ad
    return f"{ad.title} {ad.description} {ad.keywords}".lower()


def _looks_home_scale(item: SearchResultItem) -> bool:
    blob = _title_blob(item)
    return any(tok in blob for tok in _HOME_SCALE)


def _looks_portable_scale(item: SearchResultItem) -> bool:
    blob = _title_blob(item)
    return any(tok in blob for tok in _PORTABLE_SCALE)


def _has_real_price(item: SearchResultItem) -> bool:
    amount = parse_price(item.ad.price)
    return amount is not None and amount > 0


def _reorder_for_use_case(
    candidates: list[SearchResultItem],
    use_case: str,
    size_hint: str,
) -> list[SearchResultItem]:
    """Surface intent-fitting SKUs first so the LLM is less likely to latch onto megakits."""
    portable_intent = use_case in {"portable_outdoor", "rv_camping"} or size_hint in {
        "phone_powerbank",
        "day_hike",
        "weekend_camping",
        "rv_overnight",
    }
    home_intent = use_case in {"home_backup", "solar_panels"} or size_hint in {
        "apartment",
        "family_home",
        "whole_home",
    }

    def sort_key(item: SearchResultItem) -> tuple:
        priced = 0 if _has_real_price(item) else 1
        if portable_intent:
            fit = 0 if _looks_portable_scale(item) and not _looks_home_scale(item) else (
                1 if _looks_portable_scale(item) else (3 if _looks_home_scale(item) else 2)
            )
        elif home_intent:
            fit = 0 if _looks_home_scale(item) or "solar panel" in _title_blob(item) else (
                1 if _looks_portable_scale(item) else 2
            )
        else:
            fit = 1
        return (fit, priced)

    return sorted(candidates, key=sort_key)


async def plan_search_terms(
    query: str,
    settings: Settings | None = None,
) -> tuple[str, str, str, list[str]]:
    settings = settings or get_settings()
    if is_decompose_configured(settings):
        try:
            data = await complete_json(
                PLAN_SYSTEM,
                f"User request: {query}",
                settings=settings,
                max_tokens=500,
            )
            terms = [str(t).strip() for t in (data.get("search_terms") or []) if str(t).strip()]
            intent = str(data.get("intent") or query).strip() or query
            use_case = str(data.get("use_case") or "general").strip().lower() or "general"
            size_hint = str(data.get("size_hint") or "general").strip().lower() or "general"
            if terms:
                return intent, use_case, size_hint, terms[:4]
        except (LLMNotConfiguredError, Exception):
            pass

    q = query.lower()
    if any(t in q for t in ("hike", "hiking", "backpack", "camping", "portable", "trail")):
        return (
            query,
            "portable_outdoor",
            "day_hike",
            ["portable power station", "C1000", "PS100", "portable solar panel"],
        )
    if any(t in q for t in ("family", "home", "house", "backup")):
        return (
            query,
            "home_backup",
            "family_home",
            ["solar panel", "home backup power station", "solar power station"],
        )
    return query, "general", "general", [query]


def _merge_results_round_robin(batches: list[list[SearchResultItem]], limit: int) -> list[SearchResultItem]:
    """Interleave search batches so one megakit-heavy query cannot dominate the candidate list."""
    seen: set[str] = set()
    merged: list[SearchResultItem] = []
    pointers = [0] * len(batches)
    while len(merged) < limit and any(pointers[i] < len(batches[i]) for i in range(len(batches))):
        progress = False
        for i, batch in enumerate(batches):
            while pointers[i] < len(batch):
                item = batch[pointers[i]]
                pointers[i] += 1
                if item.ad.id in seen:
                    continue
                seen.add(item.ad.id)
                merged.append(item)
                progress = True
                break
            if len(merged) >= limit:
                break
        if not progress:
            break
    return merged


def _heuristic_pick(
    results: list[SearchResultItem],
    limit: int,
    use_case: str = "general",
    size_hint: str = "general",
) -> list[SearchResultItem]:
    ordered = _reorder_for_use_case(results, use_case, size_hint)
    priced = [r for r in ordered if _has_real_price(r)]
    pool = priced or ordered
    return pool[:limit]


async def write_catalog_reply(
    query: str,
    selected: list[SearchResultItem],
    intent: str = "",
    use_case: str = "general",
    size_hint: str = "general",
    settings: Settings | None = None,
) -> str:
    """Dedicated LLM pass: advise on the user's ask using the already-picked deals."""
    settings = settings or get_settings()
    if not selected:
        return build_template_reply(query, [])

    if is_decompose_configured(settings):
        lines = []
        for i, item in enumerate(selected, start=1):
            ad = item.ad
            lines.append(f"{i}. {ad.title} — {ad.price} ({ad.merchant})\n   {ad.description[:280]}")
        try:
            return await complete_text(
                WRITE_REPLY_SYSTEM,
                (
                    f"User request: {query}\n"
                    f"Intent: {intent or query}\n"
                    f"use_case: {use_case}\n"
                    f"size_hint: {size_hint}\n\n"
                    f"Selected deals (best first):\n" + "\n".join(lines) + "\n\n"
                    "Write the advice now."
                ),
                settings=settings,
                max_tokens=360,
                temperature=0.65,
            )
        except (LLMNotConfiguredError, Exception):
            pass

    return await generate_reply(query, selected, settings)


async def pick_relevant_ads(
    query: str,
    candidates: list[SearchResultItem],
    limit: int = 5,
    intent: str = "",
    use_case: str = "general",
    size_hint: str = "general",
    settings: Settings | None = None,
) -> tuple[list[SearchResultItem], str]:
    settings = settings or get_settings()
    if not candidates:
        return [], build_template_reply(query, [])

    candidates = _reorder_for_use_case(candidates, use_case, size_hint)
    selected: list[SearchResultItem] = []

    if is_decompose_configured(settings):
        lines: list[str] = []
        for item in candidates:
            ad = item.ad
            scale = "home_scale" if _looks_home_scale(item) else (
                "portable_scale" if _looks_portable_scale(item) else "unknown_scale"
            )
            lines.append(
                f"- id={ad.id} | scale={scale} | {ad.title} | price={ad.price} | "
                f"merchant={ad.merchant} | category={ad.category}\n"
                f"  {ad.description[:240]}"
            )
        try:
            data = await complete_json(
                PICK_SYSTEM,
                (
                    f"User request: {query}\n"
                    f"Intent: {intent or query}\n"
                    f"use_case: {use_case}\n"
                    f"size_hint: {size_hint}\n\n"
                    f"Candidates (already roughly ordered for this use case):\n"
                    + "\n".join(lines)
                ),
                settings=settings,
                max_tokens=500,
            )
            by_id = {item.ad.id: item for item in candidates}
            for ad_id in data.get("selected_ids") or []:
                hit = by_id.get(str(ad_id))
                if hit and hit not in selected:
                    selected.append(hit)
                if len(selected) >= limit:
                    break
        except (LLMNotConfiguredError, Exception):
            selected = []

    if not selected:
        selected = _heuristic_pick(candidates, limit, use_case=use_case, size_hint=size_hint)

    reply = await write_catalog_reply(
        query,
        selected,
        intent=intent,
        use_case=use_case,
        size_hint=size_hint,
        settings=settings,
    )
    return selected, _attach_tracking_links(reply, selected)


async def llm_catalog_search(
    query: str,
    search_service: SearchService,
    limit: int = 5,
    settings: Settings | None = None,
) -> tuple[list[SearchResultItem], str]:
    """LLM plans use-case + terms → AI Search → reorder → LLM picks ads + reply."""
    settings = settings or get_settings()
    intent, use_case, size_hint, terms = await plan_search_terms(query, settings=settings)

    # Extra portable-biased terms when hiking/camping so C1000/PS100 enter the pool.
    if use_case in {"portable_outdoor", "rv_camping"} or size_hint in {
        "phone_powerbank",
        "day_hike",
        "weekend_camping",
        "rv_overnight",
    }:
        for extra in ("C1000", "portable power station", "PS100", "PS200"):
            if extra.lower() not in {t.lower() for t in terms}:
                terms.append(extra)

    candidate_limit = max(limit * 5, 24)
    per_term = max(8, candidate_limit // max(len(terms), 1))
    batches: list[list[SearchResultItem]] = [
        search_service.search(term, limit=per_term) for term in terms
    ]
    if query.strip() and query.strip() not in terms:
        batches.append(search_service.search(query, limit=per_term))

    candidates = _merge_results_round_robin(batches, limit=candidate_limit)
    return await pick_relevant_ads(
        query,
        candidates,
        limit=limit,
        intent=intent,
        use_case=use_case,
        size_hint=size_hint,
        settings=settings,
    )
