from __future__ import annotations

from app.config import Settings, get_settings
from app.llm_client import LLMNotConfiguredError, complete_json, is_decompose_configured
from app.models import SearchResultItem
from app.pricing import parse_price
from app.replies import build_template_reply, generate_reply
from app.search import SearchService

PLAN_SYSTEM = """You turn a shopper request into catalog search queries for affiliate/product ads.
Return JSON only:
{
  "intent": "one sentence of what they want",
  "use_case": "home_backup|portable_outdoor|solar_panels|ev_charging|general",
  "size_hint": "phone_powerbank|day_hike|weekend_camping|apartment|family_home|whole_home",
  "search_terms": ["term1", "term2", "term3"],
  "avoid_terms": ["optional words that indicate wrong products"]
}

Rules for search_terms (1-4 short phrases that match product titles):
- portable / hike / backpacking / camping day-trip → "portable power station", "C1000", "PS100", "solar panel portable"
  (NOT whole-home F3800 kits, NOT "4x 400W", NOT EV adapters)
- family / home backup / house solar → "solar panel", "home backup", "power station solar panel"
- solar panels only → "solar panel", "portable solar panel"
- Do not invent brands unless the user named them.
- Prefer concrete product-class terms over repeating the full user sentence."""

PICK_SYSTEM = """You pick deals that match THIS user's use case — different queries must get different products when intent differs.
Return JSON only:
{
  "selected_ids": ["id1", "id2"],
  "reply": "2-3 sentences. Name the best pick and price with markdown [title](url) from candidates. Explain briefly why it fits THIS request."
}

Hard rules:
- selected_ids: best first, max 5, ONLY from the candidate list.
- Obey use_case and size_hint strictly:
  - portable_outdoor / day_hike / weekend_camping: choose compact portable power stations or small portable panels
    (e.g. C1000, C2000, PS100/PS200). REJECT whole-home kits (F3000/F3800, multi-kWh expansion + 4× rigid panels)
    unless no smaller option exists in the candidates.
  - family_home / whole_home / home_backup: larger home backup + panel kits are OK; still prefer a sensible family size.
- Prefer real positive prices; skip $0/blank when priced alternatives exist.
- Never invent products, prices, or URLs.
- Do NOT reuse a home-backup megakit answer for a hiking query (or vice versa).
- If nothing fits, selected_ids: [] and say what is missing."""

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
    portable_intent = use_case == "portable_outdoor" or size_hint in {
        "phone_powerbank",
        "day_hike",
        "weekend_camping",
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
                f"  {ad.description[:240]}\n  url={ad.url}"
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
                max_tokens=900,
            )
            by_id = {item.ad.id: item for item in candidates}
            selected: list[SearchResultItem] = []
            for ad_id in data.get("selected_ids") or []:
                hit = by_id.get(str(ad_id))
                if hit and hit not in selected:
                    selected.append(hit)
                if len(selected) >= limit:
                    break
            reply = str(data.get("reply") or "").strip()
            if selected and reply:
                return selected, reply
            if selected:
                return selected, await generate_reply(query, selected, settings)
            if reply:
                return [], reply
        except (LLMNotConfiguredError, Exception):
            pass

    picked = _heuristic_pick(candidates, limit, use_case=use_case, size_hint=size_hint)
    return picked, await generate_reply(query, picked, settings)


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
    if use_case == "portable_outdoor" or size_hint in {"phone_powerbank", "day_hike", "weekend_camping"}:
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
