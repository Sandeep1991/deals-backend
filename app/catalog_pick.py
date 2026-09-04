from __future__ import annotations

from app.config import Settings, get_settings
from app.llm_client import LLMNotConfiguredError, complete_json, is_decompose_configured
from app.models import SearchResultItem
from app.pricing import parse_price
from app.replies import build_template_reply, generate_reply
from app.search import SearchService

PLAN_SYSTEM = """You turn a shopper request into catalog search queries.
Return JSON only:
{
  "intent": "short phrase of what they want",
  "search_terms": ["term1", "term2"]
}

Rules:
- 1-3 short search_terms that match product titles (not the full sentence).
- For "solar charger for home" prefer terms like "solar charger", "solar panel", "solar power station"
  — not EV-only accessories unless the user asked for EV.
- Do not invent brands unless the user named them."""

PICK_SYSTEM = """You pick the best matching deals from AI Search candidates for the user.
Return JSON only:
{
  "selected_ids": ["id1", "id2"],
  "reply": "2-3 sentence helpful answer. Mention the best pick by name and price. Use markdown [title](url) only from candidates."
}

Rules:
- selected_ids: best matches first, max 5, ONLY ids from the candidate list.
- Match the user's intent closely (home solar charging / solar generators / panels — not unrelated EV adapters unless asked).
- Prefer real positive prices. Skip $0, blank, or placeholder prices when better priced options exist.
- Never invent products, prices, or URLs.
- If nothing is relevant, return selected_ids: [] and explain briefly in reply."""


async def plan_search_terms(query: str, settings: Settings | None = None) -> tuple[str, list[str]]:
    settings = settings or get_settings()
    if is_decompose_configured(settings):
        try:
            data = await complete_json(
                PLAN_SYSTEM,
                f"User request: {query}",
                settings=settings,
                max_tokens=400,
            )
            terms = [str(t).strip() for t in (data.get("search_terms") or []) if str(t).strip()]
            intent = str(data.get("intent") or query).strip() or query
            if terms:
                return intent, terms[:3]
        except (LLMNotConfiguredError, Exception):
            pass
    return query, [query]


def _merge_results(batches: list[list[SearchResultItem]], limit: int) -> list[SearchResultItem]:
    seen: set[str] = set()
    merged: list[SearchResultItem] = []
    for batch in batches:
        for item in batch:
            if item.ad.id in seen:
                continue
            seen.add(item.ad.id)
            merged.append(item)
            if len(merged) >= limit:
                return merged
    return merged


def _has_real_price(item: SearchResultItem) -> bool:
    amount = parse_price(item.ad.price)
    return amount is not None and amount > 0


def _heuristic_pick(results: list[SearchResultItem], limit: int) -> list[SearchResultItem]:
    priced = [r for r in results if _has_real_price(r)]
    pool = priced or results
    return pool[:limit]


async def pick_relevant_ads(
    query: str,
    candidates: list[SearchResultItem],
    limit: int = 5,
    settings: Settings | None = None,
) -> tuple[list[SearchResultItem], str]:
    settings = settings or get_settings()
    if not candidates:
        return [], build_template_reply(query, [])

    if is_decompose_configured(settings):
        lines: list[str] = []
        for item in candidates:
            ad = item.ad
            lines.append(
                f"- id={ad.id} | {ad.title} | price={ad.price} | merchant={ad.merchant} | "
                f"category={ad.category}\n  {ad.description[:240]}\n  url={ad.url}"
            )
        try:
            data = await complete_json(
                PICK_SYSTEM,
                f"User request: {query}\n\nCandidates:\n" + "\n".join(lines),
                settings=settings,
                max_tokens=800,
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

    picked = _heuristic_pick(candidates, limit)
    return picked, await generate_reply(query, picked, settings)


async def llm_catalog_search(
    query: str,
    search_service: SearchService,
    limit: int = 5,
    settings: Settings | None = None,
) -> tuple[list[SearchResultItem], str]:
    """LLM plans terms → AI Search candidates → LLM picks relevant ads + reply."""
    settings = settings or get_settings()
    _intent, terms = await plan_search_terms(query, settings=settings)

    candidate_limit = max(limit * 4, 20)
    batches: list[list[SearchResultItem]] = []
    for term in terms:
        batches.append(search_service.search(term, limit=candidate_limit))
    # Also search the raw query once.
    if query.strip() and query.strip() not in terms:
        batches.append(search_service.search(query, limit=candidate_limit))

    candidates = _merge_results(batches, limit=candidate_limit)
    return await pick_relevant_ads(query, candidates, limit=limit, settings=settings)
