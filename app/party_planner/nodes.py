from __future__ import annotations

import re

from app.advisory import format_missing_catalog_items
from app.llm_client import LLMNotConfiguredError, complete_json, is_decompose_configured, resolve_decompose_provider
from app.models import Ad
from app.party_planner.decompose import DECOMPOSE_SYSTEM, heuristic_decompose
from app.party_planner.state import (
    AlternativeOption,
    PlannerState,
    ProductQuote,
    ShoppingItem,
    ShoppingPlan,
)
from app.pricing import parse_price
from app.search import SearchService
from app.web_search import search_merchant_product

MERCHANTS = ["Kroger", "Walmart"]
MIN_INGREDIENT_SCORE = 0.5


def _is_relevant_match(term: str, item_name: str, ad: Ad) -> bool:
    haystack = f"{ad.title} {ad.keywords} {ad.description}".lower()
    name = item_name.lower().strip()
    if name and name in haystack:
        return True

    name_words = [w for w in name.split() if len(w) > 2]
    if len(name_words) >= 2 and all(w in haystack for w in name_words):
        return True

    term_l = term.lower().strip()
    if " " in term_l:
        term_words = [w for w in term_l.split() if len(w) > 2]
        if term_words and all(w in haystack for w in term_words):
            return True
        return term_l in haystack

    return re.search(rf"\b{re.escape(term_l)}\b", haystack) is not None


async def decompose_node(state: PlannerState) -> dict:
    query = state["query"]
    plan: ShoppingPlan | None = None
    used_fallback = False

    from app.config import get_settings

    settings = get_settings()

    if is_decompose_configured(settings):
        try:
            data = await complete_json(
                DECOMPOSE_SYSTEM,
                f"User request: {query}",
                settings=settings,
            )
            plan = ShoppingPlan.model_validate(data)
        except (LLMNotConfiguredError,):
            raise
        except Exception:
            plan = None

    if not plan or (not plan.required_items and not plan.alternative_options):
        if settings.decompose_provider.lower().strip() == "template":
            plan = heuristic_decompose(query)
            used_fallback = True
        elif not is_decompose_configured(settings):
            raise LLMNotConfiguredError(
                "Store comparison requires an LLM. Set DECOMPOSE_PROVIDER=auto (default) and configure "
                "AZURE_OPENAI_* or run Ollama locally."
            )
        else:
            plan = heuristic_decompose(query)
            used_fallback = True

    if used_fallback:
        plan.event_summary = f"{plan.event_summary} (fallback planner — configure LLM for better results)"

    return {"plan": plan}


async def fetch_prices_node(state: PlannerState, search_service: SearchService) -> dict:
    plan = state["plan"]
    if not plan:
        return {"quotes": []}

    quotes: list[ProductQuote] = []
    items_to_fetch: list[tuple[str, ShoppingItem, str | None]] = []

    for item in plan.required_items:
        items_to_fetch.append(("required", item, None))

    for option in plan.alternative_options:
        for item in option.items:
            items_to_fetch.append(("alternative", item, option.label))

    seen: set[tuple[str, str]] = set()
    for _kind, item, _alt_label in items_to_fetch:
        for merchant in MERCHANTS:
            key = (merchant, item.name)
            if key in seen:
                continue
            seen.add(key)

            quote = await _quote_item(search_service, merchant, item)
            if quote:
                quotes.append(quote)

    return {"quotes": quotes}


async def _quote_item(
    search_service: SearchService,
    merchant: str,
    item: ShoppingItem,
) -> ProductQuote | None:
    ad: Ad | None = None
    source = "search"

    for term in item.search_terms or [item.name]:
        results = search_service.search(
            term,
            limit=5,
            merchant=merchant,
            min_score=MIN_INGREDIENT_SCORE,
        )
        for result in results:
            if _is_relevant_match(term, item.name, result.ad):
                ad = result.ad
                source = "search"
                break
        if ad:
            break

    if not ad:
        ad = await search_merchant_product(merchant, item.search_terms[0] if item.search_terms else item.name)
        source = "web"

    if not ad:
        return None

    unit_price = parse_price(ad.price)
    line_total = round(unit_price * item.quantity, 2) if unit_price is not None else None

    return ProductQuote(
        item_name=item.name,
        merchant=merchant,
        ad=ad,
        unit_price=unit_price,
        line_total=line_total,
        source=source,
    )


def compare_node(state: PlannerState) -> dict:
    plan = state["plan"]
    quotes = state["quotes"]
    query = state["query"]

    if not plan:
        return {
            "comparison": None,
            "reply": "I couldn't build a shopping list from that request. Try mentioning specific foods or a party theme.",
            "ads": [],
        }

    from app.party_planner.state import MerchantBasket, StoreComparison

    baskets: list[MerchantBasket] = []

    for merchant in MERCHANTS:
        merchant_quotes = [q for q in quotes if q.merchant == merchant]
        quote_map = {q.item_name: q for q in merchant_quotes}

        selected: list[ProductQuote] = []
        subtotal = 0.0
        priced_items = 0
        total_items = 0
        alt_label: str | None = None

        for item in plan.required_items:
            quote = quote_map.get(item.name)
            if quote:
                selected.append(quote)
                total_items += 1
                if quote.line_total is not None:
                    subtotal += quote.line_total
                    priced_items += 1

        if plan.alternative_options:
            branches_by_label: dict[str, list[list[ShoppingItem]]] = {}
            for option in plan.alternative_options:
                branches_by_label.setdefault(option.label, []).append(option.items)

            best_alt_total: float | None = None
            best_alt_quotes: list[ProductQuote] = []
            best_alt_priced = 0
            best_label: str | None = None

            for label, branches in branches_by_label.items():
                for branch_items in branches:
                    option_quotes: list[ProductQuote] = []
                    option_total = 0.0
                    option_priced = 0
                    for opt_item in branch_items:
                        quote = quote_map.get(opt_item.name)
                        if not quote:
                            continue
                        option_quotes.append(quote)
                        if quote.line_total is not None:
                            option_total += quote.line_total
                            option_priced += 1

                    if not option_quotes:
                        continue
                    if best_alt_total is None or option_total < best_alt_total:
                        best_alt_total = option_total
                        best_alt_quotes = option_quotes
                        best_alt_priced = option_priced
                        best_label = label

            if best_alt_quotes:
                selected.extend(best_alt_quotes)
                alt_label = best_label
                total_items += len(best_alt_quotes)
                priced_items += best_alt_priced
                if best_alt_total is not None:
                    subtotal += best_alt_total

        partial = priced_items < total_items or total_items == 0
        basket_subtotal = round(subtotal, 2) if priced_items > 0 else None

        baskets.append(
            MerchantBasket(
                merchant=merchant,
                quotes=selected,
                alternative_label=alt_label,
                subtotal=basket_subtotal,
                subtotal_is_partial=partial and priced_items > 0,
                priced_items=priced_items,
                total_items=total_items,
            )
        )

    comparable = [b for b in baskets if b.subtotal is not None]
    recommended: str | None = None
    savings: float | None = None

    if len(comparable) >= 2:
        comparable.sort(key=lambda b: b.subtotal or 0)
        recommended = comparable[0].merchant
        savings = round((comparable[-1].subtotal or 0) - (comparable[0].subtotal or 0), 2)
    elif len(comparable) == 1:
        recommended = comparable[0].merchant

    comparison = StoreComparison(
        query=query,
        plan=plan,
        merchants=baskets,
        recommended_merchant=recommended,
        savings=savings if savings and savings > 0 else None,
    )

    reply = _format_reply(comparison)
    comparison.reply = reply
    ads = [q.ad for b in baskets for q in b.quotes]

    return {"comparison": comparison, "reply": reply, "ads": ads}


def _format_reply(comparison: StoreComparison) -> str:
    plan = comparison.plan
    lines = [
        f"**{plan.event_summary}**",
        "",
        "**Shopping list:**",
    ]

    for item in plan.required_items:
        lines.append(f"- {item.name}")

    if plan.alternative_options:
        labels = sorted({opt.label for opt in plan.alternative_options})
        for label in labels:
            lines.append(f"- {label} (cheapest option per store)")

    lines.extend(["", "**Store comparison:**", ""])

    for basket in comparison.merchants:
        lines.append(f"### {basket.merchant}")
        if basket.alternative_label:
            lines.append(f"_Includes best option for {basket.alternative_label}_")
        for quote in basket.quotes:
            price = quote.ad.price
            total = f" → ${quote.line_total:.2f}" if quote.line_total is not None else ""
            source = " (web)" if quote.source == "web" else ""
            lines.append(f"- {quote.item_name}: [{quote.ad.title}]({quote.ad.url}) — {price}{total}{source}")
        if basket.subtotal is not None:
            if basket.subtotal_is_partial:
                lines.append(
                    f"**Estimated total: ${basket.subtotal:.2f}** "
                    f"({basket.priced_items}/{basket.total_items} items priced — some items need in-store check)"
                )
            else:
                lines.append(f"**Estimated total: ${basket.subtotal:.2f}**")
        else:
            lines.append("**Estimated total: unavailable** (no prices found in catalog)")
        lines.append("")

    if comparison.recommended_merchant:
        msg = f"**Recommendation: shop at {comparison.recommended_merchant}**"
        if comparison.savings:
            others = [b for b in comparison.merchants if b.merchant != comparison.recommended_merchant and b.subtotal]
            if others:
                msg += f" — save about **${comparison.savings:.2f}** vs {others[0].merchant}"
        lines.append(msg)

    missing: list[str] = []
    priced_names = {
        q.item_name
        for b in comparison.merchants
        for q in b.quotes
        if q.source == "search" and q.line_total is not None
    }
    for item in plan.required_items:
        if item.name not in priced_names:
            missing.append(item.name)
    for option in plan.alternative_options:
        for item in option.items:
            if item.name not in priced_names and item.name not in missing:
                missing.append(item.name)

    missing_note = format_missing_catalog_items(missing)
    if missing_note:
        lines.append(missing_note)

    return "\n".join(lines)
