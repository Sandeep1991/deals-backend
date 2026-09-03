from __future__ import annotations

from app.models import CompareResponse, MerchantBasketOut, ProductQuoteOut
from app.party_planner.state import StoreComparison


def to_compare_response(comparison: StoreComparison) -> CompareResponse:
    merchants = [
        MerchantBasketOut(
            merchant=basket.merchant,
            quotes=[
                ProductQuoteOut(
                    item_name=quote.item_name,
                    merchant=quote.merchant,
                    ad=quote.ad,
                    unit_price=quote.unit_price,
                    line_total=quote.line_total,
                    source=quote.source,
                )
                for quote in basket.quotes
            ],
            alternative_label=basket.alternative_label,
            subtotal=basket.subtotal,
            subtotal_is_partial=basket.subtotal_is_partial,
            priced_items=basket.priced_items,
            total_items=basket.total_items,
        )
        for basket in comparison.merchants
    ]
    ads = [quote.ad for basket in comparison.merchants for quote in basket.quotes]
    return CompareResponse(
        query=comparison.query,
        reply=comparison.reply,
        ads=ads,
        event_summary=comparison.plan.event_summary,
        recommended_merchant=comparison.recommended_merchant,
        savings=comparison.savings,
        merchants=merchants,
    )
