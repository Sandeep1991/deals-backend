from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from pydantic import BaseModel, Field

from app.models import Ad


class ShoppingItem(BaseModel):
    name: str
    search_terms: list[str] = Field(default_factory=list)
    quantity: float = 1.0


class AlternativeOption(BaseModel):
    label: str
    items: list[ShoppingItem] = Field(default_factory=list)


class ShoppingPlan(BaseModel):
    event_summary: str
    people_count: Optional[int] = None
    required_items: list[ShoppingItem] = Field(default_factory=list)
    alternative_options: list[AlternativeOption] = Field(default_factory=list)


class ProductQuote(BaseModel):
    item_name: str
    merchant: str
    ad: Ad
    unit_price: Optional[float] = None
    line_total: Optional[float] = None
    source: str = "search"  # search | web


class MerchantBasket(BaseModel):
    merchant: str
    quotes: list[ProductQuote] = Field(default_factory=list)
    alternative_label: Optional[str] = None
    subtotal: Optional[float] = None
    subtotal_is_partial: bool = False
    priced_items: int = 0
    total_items: int = 0


class StoreComparison(BaseModel):
    query: str
    plan: ShoppingPlan
    merchants: list[MerchantBasket]
    recommended_merchant: Optional[str] = None
    savings: Optional[float] = None
    reply: str = ""


def merge_quotes(existing: list[ProductQuote], new: list[ProductQuote]) -> list[ProductQuote]:
    by_key = {(q.merchant, q.item_name): q for q in existing}
    for quote in new:
        by_key[(quote.merchant, quote.item_name)] = quote
    return list(by_key.values())


class PlannerState(TypedDict):
    query: str
    plan: Optional[ShoppingPlan]
    quotes: Annotated[list[ProductQuote], merge_quotes]
    comparison: Optional[StoreComparison]
    reply: str
    ads: list[Ad]
