from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Ad(BaseModel):
    id: str
    title: str
    description: str
    category: str
    keywords: str
    price: str
    url: str
    merchant: str = ""
    network: str = ""
    brand: str = ""
    source_key: str = ""
    expires_at: str = ""


class AdUpsert(Ad):
    pass


class SearchResultItem(BaseModel):
    ad: Ad
    score: float
    keyword_score: float = Field(serialization_alias="keywordScore")
    semantic_score: float = Field(serialization_alias="semanticScore")

    model_config = {"populate_by_name": True}


class HealthResponse(BaseModel):
    status: str
    search_configured: bool
    reply_provider: str
    decompose_configured: bool = False
    decompose_provider: str = "none"


class ShoppingItemOut(BaseModel):
    name: str
    search_terms: list[str] = []
    quantity: float = 1.0


class ProductQuoteOut(BaseModel):
    item_name: str
    merchant: str
    ad: Ad
    unit_price: Optional[float] = None
    line_total: Optional[float] = None
    source: str = "search"


class MerchantBasketOut(BaseModel):
    merchant: str
    quotes: list[ProductQuoteOut] = []
    alternative_label: Optional[str] = None
    subtotal: Optional[float] = None
    subtotal_is_partial: bool = False
    priced_items: int = 0
    total_items: int = 0


class CompareRequest(BaseModel):
    query: str


class CompareResponse(BaseModel):
    query: str
    reply: str
    ads: list[Ad]
    event_summary: str = ""
    recommended_merchant: Optional[str] = None
    savings: Optional[float] = None
    merchants: list[MerchantBasketOut] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class PreferenceSummaryOut(BaseModel):
    summary: str = ""
    preferences: list[str] = Field(default_factory=list)
    is_list_rewrite: bool = False
    prior_grocery_items: list[str] = Field(default_factory=list)
    rewrite_guidance: str = ""


class ChatRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="auto", description="auto | search | compare")
    chat_id: Optional[str] = Field(default=None, description="Browser session UUID")
    messages: list[ChatMessage] = Field(
        default_factory=list,
        description="Prior turns for session memory (browser is source of truth)",
    )
    preference_summary: Optional[PreferenceSummaryOut] = Field(
        default=None,
        description="Rolling LangChain-style preference summary for this chat_id",
    )


class ChatResponse(BaseModel):
    query: str
    reply: str
    ads: list[Ad]
    results: list[SearchResultItem] = []
    mode: str = "search"
    comparison: Optional[CompareResponse] = None
    chat_id: Optional[str] = None
    preference_summary: Optional[PreferenceSummaryOut] = None


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=20)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]


class BulkAdsRequest(BaseModel):
    ads: list[AdUpsert]


class BulkAdsResponse(BaseModel):
    uploaded: int
    ids: list[str]
