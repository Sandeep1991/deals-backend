from pydantic import BaseModel, Field


class Ad(BaseModel):
    id: str
    title: str
    description: str
    category: str
    keywords: str
    price: str
    url: str


class AdUpsert(Ad):
    pass


class SearchResultItem(BaseModel):
    ad: Ad
    score: float
    keyword_score: float = Field(serialization_alias="keywordScore")
    semantic_score: float = Field(serialization_alias="semanticScore")

    model_config = {"populate_by_name": True}


class ChatRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    query: str
    reply: str
    ads: list[Ad]
    results: list[SearchResultItem] = []


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


class HealthResponse(BaseModel):
    status: str
    search_configured: bool
    reply_provider: str
