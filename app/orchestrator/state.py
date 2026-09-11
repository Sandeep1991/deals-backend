from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

from app.models import Ad
from app.party_planner.state import ProductQuote, StoreComparison

CategoryName = Literal["grocery", "electronics", "clothing", "stationery", "other"]


class CompositeItem(BaseModel):
    name: str
    search_terms: list[str] = Field(default_factory=list)
    category: CategoryName = "other"
    quantity: float = 1.0


class CategoryResult(BaseModel):
    category: str
    items: list[CompositeItem] = Field(default_factory=list)
    quotes: list[ProductQuote] = Field(default_factory=list)
    ads: list[Ad] = Field(default_factory=list)
    comparison: Optional[StoreComparison] = None
    notes: list[str] = Field(default_factory=list)
    reply_fragment: str = ""


def merge_category_results(
    existing: list[CategoryResult],
    new: list[CategoryResult],
) -> list[CategoryResult]:
    return list(existing) + list(new)


def merge_intent_clarifications(
    existing: list[dict[str, Any]],
    new: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reducer for parallel per-intent ReAct clarify outputs."""
    return list(existing or []) + list(new or [])


class ChatTurn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class OrchestratorState(TypedDict):
    query: str
    event_summary: str
    items: list[CompositeItem]
    category_results: Annotated[list[CategoryResult], merge_category_results]
    reply: str
    ads: list[Ad]
    mode: str
    comparison: Optional[StoreComparison]
    limit: int
    chat_id: str
    history: list[ChatTurn]
    preference_summary: Optional[dict]
    clarification: Optional[dict]
    # Per-intent ReAct clarify results (parallel Send → gather).
    intent_clarifications: Annotated[list[dict[str, Any]], merge_intent_clarifications]
    # Which intent the clarify_react node should handle (set via Send payload).
    clarify_intent: str
