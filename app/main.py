from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config import ENV_FILE, get_settings
from app.compare import to_compare_response
from app.models import (
    Ad,
    BulkAdsRequest,
    BulkAdsResponse,
    ChatRequest,
    ChatResponse,
    CompareRequest,
    CompareResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from app.advisory import build_advisory_reply
from app.catalog_pick import llm_catalog_search
from app.intent import is_non_grocery_query, out_of_scope_reply
from app.llm_client import LLMNotConfiguredError, resolve_decompose_provider
from app.party_planner.graph import run_store_comparison
from app.routing import should_compare
from app.search import SearchService, SearchNotConfiguredError

settings = get_settings()
origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]

app = FastAPI(title="DealFinder API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_origin_regex=r"https://(.*\.)?azurestaticapps\.net|https://(www\.)?thetinkerer\.xyz",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_service = SearchService()

SEARCH_SETUP_HINT = (
    "Azure AI Search is not configured. "
    f"Copy {ENV_FILE.name} from .env.example and set "
    "AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY, then restart the server."
)


def require_search() -> None:
    if not search_service.is_configured:
        raise HTTPException(status_code=503, detail=SEARCH_SETUP_HINT)


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    current = get_settings()
    decompose = resolve_decompose_provider(current) or "none"
    return HealthResponse(
        status="ok",
        search_configured=search_service.is_configured,
        reply_provider=current.reply_provider,
        decompose_configured=decompose != "none",
        decompose_provider=decompose,
    )


def _compare_has_catalog_prices(compare_response: CompareResponse) -> bool:
    """True when at least one quote came from AI Search (not web-only 'See site')."""
    for basket in compare_response.merchants:
        for quote in basket.quotes:
            if quote.source == "search" and quote.unit_price is not None:
                return True
    return False


async def _catalog_search_response(query: str, limit: int) -> ChatResponse:
    # LLM plans search terms → AI Search → LLM picks relevant ads (not first-hit ranking).
    results, reply = await llm_catalog_search(query, search_service, limit=limit)
    if not results:
        if not reply:
            try:
                reply = await build_advisory_reply(query, in_catalog_scope=True)
            except Exception:
                reply = out_of_scope_reply(query)
        return ChatResponse(query=query, reply=reply, ads=[], results=[], mode="advisory")

    return ChatResponse(
        query=query,
        reply=reply,
        ads=[item.ad for item in results],
        results=results,
        mode="search",
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    require_search()

    # Grocery/planning → LLM decompose + Kroger/Walmart compare.
    # Product/affiliate queries → open AI Search (Anker Solix, etc.).
    if should_compare(request.query, request.mode):
        try:
            comparison = await run_store_comparison(request.query, search_service)
            compare_response = to_compare_response(comparison)
            if _compare_has_catalog_prices(compare_response):
                return ChatResponse(
                    query=request.query,
                    reply=compare_response.reply,
                    ads=compare_response.ads,
                    results=[],
                    mode="compare",
                    comparison=compare_response,
                )
            # No grocery catalog hits — fall back so Rakuten/other merchants can surface.
            return await _catalog_search_response(request.query, request.limit)
        except SearchNotConfiguredError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LLMNotConfiguredError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Compare failed: {exc}") from exc

    if is_non_grocery_query(request.query):
        reply = await build_advisory_reply(request.query)
        return ChatResponse(query=request.query, reply=reply, ads=[], results=[], mode="advisory")

    try:
        return await _catalog_search_response(request.query, request.limit)
    except SearchNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Search failed: {exc}") from exc


@app.post("/api/compare", response_model=CompareResponse)
async def compare(request: CompareRequest) -> CompareResponse:
    require_search()
    try:
        comparison = await run_store_comparison(request.query, search_service)
        return to_compare_response(comparison)
    except SearchNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Compare failed: {exc}") from exc


@app.post("/api/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    require_search()
    try:
        results = search_service.search(request.query, limit=request.limit)
    except SearchNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Search failed: {exc}") from exc
    return SearchResponse(query=request.query, results=results)


@app.post("/api/ads", response_model=BulkAdsResponse)
def upsert_ads(request: BulkAdsRequest) -> BulkAdsResponse:
    require_search()
    try:
        ids = search_service.upsert_ads(request.ads)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upsert failed: {exc}") from exc
    return BulkAdsResponse(uploaded=len(ids), ids=ids)


@app.put("/api/ads/{ad_id}", response_model=BulkAdsResponse)
def upsert_ad(ad_id: str, ad: Ad) -> BulkAdsResponse:
    require_search()
    if ad.id != ad_id:
        raise HTTPException(status_code=400, detail="Ad id in body must match path parameter")
    try:
        ids = search_service.upsert_ads([ad])
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upsert failed: {exc}") from exc
    return BulkAdsResponse(uploaded=len(ids), ids=ids)


@app.delete("/api/ads/{ad_id}")
def delete_ad(ad_id: str) -> dict[str, str]:
    require_search()
    try:
        search_service.delete_ad(ad_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Delete failed: {exc}") from exc
    return {"status": "deleted", "id": ad_id}
