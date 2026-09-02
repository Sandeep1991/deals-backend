from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config import ENV_FILE, get_settings
from app.models import (
    Ad,
    BulkAdsRequest,
    BulkAdsResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from app.replies import generate_reply
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
    return HealthResponse(
        status="ok",
        search_configured=search_service.is_configured,
        reply_provider=current.reply_provider,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    require_search()
    try:
        results = search_service.search(request.query, limit=request.limit)
    except SearchNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Search failed: {exc}") from exc

    reply = await generate_reply(request.query, results)
    return ChatResponse(
        query=request.query,
        reply=reply,
        ads=[item.ad for item in results],
        results=results,
    )


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
