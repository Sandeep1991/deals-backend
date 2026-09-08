# DealFinder API (deals-backend)

FastAPI backend for DealFinder: routes natural-language shopping questions through an LLM, retrieves deals from Azure AI Search, and returns compare baskets or ranked affiliate ads.

Frontend repo: [Sandeep1991/deals](https://github.com/Sandeep1991/deals)  
Ingest / affiliate feeds: [Sandeep1991/deals-ingest](https://github.com/Sandeep1991/deals-ingest)

## Architecture

```
                    ┌─────────────────────┐
  React (deals) ──► │  FastAPI /api/chat  │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  routing.should_    │
                    │  compare(query)     │
                    └─────────┬───────────┘
               ┌──────────────┼──────────────┐
               │              │              │
        grocery/party    product/affiliate   stationery /
        /meal planning   (solar, Anker…)    out-of-scope
               │              │              │
               ▼              ▼              ▼
        LangGraph planner  catalog_pick   advisory LLM
        (party_planner)    (LLM → Search  (list only)
               │            → LLM reply)
               │              │
               ▼              ▼
        Azure AI Search   Azure AI Search
        merchant=Kroger   all merchants
        / Walmart         (e.g. Anker Solix
               │           via Rakuten)
               ▼              ▼
        mode=compare      mode=search
        store baskets     ranked ads + advice
```

### `/api/chat` request flow

1. **Route** (`app/routing.py`)
   - **Product / affiliate** hints (`solar`, `anker`, `solix`, `rv`, …) → full-catalog search (product intent wins even if “camping” appears).
   - **Grocery / party / recipe** hints → Kroger vs Walmart store comparison.
   - Explicit `mode=search` | `mode=compare` overrides auto.

2. **Compare path** (`app/party_planner/`)
   - LLM decomposes the request into a shopping list (Azure OpenAI / Ollama; heuristic fallback only if LLM fails).
   - LangGraph: `decompose` → `fetch_prices` → `compare`.
   - Prices from Azure AI Search filtered by merchant; optional web fallback for missing items.
   - Returns `mode=compare` with per-store baskets. If nothing priced in catalog, falls back to catalog search.

3. **Catalog / affiliate path** (`app/catalog_pick.py`)
   - LLM plans use-case + search terms.
   - Azure AI Search retrieves candidates (all merchants).
   - LLM **picks** relevant ad IDs (use-case aware: portable vs home backup).
   - Separate LLM **reply** pass writes scenario-specific advice (not first-hit / brochure text).
   - Tracking links attached from catalog URLs (full Rakuten `murl` deep links).
   - Returns `mode=search`.

4. **Advisory path** (`app/advisory.py`)
   - Non-grocery lists (e.g. stationery) when search is not appropriate → shopping list without deal cards.

### Supporting services

| Piece | Role |
|---|---|
| Azure AI Search (`ads` index) | Grocery staples (Kroger/Walmart) + affiliate products (e.g. Anker Solix / Rakuten) |
| Azure OpenAI / Ollama (`app/llm_client.py`) | Decompose, catalog plan/pick/reply, advisory |
| `deals-ingest` | Fetches Rakuten Solix feed, filters sold-out via merchant site, upserts/deletes Search docs |

## Azure App Service deployment

1. Create **Web App** → Linux → Python 3.12
2. **Deployment Center** → connect to `Sandeep1991/deals-backend` → branch `main`
3. **Configuration** → Application settings:

   | Setting | Value |
   |---|---|
   | `AZURE_SEARCH_ENDPOINT` | `https://dealssearch.search.windows.net` |
   | `AZURE_SEARCH_API_KEY` | your key |
   | `AZURE_SEARCH_INDEX` | `ads` |
   | `AZURE_SEARCH_SEMANTIC_CONFIG` | `ads-semantic` |
   | `REPLY_PROVIDER` | `auto` |
   | `DECOMPOSE_PROVIDER` | `auto` |
   | `AZURE_OPENAI_ENDPOINT` | Foundry / OpenAI resource endpoint |
   | `AZURE_OPENAI_API_KEY` | your key |
   | `AZURE_OPENAI_DEPLOYMENT` | e.g. `gpt-4o-mini` |
   | `SCM_DO_BUILD_DURING_DEPLOYMENT` | `true` |

4. **Startup command:** `bash startup.sh`
5. Test: `https://<your-app>.azurewebsites.net/health`

### GitHub Actions deploy fails: "No subscriptions found"

The Node 20 message is only a warning — the real error is Azure permissions.

OIDC login worked, but the App Registration service principal cannot see your subscription. Fix in Azure Portal:

1. **Subscriptions** → your subscription → **Access control (IAM)** → **Add role assignment**
2. Role: **Contributor** (or **Website Contributor** on the resource group)
3. Assign access to: **User, group, or service principal**
4. Search by the **App Registration name** (paste the client ID from GitHub secret `AZUREAPPSERVICE_CLIENTID_*` if name search fails)
5. Save, wait 2–5 minutes, re-run the workflow

Also verify **App registrations** → your app → **Certificates & secrets** → **Federated credentials** includes:

```
repo:Sandeep1991/deals-backend:ref:refs/heads/main
```

If GitHub shows a subject with `@` IDs (e.g. `Sandeep1991@8342110/deals-backend@...`), add a second federated credential with that exact subject from the failed workflow log.

## Local development

```bash
./run.sh
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set AZURE_SEARCH_* and AZURE_OPENAI_*
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health + search/LLM provider status |
| POST | `/api/chat` | Routed chat: compare, catalog search, or advisory |
| POST | `/api/compare` | Force store comparison (Kroger vs Walmart) |
| POST | `/api/search` | Raw Azure AI Search (no LLM) |
| POST | `/api/ads` | Bulk upsert ads |
| PUT | `/api/ads/{id}` | Upsert single ad |
| DELETE | `/api/ads/{id}` | Delete ad |

## Environment variables

See `.env.example`. Important:

- **Search:** `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_API_KEY`, `AZURE_SEARCH_INDEX`, `AZURE_SEARCH_SEMANTIC_CONFIG`
- **LLM:** `DECOMPOSE_PROVIDER=auto`, `REPLY_PROVIDER=auto`, plus `AZURE_OPENAI_*` (or Ollama)
- **CORS:** `CORS_ORIGINS`

## Search behavior

- **Literal queries** (`shower`, `soap`): keyword search + score threshold
- **Meaning queries** (`discount`, `deal`): hybrid + semantic + vector
- **Fallback**: retries with hybrid if keyword returns nothing
- **Catalog chat**: LLM plans terms → multi-query retrieve → LLM pick → LLM reply (not raw first-hit ranking)

Tune via `MIN_RERANKER_SCORE` and `MIN_SEARCH_SCORE`.

## Key modules

| Module | Responsibility |
|---|---|
| `app/main.py` | FastAPI routes and chat orchestration |
| `app/routing.py` | Grocery compare vs product search vs advisory |
| `app/party_planner/` | LangGraph decompose → price → compare |
| `app/catalog_pick.py` | Affiliate/product LLM plan, pick, and reply |
| `app/search.py` | Azure AI Search client |
| `app/llm_client.py` | Shared Azure OpenAI / Ollama completions |
| `app/replies.py` | LLM reply helper (template only as last resort) |
| `app/advisory.py` | Out-of-catalog shopping lists |
