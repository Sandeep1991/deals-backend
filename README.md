# DealFinder API (deals-backend)

FastAPI backend for DealFinder: a LangGraph orchestrator splits each question into category items, fans out to specialized agents, retrieves deals from Azure AI Search (and optional merchant APIs), then merges one reply.

Frontend repo: [Sandeep1991/deals](https://github.com/Sandeep1991/deals)  
Ingest / affiliate feeds: [Sandeep1991/deals-ingest](https://github.com/Sandeep1991/deals-ingest)

## Architecture

```
                         ┌──────────────────────┐
   React (deals) ──────► │  FastAPI /api/chat   │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │ split_query (LLM)    │
                         │ items + categories   │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │ LangGraph Send route │
                         └──────────┬───────────┘
              ┌─────────────┬───────┴───────┬─────────────┐
              ▼             ▼               ▼             ▼
        grocery_agent  electronics    clothing_agent  stationery /
        Search→API→web  catalog_pick   Search/notes   advisory
        Kroger/Walmart  (Solix/etc.)
              │             │               │             │
              └─────────────┴───────┬───────┴─────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ merge_results (LLM)  │
                         │ mode=compare|search  │
                         │ |mixed|advisory      │
                         └──────────────────────┘
```

### `/api/chat` request flow

1. **`mode=auto` (default)** — LangGraph orchestrator in `app/orchestrator/`
   - LLM **split_query** → composite items tagged `grocery` | `electronics` | `clothing` | `stationery` | `other`
   - **Send** fan-out to category agents in parallel
   - **merge_results** builds one reply + ads (+ grocery comparison when present)

2. **Grocery agent** — Azure AI Search (merchant filter) → optional Kroger/Walmart product APIs → DuckDuckGo web fallback → Kroger vs Walmart baskets

3. **Electronics agent** — wraps `catalog_pick` (LLM plan → Search → LLM pick → LLM reply) for Solix/affiliate ads

4. **Clothing / stationery** — Search first; stationery falls back to advisory lists when empty

5. **Explicit overrides** — `mode=compare` forces grocery LangGraph; `mode=search` forces catalog_pick only

### Supporting services

| Piece | Role |
|---|---|
| Azure AI Search (`ads` index) | Grocery staples + affiliate products (e.g. Anker Solix / Rakuten) |
| Azure OpenAI / Ollama (`app/llm_client.py`) | Split, pick/reply, merge, advisory |
| Kroger / Walmart APIs (`app/merchants/`) | Optional grocery price step when env keys set |
| `deals-ingest` | Rakuten Solix feed, sold-out filter, Search upsert/delete |

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
| POST | `/api/chat` | Orchestrated chat (auto) or forced compare/search |
| POST | `/api/compare` | Force grocery store comparison |
| POST | `/api/search` | Raw Azure AI Search (no LLM) |
| POST | `/api/ads` | Bulk upsert ads |
| PUT | `/api/ads/{id}` | Upsert single ad |
| DELETE | `/api/ads/{id}` | Delete ad |

## Environment variables

See `.env.example`. Important:

- **Search:** `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_API_KEY`, `AZURE_SEARCH_INDEX`, `AZURE_SEARCH_SEMANTIC_CONFIG`
- **LLM:** `DECOMPOSE_PROVIDER=auto`, `REPLY_PROVIDER=auto`, plus `AZURE_OPENAI_*` (or Ollama)
- **Optional grocery APIs:** `KROGER_CLIENT_ID`, `KROGER_CLIENT_SECRET`, `KROGER_LOCATION_ID` (or `KROGER_ZIP_CODE`), `WALMART_API_KEY`, `WALMART_PUBLISHER_ID`
- Grocery compare only keeps a Search/API/web hit when `parse_price` succeeds; unpriced hits fall through the ladder
- **CORS:** `CORS_ORIGINS`

## Search behavior

- **Literal queries** (`shower`, `soap`): keyword search + score threshold
- **Meaning queries** (`discount`, `deal`): hybrid + semantic + vector
- **Grocery ladder:** Azure Search → merchant API (if configured) → web search
- **Electronics chat:** LLM plans terms → multi-query retrieve → LLM pick → LLM reply

Tune via `MIN_RERANKER_SCORE` and `MIN_SEARCH_SCORE`.

## Key modules

| Module | Responsibility |
|---|---|
| `app/main.py` | FastAPI routes |
| `app/orchestrator/` | LangGraph split → clarify → trip planner → Send agents → merge |
| `app/orchestrator/trip_planner.py` | Camping/road-trip breakdown across grocery/clothing/electronics/other |
| `app/orchestrator/clarify.py` | Ask-back gate (family, kids food, care items, weather, make-vs-buy) |
| `app/party_planner/` | Grocery fetch/compare subgraph helpers |
| `app/catalog_pick.py` | Electronics LLM plan/pick/reply |
| `app/merchants/` | Optional Kroger/Walmart API clients |
| `app/search.py` | Azure AI Search client |
| `app/llm_client.py` | Shared Azure OpenAI / Ollama completions |
| `app/routing.py` | Heuristic helpers (fallback split / legacy hints) |
| `app/advisory.py` | Out-of-catalog shopping lists |
