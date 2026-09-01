# DealFinder API (deals-backend)

FastAPI backend for Azure AI Search retrieval and optional RAG replies.

Frontend repo: [Sandeep1991/deals](https://github.com/Sandeep1991/deals)

## Architecture

```
React  →  FastAPI (this repo)  →  Azure AI Search
                ↓
         template / Ollama / Azure OpenAI (reply only)
```

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
cp .env.example .env   # set AZURE_SEARCH_API_KEY
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/chat` | Search + generate reply |
| POST | `/api/search` | Search only |
| POST | `/api/ads` | Bulk upsert ads |
| PUT | `/api/ads/{id}` | Upsert single ad |
| DELETE | `/api/ads/{id}` | Delete ad |

## Environment variables

See `.env.example`. Required:

- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_API_KEY`
- `AZURE_SEARCH_INDEX` (default: `ads`)
- `AZURE_SEARCH_SEMANTIC_CONFIG` (default: `ads-semantic`)

## Search behavior

- **Literal queries** (`shower`, `soap`): keyword search + score threshold
- **Meaning queries** (`discount`, `deal`): hybrid + semantic + vector
- **Fallback**: retries with hybrid if keyword returns nothing

Tune via `MIN_RERANKER_SCORE` and `MIN_SEARCH_SCORE`.
