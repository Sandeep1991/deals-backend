#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo ""
  echo "Created backend/.env from .env.example"
  echo "Edit backend/.env and set AZURE_SEARCH_API_KEY before calling /api/chat"
  echo ""
fi

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi

echo "Starting API at http://localhost:8000 (docs: http://localhost:8000/docs)"
exec .venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
