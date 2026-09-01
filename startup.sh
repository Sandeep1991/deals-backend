#!/bin/bash
# Azure App Service startup (set as startup command in portal or use as-is with Oryx)
gunicorn app.main:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind "0.0.0.0:${PORT:-8000}" \
  --timeout 120
