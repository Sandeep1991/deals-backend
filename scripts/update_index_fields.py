#!/usr/bin/env python3
"""Add network/brand/source_key/expires_at fields to the Azure AI Search ads index."""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    for root in (Path(__file__).resolve().parents[1], Path.home() / "src" / "deals-backend"):
        env = root / ".env"
        if env.exists():
            load_dotenv(env)

ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "").rstrip("/")
KEY = os.getenv("AZURE_SEARCH_API_KEY", "")
INDEX = os.getenv("AZURE_SEARCH_INDEX", "ads")
API = "2024-07-01"

NEW_FIELDS = [
    {"name": "network", "type": "Edm.String", "filterable": True, "facetable": True, "searchable": True},
    {"name": "brand", "type": "Edm.String", "filterable": True, "facetable": True, "searchable": True},
    {"name": "source_key", "type": "Edm.String", "filterable": True, "facetable": True, "searchable": False},
    {"name": "expires_at", "type": "Edm.String", "filterable": True, "sortable": True, "searchable": False},
]


def main() -> int:
    if not ENDPOINT or not KEY:
        print("Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY", file=sys.stderr)
        return 1

    url = f"{ENDPOINT}/indexes/{INDEX}?api-version={API}"
    req = urllib.request.Request(url, headers={"api-key": KEY})
    with urllib.request.urlopen(req) as resp:
        index = json.loads(resp.read().decode())

    existing = {f["name"] for f in index.get("fields", [])}
    added = []
    for field in NEW_FIELDS:
        if field["name"] not in existing:
            index["fields"].append(field)
            added.append(field["name"])

    if not added:
        print(f"Index {INDEX} already has network/brand/source_key/expires_at")
        return 0

    put = urllib.request.Request(
        url,
        data=json.dumps(index).encode(),
        headers={"Content-Type": "application/json", "api-key": KEY},
        method="PUT",
    )
    with urllib.request.urlopen(put) as resp:
        resp.read()
    print(f"Updated index {INDEX}; added fields: {', '.join(added)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
