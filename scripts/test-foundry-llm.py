#!/usr/bin/env python3
"""Test Azure AI Foundry LLM connectivity. Reads credentials from deals-backend/.env"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.llm_client import complete_json
from app.party_planner.decompose import DECOMPOSE_SYSTEM


async def main() -> int:
    settings = get_settings()
    if not settings.azure_openai_endpoint or not settings.azure_openai_api_key:
        print("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env")
        return 1

    print(f"Endpoint: {settings.azure_openai_endpoint}")
    print(f"Deployment: {settings.azure_openai_deployment}")
    print(f"API style: {settings.azure_openai_api_style}")

    query = "taco night for 4 people — which ingredients do I need?"
    try:
        data = await complete_json(DECOMPOSE_SYSTEM, f"User request: {query}", settings=settings)
        print("\nLLM decompose OK:")
        print(json.dumps(data, indent=2)[:1500])
        return 0
    except Exception as exc:
        print(f"\nFAILED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
