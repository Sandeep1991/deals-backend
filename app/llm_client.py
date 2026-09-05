from __future__ import annotations

import json
import re
from typing import Any, Optional

import httpx

from app.config import Settings, get_settings

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

LLM_SETUP_HINT = (
    "Store comparison requires an LLM for ingredient planning. "
    "Set DECOMPOSE_PROVIDER=azure_openai with AZURE_OPENAI_* credentials, "
    "or DECOMPOSE_PROVIDER=ollama with Ollama running locally."
)


class LLMNotConfiguredError(RuntimeError):
    pass


def resolve_decompose_provider(settings: Optional[Settings] = None) -> Optional[str]:
    """Return azure_openai | ollama | None based on DECOMPOSE_PROVIDER."""
    settings = settings or get_settings()
    provider = settings.decompose_provider.lower().strip()

    if provider == "template":
        return None
    if provider == "azure_openai":
        if settings.azure_openai_endpoint and settings.azure_openai_api_key:
            return "azure_openai"
        return None
    if provider == "ollama":
        return "ollama"

    # auto: prefer Azure OpenAI, then Ollama
    if settings.azure_openai_endpoint and settings.azure_openai_api_key:
        return "azure_openai"
    if settings.ollama_base_url:
        return "ollama"
    return None


def is_decompose_configured(settings: Optional[Settings] = None) -> bool:
    return resolve_decompose_provider(settings) is not None


async def complete_json(
    system: str,
    user: str,
    settings: Settings | None = None,
    max_tokens: int = 1200,
) -> dict[str, Any]:
    settings = settings or get_settings()
    provider = resolve_decompose_provider(settings)

    if provider == "azure_openai":
        text = await _azure_openai_complete(system, user, settings, max_tokens, json_mode=True)
    elif provider == "ollama":
        text = await _ollama_complete(system, user, settings, max_tokens, json_mode=True)
    else:
        raise LLMNotConfiguredError(LLM_SETUP_HINT)

    return _parse_json(text)


async def complete_text(
    system: str,
    user: str,
    settings: Settings | None = None,
    max_tokens: int = 400,
    temperature: float = 0.2,
) -> str:
    """Free-form chat completion using the same provider as decompose."""
    settings = settings or get_settings()
    provider = resolve_decompose_provider(settings)

    if provider == "azure_openai":
        return await _azure_openai_complete(
            system, user, settings, max_tokens, json_mode=False, temperature=temperature
        )
    if provider == "ollama":
        return await _ollama_complete(
            system, user, settings, max_tokens, json_mode=False, temperature=temperature
        )
    raise LLMNotConfiguredError(LLM_SETUP_HINT)


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    block = JSON_BLOCK_RE.search(text)
    if block:
        text = block.group(1).strip()
    return json.loads(text)


async def _ollama_complete(
    system: str,
    user: str,
    settings: Settings,
    max_tokens: int,
    json_mode: bool = True,
    temperature: float | None = None,
) -> str:
    temp = 0.1 if json_mode else 0.2
    if temperature is not None:
        temp = temperature
    payload: dict[str, Any] = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temp, "num_predict": max_tokens},
    }
    if json_mode:
        payload["format"] = "json"
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()


async def _azure_openai_complete(
    system: str,
    user: str,
    settings: Settings,
    max_tokens: int,
    json_mode: bool = True,
    temperature: float | None = None,
) -> str:
    base = settings.azure_openai_endpoint.rstrip("/")
    # Strip Foundry project path if user pasted project URL by mistake
    if "/api/projects/" in base:
        base = base.split("/api/projects/")[0]

    style = settings.azure_openai_api_style.lower().strip()
    if style == "auto":
        style = "v1" if ".services.ai.azure.com" in base else "legacy"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    temp = 0.1 if json_mode else 0.2
    if temperature is not None:
        temp = temperature
    body: dict[str, Any] = {
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    if style == "v1":
        url = f"{base}/openai/v1/chat/completions"
        body["model"] = settings.azure_openai_deployment
    else:
        url = (
            f"{base}/openai/deployments/{settings.azure_openai_deployment}/chat/completions"
            f"?api-version={settings.azure_openai_api_version}"
        )

    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(
            url,
            headers={"api-key": settings.azure_openai_api_key},
            json=body,
        )
        if response.status_code >= 400:
            detail = response.text[:500]
            raise RuntimeError(f"Azure LLM error {response.status_code}: {detail}")
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
