from __future__ import annotations

import httpx

from app.config import Settings, get_settings
from app.models import SearchResultItem

SYSTEM_PROMPT = """You are DealFinder, a concise shopping assistant.
Use ONLY the deals provided in the context. Do not invent products, prices, or URLs.
Keep replies to 2-3 sentences. Mention the best matching deal by name and price when relevant."""


def build_template_reply(query: str, results: list[SearchResultItem]) -> str:
    if not results:
        return (
            f'I could not find deals matching "{query}". '
            "Try searching for tea, soap, coffee, or household items."
        )

    count = len(results)
    plural = "deal" if count == 1 else "deals"
    top = results[0].ad
    intro = f'I found {count} {plural} for "{query}".'

    if results[0].score >= 1.5:
        return (
            f"{intro} The best match is [{top.title}]({top.url}) at {top.price}. "
            "Click any deal card below to visit the partner site."
        )

    return f"{intro} Here are the closest matches I found. Click a card below to visit the partner site."


def _format_context(results: list[SearchResultItem]) -> str:
    lines: list[str] = []
    for i, item in enumerate(results, start=1):
        ad = item.ad
        lines.append(
            f"{i}. {ad.title} — {ad.price} ({ad.category})\n"
            f"   {ad.description}\n"
            f"   URL: {ad.url}"
        )
    return "\n".join(lines)


async def generate_reply(query: str, results: list[SearchResultItem], settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    provider = settings.reply_provider.lower().strip()

    if not results:
        return build_template_reply(query, results)

    if provider == "ollama":
        try:
            return await _ollama_reply(query, results, settings)
        except Exception:
            return build_template_reply(query, results)

    if provider == "azure_openai":
        try:
            return await _azure_openai_reply(query, results, settings)
        except Exception:
            return build_template_reply(query, results)

    return build_template_reply(query, results)


async def _ollama_reply(query: str, results: list[SearchResultItem], settings: Settings) -> str:
    prompt = (
        f"User query: {query}\n\n"
        f"Available deals:\n{_format_context(results)}\n\n"
        "Write a helpful reply."
    )
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 150},
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()


async def _azure_openai_reply(query: str, results: list[SearchResultItem], settings: Settings) -> str:
    if not settings.azure_openai_endpoint or not settings.azure_openai_api_key:
        raise RuntimeError("Azure OpenAI is not configured")

    prompt = (
        f"User query: {query}\n\n"
        f"Available deals:\n{_format_context(results)}\n\n"
        "Write a helpful reply."
    )
    url = (
        f"{settings.azure_openai_endpoint.rstrip('/')}/openai/deployments/"
        f"{settings.azure_openai_deployment}/chat/completions"
        f"?api-version={settings.azure_openai_api_version}"
    )
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            url,
            headers={"api-key": settings.azure_openai_api_key},
            json={
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 150,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
