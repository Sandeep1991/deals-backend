from __future__ import annotations

from app.config import Settings, get_settings
from app.llm_client import LLMNotConfiguredError, complete_text, is_decompose_configured
from app.models import SearchResultItem

SYSTEM_PROMPT = """You are DealFinder, a helpful shopping assistant.
Use ONLY the deals provided in the context. Do not invent products, prices, or URLs.
Write a natural 2-4 sentence reply tailored to the user's request.
Mention the best matching deal by name and price.
Do NOT use canned phrases like "I found N deals for ..." or "Click any deal card below".
Do NOT include markdown links or raw URLs — product cards carry the tracking links."""


def build_template_reply(query: str, results: list[SearchResultItem]) -> str:
    """Last-resort fallback when no LLM is configured."""
    if not results:
        return (
            f'I could not find deals matching "{query}". '
            "Try a more specific product name or browse related categories."
        )

    top = results[0].ad
    extras = ""
    if len(results) > 1:
        extras = f" I also found {len(results) - 1} related option(s) in the cards below."
    return f"For \"{query}\", I'd start with {top.title} at {top.price}.{extras}"


def _format_context(results: list[SearchResultItem]) -> str:
    lines: list[str] = []
    for i, item in enumerate(results, start=1):
        ad = item.ad
        lines.append(
            f"{i}. {ad.title} — {ad.price} ({ad.merchant or ad.category})\n"
            f"   {ad.description}"
        )
    return "\n".join(lines)


async def generate_reply(
    query: str,
    results: list[SearchResultItem],
    settings: Settings | None = None,
) -> str:
    """Always prefer the configured decompose LLM; template only if LLM is unavailable."""
    settings = settings or get_settings()

    if not results:
        return build_template_reply(query, results)

    provider = settings.reply_provider.lower().strip()
    # template/auto → use Azure/Ollama whenever decompose LLM is configured
    use_llm = provider in {"auto", "template", "azure_openai", "ollama", ""}

    if use_llm and is_decompose_configured(settings):
        try:
            return await complete_text(
                SYSTEM_PROMPT,
                (
                    f"User query: {query}\n\n"
                    f"Selected deals (best first):\n{_format_context(results)}\n\n"
                    "Write the reply now."
                ),
                settings=settings,
                max_tokens=280,
            )
        except (LLMNotConfiguredError, Exception):
            pass

    return build_template_reply(query, results)
