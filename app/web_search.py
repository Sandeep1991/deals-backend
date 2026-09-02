from __future__ import annotations

import re
from html import unescape
from urllib.parse import quote_plus

import httpx

from app.models import Ad
from app.pricing import parse_price

MERCHANT_SITES = {
    "Walmart": "walmart.com",
    "Kroger": "kroger.com",
}

RESULT_LINK_RE = re.compile(r'class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', re.IGNORECASE)
SNIPPET_RE = re.compile(r'class="result__snippet"[^>]*>([\s\S]*?)</(?:a|td|div)>', re.IGNORECASE)
PRICE_IN_TEXT_RE = re.compile(r"\$\s?\d+(?:\.\d{2})?")


async def search_merchant_product(merchant: str, product: str) -> Ad | None:
    site = MERCHANT_SITES.get(merchant)
    if not site:
        return None

    query = f"site:{site} {product}"
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(
                url,
                headers={"User-Agent": "DealFinder/1.0 (+https://thetinkerer.xyz)"},
            )
            response.raise_for_status()
            html = response.text
    except Exception:
        return _fallback_search_ad(merchant, product)

    links = RESULT_LINK_RE.findall(html)
    snippets = SNIPPET_RE.findall(html)

    title = product.title()
    link = f"https://www.{site}/search?q={quote_plus(product)}"
    price = ""

    for href, link_title in links[:5]:
        if site not in href:
            continue
        title = unescape(link_title.strip()) or title
        link = href
        break

    for snippet in snippets[:3]:
        text = unescape(re.sub(r"<[^>]+>", " ", snippet))
        match = PRICE_IN_TEXT_RE.search(text)
        if match:
            price = match.group(0).replace(" ", "")
            break

    if not price:
        price = "See site"

    ad_id = f"web-{merchant.lower()}-{re.sub(r'[^a-z0-9]+', '-', product.lower()).strip('-')}"
    return Ad(
        id=ad_id[:120],
        title=title,
        description=f"Found via web search on {merchant} for '{product}'.",
        category="grocery",
        keywords=f"{product},{merchant},web search",
        price=price,
        url=link,
        merchant=merchant,
    )


def _fallback_search_ad(merchant: str, product: str) -> Ad:
    site = MERCHANT_SITES[merchant]
    return Ad(
        id=f"web-{merchant.lower()}-{product.lower().replace(' ', '-')[:40]}",
        title=f"{product.title()} — {merchant}",
        description=f"Search {merchant} for current pricing.",
        category="grocery",
        keywords=f"{product},{merchant}",
        price="See site",
        url=f"https://www.{site}/search?q={quote_plus(product)}",
        merchant=merchant,
    )
