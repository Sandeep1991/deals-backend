from __future__ import annotations

import re
from html import unescape
from urllib.parse import quote_plus, unquote

import httpx

from app.models import Ad
from app.pricing import parse_price

MERCHANT_SITES = {
    "Walmart": "walmart.com",
    "Kroger": "kroger.com",
}

RESULT_LINK_RE = re.compile(
    r'class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>',
    re.IGNORECASE,
)
SNIPPET_RE = re.compile(
    r'class="result__snippet"[^>]*>([\s\S]*?)</(?:a|td|div)>',
    re.IGNORECASE,
)
# DuckDuckGo often wraps outbound links as /l/?uddg=<urlencoded>
UDDG_RE = re.compile(r"[?&]uddg=([^&]+)", re.IGNORECASE)
PRICE_IN_TEXT_RE = re.compile(
    r"(?:\$\s?\d{1,4}(?:,\d{3})*(?:\.\d{2})?|\d{1,4}(?:\.\d{2})?\s*¢|"
    r"\d+\s+for\s+\$?\d+(?:\.\d{2})?)",
    re.IGNORECASE,
)


def _resolve_result_url(href: str) -> str:
    href = unescape(href.strip())
    match = UDDG_RE.search(href)
    if match:
        return unquote(match.group(1))
    if href.startswith("//"):
        return "https:" + href
    return href


def _strip_html(text: str) -> str:
    return unescape(re.sub(r"<[^>]+>", " ", text or ""))


def _extract_price(*texts: str) -> str:
    """Return the best parseable product price, skipping shipping/fee crumbs."""
    skip_near = re.compile(
        r"(shipping|delivery|fee|tax|subscribe|star|rating|app\b|download)",
        re.I,
    )
    candidates: list[float] = []
    for text in texts:
        cleaned = _strip_html(text)
        for match in PRICE_IN_TEXT_RE.finditer(cleaned):
            start = max(0, match.start() - 40)
            end = min(len(cleaned), match.end() + 40)
            window = cleaned[start:end]
            if skip_near.search(window):
                continue
            amount = parse_price(match.group(0))
            if amount is None:
                continue
            # Grocery shelf prices are rarely $1.00 flat junk from SERPs
            if amount < 1.25 or amount > 80:
                continue
            candidates.append(amount)
    if not candidates:
        return ""
    # Prefer the median-ish first reasonable hit
    amount = candidates[0]
    return f"${amount:.2f}"


async def _fetch_html(url: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            response.raise_for_status()
            return response.text
    except Exception:
        return None


def _parse_ddg_results(html: str, site: str, product: str) -> tuple[str, str, str]:
    links = RESULT_LINK_RE.findall(html)
    snippets = [_strip_html(s) for s in SNIPPET_RE.findall(html)]

    title = product.title()
    link = f"https://www.{site}/search?q={quote_plus(product)}"
    candidate_titles: list[str] = []

    for href, link_title in links[:8]:
        resolved = _resolve_result_url(href)
        if site not in resolved and site not in href:
            continue
        clean_title = _strip_html(link_title).strip() or title
        candidate_titles.append(clean_title)
        if site in resolved:
            title = clean_title
            link = resolved
            break

    price = _extract_price(*(candidate_titles[:5] + snippets[:5]))
    if not price:
        chunks = re.findall(r'class="result[\s\S]{0,1200}', html, flags=re.IGNORECASE)
        price = _extract_price(*chunks[:6])
    return title, link, price


def _parse_bing_results(html: str, site: str, product: str) -> tuple[str, str, str]:
    """Best-effort Bing HTML parse for merchant product + price snippets."""
    title = product.title()
    link = f"https://www.{site}/search?q={quote_plus(product)}"
    # Captures organic result titles/links
    items = re.findall(
        r'<li class="b_algo"[\s\S]*?<h2>\s*<a[^>]+href="([^"]+)"[^>]*>([\s\S]*?)</a>',
        html,
        flags=re.IGNORECASE,
    )
    captions = re.findall(
        r'class="b_caption"[\s\S]*?<p>([\s\S]*?)</p>',
        html,
        flags=re.IGNORECASE,
    )
    titles: list[str] = []
    for href, link_title in items[:8]:
        if site not in href:
            continue
        clean = _strip_html(link_title).strip()
        if clean:
            titles.append(clean)
        if site in href and ("/ip/" in href or "/p/" in href or "search" not in href):
            title = clean or title
            link = href
            break
    if titles and title == product.title():
        title = titles[0]
        # Keep first merchant link even if not a product detail page
        for href, _ in items:
            if site in href:
                link = href
                break

    price = _extract_price(*(titles[:5] + [_strip_html(c) for c in captions[:5]]))
    if not price:
        price = _extract_price(html[:20000])
    return title, link, price


async def search_merchant_product(merchant: str, product: str) -> Ad | None:
    site = MERCHANT_SITES.get(merchant)
    if not site:
        return None

    title = product.title()
    link = f"https://www.{site}/search?q={quote_plus(product)}"
    price = ""

    ddg_url = f"https://html.duckduckgo.com/html/?q={quote_plus(f'site:{site} {product} price')}"
    ddg_html = await _fetch_html(ddg_url)
    if ddg_html:
        title, link, price = _parse_ddg_results(ddg_html, site, product)

    if not price or parse_price(price) is None:
        bing_url = f"https://www.bing.com/search?q={quote_plus(f'site:{site} {product} price')}"
        bing_html = await _fetch_html(bing_url)
        if bing_html:
            b_title, b_link, b_price = _parse_bing_results(bing_html, site, product)
            if parse_price(b_price) is not None:
                title, link, price = b_title, b_link, b_price
            elif not ddg_html:
                title, link, price = b_title, b_link, b_price

    if not price or parse_price(price) is None:
        if not ddg_html:
            return _fallback_search_ad(merchant, product)
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
        source_key="web-search",
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
        source_key="web-search",
    )
