from __future__ import annotations

import re
from urllib.parse import quote_plus

import httpx

from app.config import Settings, get_settings
from app.merchants.base import MerchantProductClient
from app.models import Ad


class WalmartProductClient(MerchantProductClient):
    merchant = "Walmart"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def is_configured(self) -> bool:
        return bool(self.settings.walmart_api_key)

    async def search_product(self, term: str) -> Ad | None:
        if not self.is_configured():
            return None

        base = self.settings.walmart_api_base.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                params = {
                    "query": term,
                    "numItems": "5",
                    "format": "json",
                    "apiKey": self.settings.walmart_api_key,
                }
                if self.settings.walmart_publisher_id:
                    params["publisherId"] = self.settings.walmart_publisher_id
                response = await client.get(
                    base,
                    params=params,
                    headers={"Accept": "application/json"},
                )
                if response.status_code >= 400:
                    return None
                payload = response.json()
        except Exception:
            return None

        items = payload.get("items") or payload.get("data") or payload.get("products") or []
        if not items and isinstance(payload.get("response"), dict):
            items = payload["response"].get("items") or []
        if not items:
            return None

        item = items[0]
        title = item.get("name") or item.get("title") or term
        item_id = str(item.get("itemId") or item.get("id") or item.get("sku") or term)
        sale = item.get("salePrice") or item.get("price") or item.get("currentPrice")
        if isinstance(sale, dict):
            sale = sale.get("amount") or sale.get("value")
        price = f"${float(sale):.2f}" if isinstance(sale, (int, float)) else (
            sale if isinstance(sale, str) and sale.strip() else "See site"
        )
        product_url = (
            item.get("productUrl")
            or item.get("url")
            or f"https://www.walmart.com/search?q={quote_plus(term)}"
        )
        safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", item_id)[:80]
        return Ad(
            id=f"walmart-api-{safe_id}"[:120],
            title=str(title),
            description=f"Walmart API result for '{term}'.",
            category="grocery",
            keywords=f"{term},walmart,api",
            price=str(price),
            url=str(product_url),
            merchant="Walmart",
            network="",
            brand="",
            source_key="walmart-api",
        )
