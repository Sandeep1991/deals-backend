from __future__ import annotations

import re
from urllib.parse import quote_plus

import httpx

from app.config import Settings, get_settings
from app.merchants.base import MerchantProductClient
from app.models import Ad


class KrogerProductClient(MerchantProductClient):
    merchant = "Kroger"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._token: str | None = None

    def is_configured(self) -> bool:
        return bool(
            self.settings.kroger_client_id
            and self.settings.kroger_client_secret
            and self.settings.kroger_location_id
        )

    async def _access_token(self) -> str | None:
        if not self.is_configured():
            return None
        if self._token:
            return self._token
        token_url = "https://api.kroger.com/v1/connect/oauth2/token"
        data = {
            "grant_type": "client_credentials",
            "scope": "product.compact",
        }
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    token_url,
                    data=data,
                    auth=(self.settings.kroger_client_id, self.settings.kroger_client_secret),
                )
                response.raise_for_status()
                self._token = response.json().get("access_token")
                return self._token
        except Exception:
            return None

    async def search_product(self, term: str) -> Ad | None:
        token = await self._access_token()
        if not token:
            return None
        loc = self.settings.kroger_location_id
        url = (
            "https://api.kroger.com/v1/products"
            f"?filter.term={quote_plus(term)}&filter.locationId={quote_plus(loc)}&filter.limit=5"
        )
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                )
                if response.status_code >= 400:
                    return None
                payload = response.json()
        except Exception:
            return None

        data = payload.get("data") or []
        if not data:
            return None
        product = data[0]
        desc = product.get("description") or term
        product_id = str(product.get("productId") or product.get("upc") or term)
        items = product.get("items") or [{}]
        price_obj = (items[0] or {}).get("price") or {}
        amount = price_obj.get("promo") or price_obj.get("regular")
        price = f"${amount:.2f}" if isinstance(amount, (int, float)) else (str(amount) if amount else "See site")
        slug = re.sub(r"[^a-z0-9]+", "-", desc.lower()).strip("-")[:60]
        return Ad(
            id=f"kroger-api-{product_id}"[:120],
            title=desc,
            description=f"Kroger API result for '{term}'.",
            category="grocery",
            keywords=f"{term},kroger,api",
            price=price,
            url=f"https://www.kroger.com/p/{slug}/{product_id}",
            merchant="Kroger",
            network="",
            brand="",
            source_key="kroger-api",
        )
