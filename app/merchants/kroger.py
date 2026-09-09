from __future__ import annotations

import re
from urllib.parse import quote_plus

import httpx

from app.config import Settings, get_settings
from app.merchants.base import MerchantProductClient
from app.models import Ad
from app.pricing import parse_price


class KrogerProductClient(MerchantProductClient):
    merchant = "Kroger"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._token: str | None = None
        self._resolved_location: str | None = None

    def is_configured(self) -> bool:
        return bool(self.settings.kroger_client_id and self.settings.kroger_client_secret)

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

    async def _location_id(self, token: str) -> str | None:
        if self._resolved_location:
            return self._resolved_location
        if self.settings.kroger_location_id:
            self._resolved_location = self.settings.kroger_location_id
            return self._resolved_location

        zip_code = (getattr(self.settings, "kroger_zip_code", "") or "").strip()
        if not zip_code:
            return None
        url = (
            "https://api.kroger.com/v1/locations"
            f"?filter.zipCode.near={quote_plus(zip_code)}&filter.limit=1"
        )
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                )
                if response.status_code >= 400:
                    return None
                data = response.json().get("data") or []
                if not data:
                    return None
                loc = str(data[0].get("locationId") or "")
                if loc:
                    self._resolved_location = loc
                    return loc
        except Exception:
            return None
        return None

    def _ad_from_product(self, product: dict, term: str) -> Ad | None:
        desc = product.get("description") or term
        product_id = str(product.get("productId") or product.get("upc") or term)
        items = product.get("items") or [{}]
        price_obj = (items[0] or {}).get("price") or {}
        amount = price_obj.get("promo") or price_obj.get("regular")
        if isinstance(amount, (int, float)):
            price = f"${float(amount):.2f}"
        elif amount:
            price = str(amount)
        else:
            price = "See site"
        if parse_price(price) is None:
            # Prefer skipping unpriced SKUs so the ladder can try web.
            return None
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

    async def search_product(self, term: str) -> Ad | None:
        token = await self._access_token()
        if not token:
            return None
        loc = await self._location_id(token)
        if not loc:
            return None
        url = (
            "https://api.kroger.com/v1/products"
            f"?filter.term={quote_plus(term)}&filter.locationId={quote_plus(loc)}&filter.limit=8"
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
        for product in data:
            ad = self._ad_from_product(product, term)
            if ad:
                return ad
        return None
