from __future__ import annotations

from app.merchants.kroger import KrogerProductClient
from app.merchants.walmart import WalmartProductClient


def client_for_merchant(merchant: str):
    if merchant == "Kroger":
        return KrogerProductClient()
    if merchant == "Walmart":
        return WalmartProductClient()
    return None
