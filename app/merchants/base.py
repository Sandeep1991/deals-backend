from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import Ad


class MerchantProductClient(ABC):
    merchant: str

    @abstractmethod
    def is_configured(self) -> bool:
        ...

    @abstractmethod
    async def search_product(self, term: str) -> Ad | None:
        """Return a single best Ad or None if unavailable / unconfigured / error."""
        ...
