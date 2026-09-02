from __future__ import annotations

import re

_PRICE_RE = re.compile(
    r"(?P<whole>\d+)\s*(?:for|/)\s*\$?(?P<bundle>\d+(?:\.\d{2})?)|"
    r"\$?(?P<dollars>\d+(?:\.\d{2})?)|"
    r"(?P<cents>\d+)\s*¢"
)


def parse_price(price: str) -> float | None:
    """Parse ad price strings into a numeric dollar amount when possible."""
    text = price.strip().lower()
    if not text:
        return None
    if any(token in text for token in ("buy", "free", "rollback", "from")):
        # Still try numeric extraction for strings like "From $3.98"
        pass

    bundle = re.search(r"(?P<count>\d+)\s+for\s+\$?(?P<amount>\d+(?:\.\d{2})?)", text)
    if bundle:
        count = float(bundle.group("count"))
        amount = float(bundle.group("amount"))
        if count > 0:
            return round(amount / count, 2)

    cents = re.search(r"(?P<cents>\d+(?:\.\d{2})?)\s*¢", text)
    if cents:
        return round(float(cents.group("cents")) / 100, 2)

    dollars = re.search(r"\$?(?P<amount>\d+(?:\.\d{2})?)", text)
    if dollars:
        return round(float(dollars.group("amount")), 2)

    return None
