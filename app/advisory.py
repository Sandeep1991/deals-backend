from __future__ import annotations

from app.config import Settings, get_settings
from app.llm_client import LLMNotConfiguredError, complete_json, is_decompose_configured
from app.party_planner.state import ShoppingItem, ShoppingPlan

APPROVED_MERCHANTS = ("Kroger", "Walmart")

ADVISORY_SYSTEM = """You help users plan what they need to buy.
Our approved deal partners are Kroger and Walmart (grocery and household focused).

Break the user's request into a practical shopping list — include ALL items they asked about,
even if they are not grocery (e.g. stationery, school supplies, party decorations).

Return JSON with this exact shape:
{
  "event_summary": "short summary of the event or need",
  "required_items": [
    {"name": "item name", "search_terms": ["optional"], "quantity": 1}
  ],
  "alternative_options": []
}

Rules:
- required_items: every product the user should purchase.
- Use clear, short item names (e.g. "glue sticks", "wide-rule notebooks").
- quantity defaults to 1; increase for party sizes when obvious (e.g. 6 people).
- alternative_options: leave empty unless the user explicitly has OR choices.
- Do NOT invent prices or store URLs."""


def _static_stationery_items() -> ShoppingPlan:
    return ShoppingPlan(
        event_summary="Back-to-school stationery",
        required_items=[
            ShoppingItem(name="No. 2 pencils"),
            ShoppingItem(name="Crayons"),
            ShoppingItem(name="Glue sticks"),
            ShoppingItem(name="Safety scissors"),
            ShoppingItem(name="Wide-rule notebooks"),
            ShoppingItem(name="Pocket folders"),
            ShoppingItem(name="Erasers"),
            ShoppingItem(name="Backpack"),
        ],
    )


def _format_item_list(plan: ShoppingPlan) -> list[str]:
    lines: list[str] = []
    for item in plan.required_items:
        qty = f" (×{int(item.quantity)})" if item.quantity and item.quantity > 1 else ""
        lines.append(f"- {item.name}{qty}")
    for option in plan.alternative_options:
        for item in option.items:
            lines.append(f"- {item.name} ({option.label} option)")
    return lines


def _no_deals_footer(in_catalog_scope: bool = False) -> str:
    merchants = " and ".join(APPROVED_MERCHANTS)
    if in_catalog_scope:
        return (
            f"\n\n**Deals note:** Some of these items have no current deals from our approved "
            f"merchants (**{merchants}**) in our catalog right now. Prices shown above are only "
            f"for items we could match."
        )
    return (
        f"\n\n**Deals note:** We don't have current deals on these items from our approved "
        f"merchants (**{merchants}**). Our catalog focuses on grocery and household products. "
        f"You can still use the list above when shopping elsewhere."
    )


def format_advisory_reply(plan: ShoppingPlan, in_catalog_scope: bool = False) -> str:
    lines = [
        f"**{plan.event_summary}**",
        "",
        "**Items you'll need:**",
        *_format_item_list(plan),
        _no_deals_footer(in_catalog_scope=in_catalog_scope),
    ]
    if not in_catalog_scope:
        lines.append(
            "\n\nI *can* compare **grocery** deals at Kroger vs Walmart — try "
            '"lunchbox snacks which store is cheaper?" or "PBJ party deals."'
        )
    return "\n".join(lines)


async def build_advisory_plan(query: str, settings: Settings | None = None) -> ShoppingPlan:
    settings = settings or get_settings()

    if is_decompose_configured(settings):
        try:
            data = await complete_json(
                ADVISORY_SYSTEM,
                f"User request: {query}",
                settings=settings,
            )
            plan = ShoppingPlan.model_validate(data)
            if plan.required_items or plan.alternative_options:
                return plan
        except (LLMNotConfiguredError,):
            raise
        except Exception:
            pass

    # Minimal static fallback when LLM unavailable
    q = query.lower()
    if any(t in q for t in ("school", "stationery", "1st grade", "first grade", "grade")):
        plan = _static_stationery_items()
        plan.event_summary = query.strip()[:100] or plan.event_summary
        return plan

    return ShoppingPlan(
        event_summary=query.strip()[:100] or "Shopping list",
        required_items=[ShoppingItem(name="Items based on your request — configure LLM for a full breakdown")],
    )


async def build_advisory_reply(query: str, in_catalog_scope: bool = False) -> str:
    plan = await build_advisory_plan(query)
    return format_advisory_reply(plan, in_catalog_scope=in_catalog_scope)


def format_missing_catalog_items(item_names: list[str]) -> str:
    if not item_names:
        return ""
    unique = sorted(set(item_names))
    merchants = " and ".join(APPROVED_MERCHANTS)
    bullets = "\n".join(f"- {name}" for name in unique)
    return (
        f"\n\n**No current {merchants} deals in our catalog for:**\n{bullets}\n\n"
        f"_We searched approved merchants but couldn't match weekly deals for these items._"
    )
