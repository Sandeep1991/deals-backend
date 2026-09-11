from __future__ import annotations

from app.compare import to_compare_response
from app.llm_client import LLMNotConfiguredError, complete_text, is_decompose_configured
from app.models import Ad
from app.orchestrator.state import CategoryResult, OrchestratorState

MERGE_SYSTEM = """You are DealFinder. Merge category agent results into one helpful reply.
Write 3-6 sentences (or short markdown sections) that answer the user's request.
Use ONLY the provided category results — do not invent products, prices, or URLs.
When a product is listed with a markdown link, keep that exact [title](url) in your reply
so the shopper can open the deal. Do not invent or rewrite URLs.
If grocery comparison exists, briefly mention which store is cheaper when clear and name
2-4 of the priced grocery items (not just "snacks").
If grocery items could not be priced (empty quotes / "see item prices" / no dollar amounts),
say prices are unavailable right now and do NOT invent typical prices like $1.00 or $1.50.
If this turn used a preference summary rewrite, mention that the prior list was updated
for those preferences and highlight a few replacements + which store wins.
If electronics and grocery both appear, cover both needs.
If prior conversation is provided, answer as a follow-up in that thread.
If planning guidance or notes say session facts were reused (household size, power capacity,
kids food, weather, etc.), briefly say you are using those specific details when explaining
why you chose the list / offers (e.g. "Using your earlier note of 2 adults + 1 kid…").
If a category note says clarification is needed, or reply_fragment asks the user to choose
among options (including multiple lettered questions A/B/C for different intents), ask
those questions clearly and do NOT invent a shopping list, prices, or store comparison yet.
When the clarify reply includes "**Using from this chat:**", keep that section visible.
Keep lettered sections and numbered options visible.
Avoid canned phrases like "I found N deals" or "click any deal card"."""


def _flatten_ads(results: list[CategoryResult]) -> list[Ad]:
    seen: set[str] = set()
    ads: list[Ad] = []
    for result in results:
        for ad in result.ads:
            if ad.id in seen:
                continue
            seen.add(ad.id)
            ads.append(ad)
        if result.comparison:
            for basket in result.comparison.merchants:
                for quote in basket.quotes:
                    if quote.ad.id in seen:
                        continue
                    seen.add(quote.ad.id)
                    ads.append(quote.ad)
    return ads


def _pick_mode(results: list[CategoryResult]) -> str:
    if any(r.category == "clarify" or "clarification" in " ".join(r.notes).lower() for r in results):
        return "advisory"
    cats = {r.category for r in results if r.items or r.ads or r.comparison or r.notes}
    has_grocery_compare = any(r.category == "grocery" and r.comparison for r in results)
    has_electronics = any(r.category == "electronics" and r.ads for r in results)
    has_searchish = any(r.category in {"electronics", "clothing"} and r.ads for r in results)
    has_advisory = any(r.category in {"stationery", "other", "clothing"} and not r.ads for r in results)

    if has_grocery_compare and has_searchish:
        return "mixed"
    if has_grocery_compare and len(cats) == 1:
        return "compare"
    if has_searchish and not has_grocery_compare:
        return "search"
    if has_advisory and not has_grocery_compare and not has_searchish:
        return "advisory"
    if has_grocery_compare:
        return "compare"
    if has_electronics:
        return "search"
    return "mixed" if len(cats) > 1 else "search"


def _ad_line(ad: Ad) -> str:
    label = f"[{ad.title}]({ad.url})" if ad.url else ad.title
    merchant = f" ({ad.merchant})" if ad.merchant else ""
    return f"- {label} — {ad.price}{merchant}"


def _ensure_product_links(reply: str, ads: list[Ad]) -> str:
    """If the merge model named a product but dropped its URL, weave trusted links back in."""
    text = (reply or "").strip()
    if not text or not ads:
        return text
    for ad in ads:
        if not ad.url or not ad.title:
            continue
        linked = f"[{ad.title}]({ad.url})"
        if linked in text:
            continue
        if ad.title in text:
            text = text.replace(ad.title, linked, 1)
    return text


def _template_merge(query: str, summary: str, results: list[CategoryResult]) -> str:
    parts: list[str] = []
    if summary:
        parts.append(f"**{summary}**")
    # Surface reused session facts before priced results when present.
    for result in results:
        if result.category == "session_context" and result.notes:
            parts.append("**Using from this chat:**")
            for note in result.notes:
                if note.lower().startswith("reused session facts:"):
                    detail = note.split(":", 1)[-1].strip()
                    for bit in detail.split(","):
                        bit = bit.strip()
                        if bit:
                            parts.append(f"- {bit}")
                else:
                    parts.append(f"- {note}")
    for result in results:
        if result.category == "session_context":
            continue
        if result.reply_fragment:
            parts.append(result.reply_fragment)
            continue
        if result.comparison and result.comparison.reply:
            parts.append(result.comparison.reply)
            continue
        if result.ads:
            top = result.ads[0]
            linked = f"[{top.title}]({top.url})" if top.url else top.title
            parts.append(
                f"For {result.category}, start with {linked} at {top.price}."
            )
        elif result.notes:
            parts.append(" ".join(result.notes))
    return "\n\n".join(p for p in parts if p) or f'I could not build a full answer for "{query}".'


async def merge_results_node(state: OrchestratorState) -> dict:
    query = state["query"]
    summary = state.get("event_summary") or query
    results = list(state.get("category_results") or [])
    ads = _flatten_ads(results)
    mode = _pick_mode(results)

    grocery = next((r for r in results if r.category == "grocery" and r.comparison), None)
    comparison = grocery.comparison if grocery else None

    # Clarification turns: return the ask-back text as-is (no priced comparison UI).
    clarify = next(
        (
            r
            for r in results
            if r.category == "clarify"
            or (
                r.reply_fragment
                and any("clarification" in n.lower() for n in r.notes)
            )
        ),
        None,
    )
    if clarify and clarify.reply_fragment:
        return {
            "reply": clarify.reply_fragment.strip(),
            "ads": [],
            "mode": "advisory",
            "comparison": None,
        }

    # Also honor structured clarification on state (even if category_results odd)
    from app.orchestrator.clarify import clarification_from_raw

    decision = clarification_from_raw(state.get("clarification"))
    if decision and decision.needs_clarification:
        return {
            "reply": decision.reply_markdown(),
            "ads": [],
            "mode": "advisory",
            "comparison": None,
        }

    context_lines: list[str] = [f"User request: {query}", f"Summary: {summary}", ""]
    history = list(state.get("history") or [])
    if history:
        from app.orchestrator.memory import format_history_block

        block = format_history_block(history)
        if block:
            context_lines.insert(0, block)
            context_lines.insert(1, "")
    pref = state.get("preference_summary")
    if isinstance(pref, dict) and (pref.get("summary") or pref.get("preferences")):
        context_lines.insert(0, f"Preference summary: {pref.get('summary') or ''}")
        if pref.get("preferences"):
            context_lines.insert(1, f"Active preferences: {', '.join(pref['preferences'])}")
        context_lines.insert(2, "")
    decision_for_facts = clarification_from_raw(state.get("clarification"))
    if decision_for_facts and (decision_for_facts.planning_guidance or "").strip():
        context_lines.insert(
            0,
            "Planning guidance / session facts to cite when relevant:\n"
            + decision_for_facts.planning_guidance.strip(),
        )
        context_lines.insert(1, "")
    for result in results:
        context_lines.append(f"## {result.category}")
        if result.reply_fragment:
            context_lines.append(result.reply_fragment)
        if result.comparison and result.comparison.reply:
            # Keep shopping-list + linked products for the merge model
            context_lines.append(result.comparison.reply[:2400])
        for ad in result.ads[:5]:
            context_lines.append(_ad_line(ad))
        for note in result.notes:
            context_lines.append(f"- note: {note}")
        context_lines.append("")

    reply = ""
    if is_decompose_configured():
        try:
            reply = await complete_text(
                MERGE_SYSTEM,
                "\n".join(context_lines) + "\nWrite the merged reply now.",
                max_tokens=650,
                temperature=0.45,
            )
        except (LLMNotConfiguredError, Exception):
            reply = ""
    if not reply:
        reply = _template_merge(query, summary, results)

    # Prefer grocery compare reply body when mode is pure compare and merge was thin
    if mode == "compare" and comparison and comparison.reply and len(reply) < 80:
        reply = comparison.reply

    reply = _ensure_product_links(reply, ads)

    out: dict = {
        "reply": reply,
        "ads": ads,
        "mode": mode,
        "comparison": comparison,
    }
    return out


def to_chat_payload(state: OrchestratorState) -> dict:
    """Map orchestrator state into ChatResponse fields."""
    comparison = state.get("comparison")
    compare_out = to_compare_response(comparison) if comparison else None
    return {
        "query": state["query"],
        "reply": state.get("reply") or "",
        "ads": state.get("ads") or [],
        "mode": state.get("mode") or "search",
        "comparison": compare_out,
        "chat_id": state.get("chat_id") or None,
        "preference_summary": state.get("preference_summary"),
    }
