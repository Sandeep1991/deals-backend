"""Gather per-intent ReAct clarifications, dedupe similar questions, batch ask-back."""

from __future__ import annotations

from typing import Any

from app.orchestrator.clarify import ClarificationDecision, original_user_ask
from app.orchestrator.planner_clarify import (
    ClarificationNeed,
    format_multi_clarify_reply,
    looks_like_multi_clarify_answer,
)
from app.orchestrator.session_facts import (
    extract_session_facts,
    format_used_facts_markdown,
    guidance_from_facts,
    merge_unique_facts,
)
from app.orchestrator.state import CategoryResult, OrchestratorState


def dedupe_needs(needs: list[ClarificationNeed]) -> list[ClarificationNeed]:
    """Collapse same similarity_key across intents; keep distinct keys."""
    by_key: dict[str, ClarificationNeed] = {}
    no_key: list[ClarificationNeed] = []
    sources: dict[str, list[str]] = {}

    for need in needs:
        key = (need.similarity_key or "").strip()
        if not key:
            no_key.append(need)
            continue
        sources.setdefault(key, []).append(need.id)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = need
            continue
        # Prefer neutral shared wording for party size when multiple intents ask.
        question = existing.question
        if key == "party_size":
            question = (
                "Who is this for — how many adults, children (ages if useful), and pets?"
            )
        by_key[key] = ClarificationNeed(
            id=f"shared.{key}",
            intent=existing.intent,
            question=question,
            options=existing.options or need.options,
            reason=f"deduped from {', '.join(sources[key])}",
            similarity_key=key,
        )

    priority = [
        "party_size",
        "ages",
        "pets",
        "meal_count",
        "kids_food",
        "care_items",
        "weather_gear",
        "power_capacity",
        "fulfillment_path",
    ]
    ordered: list[ClarificationNeed] = []
    seen: set[str] = set()
    for key in priority:
        if key in by_key:
            ordered.append(by_key[key])
            seen.add(key)
    for key, need in by_key.items():
        if key not in seen:
            ordered.append(need)
    ordered.extend(no_key)
    return ordered[:6]


def _flatten_questions(intent_clarifications: list[dict[str, Any]]) -> list[ClarificationNeed]:
    needs: list[ClarificationNeed] = []
    for block in intent_clarifications:
        if block.get("ready") and not block.get("questions"):
            continue
        for raw in block.get("questions") or []:
            try:
                needs.append(ClarificationNeed.model_validate(raw))
            except Exception:
                continue
    return needs


def _merge_guidance(
    intent_clarifications: list[dict[str, Any]],
    *,
    session_fact_line: str = "",
) -> str:
    parts: list[str] = []
    if session_fact_line:
        parts.append(session_fact_line)
    for block in intent_clarifications:
        intent = block.get("intent") or "intent"
        guidance = (block.get("guidance") or "").strip()
        if guidance:
            parts.append(f"[{intent}] {guidance}")
    return "\n".join(parts)


async def gather_clarifications_node(state: OrchestratorState) -> dict:
    """Aggregate ReAct outputs; ask-back if needed; else attach combined guidance."""
    blocks = list(state.get("intent_clarifications") or [])  # type: ignore[arg-type]
    query = state.get("query") or ""
    history = list(state.get("history") or [])
    pref = state.get("preference_summary") if isinstance(state.get("preference_summary"), dict) else None

    used_facts = merge_unique_facts(blocks)
    if not used_facts:
        # Fallback: extract directly so gather still surfaces known session info.
        used_facts = extract_session_facts(
            query=query, history=history, preference_summary=pref
        )
    fact_line = guidance_from_facts(used_facts)
    facts_md = format_used_facts_markdown(used_facts)

    # If every intent is ready (or answering completed), proceed with guidance.
    all_ready = bool(blocks) and all(bool(b.get("ready")) for b in blocks)
    outstanding = _flatten_questions(blocks)
    # Drop outstanding questions already covered by session facts (safety net).
    known_keys = {f.key for f in used_facts}
    outstanding = [n for n in outstanding if n.similarity_key not in known_keys]

    # When answering, intents should have ready=True from react resume path.
    if looks_like_multi_clarify_answer(query, history) and all_ready and not outstanding:
        guidance = _merge_guidance(blocks, session_fact_line=fact_line)
        decision = ClarificationDecision(
            needs_clarification=False,
            resolved_choice=query.strip()[:160],
            planning_guidance=guidance,
            questions=[],
            reason="all intents ready after multi-clarify answers",
        )
        ask = original_user_ask(history, fallback=query)
        return {
            "clarification": decision.model_dump_state(),
            "event_summary": ask[:120] or decision.resolved_choice or "Shopping list",
            "intent_clarifications": blocks,
        }

    if outstanding and not all_ready:
        deduped = dedupe_needs(outstanding)
        prior_guidance = _merge_guidance(blocks, session_fact_line=fact_line)
        decision = ClarificationDecision(
            needs_clarification=True,
            question=deduped[0].question if deduped else "I need a few details.",
            options=list(deduped[0].options) if deduped else [],
            questions=[n.model_dump() for n in deduped],
            planning_guidance=prior_guidance,
            reason="gathered per-intent react questions (deduped); reused session facts",
        )
        reply = format_multi_clarify_reply(deduped, used_facts=used_facts)
        return {
            "clarification": decision.model_dump_state(),
            "category_results": [
                CategoryResult(
                    category="clarify",
                    notes=[
                        "Awaiting user clarification before pricing.",
                        *(
                            [f"Using session facts: {', '.join(f.key for f in used_facts)}"]
                            if used_facts
                            else []
                        ),
                    ],
                    reply_fragment=reply if reply else decision.reply_markdown(),
                )
            ],
            "items": [],
            "intent_clarifications": blocks,
        }

    # No questions — ready to search
    guidance = _merge_guidance(blocks, session_fact_line=fact_line) or (
        "Proceed with known session context; do not invent household size."
    )
    decision = ClarificationDecision(
        needs_clarification=False,
        planning_guidance=guidance,
        reason="all intents clear; using session facts" if used_facts else "all intents clear",
    )
    out: dict[str, Any] = {
        "clarification": decision.model_dump_state(),
        "intent_clarifications": blocks,
    }
    # Soft note for merge/agents that session facts were applied (no ask-back).
    if facts_md and not outstanding:
        out["category_results"] = [
            CategoryResult(
                category="session_context",
                notes=[f"Reused session facts: {', '.join(f'{f.label}={f.value}' for f in used_facts)}"],
                reply_fragment="",
            )
        ]
    return out
