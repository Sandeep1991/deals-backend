"""Per-intent ReAct clarification loop.

Each intent runs a short think→tool→observe loop that can emit ask_user questions
or mark_intent_ready. Questions are recorded only — HTTP does not block mid-loop;
gather_clarifications batches them for the user.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.llm_client import LLMNotConfiguredError, complete_json, is_decompose_configured
from app.orchestrator.memory import format_history_block
from app.orchestrator.planner_clarify import (
    ClarificationNeed,
    IntentName,
    clothing_planner_needs,
    electronics_planner_needs,
    grocery_planner_needs,
    looks_like_multi_clarify_answer,
    trip_planner_needs,
)
from app.orchestrator.preferences import preference_summary_from_raw
from app.orchestrator.state import ChatTurn, CompositeItem, OrchestratorState

MAX_REACT_STEPS = 4

REACT_SYSTEM = """You are a DealFinder intent clarifier using a ReAct loop for ONE shopping intent.
Decide whether you have enough information to search product offers, or must ask the user.

Return JSON only for ONE action per step:
{
  "thought": "brief reasoning",
  "tool": "get_known_context" | "ask_user" | "mark_intent_ready",
  "ask_user": {
    "id": "intent.topic",
    "similarity_key": "party_size|kids_food|care_items|weather_gear|power_capacity|fulfillment_path|ages|pets|meal_count",
    "question": "plain question",
    "options": ["opt1", "opt2"],
    "reason": "why"
  },
  "guidance": "planning guidance when marking ready (include/exclude). Cite session facts you are using."
}

Rules:
- If session facts already answer a seed question, do NOT ask_user for that topic — use the fact.
- Prefer suggested seed questions only when still unresolved after session facts.
- Use similarity_key so overlapping asks (party size / ages) can be deduped across intents.
- Do NOT ask ready-made vs DIY for electronics/power — ask capacity/use-case instead.
- mark_intent_ready when this intent can search; guidance MUST name the specific session facts used
  (e.g. "Using household 2 adults + 1 kid from chat; weekend power station from earlier answer").
- ask_user at most 2 times in this loop; then mark ready or stop with remaining questions recorded.
"""


class IntentClarification(BaseModel):
    intent: str
    ready: bool = False
    questions: list[dict[str, Any]] = Field(default_factory=list)
    guidance: str = ""
    reason: str = ""
    # Session facts reused instead of re-asking (shown to the user).
    used_facts: list[dict[str, Any]] = Field(default_factory=list)

    def model_dump_state(self) -> dict[str, Any]:
        return self.model_dump()


def _seed_needs_for_intent(
    intent: IntentName,
    *,
    query: str,
    history: list[ChatTurn] | None,
    preference_summary: dict | None,
    items: list[CompositeItem] | None,
) -> list[ClarificationNeed]:
    if intent == "trip":
        return trip_planner_needs(
            query=query, history=history, preference_summary=preference_summary
        )
    if intent == "grocery":
        return grocery_planner_needs(
            query=query,
            history=history,
            preference_summary=preference_summary,
            items=items,
        )
    if intent == "electronics":
        return electronics_planner_needs(
            query=query,
            history=history,
            preference_summary=preference_summary,
            items=items,
        )
    if intent == "clothing":
        return clothing_planner_needs(
            query=query,
            history=history,
            preference_summary=preference_summary,
            items=items,
        )
    return []


def _context_observation(
    *,
    intent: str,
    query: str,
    history: list[ChatTurn] | None,
    preference_summary: dict | None,
    items: list[CompositeItem] | None,
    seeds: list[ClarificationNeed],
    used_facts: list | None = None,
) -> str:
    from app.orchestrator.clarify import original_user_ask
    from app.orchestrator.session_facts import format_used_facts_markdown

    ask = original_user_ask(history, fallback=query)
    pref = preference_summary_from_raw(preference_summary)
    hist = format_history_block(history or [])
    item_bits = ", ".join(f"{i.name}[{i.category}]" for i in (items or [])[:12]) or "(none)"
    seed_bits = "\n".join(
        f"- {s.id} key={s.similarity_key}: {s.question} opts={s.options}" for s in seeds
    ) or "(none)"
    pref_bits = ""
    if pref and (pref.summary or pref.preferences):
        pref_bits = f"Preferences: {pref.summary}; {', '.join(pref.preferences)}\n"
    facts_md = format_used_facts_markdown(used_facts or []) or "(none yet)"
    return (
        f"Intent: {intent}\n"
        f"Original ask: {ask}\n"
        f"Latest message: {query}\n"
        f"{pref_bits}"
        f"Split items: {item_bits}\n"
        f"Already-known session facts (do NOT re-ask these):\n{facts_md}\n"
        f"Seed questions still unresolved:\n{seed_bits}\n"
        f"{hist}"
    )


async def _resolve_answers_for_intent(
    *,
    intent: str,
    query: str,
    history: list[ChatTurn] | None,
    seeds: list[ClarificationNeed],
) -> IntentClarification:
    """Map a multi-clarify user reply into guidance for one intent."""
    from app.orchestrator.clarify import original_user_ask
    from app.orchestrator.planner_clarify import MULTI_ANSWER_SYSTEM

    ask = original_user_ask(history, fallback=query)
    seed_blurb = "\n".join(
        f"{chr(ord('A') + i)}. [{s.intent}/{s.similarity_key}] {s.question} options={s.options}"
        for i, s in enumerate(seeds)
    ) or "(no prior seeds)"
    guidance = (
        f"Intent={intent}. Original: {ask}. User answers: {query.strip()}"
    )
    if is_decompose_configured():
        try:
            data = await complete_json(
                MULTI_ANSWER_SYSTEM,
                (
                    f"Focus on intent '{intent}' only.\n"
                    f"Original request:\n{ask}\n\n"
                    f"Questions (all intents; extract what matters for {intent}):\n{seed_blurb}\n\n"
                    f"User answers:\n{query.strip()}\n\n"
                    "Return JSON with planning_guidance for this intent."
                ),
                max_tokens=700,
            )
            if data.get("planning_guidance"):
                guidance = str(data["planning_guidance"])
        except (LLMNotConfiguredError, Exception):
            pass
    return IntentClarification(
        intent=intent,
        ready=True,
        questions=[],
        guidance=guidance,
        reason="resolved from multi-clarify answers",
    )


async def run_intent_react(
    *,
    intent: IntentName,
    query: str,
    history: list[ChatTurn] | None,
    preference_summary: dict | None,
    items: list[CompositeItem] | None,
) -> IntentClarification:
    """Run ReAct clarify for a single intent; emit questions or mark ready."""
    from app.orchestrator.session_facts import (
        extract_session_facts,
        facts_for_intent,
        guidance_from_facts,
    )

    intent_items = [i for i in (items or []) if _item_matches_intent(i.category, intent)]
    all_facts = extract_session_facts(
        query=query, history=history, preference_summary=preference_summary
    )
    used_facts = facts_for_intent(all_facts, intent)
    used_facts_dump = [f.model_dump_state() for f in used_facts]
    fact_guidance = guidance_from_facts(used_facts, intent=intent)

    seeds = _seed_needs_for_intent(
        intent,
        query=query,
        history=history,
        preference_summary=preference_summary,
        items=items,
    )
    # Never re-ask topics already covered by session facts.
    known_keys = {f.key for f in used_facts}
    seeds = [s for s in seeds if s.similarity_key not in known_keys]

    # Resume path: user answered a prior multi-clarify batch.
    if looks_like_multi_clarify_answer(query, history):
        from app.orchestrator.clarify import original_user_ask
        from app.orchestrator.state import ChatTurn

        ask = original_user_ask(history, fallback=query)
        # History for seed recompute includes the new answers so resolved keys drop out.
        hist_with_answers = list(history or [])
        if not (
            hist_with_answers
            and hist_with_answers[-1].role == "user"
            and hist_with_answers[-1].content.strip() == query.strip()
        ):
            hist_with_answers = hist_with_answers + [
                ChatTurn(role="user", content=query)
            ]
        refreshed = extract_session_facts(
            query=query, history=hist_with_answers, preference_summary=preference_summary
        )
        used_facts = facts_for_intent(refreshed, intent)
        used_facts_dump = [f.model_dump_state() for f in used_facts]
        fact_guidance = guidance_from_facts(used_facts, intent=intent)
        known_keys = {f.key for f in used_facts}

        remaining = [
            s
            for s in _seed_needs_for_intent(
                intent,
                query=ask,
                history=hist_with_answers,
                preference_summary=preference_summary,
                items=items,
            )
            if s.similarity_key not in known_keys
        ]
        # Seeds that were on the prior ask (for LLM mapping of lettered answers).
        hist_prior = [
            t
            for t in (history or [])
            if not (t.role == "assistant" and "before i plan" in (t.content or "").lower())
        ]
        prior_seeds = _seed_needs_for_intent(
            intent,
            query=ask,
            history=hist_prior,
            preference_summary=preference_summary,
            items=items,
        )
        resolved = await _resolve_answers_for_intent(
            intent=intent,
            query=query,
            history=history,
            seeds=prior_seeds or seeds,
        )
        guidance = "\n".join(p for p in (fact_guidance, resolved.guidance) if p)
        if remaining:
            return IntentClarification(
                intent=intent,
                ready=False,
                questions=[n.model_dump() for n in remaining],
                guidance=guidance,
                reason="partial multi-clarify — remaining intent questions",
                used_facts=used_facts_dump,
            )
        return IntentClarification(
            intent=intent,
            ready=True,
            questions=[],
            guidance=guidance,
            reason="resolved from multi-clarify answers",
            used_facts=used_facts_dump,
        )

    # Deterministic path when LLM unavailable: ask unresolved seeds or ready with facts.
    if not is_decompose_configured():
        if seeds:
            return IntentClarification(
                intent=intent,
                ready=False,
                questions=[n.model_dump() for n in seeds],
                guidance=fact_guidance,
                reason="deterministic seeds (no LLM)",
                used_facts=used_facts_dump,
            )
        return IntentClarification(
            intent=intent,
            ready=True,
            guidance=fact_guidance
            or f"Intent {intent} clear enough to search using session context.",
            reason="no unresolved seeds — using session facts",
            used_facts=used_facts_dump,
        )

    asked: list[ClarificationNeed] = []
    observations: list[str] = []
    guidance = fact_guidance

    for step in range(MAX_REACT_STEPS):
        remaining = [
            s
            for s in seeds
            if s.id not in {a.id for a in asked}
            and s.similarity_key not in {a.similarity_key for a in asked if a.similarity_key}
            and s.similarity_key not in known_keys
        ]
        obs = _context_observation(
            intent=intent,
            query=query,
            history=history,
            preference_summary=preference_summary,
            items=intent_items or items,
            seeds=remaining,
            used_facts=used_facts,
        )
        if observations:
            obs += "\nPrior observations:\n" + "\n".join(observations[-4:])
        obs += (
            f"\nAlready asked this intent: {[a.id for a in asked] or '(none)'}\n"
            f"Step {step + 1}/{MAX_REACT_STEPS}. Choose ONE tool."
        )

        try:
            data = await complete_json(REACT_SYSTEM, obs, max_tokens=600)
        except (LLMNotConfiguredError, Exception) as exc:
            observations.append(f"LLM failed: {exc}")
            break

        tool = str(data.get("tool") or "").strip()
        thought = str(data.get("thought") or "").strip()
        if thought:
            observations.append(f"thought: {thought}")

        if tool == "get_known_context":
            observations.append(
                "observation: known session facts = "
                + (", ".join(f"{f.key}:{f.value}" for f in used_facts) or "(none)")
            )
            continue

        if tool == "ask_user":
            raw_ask = data.get("ask_user") if isinstance(data.get("ask_user"), dict) else {}
            # Prefer matching seed if id/key aligns
            need = None
            ask_id = str(raw_ask.get("id") or "")
            ask_key = str(raw_ask.get("similarity_key") or "")
            if ask_key and ask_key in known_keys:
                observations.append(
                    f"observation: refused ask_user for known key {ask_key}"
                )
                continue
            for s in remaining:
                if (ask_id and s.id == ask_id) or (ask_key and s.similarity_key == ask_key):
                    need = s
                    break
            if need is None and remaining:
                need = remaining[0]
            if need is None and raw_ask.get("question"):
                need = ClarificationNeed(
                    id=ask_id or f"{intent}.custom_{step}",
                    intent=intent,  # type: ignore[arg-type]
                    question=str(raw_ask.get("question")),
                    options=[str(o) for o in (raw_ask.get("options") or [])][:4],
                    reason=str(raw_ask.get("reason") or thought or "react ask_user"),
                    similarity_key=ask_key or ask_id or f"{intent}.custom",
                )
            if need and need.similarity_key in known_keys:
                observations.append(
                    f"observation: skipped ask_user; {need.similarity_key} already known"
                )
                continue
            if need:
                asked.append(need)
                observations.append(f"observation: recorded ask_user {need.id}")
            if len(asked) >= 2:
                break
            continue

        if tool == "mark_intent_ready":
            llm_guidance = str(data.get("guidance") or "").strip()
            guidance = "\n".join(
                p
                for p in (
                    fact_guidance,
                    llm_guidance
                    or f"Intent {intent} ready to search with known session facts.",
                )
                if p
            )
            return IntentClarification(
                intent=intent,
                ready=True,
                questions=[n.model_dump() for n in asked],
                guidance=guidance,
                reason="mark_intent_ready",
                used_facts=used_facts_dump,
            )

        observations.append(f"observation: unknown tool {tool!r}")

    # Finished loop with outstanding questions
    final_questions = asked or seeds
    if final_questions:
        return IntentClarification(
            intent=intent,
            ready=False,
            questions=[n.model_dump() for n in final_questions],
            guidance=fact_guidance,
            reason="react emitted questions",
            used_facts=used_facts_dump,
        )
    return IntentClarification(
        intent=intent,
        ready=True,
        questions=[],
        guidance=guidance or fact_guidance or f"Intent {intent} proceeding with session facts.",
        reason="react complete",
        used_facts=used_facts_dump,
    )


def _item_matches_intent(category: str, intent: str) -> bool:
    if intent == "trip":
        return category in {"grocery", "clothing", "electronics", "other"}
    if intent == "other":
        return category in {"other", "stationery"}
    return category == intent


async def clarify_react_node(state: OrchestratorState) -> dict:
    """LangGraph node for one intent (payload includes clarify_intent)."""
    intent = str(state.get("clarify_intent") or "other")  # type: ignore[arg-type]
    if intent not in {"trip", "grocery", "electronics", "clothing", "other"}:
        intent = "other"
    result = await run_intent_react(
        intent=intent,  # type: ignore[arg-type]
        query=state.get("query") or "",
        history=list(state.get("history") or []),
        preference_summary=state.get("preference_summary")
        if isinstance(state.get("preference_summary"), dict)
        else None,
        items=list(state.get("items") or []),
    )
    return {"intent_clarifications": [result.model_dump_state()]}
