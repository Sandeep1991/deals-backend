#!/usr/bin/env python3
"""Verify per-intent ReAct clarify: dedupe, remaining asks, search fan-out wiring."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.orchestrator.gather_clarify import dedupe_needs, gather_clarifications_node
from app.orchestrator.graph import _after_gather, _route_by_category, _route_clarify_intents, build_orchestrator_graph
from app.orchestrator.intent_react import run_intent_react
from app.orchestrator.planner_clarify import (
    ClarificationNeed,
    active_intents,
    electronics_planner_needs,
    format_multi_clarify_reply,
    grocery_planner_needs,
    trip_planner_needs,
)
from app.orchestrator.state import ChatTurn, CompositeItem


CAMPING_ASK = (
    "Help me pack for a family camping weekend — groceries plus power for lots of electronics"
)


def test_session_facts_reuse() -> None:
    from app.orchestrator.session_facts import extract_session_facts, format_used_facts_markdown

    history = [
        ChatTurn(role="user", content=CAMPING_ASK),
        ChatTurn(
            role="assistant",
            content="Before I plan **Trip**, I need a few details:\n\n**A.** Who is coming?",
        ),
        ChatTurn(role="user", content="A: 2 adults + 1 kid age 5. Same food for everyone. None of these care items."),
    ]
    facts = extract_session_facts(
        query="Also need a weekend power station",
        history=history,
        preference_summary={"summary": "", "preferences": ["household:2 adults"]},
    )
    keys = {f.key for f in facts}
    assert "party_size" in keys, keys
    assert "kids_food" in keys, keys
    assert "care_items" in keys, keys
    assert "power_capacity" in keys, keys
    md = format_used_facts_markdown(facts)
    assert "Using from this chat" in md
    assert "2 adult" in md.lower() or "household" in md.lower()
    print("OK session facts:", sorted(keys))


async def test_react_skips_known_party_size() -> None:
    history = [
        ChatTurn(
            role="user",
            content="Family camping for 2 adults and 1 child age 4 near Asheville this weekend, cool nights",
        ),
    ]
    items = [
        CompositeItem(name="bottled water", category="grocery"),
        CompositeItem(name="portable power station", category="electronics"),
    ]
    trip = await run_intent_react(
        intent="trip",
        query=history[0].content,
        history=[],
        preference_summary=None,
        items=items,
    )
    keys = {q.get("similarity_key") for q in trip.questions}
    assert "party_size" not in keys, keys
    assert trip.used_facts, trip
    used_keys = {f.get("key") for f in trip.used_facts}
    assert "party_size" in used_keys
    assert "weather_gear" in used_keys  # cool nights / Asheville weekend
    # Still may ask kids food / care if not stated
    print("OK react skips known:", "questions", keys, "used", used_keys)


def test_active_intents_and_seeds() -> None:
    items = [
        CompositeItem(name="bottled water", category="grocery"),
        CompositeItem(name="portable power station", category="electronics"),
    ]
    intents = active_intents(CAMPING_ASK, [], items)
    assert "trip" in intents, intents
    assert "electronics" in intents, intents
    assert "grocery" in intents, intents

    trip = trip_planner_needs(query=CAMPING_ASK, history=None, preference_summary=None)
    grocery = grocery_planner_needs(
        query=CAMPING_ASK, history=None, preference_summary=None, items=items
    )
    electronics = electronics_planner_needs(
        query=CAMPING_ASK, history=None, preference_summary=None, items=items
    )
    all_needs = trip + grocery + electronics
    keys = [n.similarity_key for n in all_needs]
    assert "party_size" in keys
    assert keys.count("party_size") >= 2  # trip + grocery before dedupe
    assert "power_capacity" in keys
    assert "kids_food" in keys
    assert "care_items" in keys
    assert "weather_gear" in keys

    deduped = dedupe_needs(all_needs)
    dkeys = [n.similarity_key for n in deduped]
    assert dkeys.count("party_size") == 1, dkeys
    assert "power_capacity" in dkeys
    assert "kids_food" in dkeys
    assert "care_items" in dkeys
    assert "weather_gear" in dkeys
    # Distinct intents still represented after dedupe (shared party_size keeps one intent tag)
    reply = format_multi_clarify_reply(deduped)
    assert "Before I plan" in reply
    assert "Electronics" in reply or "power" in reply.lower()
    print("OK seeds+dedupe:", dkeys)


async def test_partial_answer_remaining() -> None:
    history = [
        ChatTurn(role="user", content=CAMPING_ASK),
        ChatTurn(
            role="assistant",
            content=(
                "Before I plan **Trip / camping + Electronics / power**, I need a few details:\n\n"
                "**A. Trip / camping** — Who is coming?\n"
                "**B. Electronics / power** — What power setup?\n\n"
                "Reply with answers per letter and I'll continue planning."
            ),
        ),
    ]
    items = [
        CompositeItem(name="bottled water", category="grocery"),
        CompositeItem(name="portable power station", category="electronics"),
    ]
    # Only party size answered — power + other trip themes should remain.
    answer = "A: 2 adults + 1 kid age 5"
    trip_result = await run_intent_react(
        intent="trip",
        query=answer,
        history=history,
        preference_summary=None,
        items=items,
    )
    elec_result = await run_intent_react(
        intent="electronics",
        query=answer,
        history=history,
        preference_summary=None,
        items=items,
    )
    trip_keys = {q.get("similarity_key") for q in trip_result.questions}
    elec_keys = {q.get("similarity_key") for q in elec_result.questions}
    assert not trip_result.ready or trip_keys, trip_result
    assert "party_size" not in trip_keys, trip_keys
    assert trip_keys & {"kids_food", "care_items", "weather_gear"}, trip_keys
    assert not elec_result.ready
    assert "power_capacity" in elec_keys, elec_keys
    print("OK partial answer remaining:", "trip", trip_keys, "elec", elec_keys)


async def test_gather_and_route() -> None:
    items = [
        CompositeItem(name="bottled water", category="grocery"),
        CompositeItem(name="portable power station", category="electronics"),
    ]
    # Simulate parallel ReAct outputs with overlapping party_size.
    blocks = [
        {
            "intent": "trip",
            "ready": False,
            "questions": [
                ClarificationNeed(
                    id="trip.family",
                    intent="trip",
                    question="Who is coming?",
                    options=["Adults only"],
                    similarity_key="party_size",
                ).model_dump(),
                ClarificationNeed(
                    id="trip.weather",
                    intent="trip",
                    question="Weather?",
                    options=["Warm"],
                    similarity_key="weather_gear",
                ).model_dump(),
            ],
            "guidance": "",
        },
        {
            "intent": "grocery",
            "ready": False,
            "questions": [
                ClarificationNeed(
                    id="grocery.party_size",
                    intent="grocery",
                    question="How many people for food?",
                    options=["Adults only"],
                    similarity_key="party_size",
                ).model_dump(),
            ],
            "guidance": "",
        },
        {
            "intent": "electronics",
            "ready": False,
            "questions": [
                ClarificationNeed(
                    id="electronics.power",
                    intent="electronics",
                    question="Power setup?",
                    options=["Weekend station"],
                    similarity_key="power_capacity",
                ).model_dump(),
            ],
            "guidance": "",
        },
    ]
    state = {
        "query": CAMPING_ASK,
        "history": [],
        "intent_clarifications": blocks,
        "items": items,
        "clarification": None,
    }
    gathered = await gather_clarifications_node(state)  # type: ignore[arg-type]
    clar = gathered["clarification"]
    assert clar["needs_clarification"] is True
    keys = [q["similarity_key"] for q in clar["questions"]]
    assert keys.count("party_size") == 1
    assert "power_capacity" in keys
    assert "weather_gear" in keys
    assert _after_gather({**state, **gathered}) == "merge_results"  # type: ignore[arg-type]

    # Ready path → plan_trip → dual search fan-out
    ready_state = {
        "query": CAMPING_ASK,
        "history": [],
        "items": items,
        "intent_clarifications": [
            {"intent": "trip", "ready": True, "questions": [], "guidance": "2 adults + kid"},
            {"intent": "electronics", "ready": True, "questions": [], "guidance": "weekend power"},
            {"intent": "grocery", "ready": True, "questions": [], "guidance": "family food"},
        ],
        "clarification": {
            "needs_clarification": False,
            "planning_guidance": "[trip] 2 adults\n[electronics] weekend power",
        },
        "event_summary": CAMPING_ASK,
        "limit": 5,
        "chat_id": "",
        "preference_summary": None,
        "category_results": [],
        "ads": [],
        "mode": "search",
        "comparison": None,
        "reply": "",
        "clarify_intent": "",
    }
    assert _after_gather(ready_state) == "plan_trip"  # type: ignore[arg-type]
    sends = _route_by_category(ready_state)  # type: ignore[arg-type]
    nodes = {s.node for s in sends}
    assert "grocery_agent" in nodes and "electronics_agent" in nodes, nodes

    clarify_sends = _route_clarify_intents(ready_state)  # type: ignore[arg-type]
    clarify_intents = {s.arg.get("clarify_intent") for s in clarify_sends}
    assert "trip" in clarify_intents and "electronics" in clarify_intents, clarify_intents
    print("OK gather+route: ask keys", keys, "search", nodes)


def test_graph_compiles() -> None:
    class _DummySearch:
        pass

    graph = build_orchestrator_graph(_DummySearch())  # type: ignore[arg-type]
    assert graph is not None
    print("OK graph compiles")


async def main() -> None:
    test_active_intents_and_seeds()
    test_session_facts_reuse()
    await test_react_skips_known_party_size()
    await test_partial_answer_remaining()
    await test_gather_and_route()
    test_graph_compiles()
    print("\nAll per-intent clarify checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
