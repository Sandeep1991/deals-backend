from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.orchestrator.agents.clothing import clothing_agent_node
from app.orchestrator.agents.electronics import electronics_agent_node
from app.orchestrator.agents.grocery import grocery_agent_node
from app.orchestrator.agents.stationery import advisory_agent_node, stationery_agent_node
from app.orchestrator.clarify import clarification_from_raw
from app.orchestrator.gather_clarify import gather_clarifications_node
from app.orchestrator.intent_react import clarify_react_node
from app.orchestrator.merge import merge_results_node, to_chat_payload
from app.orchestrator.planner_clarify import active_intents
from app.orchestrator.split import split_query_node
from app.orchestrator.state import CompositeItem, OrchestratorState
from app.orchestrator.trip_planner import plan_trip_node
from app.search import SearchService


def _agent_payload(state: OrchestratorState, cat_items: list[CompositeItem]) -> dict:
    clarification = state.get("clarification")
    # Prefer combined gather guidance; fall back to merged per-intent guidance.
    if isinstance(clarification, dict) and not (clarification.get("planning_guidance") or "").strip():
        parts = []
        for block in state.get("intent_clarifications") or []:
            g = (block.get("guidance") or "").strip()
            if g:
                parts.append(f"[{block.get('intent')}] {g}")
        if parts:
            clarification = {**clarification, "planning_guidance": "\n".join(parts)}
    return {
        "query": state["query"],
        "event_summary": state.get("event_summary") or state["query"],
        "items": cat_items,
        "category_results": [],
        "reply": "",
        "ads": [],
        "mode": "",
        "comparison": None,
        "limit": int(state.get("limit") or 5),
        "chat_id": state.get("chat_id") or "",
        "history": list(state.get("history") or []),
        "preference_summary": state.get("preference_summary"),
        "clarification": clarification,
        "intent_clarifications": list(state.get("intent_clarifications") or []),
        "clarify_intent": "",
    }


def _clarify_payload(state: OrchestratorState, intent: str) -> dict:
    payload = _agent_payload(state, list(state.get("items") or []))
    payload["clarify_intent"] = intent
    payload["intent_clarifications"] = []
    payload["category_results"] = []
    return payload


def _route_clarify_intents(state: OrchestratorState) -> list[Send]:
    """Fan out one ReAct clarify loop per active intent."""
    query = state.get("query") or ""
    history = list(state.get("history") or [])
    items = list(state.get("items") or [])
    intents = active_intents(query, history, items)
    if not intents:
        # Always run at least grocery-style clarify when split produced grocery items
        cats = {i.category for i in items}
        if "electronics" in cats:
            intents.add("electronics")
        if "grocery" in cats:
            intents.add("grocery")
        if "clothing" in cats:
            intents.add("clothing")
        if not intents:
            intents.add("grocery")
    return [Send("clarify_react", _clarify_payload(state, intent)) for intent in sorted(intents)]


def _route_by_category(state: OrchestratorState) -> list[Send]:
    items = list(state.get("items") or [])
    by_cat: dict[str, list[CompositeItem]] = {}
    for item in items:
        by_cat.setdefault(item.category, []).append(item)

    sends: list[Send] = []

    if by_cat.get("grocery"):
        sends.append(Send("grocery_agent", _agent_payload(state, by_cat["grocery"])))
    if by_cat.get("electronics"):
        sends.append(Send("electronics_agent", _agent_payload(state, by_cat["electronics"])))
    if by_cat.get("clothing"):
        sends.append(Send("clothing_agent", _agent_payload(state, by_cat["clothing"])))

    stationery_items = list(by_cat.get("stationery") or [])
    other_items = list(by_cat.get("other") or [])
    if stationery_items or (other_items and not sends):
        combined = stationery_items + (other_items if stationery_items else [])
        if not combined and other_items:
            combined = other_items
        if combined:
            sends.append(Send("stationery_agent", _agent_payload(state, combined)))
    elif other_items and sends:
        sends.append(Send("advisory_agent", _agent_payload(state, other_items)))

    if not sends:
        sends.append(Send("advisory_agent", _agent_payload(state, items or [])))

    return sends


def _after_gather(state: OrchestratorState) -> str:
    """Ask-back via merge, or expand trip then search."""
    decision = clarification_from_raw(state.get("clarification"))
    if decision and decision.needs_clarification:
        return "merge_results"
    return "plan_trip"


def build_orchestrator_graph(search_service: SearchService):
    async def grocery_agent(state: OrchestratorState) -> dict:
        return await grocery_agent_node(state, search_service)

    async def electronics_agent(state: OrchestratorState) -> dict:
        return await electronics_agent_node(state, search_service)

    async def clothing_agent(state: OrchestratorState) -> dict:
        return await clothing_agent_node(state, search_service)

    async def stationery_agent(state: OrchestratorState) -> dict:
        return await stationery_agent_node(state, search_service)

    async def advisory_agent(state: OrchestratorState) -> dict:
        return await advisory_agent_node(state, search_service)

    graph = StateGraph(OrchestratorState)
    graph.add_node("split_query", split_query_node)
    graph.add_node("clarify_react", clarify_react_node)
    graph.add_node("gather_clarifications", gather_clarifications_node)
    graph.add_node("plan_trip", plan_trip_node)
    graph.add_node("grocery_agent", grocery_agent)
    graph.add_node("electronics_agent", electronics_agent)
    graph.add_node("clothing_agent", clothing_agent)
    graph.add_node("stationery_agent", stationery_agent)
    graph.add_node("advisory_agent", advisory_agent)
    graph.add_node("merge_results", merge_results_node)

    graph.add_edge(START, "split_query")
    graph.add_conditional_edges("split_query", _route_clarify_intents)
    graph.add_edge("clarify_react", "gather_clarifications")
    graph.add_conditional_edges("gather_clarifications", _after_gather)
    graph.add_conditional_edges("plan_trip", _route_by_category)
    for node in (
        "grocery_agent",
        "electronics_agent",
        "clothing_agent",
        "stationery_agent",
        "advisory_agent",
    ):
        graph.add_edge(node, "merge_results")
    graph.add_edge("merge_results", END)

    return graph.compile()


async def run_orchestrator(
    query: str,
    search_service: SearchService,
    limit: int = 5,
    chat_id: str = "",
    history: list | None = None,
    preference_summary: dict | None = None,
) -> OrchestratorState:
    from app.orchestrator.memory import normalize_history

    graph = build_orchestrator_graph(search_service)
    result = await graph.ainvoke(
        {
            "query": query,
            "event_summary": "",
            "items": [],
            "category_results": [],
            "reply": "",
            "ads": [],
            "mode": "search",
            "comparison": None,
            "limit": limit,
            "chat_id": chat_id or "",
            "history": normalize_history(history, current_query=query),
            "preference_summary": preference_summary,
            "clarification": None,
            "intent_clarifications": [],
            "clarify_intent": "",
        }
    )
    return result  # type: ignore[return-value]


__all__ = ["build_orchestrator_graph", "run_orchestrator", "to_chat_payload"]
