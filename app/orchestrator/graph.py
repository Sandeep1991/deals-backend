from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.orchestrator.agents.clothing import clothing_agent_node
from app.orchestrator.agents.electronics import electronics_agent_node
from app.orchestrator.agents.grocery import grocery_agent_node
from app.orchestrator.agents.stationery import advisory_agent_node, stationery_agent_node
from app.orchestrator.merge import merge_results_node, to_chat_payload
from app.orchestrator.split import split_query_node
from app.orchestrator.state import CompositeItem, OrchestratorState
from app.search import SearchService


def _route_by_category(state: OrchestratorState) -> list[Send]:
    items = list(state.get("items") or [])
    query = state["query"]
    summary = state.get("event_summary") or query
    limit = int(state.get("limit") or 5)

    by_cat: dict[str, list[CompositeItem]] = {}
    for item in items:
        by_cat.setdefault(item.category, []).append(item)

    sends: list[Send] = []

    def _payload(cat_items: list[CompositeItem]) -> dict:
        return {
            "query": query,
            "event_summary": summary,
            "items": cat_items,
            "category_results": [],
            "reply": "",
            "ads": [],
            "mode": "",
            "comparison": None,
            "limit": limit,
        }

    if by_cat.get("grocery"):
        sends.append(Send("grocery_agent", _payload(by_cat["grocery"])))
    if by_cat.get("electronics"):
        sends.append(Send("electronics_agent", _payload(by_cat["electronics"])))
    if by_cat.get("clothing"):
        sends.append(Send("clothing_agent", _payload(by_cat["clothing"])))

    stationery_items = list(by_cat.get("stationery") or [])
    other_items = list(by_cat.get("other") or [])
    if stationery_items or (other_items and not sends):
        # Stationery agent also handles leftover "other" when it's the only path,
        # or stationery items alongside others.
        combined = stationery_items + (other_items if stationery_items else [])
        if not combined and other_items:
            combined = other_items
        if combined:
            sends.append(Send("stationery_agent", _payload(combined)))
    elif other_items and sends:
        # Mixed query with leftover "other" → advisory for those leftovers
        sends.append(Send("advisory_agent", _payload(other_items)))

    if not sends:
        sends.append(Send("advisory_agent", _payload(items or [])))

    return sends


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
    graph.add_node("grocery_agent", grocery_agent)
    graph.add_node("electronics_agent", electronics_agent)
    graph.add_node("clothing_agent", clothing_agent)
    graph.add_node("stationery_agent", stationery_agent)
    graph.add_node("advisory_agent", advisory_agent)
    graph.add_node("merge_results", merge_results_node)

    graph.add_edge(START, "split_query")
    graph.add_conditional_edges("split_query", _route_by_category)
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
) -> OrchestratorState:
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
        }
    )
    return result  # type: ignore[return-value]


__all__ = ["build_orchestrator_graph", "run_orchestrator", "to_chat_payload"]
