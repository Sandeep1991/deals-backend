from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.party_planner.nodes import compare_node, decompose_node, fetch_prices_node
from app.party_planner.state import PlannerState, StoreComparison
from app.search import SearchService


def build_graph(search_service: SearchService):
    async def fetch_prices(state: PlannerState) -> dict:
        return await fetch_prices_node(state, search_service)

    graph = StateGraph(PlannerState)
    graph.add_node("decompose", decompose_node)
    graph.add_node("fetch_prices", fetch_prices)
    graph.add_node("compare", compare_node)

    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "fetch_prices")
    graph.add_edge("fetch_prices", "compare")
    graph.add_edge("compare", END)

    return graph.compile()


async def run_store_comparison(
    query: str,
    search_service: SearchService,
) -> StoreComparison:
    graph = build_graph(search_service)
    result = await graph.ainvoke(
        {
            "query": query,
            "plan": None,
            "quotes": [],
            "comparison": None,
            "reply": "",
            "ads": [],
        }
    )
    comparison = result.get("comparison")
    if comparison is None:
        from app.party_planner.decompose import heuristic_decompose
        from app.party_planner.state import ShoppingPlan

        return StoreComparison(
            query=query,
            plan=heuristic_decompose(query),
            merchants=[],
            reply=result.get("reply", "Unable to compare stores."),
        )
    if not comparison.reply:
        comparison.reply = result.get("reply", "")
    return comparison
