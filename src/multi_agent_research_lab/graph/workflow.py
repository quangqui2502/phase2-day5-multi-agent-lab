"""LangGraph workflow — kết nối supervisor và các worker agents."""

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState


def _dict_to_state(d: dict) -> ResearchState:
    return ResearchState.model_validate(d)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(self) -> None:
        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent()
        self._analyst = AnalystAgent()
        self._writer = WriterAgent()

    def build(self) -> object:
        """Tạo LangGraph graph với 4 nodes và conditional routing."""
        graph = StateGraph(dict)

        def supervisor_node(state: dict) -> dict:
            return self._supervisor.run(_dict_to_state(state)).model_dump()

        def researcher_node(state: dict) -> dict:
            return self._researcher.run(_dict_to_state(state)).model_dump()

        def analyst_node(state: dict) -> dict:
            return self._analyst.run(_dict_to_state(state)).model_dump()

        def writer_node(state: dict) -> dict:
            return self._writer.run(_dict_to_state(state)).model_dump()

        graph.add_node("supervisor", supervisor_node)
        graph.add_node("researcher", researcher_node)
        graph.add_node("analyst", analyst_node)
        graph.add_node("writer", writer_node)

        # Supervisor quyết định đi đâu dựa vào route_history[-1]
        def route(state: dict) -> str:
            history = state.get("route_history", [])
            return history[-1] if history else "done"

        graph.add_conditional_edges(
            "supervisor",
            route,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )

        # Sau mỗi worker → quay lại supervisor
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        graph.set_entry_point("supervisor")
        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Chạy graph và trả về final state."""
        compiled = self.build()
        result = compiled.invoke(state.model_dump())
        return _dict_to_state(result)
