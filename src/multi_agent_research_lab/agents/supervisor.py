"""Supervisor — quyết định agent nào chạy tiếp theo."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Ghi route tiếp theo vào state.route_history.

        Guardrails:
        - max_iterations: dừng nếu vòng lặp quá dài
        - fallback: nếu agent trước để lại lỗi, bỏ qua bước đó và đi tiếp
        """
        settings = get_settings()

        # Guardrail: max iterations
        if state.iteration >= settings.max_iterations:
            state.errors.append(f"Stopped: reached max_iterations={settings.max_iterations}")
            state.record_route("done")
            return state

        # Fallback: nếu có lỗi nhưng vẫn còn iteration, cố gắng đi tiếp
        if state.errors:
            # Nếu research thất bại (notes rỗng dù đã route researcher) → bỏ qua, đi analyst
            researcher_ran = "researcher" in state.route_history
            analyst_ran = "analyst" in state.route_history

            if researcher_ran and state.research_notes is None and not analyst_ran:
                state.research_notes = "Research unavailable due to error. Proceeding with limited context."
            if analyst_ran and state.analysis_notes is None and state.final_answer is None:
                state.analysis_notes = "Analysis unavailable due to error."

        # Routing logic theo thứ tự
        if state.research_notes is None:
            state.record_route("researcher")
        elif state.analysis_notes is None:
            state.record_route("analyst")
        elif state.final_answer is None:
            state.record_route("writer")
        else:
            state.record_route("done")

        state.add_trace_event("supervisor_routed", {
            "route": state.route_history[-1],
            "iteration": state.iteration,
        })
        return state
