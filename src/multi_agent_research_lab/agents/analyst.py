"""Analyst agent — phân tích research notes thành insights có cấu trúc."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        resp = self._llm.complete(
            system_prompt=(
                "You are a critical analyst. Given research notes, extract:\n"
                "1. Key claims (what is being argued)\n"
                "2. Supporting evidence (what backs it up)\n"
                "3. Weak points or gaps (what's missing or uncertain)\n"
                "Be concise and structured."
            ),
            user_prompt=(
                f"Query: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes}"
            ),
        )
        state.analysis_notes = resp.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=resp.content,
                metadata={
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                },
            )
        )
        state.add_trace_event("analyst_done", {})
        return state
