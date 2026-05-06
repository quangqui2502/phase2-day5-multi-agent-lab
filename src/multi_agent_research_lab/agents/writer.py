"""Writer agent — viết câu trả lời cuối từ research + analysis."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        resp = self._llm.complete(
            system_prompt=(
                f"You are a skilled writer for {state.request.audience}. "
                "Using the research notes and analysis provided, write a clear, "
                "well-structured answer of approximately 500 words. "
                "Include citations where relevant."
            ),
            user_prompt=(
                f"Query: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes}\n\n"
                f"Analysis:\n{state.analysis_notes}"
            ),
        )
        state.final_answer = resp.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=resp.content,
                metadata={
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                },
            )
        )
        state.add_trace_event("writer_done", {})
        return state
