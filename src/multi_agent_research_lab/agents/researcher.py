"""Researcher agent — tìm kiếm thông tin và tổng hợp research notes."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._search = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        # 1. Tìm kiếm documents liên quan
        docs = self._search.search(
            query=state.request.query,
            max_results=state.request.max_sources,
        )
        state.sources.extend(docs)

        # 2. Dùng LLM tổng hợp các snippet thành research notes
        snippets = "\n\n".join(
            f"[{i+1}] {doc.title}\n{doc.snippet}" for i, doc in enumerate(docs)
        )
        resp = self._llm.complete(
            system_prompt=(
                "You are a research assistant. Given source snippets, "
                "write concise, factual research notes in bullet points. "
                "Include key facts and cite sources by number [1], [2], etc."
            ),
            user_prompt=f"Query: {state.request.query}\n\nSources:\n{snippets}",
        )
        state.research_notes = resp.content

        # 3. Ghi lại kết quả vào agent_results để trace sau
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=resp.content,
                metadata={
                    "sources_found": len(docs),
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                },
            )
        )
        state.add_trace_event("researcher_done", {"sources": len(docs)})
        return state
