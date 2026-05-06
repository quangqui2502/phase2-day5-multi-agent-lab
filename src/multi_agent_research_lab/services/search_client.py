"""Search client abstraction for ResearcherAgent."""

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Mock search client — trả về documents giả để test không cần Tavily API."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        mock_docs = [
            SourceDocument(
                title=f"Overview of: {query}",
                url="https://example.com/overview",
                snippet=(
                    f"{query} is a rapidly evolving topic. "
                    "Recent research highlights its impact on modern AI systems, "
                    "particularly in improving accuracy and contextual understanding."
                ),
            ),
            SourceDocument(
                title=f"Deep dive: {query} techniques",
                url="https://example.com/techniques",
                snippet=(
                    "Key techniques include graph-based retrieval, dense vector search, "
                    "and hybrid approaches combining lexical and semantic matching."
                ),
            ),
            SourceDocument(
                title=f"Limitations and challenges of {query}",
                url="https://example.com/challenges",
                snippet=(
                    "Current limitations include high computational cost, "
                    "difficulty in maintaining up-to-date knowledge graphs, "
                    "and challenges with multi-hop reasoning."
                ),
            ),
        ]
        return mock_docs[:max_results]
