"""Benchmark: đo latency, cost, quality cho single-agent vs multi-agent."""

from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]


def _total_cost(state: ResearchState) -> float:
    """Cộng cost từ tất cả agent_results trong state."""
    total = 0.0
    for result in state.agent_results:
        meta = result.metadata
        input_tok = meta.get("input_tokens") or 0
        output_tok = meta.get("output_tokens") or 0
        total += input_tok / 1000 * 0.000150 + output_tok / 1000 * 0.000600
    return total


def _citation_coverage(state: ResearchState) -> float:
    """Tỉ lệ câu trong final_answer có citation [N]. Giá trị 0.0 - 1.0."""
    if not state.final_answer:
        return 0.0
    sentences = [s.strip() for s in state.final_answer.split(".") if s.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if "[" in s and "]" in s)
    return cited / len(sentences)


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Chạy runner, đo latency + cost + citation coverage."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    cost = _total_cost(state)
    citation_cov = _citation_coverage(state)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost if cost > 0 else None,
        notes=f"citation_coverage={citation_cov:.0%}",
    )
    return state, metrics
