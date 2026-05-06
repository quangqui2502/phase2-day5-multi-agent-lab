"""Command-line entrypoint for the lab starter."""

import json
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _baseline_runner(query: str) -> ResearchState:
    """Single-agent runner dùng cho benchmark."""
    llm = LLMClient()
    resp = llm.complete(
        system_prompt=(
            "You are a research assistant. Given a query, provide a thorough, "
            "well-structured answer of approximately 500 words."
        ),
        user_prompt=query,
    )
    from multi_agent_research_lab.core.schemas import AgentName, AgentResult
    state = ResearchState(request=ResearchQuery(query=query))
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
    return state


def _multi_agent_runner(query: str) -> ResearchState:
    """Multi-agent runner dùng cho benchmark."""
    state = ResearchState(request=ResearchQuery(query=query))
    return MultiAgentWorkflow().run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline using a real LLM call."""
    _init()
    start = time.perf_counter()
    state = _baseline_runner(query)
    latency = time.perf_counter() - start

    cost = sum(
        (r.metadata.get("input_tokens") or 0) / 1000 * 0.000150
        + (r.metadata.get("output_tokens") or 0) / 1000 * 0.000600
        for r in state.agent_results
    )
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))
    console.print(f"[dim]Latency: {latency:.2f}s | Cost: ${cost:.6f}[/dim]")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""
    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(Panel.fit(result.final_answer or "", title="Multi-Agent Result"))
    console.print(f"[dim]Route: {' → '.join(result.route_history)}[/dim]")


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output markdown file")] = Path("reports/benchmark_report.md"),
) -> None:
    """So sánh single-agent baseline vs multi-agent, lưu report ra file."""
    _init()

    console.print("[bold]Running baseline...[/bold]")
    _, metrics_baseline = run_benchmark("single-agent", query, _baseline_runner)

    console.print("[bold]Running multi-agent...[/bold]")
    _, metrics_multi = run_benchmark("multi-agent", query, _multi_agent_runner)

    # In bảng so sánh ra terminal
    table = Table(title="Benchmark Results")
    table.add_column("Run")
    table.add_column("Latency", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Notes")
    for m in [metrics_baseline, metrics_multi]:
        cost = f"${m.estimated_cost_usd:.5f}" if m.estimated_cost_usd else "-"
        table.add_row(m.run_name, f"{m.latency_seconds:.2f}s", cost, m.notes)
    console.print(table)

    # Lưu report markdown
    output.parent.mkdir(parents=True, exist_ok=True)
    report = render_markdown_report([metrics_baseline, metrics_multi])
    output.write_text(report)
    console.print(f"[green]Report saved to {output}[/green]")


@app.command()
def trace(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("reports/trace_sample.json"),
) -> None:
    """Chạy multi-agent và export toàn bộ trace ra JSON."""
    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    result = MultiAgentWorkflow().run(state)

    report = {
        "query": query,
        "route_history": result.route_history,
        "iterations": result.iteration,
        "trace": result.trace,
        "agent_results": [
            {
                "agent": r.agent,
                "input_tokens": r.metadata.get("input_tokens"),
                "output_tokens": r.metadata.get("output_tokens"),
                "preview": r.content[:200] + "..." if len(r.content) > 200 else r.content,
            }
            for r in result.agent_results
        ],
        "errors": result.errors,
        "sources_count": len(result.sources),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    console.print(f"[green]Trace saved to {output}[/green]")

    # In tóm tắt ra terminal
    table = Table(title="Trace Summary")
    table.add_column("Step")
    table.add_column("Agent")
    table.add_column("Event")
    table.add_column("Detail")
    for i, event in enumerate(result.trace, 1):
        payload = event["payload"]
        detail = ", ".join(f"{k}={v}" for k, v in payload.items())
        agent = payload.get("route", event["name"].replace("_routed", "").replace("_done", ""))
        table.add_row(str(i), agent, event["name"], detail)
    console.print(table)


if __name__ == "__main__":
    app()
