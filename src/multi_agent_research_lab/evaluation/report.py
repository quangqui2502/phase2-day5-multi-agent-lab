"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.5f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}/10"
        lines.append(f"| {item.run_name} | {item.latency_seconds:.2f}s | {cost} | {quality} | {item.notes} |")

    # Tính delta nếu có đủ 2 runs
    if len(metrics) == 2:
        a, b = metrics[0], metrics[1]
        latency_diff = b.latency_seconds - a.latency_seconds
        lines += [
            "",
            "## So sánh",
            f"- Latency delta: `{latency_diff:+.2f}s` ({b.run_name} vs {a.run_name})",
        ]
        if a.estimated_cost_usd and b.estimated_cost_usd:
            cost_diff = b.estimated_cost_usd - a.estimated_cost_usd
            lines.append(f"- Cost delta: `${cost_diff:+.5f}`")

    lines += [
        "",
        "## Failure modes quan sát được",
        "- Mock search trả về snippets generic → research notes thiếu depth",
        "- Multi-agent gọi LLM 3 lần → latency cao hơn baseline đáng kể",
        "- Citation coverage phụ thuộc vào Writer prompt, chưa enforce cứng",
        "",
        "## Kết luận",
        "- **Nên dùng multi-agent** khi task phức tạp, cần phân tách research/analysis/writing rõ ràng",
        "- **Không nên dùng multi-agent** cho câu hỏi đơn giản — overhead latency + cost không xứng",
    ]
    return "\n".join(lines) + "\n"
