# Benchmark Report

| Run | Latency (s) | Cost (USD) | Quality | Notes |
|---|---:|---:|---:|---|
| single-agent | 10.26s | $0.00044 |  | citation_coverage=0% |
| multi-agent | 16.27s | $0.00078 |  | citation_coverage=0% |

## So sánh
- Latency delta: `+6.01s` (multi-agent vs single-agent)
- Cost delta: `$+0.00034`

## Failure modes quan sát được
- Mock search trả về snippets generic → research notes thiếu depth
- Multi-agent gọi LLM 3 lần → latency cao hơn baseline đáng kể
- Citation coverage phụ thuộc vào Writer prompt, chưa enforce cứng

## Kết luận
- **Nên dùng multi-agent** khi task phức tạp, cần phân tách research/analysis/writing rõ ràng
- **Không nên dùng multi-agent** cho câu hỏi đơn giản — overhead latency + cost không xứng
