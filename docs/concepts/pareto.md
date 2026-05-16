# Pareto frontiers

InferenceBench does not report single headline numbers. Every result is a tuple of throughput, latency, cost, energy, and quality. Comparison is done on the Pareto frontier across those axes.

## Why not a single number

A single-number benchmark is a benchmark waiting to be reward-hacked. If we publish only `throughput_tok_per_s`, every engine vendor will optimize for that one number at the expense of latency, quality, or cost. The same pattern appears in MLPerf, LMSYS Arena, and every public LLM leaderboard.

A Pareto frontier resists this pressure because moving up on one axis costs you another. The frontier itself is the reportable artifact, not any single point on it.

## What we report

| Axis | Metric examples |
|---|---|
| Latency | `ttft_p50_ms`, `ttft_p99_ms`, `tpot_p50_ms` |
| Throughput | `throughput_tok_per_s`, `goodput_at_slo` |
| Cost | `cost_usd_per_million_tokens` |
| Energy | `power_avg_w`, `joules_per_token` |
| Quality | `quality_score` (suite-defined; null where N/A) |

Every envelope carries enough of these to plot a point in the 5-dimensional space. `bench compare` projects to 2D scatter plus a per-axis table.

## Example

```bash
bench compare run-a.json run-b.json run-c.json --report pareto
```

Expected output:

```
Pareto frontier across (latency, throughput, cost, energy):
  *  Maverick fp8  ttft_p50=142ms  tput=1842  $0.18/Mtok  0.32 J/tok
  *  Maverick nvfp4 ttft_p50=118ms  tput=2100  $0.21/Mtok  0.41 J/tok
     Maverick fp16 ttft_p50=168ms  tput=1340  $0.31/Mtok  0.52 J/tok  (dominated)
```

The `*` marker indicates points on the frontier. The `(dominated)` annotation calls out points that lose on every axis.

## Goodput at SLO

`goodput_at_slo` is the throughput rate at which the SLO template is still satisfied. This is the metric that matters for production planning — peak throughput is irrelevant if your tail latency blows past your service level.

The SLO template is recorded in `envelope.slo_template`. Common templates:

| Template | Target |
|---|---|
| `llm.standard` | TTFT p99 < 300 ms, TPOT p99 < 50 ms |
| `voice.realtime` | TTFT p99 < 100 ms, end-to-end p95 < 250 ms |
| `agent.tool-call` | TTFT p99 < 500 ms, completion p99 < 5 s |

## Frontiers, not points

A vendor showing one of their points off the frontier is showing you the wrong thing. Always ask for the frontier.

## See also

- [Methodology](methodology.md)
- [Vendor neutrality](vendor-neutrality.md)
- [bench compare](../cli/bench-compare.md)
