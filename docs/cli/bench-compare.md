# bench compare

Pareto-frontier comparison across two or more signed envelopes. Computes frontiers for the canonical metric pairs (quality-vs-cost, throughput-vs-latency, throughput-vs-energy) and renders the result as a Rich table, JSON, or a Pareto-only filtered table.

## Synopsis

```bash
bench compare <envelope-path>... [--report table|pareto|json] [--verify]
```

At least two local envelope paths are required. Remote URIs (`hf://`, `https://`) are not loaded directly — use [`bench fetch`](bench-fetch.md) first.

## Example: compare two sweep points

```bash
bench compare \
  ./results/c1-814953250c16.json \
  ./results/c16-60be8efd6d21.json \
  --report table
```

Expected output:

```
                        Benchmark comparison
 Suite                    Model                          Engine       Throughput tok/s  TTFT p99 ms  J/tok  Pareto?
 llm.inference.chatbot... meta-llama/Llama-3.1-8B-Inst.  vllm 0.21.0  1,384.2           64.71        0.70   yes
 llm.inference.chatbot... meta-llama/Llama-3.1-8B-Inst.  vllm 0.21.0  122.17            14.95        7.24   yes
```

Both points land on the frontier — the conc=1 envelope wins on TTFT, the conc=16 envelope wins on throughput and energy.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--report` | `table` | Output format: `table`, `pareto` (Pareto-only rows), or `json`. |
| `--verify` | off | Verify each envelope's signature before comparing; exits 1 on signature failure. |

## Report formats

| Format | What you get |
|---|---|
| `table` | All envelopes, sorted by throughput desc, with a `Pareto?` column. Frontier rows are bolded. |
| `pareto` | Same table but only rows that are on the frontier of at least one metric pair. |
| `json` | One JSON object per envelope plus a `pareto` index by metric pair. Pipe into jq. |

## Pareto pairs

| Label | x (maximise) | y (minimise) |
|---|---|---|
| quality_vs_cost | `goodput_at_slo` (falls back to `req_per_s_passing`) | `cost_usd_per_million_tokens` |
| throughput_vs_latency | `throughput_tok_per_s` | `ttft_p99_ms` |
| throughput_vs_energy | `throughput_tok_per_s` | `joules_per_token` |

A run is "on the Pareto frontier" if there is no other run that dominates it on both axes of at least one pair.

## See also

- [Pareto frontiers](../concepts/pareto.md)
- [bench diff](bench-diff.md) — for the focused two-envelope regression check
- [Recipes: cross-model corpus](../recipes/cross-model.md)
