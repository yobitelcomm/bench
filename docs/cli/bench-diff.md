# bench diff

Per-metric delta between two envelopes — a baseline and a candidate. Each metric is classified as `improvement`, `regression`, `no_change`, `unknown`, or `missing` using a direction-aware policy (lower-is-better for latencies / cost / energy; higher-is-better for throughput / quality / goodput).

Sharper than [`bench compare`](bench-compare.md), which renders Pareto frontiers across many runs. `bench diff` is the canonical "did my optimisation actually help?" command — and the canonical CI regression check via `--strict`.

## Synopsis

```bash
bench diff <baseline.json> <candidate.json> [--tolerance 0.02] [--report table|json] [--strict] [--verify]
```

## Example: kernel change regression check

```bash
bench diff \
  baseline/c16-60be8efd6d21.json \
  candidate/c16-60be8efd6d21.json \
  --strict
```

Expected output (truncated, with one regression):

```
                                    Envelope diff
 Metric                       Baseline  Candidate  Δ abs    Δ rel    Verdict
 ttft_p99_ms                  64.71     78.50      +13.79   +21.31%  ↑ regression
 throughput_tok_per_s         1,384.2   1,402.7    +18.50   +1.34%   ≈
 joules_per_token             0.70      0.68       -0.02    -2.86%   ↓ improvement
 ok_rate                      1.000     1.000      +0.00    +0.00%   ≈
```

Exit code is `1` when `--strict` is set and any metric is classified as a regression, `0` otherwise.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--tolerance` | `0.02` | Relative-delta band (`±2 %`) inside which a metric is `no_change`. |
| `--report` | `table` | Output format: `table` or `json`. |
| `--strict` | off | Exit 1 if any metric is a regression. Use this in CI. |
| `--verify` | off | Verify both envelopes' signatures before diffing. |

## Direction policy

| Lower is better | Higher is better |
|---|---|
| `ttft_*`, `tpot_*`, `total_*` ms percentiles | `throughput_tok_per_s` |
| `joules_per_token`, `energy_joules_total` | `req_per_s_passing`, `req_per_s_all` |
| `power_avg_w`, `power_peak_w` | `compliance_rate`, `ok_rate` |
| `cost_usd_per_million_tokens` | `goodput_at_slo` |

Metrics not in either set are tagged `unknown` — the delta is still rendered but no verdict is emitted.

## Context warnings

If the baseline and candidate envelopes differ on `suite_id`, `model.id`, `engine.name`, `engine.version`, `quantization.format`, or the hardware fingerprint, `bench diff` still runs but prints a yellow warning. Diffing across contexts is supported; interpret the deltas with care.

## See also

- [bench compare](bench-compare.md) — Pareto across many runs
- [Recipes: regression check](../recipes/regression-check.md)
