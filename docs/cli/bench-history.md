# bench history

Time-series view of one metric across a directory of envelopes. Sort chronologically by envelope `timestamp`, optionally filter to one model / suite / engine, and render the value plus per-step delta and a sparkline summary.

## Synopsis

```bash
bench history <dir> [--metric KEY] [--filter-model ID] [--filter-suite ID]
                     [--filter-engine NAME] [--json]
```

## Example: throughput trend across the cross-model corpus

```bash
bench history ./validation-runs/2026-05-16-cross-model-corpus/corpus/all \
  --metric throughput_tok_per_s \
  --filter-model meta-llama/Llama-3.1-8B-Instruct \
  --filter-suite llm.inference.chatbot-short
```

Expected output:

```
                              History: throughput_tok_per_s
 #  Timestamp         Model                              Engine       Run ID   throughput_tok_per_s   Δ vs prev    Δ vs prev (rel%)   Trend
 1  2026-05-16 14:02  meta-llama/Llama-3.1-8B-Instruct   vllm v0.21.0 01j7q5c1        122.2            -            -                 -
 2  2026-05-16 14:08  meta-llama/Llama-3.1-8B-Instruct   vllm v0.21.0 01j7q5c4        580.3            +458.1       +374.93%           ↑
 3  2026-05-16 14:14  meta-llama/Llama-3.1-8B-Instruct   vllm v0.21.0 01j7q5c6      1,384.2            +803.9       +138.53%           ↑
 4  2026-05-16 14:21  meta-llama/Llama-3.1-8B-Instruct   vllm v0.21.0 01j7q5c8      1,312.3            -71.9        -5.19%             ↓
▁▃▆█  min=122.20  median=580.30  max=1,384.20
```

## Flags

| Flag | Default | Description |
|---|---|---|
| `--metric` | `throughput_tok_per_s` | Metric key (any leaf in `envelope.metrics`). Missing values render as `-`. |
| `--filter-model` | `""` | Only include envelopes whose `model.id` matches exactly. |
| `--filter-suite` | `""` | Only include envelopes whose `suite_id` matches exactly. |
| `--filter-engine` | `""` | Only include envelopes whose `engine.name` matches exactly. |
| `--json` | off | Emit a JSON document (`metric`, `filter`, `series`, `stats`) instead of the table + sparkline. |

## Behaviour

- Envelopes that fail JSON decode or `Envelope.model_validate()` are silently skipped (same policy as `bench summary`).
- Series is sorted by envelope `timestamp`; the trailing sparkline uses 8-level Unicode blocks (`▁▂▃▄▅▆▇█`).
- With no matches the table is suppressed and the command prints `no matches`.

## See also

- [bench summary](bench-summary.md) — tabulate the same directory by suite
- [bench diff](bench-diff.md) — point-in-time delta between two specific envelopes
