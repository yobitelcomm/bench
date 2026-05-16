# bench summary

One-glance table of every envelope in a directory (or a single envelope file). Recursively walks `*.json`, validates against the `Envelope` schema, groups by `suite_id`, and renders one Rich table per suite sorted by throughput descending. The natural follow-up to a sweep.

## Synopsis

```bash
bench summary <path> [--json]
```

## Example: tabulate a cross-model sweep corpus

```bash
bench summary ./validation-runs/2026-05-16-cross-model-corpus/corpus/all
```

Expected output (excerpt):

```
                                       Suite: llm.inference.chatbot-short
 Model                              Engine       Quant  Hardware              Throughput  TTFT p50/p99  J/tok   Run ID short
 meta-llama/Llama-3.1-8B-Instruct   vllm 0.21.0  fp16   NVIDIA H100 80GB HBM3  1,384.2     41.69/64.71   0.70    01j7q5c6
 Qwen/Qwen2.5-7B-Instruct           vllm 0.21.0  fp16   NVIDIA H100 80GB HBM3  1,362.3     40.98/63.42   0.71    01j7q5d2
 Qwen/Qwen2.5-7B-Instruct           vllm 0.21.0  fp16   NVIDIA H100 80GB HBM3  607.83      26.55/-       1.55    01j7q5e1
 meta-llama/Llama-3.1-8B-Instruct   vllm 0.21.0  fp16   NVIDIA H100 80GB HBM3  580.29      22.75/32.18   1.63    01j7q5e9
 ...
8 envelopes loaded, 0 skipped (validation failure), 1 suites
```

## Flags

| Flag | Default | Description |
|---|---|---|
| `--json` | off | Emit `{"suites": {<id>: [...]}, "skipped": N}` on stdout for piping into jq. |

## Behaviour

- Envelopes that fail JSON decode or `Envelope.model_validate()` are silently skipped; the trailer reports the skipped count.
- A path that points at a single `*.json` file is treated as a one-element directory and rendered with the same table.
- Sort key is `throughput_tok_per_s` descending; missing throughput sorts to the bottom.

## See also

- [bench compare](bench-compare.md) — Pareto frontier across the same corpus
- [Recipes: cross-model corpus](../recipes/cross-model.md)
