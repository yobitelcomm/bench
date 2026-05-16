# bench cost

Compare model cost across providers using the in-process pricing registry shipped with the `inferencebench-llm` plugin. Renders a per-provider table with input, output, and a blended rate.

## Synopsis

```bash
bench cost <model> [--providers a,b,c] [--input-token-share 0.75]
```

## Example: cross-provider price for Llama-3.1-8B-Instruct

```bash
bench cost llama-3.1-8b-instruct --input-token-share 0.75
```

Expected output (excerpt):

```
                  Cost for llama-3.1-8b-instruct  (blend = 0.75 input + 0.25 output, suite=intelligence-index)
 Provider     Input $/Mtok  Output $/Mtok  Blended (3:1) $/Mtok  Notes
 groq         $0.05         $0.08          $0.06                 -
 together     $0.18         $0.18          $0.18                 -
 fireworks    $0.20         $0.20          $0.20                 -
```

Rows are sorted by blended rate ascending.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--suite` | `intelligence-index` | Suite hint for the comparison (informational in Phase 1). |
| `--providers` | `""` | Comma-separated provider filter (e.g. `together,groq`). Empty = all providers. |
| `--input-token-share` | `0.75` | Share of input tokens in the blended rate. The complement (`0.25`) is the output share. Clamped to `[0.0, 1.0]`. |

Common blends: `0.75` (chat / RAG workloads, the default), `0.5` (summarisation), `0.25` (codegen).

## Behaviour

- The registry lives in `inferencebench_llm.pricing`. If the plugin isn't installed, `bench cost` exits 2 with a hint.
- Unknown models exit 1 and print up to 5 close-match suggestions from the registry.

## See also

- [bench compare](bench-compare.md) — Pareto comparison once you've benched against multiple providers
- [llm.inference plugin reference](../plugins/llm-inference.md)
