# bench replay

Reproduce a benchmark run from an existing signed envelope. The envelope already records every input needed to re-run (`suite_id`, `model.id`, `engine.name`, `dataset.id`, `seed`, `slo_template`, quantization); what it deliberately does NOT record is the live engine endpoint — that is host-specific and would make envelopes non-portable. `bench replay` therefore requires a fresh `--base-url` and produces a NEW envelope that you can diff against the original.

## Synopsis

```bash
bench replay <source-envelope.json> --base-url <URL> [--output DIR] [--no-verify]
```

## Example: replay a published Llama envelope on your own H100

```bash
bench fetch hf://datasets/yobitel-bench-results/llama-3.1-8b__chatbot-short__abcdef123456

bench replay ~/.cache/inferencebench/fetched/3f9c1a2b8e7d.json \
  --base-url http://localhost:8000/v1 \
  --output ./replay-results
```

Expected output (excerpt):

```
                              Replay summary
 field           source                            replay                           match
 suite_id        llm.inference.chatbot-short       llm.inference.chatbot-short      yes
 model.id        meta-llama/Llama-3.1-8B-Instruct  meta-llama/Llama-3.1-8B-Instruct yes
 engine.name     vllm                              vllm                             yes
 dataset.id      chatbot-short                     chatbot-short                    yes
 seed            42                                42                               yes

                 Headline metrics (source vs replay)
 metric                       source   replay
 throughput_tok_per_s         1384     1402
 ttft_p50_ms                  41.69    40.85
 joules_per_token             0.70     0.69
 ok_rate                      1.000    1.000
```

Use [`bench compare`](bench-compare.md) or [`bench diff`](bench-diff.md) for the full Pareto / regression view.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--base-url` | `""` | **Required.** Engine base URL (e.g. `http://localhost:8000/v1`). Envelopes are host-agnostic and do not store live endpoints. |
| `--output` | `./replay-results` | Output directory for the replay envelope. |
| `--signing-mode` | `dev` | `dev` (local cosign key) or `keyless` (Sigstore OIDC). |
| `--dev-key` | `cosign.key` | Path to local cosign signing key when `--signing-mode=dev`. |
| `--verify` / `--no-verify` | on | Verify the source envelope's signature before replaying. Use `--no-verify` only for local unsigned fixtures. |

## Failure modes

- A failed signature verification on the source envelope exits `1` before the engine is touched (unless `--no-verify`).
- If the plugin no longer ships the benchmark id recorded on the source envelope (e.g. plugin upgraded and dropped it), the command exits `1` with a "pin to the plugin version" hint.

## See also

- [Recipes: reproducibility](../recipes/reproducibility.md)
- [Reproducibility concept](../concepts/reproducibility.md)
- [bench verify](bench-verify.md)
