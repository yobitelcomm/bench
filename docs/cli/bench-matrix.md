# bench matrix

Run one benchmark across multiple endpoints from a single YAML config. Each target × concurrency point produces a signed envelope under `--output`, and the trailing Rich table summarises throughput, TTFT, and `ok_rate` per pair.

`bench matrix` automates the "run-it-N-times" shape that `bench run` covers for a single endpoint — useful for vLLM-vs-vLLM-vs-hosted comparisons captured in one command.

## Synopsis

```bash
bench matrix <config.yaml> --output DIR [--signing-mode dev|keyless] [--dev-key PATH]
                           [--continue-on-error/--no-continue-on-error]
```

## Example: Llama vs Qwen on two vLLM endpoints

`matrix.yaml`:

```yaml
schema: inferencebench.matrix.v1
suite_id: llm.inference.chatbot-short
duration_s: 60
sweep: [1, 16]
targets:
  - name: llama-vllm
    model: meta-llama/Llama-3.1-8B-Instruct
    engine: vllm
    base_url: http://localhost:8000/v1
    quant: fp16
  - name: qwen-vllm
    model: Qwen/Qwen2.5-7B-Instruct
    engine: vllm
    base_url: http://localhost:8001/v1
    quant: fp16
```

```bash
bench matrix matrix.yaml --output ./matrix-results
```

Expected output (real conc=16 numbers from `validation-runs/2026-05-16-cross-model-corpus/`):

```
                                 Matrix results
 target        point  throughput_tok_per_s  ttft_p50_ms  ok_rate  envelope                       status
 llama-vllm    1      122.2                 13.98        1.000    llama-vllm-c1-814953250c16.json   ✓
 llama-vllm    16     1384                  41.69        1.000    llama-vllm-c16-60be8efd6d21.json  ✓
 qwen-vllm     1      120.0                 13.40        1.000    qwen-vllm-c1-07b69e640395.json    ✓
 qwen-vllm     16     1362                  40.98        1.000    qwen-vllm-c16-8d7ef1b17fb7.json   ✓
```

Envelope filenames are prefixed with `<target-name>-c<point>-<content_hash[:12]>.json`.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--output` | required | Output directory for produced envelopes. |
| `--signing-mode` | `dev` | `dev` (local cosign key) or `keyless` (Sigstore OIDC). |
| `--dev-key` | `./cosign.key` | Path to local cosign signing key (used when `--signing-mode=dev`). |
| `--continue-on-error` / `--no-continue-on-error` | on | Keep going past failed targets. With `--no-continue-on-error`, stop the matrix on the first failure. |

## Config schema

| Field | Required | Description |
|---|---|---|
| `schema` | yes (`inferencebench.matrix.v1`) | Schema identifier. |
| `suite_id` | yes | Fully-qualified benchmark id (e.g. `llm.inference.chatbot-short`). |
| `duration_s` | optional (default `60`) | Per-point measurement duration in seconds. |
| `sweep` | yes | Non-empty list of positive integer concurrency points. |
| `targets[].name` | yes | Unique short label used as the envelope filename prefix. |
| `targets[].model` | yes | Model id passed to the plugin. |
| `targets[].engine` | yes | Engine kind (e.g. `vllm`). |
| `targets[].base_url` | optional | Endpoint URL. |
| `targets[].quant` | optional | Quantization format string recorded on the envelope. |
| `targets[].api_key_env` | optional | Env var to read for the API key. Target is skipped (yellow warning) if the var is unset. |
| `targets[].extra` | optional | Extra `RunContext.extra` keys forwarded to the plugin. |

## Adding a hosted-OpenAI target

```yaml
  # - name: openai-gpt4o
  #   model: gpt-4o-mini
  #   engine: openai
  #   base_url: https://api.openai.com/v1
  #   api_key_env: OPENAI_API_KEY
```

Phase 1 ships vLLM, SGLang (skeleton), llama.cpp, and provider-hosted engines via the OpenAI-compatible kind. Set the env var before invoking `bench matrix`; targets whose env var is missing are skipped with a warning rather than failing the whole matrix.

## Exit codes

- `0` — every pair produced an envelope (or was skipped because of a missing API key).
- `1` — at least one pair errored, or no envelopes were produced.
- `2` — invalid YAML or missing `--output`.

## See also

- [bench run](bench-run.md) — single-endpoint version
- [bench summary](bench-summary.md) — tabulate the resulting envelopes
- [Recipes: multi-vendor matrix](../recipes/multi-vendor-matrix.md)
