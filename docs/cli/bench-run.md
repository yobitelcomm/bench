# bench run

Execute a benchmark and produce a signed envelope. Supports single-point runs, closed-loop concurrency sweeps (`--sweep`), open-loop RPS sweeps (`--rps-sweep`), and a "run every benchmark this plugin ships" mode (`--all-benchmarks`).

## Synopsis

```bash
bench run <suite-id> [OPTIONS]
```

`<suite-id>` is either a plugin id (`llm.inference`) or a fully-qualified benchmark id (`llm.inference.sharegpt-v3`). When a plugin id is given without `--all-benchmarks`, the plugin's first registered spec is used.

## Example: concurrency sweep on Llama-3.1-8B

```bash
bench run llm.inference.chatbot-short \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --engine vllm \
  --hardware h100 \
  --quant fp16 \
  --sweep 1,4,16,64 \
  --base-url http://localhost:8000/v1 \
  --output ./results
```

Expected output (Rich table, abridged):

```
                       Sweep results (concurrency)
 conc  throughput_tok_per_s  ttft_p50_ms  tpot_p50_ms  ok_rate  J/tok  envelope
 1     122.2                 13.98        6.48         1.000    7.239  c1-814953250c16.json
 4     580.3                 22.75        6.59         1.000    1.631  c4-4a7ac8857dbf.json
 16    1384                  41.69        10.94        1.000    0.700  c16-60be8efd6d21.json
 64    1312                  86.92        46.91        1.000    0.691  c64-fed81eb00398.json
```

One signed envelope JSON is written per sweep point under `--output`.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `""` | Provider-prefixed model id (e.g. `meta-llama/Llama-3.1-8B-Instruct`). |
| `--engine` | `vllm` | Inference engine. vLLM ships; SGLang skeleton present. |
| `--hardware` | `h100` | Hardware class string recorded on the envelope. |
| `--quant` | `fp16` | Quantization format: `fp16`, `fp8`, `nvfp4`, `awq-int4`, etc. |
| `--concurrency` | `1` | Comma-separated levels (single-point uses the first; sweeps use `--sweep`). |
| `--rps` | `0.0` | Open-loop arrival rate (req/s); switches to open-loop driver. |
| `--sweep` | `""` | Closed-loop concurrency points, one envelope per point. Mutually exclusive with `--concurrency` and `--rps-sweep`. |
| `--rps-sweep` | `""` | Open-loop RPS points, one envelope per point. Mutually exclusive with `--rps` and `--sweep`. |
| `--all-benchmarks` | off | Run every spec the plugin exposes. Mutually exclusive with `--list`, `--sweep`, `--rps-sweep`. |
| `--list` | off | Print this plugin's bundled benchmark ids and exit. |
| `--dataset` | `""` | Dataset id override (falls back to the spec default). |
| `--duration` | `300` | Measurement duration in seconds. |
| `--slo-template` | `llm.standard` | SLO template id. |
| `--seed` | `42` | Random seed. |
| `--base-url` | `""` | Engine base URL (e.g. `http://localhost:8000/v1`). |
| `--output` | `./results` | Directory for the signed envelope(s). |
| `--signing-mode` | `dev` | `dev` (local cosign key) or `keyless` (Sigstore OIDC). |
| `--dev-key` | `cosign.key` | Path to local cosign signing key when `--signing-mode=dev`. |
| `--strict` | off | Treat `plugin.validate()` warnings as fatal. |
| `--prices-file` | `""` | Path to a custom prices YAML used by the plugin's registry-cost fallback when LiteLLM doesn't report a provider cost. Forwarded to `RunContext.extra['prices_file']`. |
| `--judge-model` | `""` | LLM-as-judge model id. Only honoured when the spec selects `scoring: judge_llm`. Forwarded to `RunContext.extra['judge_model']`. |
| `--judge-max-questions` | `0` | Cap on the number of questions sent to the judge (`0` = no cap). Only the judged questions contribute to the accuracy metric. Forwarded to `RunContext.extra['judge_max_questions']`. |

## Sweep semantics

`--sweep` produces N envelopes — one per concurrency. The sweep table at the end is a quick readout; the canonical record is the per-point JSON. Sweep exit code is `0` only if every point landed `ok_rate >= 0.95`.

See [Recipes: concurrency sweep](../recipes/concurrency-sweep.md) for the end-to-end workflow on real H100 numbers.

## What the harness does

1. Resolves the plugin via the `inferencebench.plugins` entry-point group.
2. Validates the spec against the run context.
3. Drives traffic at each requested concurrency / RPS.
4. Samples NVML and (when available) RAPL telemetry the entire run.
5. Collects the hardware fingerprint, software provenance, dataset hash, seed.
6. Builds an envelope and signs it (dev key by default).

## Output

```
./results/
  c1-<hash>.json
  c4-<hash>.json
  c16-<hash>.json
  c64-<hash>.json
```

The first 12 hex of the envelope's `content_hash` prefixes each filename.

## See also

- [bench doctor](bench-doctor.md) — run before `bench run` to catch unsafe hardware state
- [bench summary](bench-summary.md) — tabulate a directory of envelopes
- [Recipes: concurrency sweep](../recipes/concurrency-sweep.md)
- [Reproducibility](../concepts/reproducibility.md)
