# bench run

Run a benchmark suite and produce a signed envelope.

```bash
bench run <suite-id> [OPTIONS]
```

## Example

```bash
bench run llm.inference \
  --model meta-llama/Llama-4-Maverick \
  --engine vllm \
  --hardware h100 \
  --quant fp8 \
  --concurrency 1,4,16,64 \
  --duration 300 \
  --slo-template llm.standard \
  --seed 42
```

Expected output:

```
Run id:    01J7Q5C6...
Model:     meta-llama/Llama-4-Maverick @ fp8 on H100-SXM5-80GB
Engine:    vllm 0.7.2
Metrics:
  ttft_p50_ms          142.0
  ttft_p99_ms          280.3
  tpot_p50_ms           18.5
  throughput_tok_s    1842.1
  goodput_at_slo       142.3 req/s
Envelope: ~/.cache/inferencebench/runs/01J7Q5C6.../envelope.json
Signed:   sigstore-cosign (rekor log index 12345)
```

## Arguments

| Argument | Required | Description |
|---|---|---|
| `suite-id` | yes | Suite identifier, e.g. `llm.inference`. |

## Options

| Option | Default | Description |
|---|---|---|
| `--model` | `""` | Provider-prefixed model id, e.g. `meta-llama/Llama-4-Maverick`. |
| `--engine` | `vllm` | Inference engine. Phase 1 ships vLLM only. |
| `--hardware` | `h100` | Hardware class for documentation purposes. |
| `--quant` | `fp16` | Quantization format: `fp16`, `fp8`, `nvfp4`, `awq-int4`, etc. |
| `--concurrency` | `1` | Comma-separated concurrency levels (e.g. `1,4,16,64`). |
| `--dataset` | `""` | Dataset id (defaults to suite default). |
| `--duration` | `300` | Measurement duration in seconds, per concurrency. |
| `--slo-template` | `llm.standard` | SLO template id. |
| `--seed` | `42` | Random seed for reproducibility. |
| `--output` | auto | Output path for the signed envelope. |

## What the harness does

1. Validates the plugin and the dataset hash.
2. Runs three warm-up iterations and discards them.
3. Enforces the convergence gate (coefficient of variation < 5% across the last 30 requests).
4. Drives traffic with an open-loop Poisson driver at each concurrency.
5. Samples NVML (50 ms) and RAPL (100 ms) telemetry the entire run.
6. Collects the hardware fingerprint, software provenance, dataset hash, seed.
7. Builds an envelope and signs it.

## Where results land

```
~/.cache/inferencebench/runs/<run-id>/
  envelope.json
  traces.parquet
  doctor-report.json
  logs/
```

The path can be overridden with `--output`.

## Phase 1 status

`bench run` is a stub in v0.0.0. The harness wires in during the v0.1 release. The current stub prints the parsed arguments and exits 0.

## See also

- [bench doctor](bench-doctor.md) — run before `bench run` to catch unsafe hardware state
- [Reproducibility](../concepts/reproducibility.md)
- [llm.inference plugin reference](../plugins/llm-inference.md)
