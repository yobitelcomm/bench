# bench profile

Re-run a benchmark from an existing envelope with high-frequency NVML (10 ms) and RAPL (25 ms) sampling, and a short default duration. Where [`bench replay`](bench-replay.md) is tuned for reproducibility, `bench profile` is tuned for diagnosis — answering "where did my throughput go?".

## Synopsis

```bash
bench profile <envelope.json> --base-url URL [--duration SECONDS]
                              [--output DIR] [--signing-mode dev|keyless]
                              [--dev-key PATH] [--verify/--no-verify]
```

`--base-url` is required: envelopes are deliberately host-agnostic, so you must point this run at a live engine.

## Example: diagnose a Llama-3.1-8B sweep point

```bash
bench profile ./results/c16-60be8efd6d21.json \
  --base-url http://localhost:8000/v1 \
  --duration 30 \
  --output ./profile-results
```

Expected output (excerpt):

```
                                  Profile summary
 field          source                              profile                             match
 envelope       ./results/c16-60be8efd6d21.json     profile-c16-fc41a902c8de.json       no
 suite_id       llm.inference.chatbot-short         llm.inference.chatbot-short          yes
 model.id       meta-llama/Llama-3.1-8B-Instruct    meta-llama/Llama-3.1-8B-Instruct     yes
 engine.name    vllm                                vllm                                 yes
 quantization   fp16                                fp16                                 yes
 dataset.id     chatbot-short                       chatbot-short                        yes
 seed           42                                  42                                   yes

                          Headline metrics (source vs profile)
 metric                 source     profile
 throughput_tok_per_s   1384       1376
 ttft_p50_ms            41.69      42.10
 ok_rate                1.000      1.000
 joules_per_token       0.700      0.704

                              Profiling breakdown
 metric                  value    note
 % time on host          18.42%   100 - avg GPU util (81.58%)
 Energy GPU vs CPU+DRAM  6.124    9.43 kJ / 1.54 kJ
 Avg power under load    412.6 W  samples where util_gpu > 50%
 NVML sample count       3000     interval=10 ms
 RAPL sample count       1200     interval=25 ms
```

The profile envelope is written under `--output` (defaults to `./profile-results/`) and prefixed `profile-<content_hash[:12]>.json`.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--base-url` | required | Engine URL for the profile run. Envelopes are host-agnostic; the profile target lives here. |
| `--duration` | `30` | Measurement duration in seconds. Short by design — profiling is for inspection, not steady-state metrics. |
| `--output` | `./profile-results` | Output directory for the new profile envelope. |
| `--signing-mode` | `dev` | `dev` (local cosign key) or `keyless` (Sigstore OIDC). |
| `--dev-key` | unset | Path to local cosign signing key (used when `--signing-mode=dev`). |
| `--verify` / `--no-verify` | on | Verify the source envelope's signature before profiling. Use `--no-verify` for unsigned local fixtures only. |

## What gets overridden vs `bench replay`

- NVML sample interval: 10 ms (vs the run-time default).
- RAPL sample interval: 25 ms.
- Default `--duration` is short (30 s) and explicit.

Steady-state metrics from a profile run are not directly comparable to a normal run — telemetry overhead is higher and the window is shorter. Treat the metric table as a sanity check; the profiling breakdown is the load-bearing output.

## See also

- [bench replay](bench-replay.md) — same plumbing, reproducibility-tuned
- [bench doctor](bench-doctor.md) — confirm the host is healthy before profiling
