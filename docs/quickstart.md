# Quickstart

You will:

1. Install the CLI and the `llm.inference` plugin.
2. Run a hardware diagnostic.
3. Run a benchmark and produce a signed envelope.
4. Verify it.
5. (Optional) Publish the result to Hugging Face Hub.

Total time: roughly 5 minutes plus the benchmark itself.

## 1. Install

```bash
pip install inferencebench inferencebench-llm
```

Verify:

```bash
bench --version
```

Expected output:

```
bench 0.0.0
```

## 2. Check the hardware

```bash
bench doctor
```

Expected output (on a healthy H100 node):

```
Hardware diagnostic
Check                Status   Detail
NVML available       PASS     12 GPUs visible
Driver version       PASS     560.35.03
ECC enabled          PASS     enabled on all GPUs
Persistence mode     PASS     enabled
Thermal headroom     PASS     all GPUs < 75 degC
Clock state          PASS     no throttling flags
OK — all checks passed.
```

`bench doctor` refuses with exit code 1 if it detects thermal throttling, ECC errors, or driver drift. Pass `--strict` to also fail on warnings.

## 3. Run a benchmark

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

The harness:

1. Discards three warm-up runs.
2. Waits for the convergence gate (CoV < 5% across the last 30 requests).
3. Drives 300 seconds of Poisson-arrival load at each concurrency.
4. Samples NVML and RAPL telemetry the entire time.
5. Hashes the hardware fingerprint and software provenance.
6. Writes a signed envelope.

Expected output (truncated):

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
  joules_per_token       0.32
Envelope: ~/.cache/inferencebench/runs/01J7Q5C6.../envelope.json
Signed:   sigstore-cosign (rekor log index 12345)
```

!!! note "Phase 1 status"
    `bench run` is currently a stub. The full harness wires in during the v0.1 release. The output shape above is what v0.1 will print.

## 4. Verify the envelope

```bash
bench verify ~/.cache/inferencebench/runs/latest/envelope.json
```

Expected output:

```
OK  ~/.cache/inferencebench/runs/latest/envelope.json
  method:           sigstore-cosign
  content_hash:     8b1a...e2c4
  suite:            llm.inference v1.0.0
  model:            meta-llama/Llama-4-Maverick
  engine:           vllm v0.7.2
  rekor_log_index:  12345
```

Verification recomputes the content hash, checks the Sigstore signature, and confirms the Rekor inclusion proof. Any mismatch is a hard failure.

## 5. Publish to Hugging Face Hub (optional)

```bash
export HF_TOKEN=hf_xxx
bench publish ~/.cache/inferencebench/runs/latest --to hf
```

Expected output:

```
Published: https://huggingface.co/datasets/Yobitel/llama-4-maverick__llm-inference__01j7q5c6
```

The published dataset repo contains the signed envelope, the raw traces parquet, and a rendered README with the headline metrics.

## Next steps

- [What is in an envelope](concepts/envelope.md)
- [Why we report Pareto frontiers](concepts/pareto.md)
- [llm.inference plugin reference](plugins/llm-inference.md)
- [Publishing to Hugging Face Hub](integrations/huggingface-hub.md)
