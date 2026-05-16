# Plugin: llm.inference

The `llm.inference` plugin benchmarks LLM serving systems. Phase 1 ships with vLLM on Linux H100; SGLang, TensorRT-LLM, llama.cpp, and MLX are deferred to Phase 2.

```bash
pip install inferencebench inferencebench-llm
bench run llm.inference --model meta-llama/Llama-4-Maverick --engine vllm --quant fp8
```

## What it measures

The plugin drives prompts through a serving endpoint and measures:

- **Time-to-first-token (TTFT).** Latency from request submission to first decoded token, in ms. Reported as `p50` and `p99`.
- **Time-per-output-token (TPOT).** Latency between successive decoded tokens, in ms. Reported as `p50` and `p99`.
- **Throughput.** Tokens produced per second across all concurrent requests.
- **Goodput at SLO.** Tokens-per-second the system can sustain while still satisfying the SLO template.
- **Power.** Average wall power across the GPUs, in watts.
- **Energy per token.** `power_avg_w / throughput_tok_per_s`, in joules per token.
- **Cost.** USD per million tokens, computed against a published pricing snapshot when the provider is a hosted endpoint.

## Datasets

Phase 1 ships:

| Dataset id | Description | Size |
|---|---|---|
| `sharegpt-v3` | A canonical-ordered subset of ShareGPT V3 conversations | 10K turns |

Additional datasets land in Phase 2.

## Engines

| Engine | Status |
|---|---|
| `vllm` | Phase 1 |
| `sglang` | Phase 2 |
| `trtllm` | Phase 2 |
| `llama.cpp` | Phase 2 |
| `mlx` | Phase 2 |

## SLO templates

| Template | TTFT p99 | TPOT p99 |
|---|---|---|
| `llm.standard` | 300 ms | 50 ms |
| `llm.realtime` | 100 ms | 30 ms |
| `llm.batch` | n/a | 200 ms |

## Example run

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
  power_avg_w          612
  joules_per_token       0.32
```

## Methodology

Three warm-up runs are discarded. The convergence gate requires CoV < 5% across the last 30 requests before measurement begins. The driver is open-loop Poisson at the requested concurrency. Percentile reports include 95% bootstrap CIs (1000 resamples).

For cross-engine comparisons, three independent process launches with different seeds are required. The plugin enforces this when more than one engine is in the comparison.

## Known limitations (Phase 1)

- vLLM only. SGLang/TensorRT-LLM/llama.cpp/MLX support is Phase 2.
- Linux x86_64 H100 only. Other hardware classes pass the driver but lack tuned engine configs.
- No vision-language models. Multi-modal prompts are Phase 2.
- The cost figure assumes the listed provider's published pricing snapshot; promotional pricing is not reflected.

## See also

- [Methodology](../concepts/methodology.md)
- [Reproducibility](../concepts/reproducibility.md)
- [bench run](../cli/bench-run.md)
