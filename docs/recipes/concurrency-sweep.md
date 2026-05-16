# Recipe: concurrency sweep

The textbook InferenceBench story. Sweep concurrency over `1, 4, 16, 64` on Llama-3.1-8B-Instruct and watch throughput climb from 122 → 1384 tok/s while energy per token drops from 7.24 → 0.70 J/tok.

The numbers below are real: captured on a single H100-80GB-HBM3 via vLLM 0.21.0, fp16, `llm.inference.chatbot-short`, 2026-05-16. The envelopes ship in the repo at `validation-runs/2026-05-16-cross-model-corpus/corpus/llama-3.1-8b/`.

## 1. Pre-flight

```bash
bench doctor
```

Refuse to benchmark if any GPU is throttling, persistence mode is off, or the driver is below floor.

## 2. Sweep

```bash
bench run llm.inference.chatbot-short \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --engine vllm \
  --quant fp16 \
  --sweep 1,4,16,64 \
  --base-url http://localhost:8000/v1 \
  --output ./results
```

One signed envelope per point. The trailing summary table:

```
                       Sweep results (concurrency)
 conc  throughput_tok_per_s  ttft_p50_ms  ttft_p99_ms  tpot_p50_ms  ok_rate  J/tok
 1     122.2                 13.98        14.95        6.48         1.000    7.239
 4     580.3                 22.75        32.18        6.59         1.000    1.631
 16    1,384.2               41.69        64.71        10.94        1.000    0.700
 64    1,312.3               86.92        464.7        46.91        1.000    0.691
```

A few things worth reading off this table:

- Throughput peaks around conc=16. Going to conc=64 doesn't buy more tokens/s but it does blow `ttft_p99` from 65 ms to 465 ms — pure queueing latency.
- `joules_per_token` flattens at conc=16. Past that you're paying TTFT for nothing.
- `ok_rate=1.000` at every point — the engine isn't dropping requests.

The Pareto frontier on this corpus has two members: conc=1 (best TTFT) and conc=16 (best throughput, best energy). conc=4 and conc=64 are dominated.

## 3. Frontier view

```bash
bench compare ./results/c1-*.json ./results/c4-*.json ./results/c16-*.json ./results/c64-*.json --report pareto
```

Renders the Pareto-only rows (conc=1 and conc=16 here) bolded.

## 4. Static site

```bash
bench leaderboard --build --envelopes ./results --out ./site
```

The renderer emits a per-suite page and a top-level `index.html` you can drop straight into GitHub Pages or any static host.

## 5. Optional: publish

```bash
export HF_TOKEN=hf_xxx
for f in ./results/*.json; do
  bench publish "$f" --to hf --tag llama-3.1-8b-sweep
done
```

Each envelope becomes its own HF dataset repo; the `--tag` lets you find them later.

## Where to go next

- [bench diff](../cli/bench-diff.md) — guard the conc=16 number against regressions in CI
- [Recipes: cross-model](cross-model.md) — repeat this sweep on Qwen2.5-7B and compare
- [Pareto frontiers](../concepts/pareto.md) — why we sweep instead of reporting one number
