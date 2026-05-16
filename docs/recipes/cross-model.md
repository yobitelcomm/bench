# Recipe: cross-model comparison

Same hardware, same engine, same benchmark, two models. Llama-3.1-8B-Instruct vs Qwen2.5-7B-Instruct on the `llm.inference.chatbot-short` corpus. The numbers below come from the bundled corpus at `validation-runs/2026-05-16-cross-model-corpus/corpus/all/` — captured on H100-80GB-HBM3 via vLLM 0.21.0, fp16, on 2026-05-16.

## 1. Run the sweep on each model

```bash
bench run llm.inference.chatbot-short \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --engine vllm --quant fp16 \
  --sweep 1,4,16,64 \
  --base-url http://localhost:8000/v1 \
  --output ./corpus/llama-3.1-8b

bench run llm.inference.chatbot-short \
  --model Qwen/Qwen2.5-7B-Instruct \
  --engine vllm --quant fp16 \
  --sweep 1,4,16,64 \
  --base-url http://localhost:8000/v1 \
  --output ./corpus/qwen-2.5-7b
```

Eight signed envelopes total. Stash them under one `./corpus/all/` directory for the next two steps.

## 2. Tabulate

```bash
bench summary ./corpus/all
```

Expected output (excerpt — conc=16 row from each model side by side):

```
                        Suite: llm.inference.chatbot-short
 Model                              Engine       Quant  Hardware              Throughput  TTFT p50/p99  J/tok  Run ID short
 meta-llama/Llama-3.1-8B-Instruct   vllm 0.21.0  fp16   NVIDIA H100 80GB HBM3  1,384.2     41.69/64.71   0.70   01j7q5c6
 Qwen/Qwen2.5-7B-Instruct           vllm 0.21.0  fp16   NVIDIA H100 80GB HBM3  1,362.3     40.98/63.42   0.71   01j7q5d2
 ...
8 envelopes loaded, 0 skipped (validation failure), 1 suites
```

At conc=16 the two models are within 1.5 % on throughput and within 1 % on TTFT — both at roughly 1380 tok/s and 0.70 J/tok. Qwen's `joules_per_token` at conc=1 is meaningfully better (6.28 vs 7.24), and that lead narrows as concurrency rises and both models saturate the GPU.

## 3. Pareto comparison

```bash
bench compare ./corpus/all/*.json --report pareto
```

Renders only the Pareto-optimal rows. At conc=16 both models land on the throughput-vs-energy frontier; at conc=1 Qwen wins on energy while Llama wins on TTFT.

For the full JSON dump (suitable for piping into jq or feeding a custom visualisation):

```bash
bench compare ./corpus/all/*.json --report json | jq '.runs[] | {model: .model_id, tput: .metrics.throughput_tok_per_s, j_per_tok: .metrics.joules_per_token, pareto: .pareto}'
```

## 4. Cost-aware view

Pricing isn't carried on the perf envelope. Pull it from the registry instead:

```bash
bench cost llama-3.1-8b-instruct
bench cost qwen-2.5-7b-instruct
```

Compare blended-rate columns at the same `--input-token-share` to make the comparison apples-to-apples.

## Reading the result

Two models, same suite, same hardware, same engine. Where they tie (conc=16 throughput), you know it's the engine and the GPU doing the work — neither model has an architectural edge here. Where they diverge (conc=1 energy), you have a real signal worth tracing back to model architecture and tokenizer behaviour.

This is exactly the comparison a signed envelope makes credible. The hardware fingerprint and software provenance match across both runs; the only changed variable is the model. Anyone can re-run the same sweep on the same engine version and check whether the gap holds.

## Where to go next

- [bench summary reference](../cli/bench-summary.md)
- [bench compare reference](../cli/bench-compare.md)
- [Recipes: concurrency sweep](concurrency-sweep.md) — the single-model variant
- [Pareto frontiers](../concepts/pareto.md)
