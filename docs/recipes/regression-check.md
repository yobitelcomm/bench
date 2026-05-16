# Recipe: regression check

The canonical "did my kernel / config / engine-version change actually help, or did it sneak a regression past me?" workflow. Save a baseline envelope, change something, run the same benchmark again, then diff.

## 1. Capture the baseline

```bash
bench run llm.inference.chatbot-short \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --engine vllm \
  --quant fp16 \
  --concurrency 16 \
  --base-url http://localhost:8000/v1 \
  --output ./baseline
```

Tuck the envelope away in version control or an artifact store:

```bash
cp ./baseline/c16-*.json ./benchmarks/baselines/llama-3.1-8b-conc16.json
```

## 2. Change one variable

Bump the engine, change a quant format, swap the dataset, change the kernel — anything you want to evaluate. Run again with the same flags into a fresh output dir:

```bash
bench run llm.inference.chatbot-short \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --engine vllm \
  --quant fp16 \
  --concurrency 16 \
  --base-url http://localhost:8000/v1 \
  --output ./candidate
```

## 3. Diff

```bash
bench diff ./benchmarks/baselines/llama-3.1-8b-conc16.json ./candidate/c16-*.json
```

Each metric is rendered with absolute delta, relative delta, and a direction-aware verdict (`improvement` / `regression` / `no_change`). Regressions are sorted to the top of the table; improvements come next; unchanged metrics fall to the bottom.

Default tolerance is `±2 %`. Tweak with `--tolerance 0.01` for tighter or `--tolerance 0.05` for looser.

## 4. Wire it into CI

Add `--strict` and the command exits `1` the moment any metric regresses. This is what you want in a GitHub Actions step:

```yaml
- name: Inference benchmark regression check
  run: |
    bench run llm.inference.chatbot-short \
      --concurrency 16 \
      --base-url ${{ secrets.ENGINE_URL }} \
      --output ./candidate
    bench diff \
      ./benchmarks/baselines/llama-3.1-8b-conc16.json \
      ./candidate/c16-*.json \
      --strict
```

The build fails if `ttft_p99_ms` worsens by more than 2 %, if `throughput_tok_per_s` drops, if `joules_per_token` climbs, or if `ok_rate` slips below the baseline.

## Context warnings

If the baseline and candidate envelopes differ on suite, model, engine version, quant format, or hardware fingerprint, `bench diff` still runs but prints a yellow warning. That's deliberate — sometimes you want to compare across contexts (e.g. fp16 baseline vs fp8 candidate). When you do, interpret the deltas with that in mind.

## Where to go next

- [bench diff reference](../cli/bench-diff.md) — every flag and the direction policy
- [Recipes: concurrency sweep](concurrency-sweep.md) — sweep multiple points before locking in a baseline
- [GitHub Actions integration](../integrations/github-actions.md)
