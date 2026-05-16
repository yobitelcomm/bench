# InferenceBench

Vendor-neutral, hardware-fingerprinted, Sigstore-signed AI benchmarks for inference systems.

```bash
pip install -e ./cli -e ./envelope -e ./harness
bench run llm.inference.chatbot-short \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --engine vllm --quant fp16 --sweep 1,4,16,64 \
  --base-url http://localhost:8000/v1
```

InferenceBench is a CLI plus a reproducibility envelope. Every result captures the exact hardware, software stack, dataset, and seed, then signs the bundle so anyone can verify it independently.

## What you get

- **A `bench` CLI.** 17 commands grouped around three things: running benchmarks (`run`, `replay`, `doctor`, `list`, `history`), reasoning about results (`compare`, `diff`, `summary`, `cost`, `schema`), and moving envelopes around (`fetch`, `publish`, `verify`, `export`, `leaderboard`, `plugin`, `plugins`). See the [CLI overview](cli/overview.md).
- **A signed envelope per result.** Hardware fingerprint, software provenance, dataset hash, seed, metrics, signature.
- **Pareto outputs.** Throughput, latency, cost, energy, and quality together. No single headline number.
- **Hugging Face Hub publishing.** `bench publish --to hf` mints a citable dataset repo.

## Recipes — start here

Four end-to-end workflows built from the commands above. The numbers in each recipe come from a real corpus captured on a single H100-80GB-HBM3 in May 2026 — they're the same envelopes that ship under `validation-runs/` in the repo.

- **[Concurrency sweep](recipes/concurrency-sweep.md).** Throughput climbs 122 → 1384 tok/s on Llama-3.1-8B as J/tok drops from 7.24 to 0.70. The textbook story for `bench run --sweep`.
- **[Regression check](recipes/regression-check.md).** Capture a baseline, change a variable, `bench diff --strict`. Drop the diff into CI to fail the build on any regression.
- **[Verify and replay](recipes/reproducibility.md).** Anyone can verify a signed envelope and replay it on their own hardware.
- **[Cross-model comparison](recipes/cross-model.md).** Llama-3.1-8B vs Qwen2.5-7B on the same suite — both hit ~1380 tok/s at conc=16 with a slightly different energy profile.

## What this is not (yet)

- A SaaS. (Phase 2.)
- Multi-modal. Phase 1 ships the `llm.inference` plugin only.
- Multi-vendor at GA. Phase 1 ships with H100 coverage from one cluster; MI300X, RTX 5090, and M5 Max are deferred until partnerships land. Engine breadth is vLLM today, with an SGLang skeleton in tree.

## Next steps

- [Install the CLI](install.md)
- [Run your first benchmark in 5 minutes](quickstart.md)
- [Read the envelope concept](concepts/envelope.md)
- [Browse the CLI reference](reference/cli-reference.md)

## Project status

Phase 1 is active. PyPI release is pending — install from a git clone for now. The CLI is open source under Apache 2.0. Source lives at [github.com/yobitelcomm/bench](https://github.com/yobitelcomm/bench).
