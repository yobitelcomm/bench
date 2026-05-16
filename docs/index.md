# InferenceBench

Vendor-neutral, hardware-fingerprinted, Sigstore-signed AI benchmarks for inference systems.

```bash
pip install inferencebench
bench run llm.inference --model meta-llama/Llama-4-Maverick --engine vllm --hardware h100
```

InferenceBench is a CLI plus a reproducibility envelope. Every result captures the exact hardware, software stack, dataset, and seed, then signs the bundle with Sigstore so anyone can verify it independently.

## What you get

- **A `bench` CLI.** One binary, one set of verbs: `run`, `compare`, `publish`, `verify`, `doctor`, `plugin`.
- **A signed envelope per result.** Hardware fingerprint, software provenance, dataset hash, seed, metrics, Sigstore signature.
- **Pareto outputs.** Throughput, latency, cost, energy, and quality together. No single headline number.
- **Hugging Face Hub publishing.** `bench publish --to hf` mints a citable dataset repo.

## What this is not (yet)

- A SaaS. (Phase 2.)
- Multi-modal. Phase 1 ships the `llm.inference` plugin only.
- Multi-vendor at GA. Phase 1 ships with H100 coverage from one cluster; MI300X, RTX 5090, and M5 Max are deferred until partnerships land.

## Next steps

- [Install the CLI](install.md)
- [Run your first benchmark in 5 minutes](quickstart.md)
- [Read the envelope concept](concepts/envelope.md)
- [Browse the CLI reference](reference/cli-reference.md)

## Project status

Phase 1 is active. Target for v0.1 on PyPI: late 2026. The CLI is open source under Apache 2.0. Source lives at [github.com/yobitelcomm/bench](https://github.com/yobitelcomm/bench).
