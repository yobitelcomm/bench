# bench — InferenceBench Suite

> Vendor-neutral, hardware-fingerprinted, Sigstore-signed AI benchmarks.

```bash
pip install inferencebench
bench run llm.inference --model meta-llama/Llama-4-Maverick --engine vllm --hardware h100
```

**Status**: Phase 1 active development. v0.1 target: 2026-11-15.

This repo contains:

- **`bench` CLI** (Apache 2.0) — `pip install inferencebench`
- **`harness/`** — core measurement engine
- **`envelope/`** — canonical Sigstore-signed result envelope
- **`plugins/`** — per-modality benchmark plugins (Phase 1: `llm-inference` only)

## What this is

A benchmark suite that's:

- **Vendor-neutral.** Every benchmark runs on multiple inference engines and hardware vendors.
- **Hardware-fingerprinted.** Every result captures DMI UUID, GPU PCI IDs + serials, BIOS settings, drivers, CUDA, NCCL — SHA-256'd into the envelope.
- **Sigstore-signed.** Every result is signed via keyless OIDC and logged in Rekor. Anyone can `bench verify` a published result.
- **Hugging Face Hub-native.** `bench publish --to hf` mints a permanent, citable Dataset repo.
- **Pareto-framed.** No single headline numbers — always full distribution + cost + power + quality.

## What this is NOT (yet)

- A SaaS — that's Phase 2.
- Multi-modal — Phase 1 ships `llm.inference` only.
- Multi-vendor at GA — Phase 1 ships with TestBM H100 coverage; MI300X, RTX 5090, M5 Max deferred.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the realistic phased roadmap.

## Quickstart (Phase 1, end of November 2026)

```bash
pip install inferencebench
bench doctor              # hardware diagnostic, refuses if thermal throttling
bench run llm.inference --model meta-llama/Llama-4-Maverick \
                        --engine vllm --quant fp8 --hardware h100
bench verify ~/.cache/inferencebench/runs/latest/envelope.json
bench publish ~/.cache/inferencebench/runs/latest --to hf
```

## Contributing

See [CONVENTIONS.md](CONVENTIONS.md) and [TICKETS/phase-1/](TICKETS/phase-1/). Solo-engineer development pace; PRs welcome but please open an issue first to discuss scope.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Citation

If you use InferenceBench in academic work, please cite:

```bibtex
@misc{inferencebench2026,
  title = {InferenceBench: A Vendor-Neutral Signed-Envelope Benchmark Suite for AI Systems},
  author = {Yobitel team},
  year = {2026},
  url = {https://github.com/yobitelcomm/bench},
}
```
