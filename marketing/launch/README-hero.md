<div align="center">

```
  ___        __                            ____                  _
 |_ _|_ __  / _| ___ _ __ ___ _ __   ___ | __ )  ___ _ __   ___| |__
  | || '_ \| |_ / _ \ '__/ _ \ '_ \ / __||  _ \ / _ \ '_ \ / __| '_ \
  | || | | |  _|  __/ | |  __/ | | | (__ | |_) |  __/ | | | (__| | | |
 |___|_| |_|_|  \___|_|  \___|_| |_|\___||____/ \___|_| |_|\___|_| |_|
```

# InferenceBench

**Vendor-neutral AI benchmarks. Every result signed.**

[![CI](https://img.shields.io/github/actions/workflow/status/yobitelcomm/bench/ci.yml?branch=main&label=CI)](https://github.com/yobitelcomm/bench/actions)
[![PyPI](https://img.shields.io/pypi/v/inferencebench.svg)](https://pypi.org/project/inferencebench/)
[![Discord](https://img.shields.io/discord/000000000000000000?label=discord&logo=discord)](https://yobitelcomm.github.io/bench/community)
[![GitHub stars](https://img.shields.io/github/stars/yobitelcomm/bench?style=social)](https://github.com/yobitelcomm/bench)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

</div>

<!-- TODO: human to fill in — replace the Discord server ID `000000000000000000`
     in the badge URL above with the real server ID once the Discord is provisioned.
     The CI badge assumes the workflow file is `.github/workflows/ci.yml`; update
     `ci.yml` if the actual filename differs. -->

```bash
pip install inferencebench
bench run llm.inference.chatbot-short --model Qwen/Qwen2.5-72B-Instruct --engine vllm --hardware h100
bench verify ~/.cache/inferencebench/runs/latest/envelope.json
```

**Reference run shipped with v0.1.0** — Qwen2.5-72B-Instruct on 4×H100 (TP=4, BF16, 8K ctx, vLLM 0.22):

| concurrency | throughput | TTFT p50 | joules/token |
|---:|---:|---:|---:|
| 1 | 56 tok/s | 24 ms | 37 |
| 4 | 234 tok/s | 46 ms | 9.0 |
| 16 | **891 tok/s** | 47 ms | **2.5** |

Full Pareto frontier, hardware fingerprint, and signed envelope in [`validation-runs/2026-06-13-llm-h100/`](validation-runs/2026-06-13-llm-h100/). Verify yourself with `bench audit validation-runs/2026-06-13-llm-h100/`.

![demo](docs/demo.gif)

<!-- TODO: human to fill in — record a 20-30s asciinema cast of the three-command
     flow above and convert to docs/demo.gif before publishing. -->

## Why this exists

- **Today's AI benchmarks are either single-vendor, single-modality, or unreproducible.** None ship results you can independently verify byte-for-byte.
- **A benchmark you cannot reproduce is a marketing artifact.** InferenceBench signs every result with Sigstore and logs it in Rekor, so anyone can run `bench verify` and get a pass/fail rooted in a public transparency log.
- **Single numbers get reward-hacked.** Every result here is a full distribution plus cost plus power plus quality, on a Pareto frontier, not a leaderboard rank.

## How it compares

| | `bench` | NVIDIA AIPerf | MLPerf Inference | Artificial Analysis |
|---|---|---|---|---|
| Open source | Yes (Apache 2.0) | Yes | Yes | No |
| Vendor-neutral by design | Yes | NVIDIA-focused | Multi-vendor, fixed task list | Aggregator, closed methodology |
| Signed result envelopes | Yes (Sigstore + Rekor) | No | No | No |
| Hardware fingerprint per result | Yes | Partial | Partial | No |
| Multi-modality target | Yes (Phase 2+) | LLM | Multi (fixed) | LLM-focused |
| Reproducible by a third party | Yes | Partial | Yes (submission rounds) | No |

Not a competitive claim — these projects solve different problems. The table is about which property each project chooses to optimize for.

## Quickstart

```bash
# install
pip install inferencebench

# refuses to run if thermal throttling, ECC errors, or driver drift
bench doctor

# run an LLM inference benchmark
bench run llm.inference.sharegpt-v3 \
  --model Qwen/Qwen2.5-72B-Instruct \
  --engine vllm --quant bf16 \
  --hardware h100 \
  --duration 300 \
  --slo-template llm.standard

# verify the signed envelope locally
bench verify ~/.cache/inferencebench/runs/latest/envelope.json

# publish to Hugging Face Hub as a permanent, citable dataset entry
bench publish ~/.cache/inferencebench/runs/latest --to hf
```

## What is measured

Every run emits, at minimum:

- **TTFT** (time to first token) — P50, P95, P99, full distribution
- **TPOT** (time per output token) — P50, P95, P99, full distribution
- **Throughput** — tokens/sec, requests/sec
- **Goodput-at-SLO** — throughput conditioned on a latency template (`llm.standard`, `voice.realtime`, etc.)
- **Power** — average watts (NVML at 50ms, RAPL at 100ms)
- **Joules per token** — derived efficiency metric
- **$ per million tokens** — derived cost, from the engine's published or measured cost basis
- **Quality** — task-specific, when applicable (null when not)

The full schema is in [`envelope/`](envelope/) and versioned as `envelope.v1.json`. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the data flow.

## Phase 1 scope (honest)

This is v0.1. Known limits:

- **One engine shipped end-to-end:** vLLM. SGLang, TensorRT-LLM, llama.cpp and MLX are Phase 2.
- **One hardware tier shipped end-to-end:** NVIDIA H100. AMD MI300X, RTX 5090 and Apple silicon are Phase 2, gated on partnerships and hardware access.
- **Six modality plugins, reference envelopes for each:** `llm.inference`, `llm.quality`, `llm.mt`, `code.generation`, `vision.understanding`, `embeddings.retrieval`, `voice.transcription`. 3D, world-models, agents, robotics and chip kernels are sketched in [`ARCHITECTURE.md`](ARCHITECTURE.md) but not in v0.1.
- **No SaaS, no cloud orchestration.** The CLI runs locally; results go to disk or to Hugging Face Hub. Studio and Enterprise tiers are explicitly deferred.

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the phased roadmap.

## Contributing

External PRs welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`CONVENTIONS.md`](CONVENTIONS.md) first, then look at open tickets in [`TICKETS/phase-1/`](TICKETS/phase-1/). The most useful contributions today:

- Hardware coverage on a vendor we don't have access to (MI300X, RTX 5090, Apple M-series)
- Methodology critiques on the `llm.inference` plugin — open an issue using the `methodology-issue.md` template
- New engine drivers under `plugins/llm-inference/`

By contributing you agree to release your work under Apache 2.0 and to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Citation

```bibtex
@misc{inferencebench2026,
  title  = {InferenceBench: A Vendor-Neutral Signed-Envelope Benchmark Suite for AI Systems},
  author = {Yobitel team},
  year   = {2026},
  url    = {https://github.com/yobitelcomm/bench},
}
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
