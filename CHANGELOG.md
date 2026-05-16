# Changelog

All notable changes to InferenceBench Suite will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1] — 2026-05-16

First public preview release. Everything below is subject to change.

### Added

#### CLI (`inferencebench` / `bench`)

- `bench run` — execute a benchmark from any installed plugin, emit a signed envelope.
  Flags: `--model`, `--engine`, `--base-url`, `--rps`, `--concurrency`, `--duration`,
  `--quant`, `--signing-mode {dev,keyless}`, `--dev-key`, `--output`, `--list`, `--strict`.
- `bench verify` — verify a signed envelope's Sigstore or dev-key signature plus
  content hash. Exits non-zero on any tamper or signature failure.
- `bench compare` — Pareto-frontier comparison across two or more envelopes on three
  axes (quality vs cost, throughput vs latency, throughput vs energy). Output as
  Rich table, JSON, or pareto-only filtered.
- `bench cost` — provider price comparison from the bundled pricing registry, with
  blended-rate column (`--input-token-share`) and close-match suggestions.
- `bench doctor` — pre-run hardware diagnostic: GPU thermal/ECC/memory/throttle,
  driver version, CUDA version.
- `bench publish` — publish a signed envelope to Hugging Face Hub or a local mirror.
  `--dry-run` plans without touching the network.
- `bench leaderboard --build` — render a static HTML+CSS leaderboard from a directory
  of signed envelopes. Framework-free, GitHub-Pages-ready.
- `bench plugin {list,init,info,install}` — manage benchmark plugins. `init` scaffolds
  a new plugin package skeleton.

#### Envelope (`inferencebench-envelope`)

- Canonical signed-envelope spec v1 (Pydantic v2 models, JSON Schema available).
- `Envelope.content_hash()` deterministic SHA-256 over canonical JSON.
- Sigstore keyless OIDC signing + ed25519 dev-key signing.
- `verify_envelope()` covers both methods plus tamper detection.

#### Harness (`inferencebench-harness`)

- `OpenLoopDriver` (Poisson arrival) + `ClosedLoopDriver` (bounded concurrency).
- `NVMLSampler` for GPU power/util/memory; `RAPLSampler` for CPU/DRAM energy.
- Hardware fingerprint collector: GPU, CPU, memory, BIOS, NUMA, driver, CUDA, NCCL.
- `ConvergenceGate` (CoV-based steady-state detection with warmup discard).
- `BenchmarkRun` top-level orchestrator wires driver + samplers + gate + metrics.
- Goodput-at-SLO + percentiles + power/energy summariser (trapezoidal integration).
- `ModelClient` LiteLLM wrapper with streaming TTFT measurement and provider-cost
  passthrough; tiktoken fallback for missing usage.
- `bench doctor` runtime (`run_diagnostic`).

#### LLM Inference Plugin (`inferencebench-llm`)

- vLLM engine adapter (OpenAI-compatible HTTP; version probe via `/version`).
- Sample benchmark spec `llm.inference.sharegpt-v3`.
- Dataset loader: `builtin://`, `file://`, `hf://` URIs with offline fallback.
- Pricing registry covering OpenAI, Anthropic, Google, plus Llama-3.1-8B/70B and
  Llama-4-Maverick on Together, Fireworks, Groq.

#### Integrations

- `inferencebench-hf-publisher` — publish envelopes to Hugging Face Hub as dataset
  repos under `yobitel-bench-results` (deterministic repo-id, README rendering,
  optional model-card backlink).
- `inferencebench-leaderboard` — Jinja2 + plain HTML/CSS/JS static-site renderer
  with Pareto-frontier classification per category.

#### Release infrastructure

- GitHub Actions release workflow (`.github/workflows/release.yml`) with PyPI
  Trusted Publisher OIDC flow and Sigstore signing for all six wheels.
- Reusable GitHub Action `yobitelcomm/bench/.github/actions/bench-action` for
  third-party CI integration.
- Example CI workflow under `examples/ci-integration/`.

#### Documentation

- mkdocs-material site under `docs/`: install, quickstart, CLI reference,
  envelope concept, plugin authoring guide, FAQ, community pages.

### Validated against real hardware

- 8 × NVIDIA H100 80GB HBM3 (TestBM) with vLLM 0.21.0 on Linux 6.17.
- Real `Llama-3.1-8B-Instruct` end-to-end: signed envelope produced, verified,
  Pareto-compared, leaderboard rendered, HF publish dry-run successful.
- Workspace: 252 tests pass, ruff clean, mypy strict clean across 56 source files.

### Known limitations

- Phase 1 supports only the vLLM engine; SGLang, TensorRT-LLM, llama.cpp, and MLX
  are deferred.
- Memory channels / speed and `dmi_uuid` fingerprint fields fall back to defaults
  on hosts without `dmidecode` root access. Energy fingerprint is still valid;
  the hardware fingerprint becomes less unique.
- RAPL CPU/DRAM telemetry requires `CAP_SYS_RAWIO` or membership in the `rapl`
  group on modern kernels; a single info log fires when domains are unreadable.
- Hosted leaderboard browse mode (`bench leaderboard <category>`) is Phase 2+;
  local rendering via `--build` ships.
- Sigstore keyless signing only works inside GitHub Actions where an OIDC token
  is available; local runs use dev-key signing.

[Unreleased]: https://github.com/yobitelcomm/bench/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/yobitelcomm/bench/releases/tag/v0.0.1
