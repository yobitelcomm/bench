# Changelog

All notable changes to InferenceBench Suite will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- 8 remaining wheel uploads to PyPI (`-hf-publisher`, `-leaderboard`, `-quality`,
  `-mt`, `-code`, `-voice`, `-embeddings`, `-vision`) — pending PyPI's
  new-project-creation rate-limit refresh; ship target v0.0.3.
- Sigstore keyless re-sign of the marathon corpus via the release.yml workflow,
  once Trusted Publisher OIDC is configured on pypi.org.

## [0.0.2] — 2026-05-18

First PyPI release. 4 core packages live: `inferencebench`,
`inferencebench-envelope`, `inferencebench-harness`, `inferencebench-llm`.
First HF corpus published: 50 signed envelopes across 9 models / 5 vendors
under [huggingface.co/Yobitel](https://huggingface.co/Yobitel).

### Public install

```bash
pip install inferencebench inferencebench-llm
# or
uv tool install inferencebench --with inferencebench-llm
```

The full plugin set (vision, voice, mt, code, quality, embeddings, plus the
hf-publisher and leaderboard integrations) is built and tested in tree;
PyPI publication of the remaining 8 packages is pending a rate-limit refresh
and lands in v0.0.3. Install from clone today via `uv sync --all-packages`.

### Validated against real hardware

- **50 signed envelopes** on 8×H100 across Meta (Llama 8B/70B), Alibaba (Qwen
  2.5-7B/Coder-7B/VL-7B), Mistral (7B-v0.3), Microsoft (Phi-3.5-mini),
  DeepSeek (Coder-V2-Lite), Google (Gemma-2-9B).
- Suites exercised: perf, factual recall, arithmetic, multi-turn persona,
  translation chrF, code-generation pass@1, vision OCR / chart-QA.
- Full corpus + leaderboard at <https://huggingface.co/Yobitel> and recipe at
  [docs/recipes/multi-vendor-marathon.md](docs/recipes/multi-vendor-marathon.md).
- Trust anchor for the corpus: `trust/cosign-2026-05-18-marathon.pub`
  (mirrored at <https://huggingface.co/datasets/Yobitel/bench-trust-anchors>).

### Capabilities + scope expansion

### Added — CLI commands (26 total in this release)

- `bench audit <dir>` — verify every envelope in a directory + signature check.
- `bench attest <envelope>` — markdown/JSON attestation slip for compliance.
- `bench bundle create/extract` — one-file shareable zip with standalone verifier.
- `bench cache list/clear/path` — manage `~/.cache/inferencebench/fetched/`.
- `bench ci init/validate` — generate or validate a GH Actions regression workflow.
- `bench cluster run/status/sync` — distributed runner coordinator over `bench server`.
- `bench coverage <dir>` — report per-envelope metric completeness.
- `bench dashboard <dir>` — live HTTP leaderboard with auto-rescan.
- `bench fixtures list/fetch/clear/path` — download real public datasets (FLORES,
  HumanEval, GSM8K, TruthfulQA-MC, MS MARCO) into a local cache.
- `bench history <dir>` — time-series of one metric across a corpus with sparkline.
- `bench matrix <yaml>` — run one benchmark across multiple endpoints in one command.
- `bench plugin discover` — query a curated registry of known plugins.
- `bench profile <envelope>` — re-run at 10ms NVML / 25ms RAPL with profiling breakdown.
- `bench schema --target {envelope,benchmark-spec,mirror-index}` — emit JSON Schema.
- `bench server` — minimal stdlib HTTP API for distributed envelope ingestion.
- `bench spec validate/show/lint` — schema check for community-authored YAMLs.
- `bench tour` — 10-step end-to-end install validation.
- `bench watch <dir>` — watcher that rebuilds the leaderboard on new envelopes.

- `bench run --sweep / --rps-sweep` — multi-point sweep in a single invocation,
  one envelope per operating point.
- `bench run --all-benchmarks` — execute every spec the plugin exposes.
- `bench run --prices-file <path>` — override the pricing registry for the
  envelope's cost synthesis.
- `bench summary <dir> [--json]` — Rich-table corpus overview, suite-grouped.
- `bench fetch <uri>` — resolve `hf://`, `https://`, `file://`, plain path to
  a local cache (`~/.cache/inferencebench/fetched/`). Cache-aware (`--force`).
- `bench replay <envelope>` — re-run a benchmark from a signed envelope
  (reproducibility-first; refuses to replay a tampered envelope by default).
- `bench diff <a> <b>` — per-metric delta with regression/improvement verdicts
  and `--strict` mode for CI gates. Tolerance band `--tolerance` (default 2%).
- `bench list` — catalogue every benchmark across every installed plugin.
- `bench schema --target {envelope,benchmark-spec,mirror-index}` — emit JSON
  Schema for non-Python consumers. `--version` prints the envelope schema tag.
- `bench history <dir>` — time-series of one metric across a corpus with a
  unicode sparkline.
- `bench export <envelope> --format {markdown,csv,slack}` — share-friendly
  conversions.
- `bench profile <envelope>` — re-run with 10ms NVML / 25ms RAPL telemetry
  plus a profiling breakdown (% time on host, GPU vs CPU+DRAM energy ratio).
- `bench cache list/clear/path` — manage the local fetch cache.
- `bench bundle create/extract` — single-file shareable artifact containing
  the envelope, neighbouring samples, a `signature_info.json`, and a
  standalone `verify.py` script with no dependency on `inferencebench-envelope`
  (Python 3.12 + `cryptography` only).

### Added — engines, plugins, pricing

- **`inferencebench-quality` plugin** — exact/substring/F1/judge-llm scoring +
  multi-turn `persona-consistency` benchmark with drift detection.
- **`inferencebench-mt` plugin** — chrF, BLEU, exact-match scorers; FLORES-200
  mini fixtures for en-fr/de/es/ja.
- **`inferencebench-code` plugin** — pass@1 / pass@k scoring against a
  sandboxed subprocess test runner (forbidden-import pre-screen). humaneval-mini
  + mbpp-mini bundled.
- **`inferencebench-voice` plugin** — Whisper-compatible audio path
  (multipart POST to `/audio/transcriptions`); WER/CER/exact_match. Bundled
  fleurs-mini + long-form + code-switched + accented (20 small WAVs total).
- **`inferencebench-embeddings` plugin** — recall@k / MRR / NDCG; beir-mini +
  long-doc + msmarco-style + query-expansion bundled.
- **`inferencebench-vision` plugin** — OpenAI-compatible multimodal request
  shape (text + base64 image); 5 bundled PNGs each for OCR and chart-QA.
- **5-engine matrix**: vLLM (live-validated), SGLang, llama.cpp, TensorRT-LLM,
  MLX (the latter four as skeletons with HTTP integration tests).
- **Per-GPU SLO multipliers**: h200 0.6×, h100 1.0× (anchor), a100 1.5×, l4
  2.5×, rtx-5090 1.2×, rtx-4090 1.8×, rtx-3090 3.0×, mi300x 1.2×, m-series 5×,
  cpu 20×. Envelope carries `slo_hardware_class` + `slo_template_resolved`.
- **Pricing registry now loaded from `prices.yaml`** at module import (with a
  hardcoded Python fallback if the YAML is missing). New `load_pricing()` and
  `set_pricing()` public functions. `bench cost --validate-prices <path>`
  validates user-supplied pricing files.
- **Cost synthesis from registry** when LiteLLM reports zero cost: envelope
  now carries `cost_usd_per_million_tokens` using the cheapest registered
  provider's blended (3:1) rate, plus a new string metric `cost_source`
  identifying the origin (`provider` or `registry:<provider>`).
- **`bench publish --to local`** now writes a self-describing `index.json`
  (`inferencebench.mirror.v1` schema) at the mirror root.

### Changed

- `Envelope.metrics` type widened to `dict[str, float | int | str | None]` to
  accept the new `cost_source` string metric. Forward-compatible — old
  consumers reading numeric keys are unaffected.
- `bench plugin init` now scaffolds a fully-runnable plugin (echo engine +
  dev-key signing) instead of a stub.

### Validated against real hardware

- Cross-model concurrency sweep on 1 × H100: Llama-3.1-8B and Qwen2.5-7B at
  concurrencies [1, 4, 16, 64]. Eight signed envelopes produced + verified +
  Pareto-compared. Llama c16 wins the throughput-vs-energy Pareto (1384 tok/s,
  0.70 J/tok). Numbers feed `docs/recipes/concurrency-sweep.md` and
  `docs/recipes/cross-model.md`.

### Workspace

- 861 tests pass · ruff clean · mypy strict clean across 120 source files
  · `mkdocs build --strict` clean.
- 12 wheels build via `scripts/release_dry_run.py`; 4 published, 8 pending.
- Docker image + `Dockerfile` + `docker.md` recipe for container-based usage.
- Self-regression CI (`.github/workflows/self-regression.yml`) on every PR.
- Plugin registry at `tools/plugin-registry/registry.json` discoverable via
  `bench plugin discover`.

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
