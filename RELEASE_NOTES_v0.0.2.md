# v0.0.2 — 2026-05-18

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
