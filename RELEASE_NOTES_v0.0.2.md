# v0.0.2 — 2026-05-17

Capabilities + scope expansion. Engine matrix grew, second plugin landed.

### Added — CLI commands (now 19 total)

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

- **`inferencebench-quality` plugin** — second plugin in the workspace.
  Two bundled benchmarks (`llm.quality.factual-mini`, `llm.quality.reasoning-mini`)
  with deterministic exact-match / substring-match / token-F1 scoring against
  bundled fixture answers. Emits accuracy + bootstrap CIs alongside the
  perf/cost/energy stack.
- **SGLang engine adapter** — OpenAI-compatible HTTP probe (`/get_server_info`
  with `/v1/models` fallback) and ModelClient builder.
- **llama.cpp engine adapter** — `/props` probe with `/v1/models` fallback,
  CPU + GGUF quantized inference coverage.
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

- 441 tests pass · ruff clean · mypy strict clean across 73 source files.
- Docs site (`docs/`) refreshed: every command documented, four new recipe
  pages, `mkdocs build --strict` clean.

---

## Installation

PyPI publishing is pending Trusted Publisher setup. Install from clone for now:

```bash
git clone https://github.com/yobitelcomm/bench
cd bench
uv sync --all-packages --dev --prerelease=allow
uv run bench --help
```

## Verifying releases

Every signed envelope produced by this version verifies with:

```bash
bench verify <envelope.json> --dev-public-key cosign.pub
```

For keyless Sigstore signatures (CI-produced), use `bench verify` without `--dev-public-key`.
