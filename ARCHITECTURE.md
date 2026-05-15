# ARCHITECTURE.md — system design

> Code-actionable architecture. If you're about to make a structural decision, read this first and update it (with an ADR) if you're changing it.

## Mental model

InferenceBench is a **measurement pipeline** with three layers:

1. **Harness layer** — drives requests, captures telemetry, emits raw measurements
2. **Envelope layer** — packages measurements with provenance and signs them
3. **Service layer** — stores, compares, publishes, monetizes (**Phase 2+ only**)

The CLI is a thin wrapper that exposes layers 1+2 locally. Phase 1 ships layers 1+2 only. Studio (Phase 2+) becomes a thin web layer over layer 3 backed by the same layers 1+2 running in the cloud.

## Phase 1 component diagram

```
                         ┌─────────────────────────┐
                         │  bench CLI (Python 3.12)│
                         │  Typer + Rich           │
                         └────────────┬────────────┘
                                      │
                ┌─────────────────────┼─────────────────────┐
                │                     │                      │
        ┌───────┴──────┐      ┌───────┴──────┐      ┌───────┴──────┐
        │   Harness    │      │   Plugins    │      │   Envelope   │
        │ (eval engine)│      │ (Phase 1:    │      │  (Pydantic + │
        │              │      │  llm.        │      │   Sigstore)  │
        │              │      │  inference)  │      │              │
        └───────┬──────┘      └──────────────┘      └───────┬──────┘
                │                                            │
                │     ┌──────────────────────────────┐       │
                └────►│  LiteLLM (model invocation)  │       │
                      └──────────────────────────────┘       │
                                                              │
                                                  ┌───────────┴──────────┐
                                                  │   Sigstore           │
                                                  │   (cosign+rekor)     │
                                                  └───────────┬──────────┘
                                                              │
                                                              ▼
                                              Local disk OR
                                              Hugging Face Hub
                                                  │
                                                  ▼
                                          docs.yobitel.com/bench
                                          (static-export leaderboard
                                           hosted on GitHub Pages)
```

Phase 2+ adds services: API, orchestrator, runner pool, judge fleet, store (PG+CH+S3), Studio web. Those are NOT present in Phase 1.

## Components in detail (Phase 1 scope)

### CLI (`/cli`)

Typer-based, single `bench` binary. Commands map to verbs: `run`, `compare`, `publish`, `verify`, `leaderboard`, `doctor`, `cost`, `plugin`.

Phase 1 subcommand structure:
```
bench run <suite-id> [--model] [--engine] [--hardware] [--quant] [--concurrency] [--slo-template]
bench compare <run-ids...> [--report pareto|table|json]
bench publish <run-id> [--to hf|local] [--workspace] [--tag]
bench verify <envelope-uri>
bench leaderboard [<category>]
bench doctor [--strict]
bench plugin {list|init|install|info}
```

Plugin discovery uses Python entrypoints registered under `inferencebench.plugins`. Each plugin is its own pip package (Phase 1: only `inferencebench-llm`). Core CLI is plugin-less and discovers them at runtime.

Output: JSON (canonical) by default. `--format md|csv|parquet|bench-archive` for alternates. `.bench` archive is a tar.zst bundle of `(envelope.json, raw-traces.parquet, logs.tar, signature.sig, certificate.pem)`.

### Harness (`/harness`)

Pure Python library, no CLI. Public API:

```python
from inferencebench.harness import BenchmarkRun, OpenLoopDriver, ClosedLoopDriver
from inferencebench.harness.telemetry import NVMLSampler, RAPLSampler
from inferencebench.harness.metrics import Percentiles, BootstrapCI, GoodputAtSLO

run = BenchmarkRun(
    suite_id="llm.inference",
    model_endpoint=...,
    dataset=...,
    driver=OpenLoopDriver(arrival="poisson", rps=10.0, duration=300),
    telemetry=[NVMLSampler(50_ms), RAPLSampler(100_ms)],
    seed=42,
)
result = run.execute()
```

Drivers Phase 1: open-loop (Poisson arrivals at fixed RPS), closed-loop (bounded concurrency). Batch driver deferred. Three warm-up runs are discarded automatically. Convergence gate: coefficient of variation <5% across last 30 requests before measurement begins.

Percentiles use bootstrap CI on the percentile estimator (1000 resamples, 95% CI), never Gaussian assumptions. Three independent process launches with different seeds are required for cross-engine comparison.

### Envelope (`/envelope`)

The canonical signed result envelope. JSON schema versioned (`envelope.v1.json`). Fields:

```json
{
  "envelope_version": "v1",
  "suite_id": "llm.inference",
  "suite_version": "1.0.0",
  "run_id": "uuidv7",
  "timestamp": "ISO8601",
  "model": {
    "id": "meta-llama/Llama-4-Maverick",
    "revision": "git-sha",
    "provider": "vllm-local|together|...",
    "endpoint_hash": "sha256"
  },
  "engine": {
    "name": "vllm",
    "version": "0.7.2",
    "config_hash": "sha256",
    "image_digest": "sha256:..."
  },
  "quantization": {"format": "fp8", "method": "..."},
  "hardware_fingerprint": {
    "fingerprint_sha256": "sha256",
    "dmi_uuid": "...",
    "gpus": [{"model": "H100-SXM5-80GB", "pci_id": "...", "serial": "...", "vbios": "..."}],
    "cpu": {"model": "...", "microcode": "..."},
    "memory": {"channels": 12, "speed_mts": 4800, "ecc": true},
    "bios": {"version": "...", "resizable_bar": true, "above_4g": true},
    "numa": {...},
    "driver": "560.35.03",
    "cuda": "12.6",
    "nccl": "2.22.3"
  },
  "software_provenance": {
    "image_digest": "sha256:...",
    "pip_freeze_hash": "sha256",
    "git_commit": "...",
    "nvidia_smi_q_hash": "sha256"
  },
  "dataset": {"id": "sharegpt-v3", "hash": "sha256"},
  "seed": 42,
  "driver_options": {...},
  "metrics": {
    "ttft_p50_ms": 142.0,
    "ttft_p99_ms": 280.3,
    "tpot_p50_ms": 18.5,
    "throughput_tok_per_s": 1842.1,
    "goodput_at_slo": 142.3,
    "power_avg_w": 612,
    "joules_per_token": 0.32,
    "cost_usd_per_million_tokens": 0.18,
    "quality_score": null
  },
  "distributions": {
    "ttft_ms": "histogram or path-to-parquet",
    "tpot_ms": "histogram or path-to-parquet"
  },
  "slo_template": "voice.realtime|llm.standard|...",
  "warnings": ["..."],
  "signature": {
    "method": "sigstore-cosign",
    "certificate": "PEM",
    "rekor_log_index": 12345,
    "bundle": "..."
  }
}
```

Signing: keyless OIDC for OSS (GitHub identity), dev key locally for testing. HSM-backed for Enterprise deferred to Phase 4+. Verification: `bench verify` resolves the certificate chain, validates against Rekor, recomputes the envelope content hash, returns pass/fail.

### Plugins (`/plugins`)

Each plugin is its own package implementing a minimum interface:

```python
class Plugin:
    suite_id: str          # e.g. "llm.inference"
    version: str
    description: str

    def list_benchmarks(self) -> list[BenchmarkSpec]: ...
    def run(self, spec: BenchmarkSpec, context: RunContext) -> RawResult: ...
    def validate(self, result: RawResult) -> ValidationReport: ...
    def render_leaderboard(self, results: list[Envelope]) -> LeaderboardView: ...
```

Plugins live under `plugins/<modality>/` and publish to PyPI independently. Versioning: SemVer. Breaking changes bump major, methodology tweaks bump minor. Phase 1 ships ONE plugin: `plugins/llm-inference/`.

### Integrations (`/integrations`)

Phase 1: only **hf-publisher** and **github-actions** (reusable workflow `yobitelcomm/bench-action@v1`). Phase 2+: W&B, MLflow, Slack, Linear/Jira, Grafana, OpenTelemetry.

### Infra (`/infra`)

Phase 1: GitHub Actions workflows only (CI, nightly GPU bench on TestBM, release). Terraform + Helm deferred to Phase 2+.

## Data flow for one benchmark run (Phase 1, local CLI only)

1. User: `bench run llm.inference --model X --engine vllm --hardware h100`
2. CLI builds `BenchmarkSpec`, validates plugin, computes dataset hash
3. CLI invokes harness locally (no cloud orchestration in Phase 1)
4. `bench doctor` checks hardware: refuses if thermal throttling, ECC errors, driver drift
5. Harness: 3 discarded warmups, convergence gate, then measurement
6. Harness: drives requests via LiteLLM, samples NVML+RAPL telemetry, collects metrics
7. Plugin: scores results (no judge fleet yet — local-only quality metrics)
8. EnvelopeBuilder: collects all metrics + fingerprint + provenance into envelope JSON
9. Signing: keyless OIDC cosign sign, Rekor log entry
10. Persist: writes envelope to `~/.cache/inferencebench/runs/<run-id>/envelope.json`
11. CLI displays result summary
12. User: `bench publish <run-id> --to hf` → integrations/hf-publisher mirrors envelope to a HF dataset repo under `yobitel-bench-results/`

## Key design decisions (with rationale)

- **Python for CLI + harness**: ecosystem (every inference engine has Python bindings), team expertise. Go sidecars deferred until Python GIL bottlenecks measurement.
- **Sigstore over self-signed**: third-party verification is the entire point; nobody trusts self-signed.
- **LiteLLM for model invocation**: 100+ providers behind one interface. We add caching, instrumentation, cost extraction.
- **Plugins as separate packages**: keeps dependency tree per modality clean. Phase 1 just ships `inferencebench-llm`.
- **Monorepo over polyrepo**: solo engineer, fast iteration, atomic cross-component changes. Re-evaluate at team 5+.
- **No DB in Phase 1**: SQLite for local cache, HF Hub for public publishing. Postgres/ClickHouse deferred.
- **mkdocs-material over Docusaurus**: simpler for solo dev, fewer moving parts. Switch to Docusaurus only if team grows.

## ADRs

Major architectural decisions live in `memory/adr/` numbered chronologically. Open one with `/new-adr <title>` from Claude Code.
