# Contributing

External contributions are welcome. The project is in early Phase 1, so please open an issue before starting non-trivial work.

The canonical contributing guide lives in the repository at [CONTRIBUTING.md](https://github.com/yobitelcomm/bench/blob/main/CONTRIBUTING.md). Highlights below.

## Getting started

```bash
git clone https://github.com/yobitelcomm/bench
cd bench
uv sync --all-extras --dev
pre-commit install
make all
```

The `make all` target runs lint, type check, and the full test suite.

## What we welcome

- Bug fixes for documented issues
- New plugins, following the methodology review process
- Methodology improvements for existing benchmarks
- Hardware support for vendors we do not yet cover (MI300X, RTX 5090, M5 Max)
- Documentation fixes and improvements

## What we do not accept

- Changes that compromise vendor neutrality
- Benchmarks without signed envelopes
- New benchmarks without a methodology review
- Code without tests
- Changes that bypass the convergence gate or the warm-up discipline

## Workflow

1. Find or open an issue.
2. Branch off `main` with the project naming scheme: `<type>/<scope>/<ticket-id>-<short-description>`.
3. Write tests first when the spec is clear.
4. Open a PR using the template. CI must be green.
5. A maintainer reviews and merges.

## Conventional Commits

We enforce [Conventional Commits](https://www.conventionalcommits.org/). Examples:

```
feat(plugin-llm): add SGLang engine support
fix(envelope): correct content_hash canonical ordering
docs(quickstart): clarify HF Hub publish flow
```

## Contributing a new plugin

A plugin is a Python package that registers an `inferencebench.plugins` entry point and implements the four-method plugin contract. The CLI ships a scaffolder that produces an end-to-end runnable package — including a `smoke` benchmark you can run immediately to produce a signed envelope.

### 1. Scaffold

```bash
uv run bench plugin init my-modality --kind both --modality llm
```

This creates `plugins/my-modality/` with:

```
plugins/my-modality/
  pyproject.toml          # name: inferencebench-my-modality; entry point wired up
  README.md
  src/inferencebench_my_modality/
    __init__.py
    schemas.py            # BenchmarkSpec + RunContext pydantic models
    plugin.py             # MyModalityPlugin class — the four contract methods
  tests/
    test_plugin.py        # asserts the smoke benchmark produces a signed envelope
```

Install it into the workspace and run the smoke benchmark:

```bash
uv pip install -e ./plugins/my-modality
cosign generate-key-pair
uv run bench run my-modality.smoke --signing-mode dev --dev-key cosign.key
```

You should see a signed envelope under `~/.cache/inferencebench/runs/`.

### 2. The plugin contract

Your plugin class implements four methods. The scaffolded class has working stubs for each:

| Method | Purpose |
|---|---|
| `list_benchmarks() -> list[BenchmarkSpec]` | Return every benchmark this plugin exposes. Used by `bench list`. |
| `get_benchmark(benchmark_id: str) -> BenchmarkSpec` | Resolve one spec by id. Used by `bench run <id>`. |
| `validate(spec, context) -> list[str]` | Return human-readable errors. Empty list = OK. Runs before `run`. |
| `run(spec, context) -> Envelope` | Execute the benchmark and return a signed envelope. |

The reference implementation is [`plugins/llm-inference/`](https://github.com/yobitelcomm/bench/tree/main/plugins/llm-inference/) — it shows how to plumb a real workload (vLLM, SGLang, llama.cpp, MLX), drive the convergence gate, capture NVML/RAPL telemetry, and emit a Sigstore-signed envelope.

For a smaller example, see [`plugins/llm-quality/`](https://github.com/yobitelcomm/bench/tree/main/plugins/llm-quality/) — deterministic fixture scoring, no engine integration, LLM-as-judge deferred to Phase 2.

### 3. Naming

- Package name: `inferencebench-<short-name>` (PyPI distribution).
- Python module: `inferencebench_<short_name>`.
- Entry point id: `<short-name>` (lowercase, hyphens, must match `[a-z][a-z0-9-]*`).
- Benchmark ids: `<short-name>.<benchmark>` (e.g. `voice.asr-librispeech`).

### 4. Methodology review

New plugins go through methodology review before they merge. Open a "Benchmark suggestion" issue first using the [benchmark issue template](https://github.com/yobitelcomm/bench/blob/main/.github/ISSUE_TEMPLATE/benchmark.yml). The validator checks dataset license, contamination risk, vendor bias, and scoring metric robustness.

## See also

- [Code of conduct](code-of-conduct.md)
- [Security policy](security.md)
- [CONTRIBUTING.md on GitHub](https://github.com/yobitelcomm/bench/blob/main/CONTRIBUTING.md)
- [Methodology](../concepts/methodology.md)
- [`bench plugin` reference](../cli/bench-plugin.md)
