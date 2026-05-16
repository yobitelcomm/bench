# Plugins overview

A plugin packages one benchmark suite: the dataset, the driver, the metrics, and the leaderboard rendering. Plugins are independent Python packages discovered by the CLI through entry points.

```bash
pip install inferencebench-llm
bench plugin list
```

Expected output:

```
Installed plugins
Name             Module                          Distribution
llm.inference    inferencebench_llm:plugin       inferencebench-llm
```

## Phase 1 plugins

| Plugin | Suite id | Status |
|---|---|---|
| `inferencebench-llm` | `llm.inference` | Phase 1, vLLM on Linux H100 only |

The roster expands in Phase 2: voice, video, image, embeddings, time-series, robotics, agents.

## Plugin contract

Every plugin implements the same minimum interface:

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

The CLI hands the plugin a `RunContext` with hardware fingerprint, driver options, and a writable cache directory. The plugin returns a `RawResult` that the CLI then packages into an envelope and signs.

## Discovery

Plugins register via Python entry points:

```toml
[project.entry-points."inferencebench.plugins"]
"llm.inference" = "inferencebench_llm:plugin"
```

The CLI calls `importlib.metadata.entry_points(group="inferencebench.plugins")` at startup. No plugins ship in the core CLI; the discovery system must work with zero plugins installed and return an empty list, not an error.

## Writing a new plugin

Scaffold a new plugin package:

```bash
bench plugin init voice --kind both --modality voice
```

This creates a `inferencebench-voice/` directory with a stub `Plugin` class, a methodology page template, and a snapshot test fixture. From there you wire the dataset, the driver, and the metrics.

A new plugin must pass the methodology review before its results land on a public leaderboard.

## See also

- [llm.inference plugin](llm-inference.md)
- [Methodology](../concepts/methodology.md)
- [bench plugin](../cli/bench-plugin.md)
