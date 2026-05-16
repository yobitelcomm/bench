# CLI reference

This is a hand-curated reference. Autogeneration from Typer help is on the v0.2 roadmap; until then, this page mirrors `bench --help` and each subcommand's `--help`.

## bench

```
Usage: bench [OPTIONS] COMMAND [ARGS]...

  InferenceBench Suite — vendor-neutral, signed-envelope AI benchmarks.

Options:
  --version          Show version and exit.
  -v, --verbose      Verbose logging (DEBUG level).
  --help             Show this message and exit.

Commands:
  run          Run a benchmark and produce a signed envelope.
  compare      Compare benchmark runs (Pareto frontier).
  publish      Publish a signed envelope (HF Hub, local).
  verify       Verify a signed envelope's signature + content.
  leaderboard  Browse public leaderboards.
  doctor       Diagnose hardware health before benchmarking.
  cost         Compare model cost across providers.
  plugin       Manage benchmark plugins.
  plugins      List installed plugins (shorthand for "bench plugin list").
```

## bench run

```
Usage: bench run [OPTIONS] SUITE_ID

  Run a benchmark from the named suite.

Arguments:
  SUITE_ID  Suite identifier, e.g. 'llm.inference'.  [required]

Options:
  --model TEXT           Model id (provider-prefixed).
  --engine TEXT          Inference engine (vllm, sglang, ...). [default: vllm]
  --hardware TEXT        Hardware class (h100, h200, ...). [default: h100]
  --quant TEXT           Quantization format (fp16, fp8, nvfp4, ...). [default: fp16]
  --concurrency TEXT     Comma-separated concurrency levels. [default: 1]
  --dataset TEXT         Dataset id (e.g. sharegpt-v3).
  --duration INTEGER     Measurement duration in seconds. [default: 300]
  --slo-template TEXT    SLO template id. [default: llm.standard]
  --seed INTEGER         Random seed. [default: 42]
  --output TEXT          Output path for the signed envelope.
```

See [bench run](../cli/bench-run.md) for examples.

## bench compare

```
Usage: bench compare [OPTIONS] RUN_IDS...

  Compare two or more benchmark runs.

Arguments:
  RUN_IDS...  One or more run IDs / envelope paths to compare.  [required]

Options:
  --report TEXT  Report format: pareto, table, json. [default: pareto]
```

See [bench compare](../cli/bench-compare.md) for examples.

## bench publish

```
Usage: bench publish [OPTIONS] RUN_ID

  Publish a signed envelope.

Arguments:
  RUN_ID  Run ID or envelope path to publish.  [required]

Options:
  --to TEXT          Target: hf, local, studio. [default: hf]
  --workspace TEXT   Workspace (Studio only).
  --tag TEXT         Optional tag for this publish.
```

See [bench publish](../cli/bench-publish.md) for examples.

## bench verify

```
Usage: bench verify [OPTIONS] ENVELOPE_URI

  Verify a signed envelope. Exits 0 on success, non-zero on failure.

Arguments:
  ENVELOPE_URI  Envelope URI: local path, hf://datasets/..., or https://...  [required]

Options:
  --dev-public-key PATH  Path to ed25519 public key for dev-signed envelopes.
```

See [bench verify](../cli/bench-verify.md) for examples.

## bench doctor

```
Usage: bench doctor [OPTIONS]

  Run hardware diagnostic. Exit 0 if OK, 1 otherwise.

Options:
  --strict  Refuse if any check returns FAIL or WARN.
```

See [bench doctor](../cli/bench-doctor.md) for examples.

## bench plugin

```
Usage: bench plugin [OPTIONS] COMMAND [ARGS]...

  Manage benchmark plugins.

Commands:
  list     List installed plugins.
  init     Scaffold a new plugin package.
  install  Install a plugin from PyPI.
  info     Show details for a specific plugin.
```

See [bench plugin](../cli/bench-plugin.md) for examples.

## bench leaderboard

```
Usage: bench leaderboard [OPTIONS] [CATEGORY]

  Show the public leaderboard for a category.

Arguments:
  [CATEGORY]  Category id (e.g. llm.inference). Omit to list categories.
```

Phase 2.

## bench cost

```
Usage: bench cost [OPTIONS] MODEL

  Compare model cost across providers.

Arguments:
  MODEL  Model id (e.g. llama-4-maverick).  [required]

Options:
  --suite TEXT      Suite for the cost comparison. [default: intelligence-index]
  --providers TEXT  Comma-separated provider list.
```

Phase 2.

## See also

- [CLI overview](../cli/overview.md)
- [Envelope schema](envelope-schema.md)
