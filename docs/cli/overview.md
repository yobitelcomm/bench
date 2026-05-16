# CLI overview

The `bench` CLI is a Typer app. Every subcommand maps to a verb you can run on a benchmark or an envelope.

```bash
bench --help
```

Expected output:

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

## Commands at a glance

| Command | Purpose | Page |
|---|---|---|
| `bench run` | Execute a benchmark suite, produce a signed envelope | [bench run](bench-run.md) |
| `bench compare` | Compare two or more runs, render Pareto frontier | [bench compare](bench-compare.md) |
| `bench publish` | Mirror an envelope to Hugging Face Hub or local | [bench publish](bench-publish.md) |
| `bench verify` | Validate signature + content hash + Rekor entry | [bench verify](bench-verify.md) |
| `bench doctor` | Hardware diagnostic, refuses on unsafe state | [bench doctor](bench-doctor.md) |
| `bench plugin` | List, init, install, info on plugins | [bench plugin](bench-plugin.md) |
| `bench leaderboard` | Browse public leaderboards (Phase 2) | — |
| `bench cost` | Cross-provider cost comparison (Phase 2) | — |

## Global options

| Flag | Effect |
|---|---|
| `--version` | Print version and exit. |
| `-v`, `--verbose` | Enable DEBUG logging. |
| `--help` | Show help for the command. |

## Output formats

`bench` defaults to canonical JSON for machine-readable output and Rich tables for the terminal. Most commands accept `--format json|md|csv|parquet` for alternates. Where applicable, `--format bench-archive` emits a `.bench` file — a tar.zst bundle of the envelope, raw traces, logs, signature, and certificate.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Operational failure (verification failed, hardware refused, etc.) |
| 2 | Bad invocation (missing flag, invalid value) |
