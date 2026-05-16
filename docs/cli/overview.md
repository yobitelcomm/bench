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
  cost         Compare model cost across providers.
  diff         Per-metric delta between two envelopes.
  doctor       Diagnose hardware health before benchmarking.
  export       Export an envelope as markdown / CSV / Slack snippet.
  fetch        Fetch a signed envelope from a remote URI.
  history      Time-series view of one metric across an envelope corpus.
  leaderboard  Browse public leaderboards.
  list         List every benchmark across every installed plugin.
  plugin       Manage benchmark plugins.
  plugins      List installed plugins (shorthand for "bench plugin list").
  publish      Publish a signed envelope (HF Hub, local).
  replay       Replay a benchmark from an existing envelope.
  schema       Emit JSON Schema for envelopes / benchmark specs / mirror index.
  summary      Summarise envelopes in a directory or file.
  verify       Verify a signed envelope's signature + content.
```

## Commands at a glance

| Command | Purpose | Page |
|---|---|---|
| `bench run` | Execute a benchmark, produce a signed envelope (supports `--sweep`, `--rps-sweep`, `--all-benchmarks`) | [bench run](bench-run.md) |
| `bench compare` | Pareto-frontier comparison across N envelopes | [bench compare](bench-compare.md) |
| `bench cost` | Provider-cost comparison from the pricing registry | [bench cost](bench-cost.md) |
| `bench diff` | Per-metric delta between two envelopes (regression detection) | [bench diff](bench-diff.md) |
| `bench doctor` | Pre-run hardware diagnostic | [bench doctor](bench-doctor.md) |
| `bench export` | Render an envelope as markdown / CSV / Slack | [bench export](bench-export.md) |
| `bench fetch` | Download an envelope to local cache (`hf://`, `https://`, `file://`) | [bench fetch](bench-fetch.md) |
| `bench history` | Time-series of one metric across an envelope corpus (with sparkline) | — |
| `bench leaderboard` | Render a static HTML site from a directory of envelopes | [bench leaderboard](bench-leaderboard.md) |
| `bench list` | Catalogue every benchmark across every installed plugin | [bench list](bench-list.md) |
| `bench plugin` | `list`, `init`, `install`, `info` subcommands | [bench plugin](bench-plugin.md) |
| `bench plugins` | Shorthand for `bench plugin list` | [bench plugin](bench-plugin.md) |
| `bench publish` | Publish to HF Hub or local mirror | [bench publish](bench-publish.md) |
| `bench replay` | Re-run a benchmark from a signed envelope | [bench replay](bench-replay.md) |
| `bench schema` | Emit JSON Schema for envelope / benchmark-spec / mirror-index | — |
| `bench summary` | Tabulate envelopes in a directory (`--json` for jq) | [bench summary](bench-summary.md) |
| `bench verify` | Verify a signed envelope's signature + content hash | [bench verify](bench-verify.md) |

## Global options

| Flag | Effect |
|---|---|
| `--version` | Print version and exit. |
| `-v`, `--verbose` | Enable DEBUG logging. |
| `--help` | Show help for the command. |

## Output formats

`bench` defaults to Rich tables for terminal output. Commands that emit machine-readable data accept a flag for it: `bench compare --report json`, `bench diff --report json`, `bench summary --json`, `bench list --json`. Envelopes themselves are always canonical JSON on disk.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Operational failure (verification failed, hardware refused, regression with `--strict`, etc.) |
| 2 | Bad invocation (missing flag, invalid value, envelope not found) |
