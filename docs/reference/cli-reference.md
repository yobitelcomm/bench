# CLI reference

This is a hand-curated reference. Autogeneration from Typer help is on the v0.2 roadmap; until then, this page mirrors `bench --help` and each subcommand's `--help`. For end-to-end stories see the [recipes](../recipes/concurrency-sweep.md); for per-command examples and expected output see the [CLI pages](../cli/overview.md).

## bench

```
Usage: bench [OPTIONS] COMMAND [ARGS]...

  InferenceBench Suite — vendor-neutral, signed-envelope AI benchmarks.

Options:
  --version          Show version and exit.
  -v, --verbose      Verbose logging (DEBUG level).
  --help             Show this message and exit.

Commands:
  compare      Compare benchmark runs (Pareto frontier).
  cost         Compare model cost across providers.
  diff         Per-metric delta between two envelopes.
  doctor       Diagnose hardware health before benchmarking.
  export       Export an envelope as markdown / CSV / Slack snippet.
  fetch        Fetch a signed envelope from a remote URI.
  leaderboard  Browse public leaderboards.
  list         List every benchmark across every installed plugin.
  plugin       Manage benchmark plugins.
  plugins      List installed plugins (shorthand for "bench plugin list").
  publish      Publish a signed envelope (HF Hub, local).
  replay       Replay a benchmark from an existing envelope.
  run          Run a benchmark and produce a signed envelope.
  summary      Summarise envelopes in a directory or file.
  verify       Verify a signed envelope's signature + content.
```

## bench compare

```
Usage: bench compare [OPTIONS] RUN_IDS...

Arguments:
  RUN_IDS...  Two or more local envelope paths to compare. [required]

Options:
  --report TEXT  table | pareto | json. [default: table]
  --verify       Verify each envelope's signature before comparing.
```

See [bench compare](../cli/bench-compare.md).

## bench cost

```
Usage: bench cost [OPTIONS] MODEL

Arguments:
  MODEL  Model id (e.g. llama-3.1-8b-instruct). [required]

Options:
  --suite TEXT                 Suite hint. [default: intelligence-index]
  --providers TEXT             Comma-separated provider filter.
  --input-token-share FLOAT    Input share of the blended rate (0.0..1.0). [default: 0.75]
```

See [bench cost](../cli/bench-cost.md).

## bench diff

```
Usage: bench diff [OPTIONS] BASELINE_PATH CANDIDATE_PATH

Arguments:
  BASELINE_PATH   Path to the baseline envelope. [required]
  CANDIDATE_PATH  Path to the candidate envelope. [required]

Options:
  --tolerance FLOAT   Relative-delta band for "no_change". [default: 0.02]
  --report TEXT       table | json. [default: table]
  --verify            Verify both envelopes before diffing.
  --strict            Exit 1 if any metric is a regression.
```

See [bench diff](../cli/bench-diff.md).

## bench doctor

```
Usage: bench doctor [OPTIONS]

Options:
  --strict  Treat WARN as failure. Default fails only on FAIL.
```

See [bench doctor](../cli/bench-doctor.md).

## bench export

```
Usage: bench export [OPTIONS] ENVELOPE_PATH

Arguments:
  ENVELOPE_PATH  Path to the envelope JSON. [required]

Options:
  --format TEXT  markdown | csv | slack. [default: markdown]
  --out PATH     Write to this file instead of stdout.
  --metric TEXT  Restrict to this metric key. Repeatable.
```

See [bench export](../cli/bench-export.md).

## bench fetch

```
Usage: bench fetch [OPTIONS] URI

Arguments:
  URI  hf://datasets/OWNER/REPO[/FILE] | https://... | file://... | local path. [required]

Options:
  --out PATH               Local destination path. [default: cache]
  --force / --no-force     Re-download even if a cached copy exists.
```

See [bench fetch](../cli/bench-fetch.md).

## bench leaderboard

```
Usage: bench leaderboard [OPTIONS] [CATEGORY]

Arguments:
  [CATEGORY]  Reserved for Phase 2+ hosted browse mode.

Options:
  --build / --no-build  Render a static site from --envelopes into --out.
  --envelopes / -i DIR  Directory of signed envelopes. Required with --build.
  --out / -o DIR        Destination directory for the rendered site.
  --base-url TEXT       URL prefix for generated links. [default: /]
```

See [bench leaderboard](../cli/bench-leaderboard.md).

## bench list

```
Usage: bench list [OPTIONS]

Options:
  --plugin TEXT  Filter to a single plugin (e.g. llm.inference).
  --json         Emit JSON instead of a Rich table.
```

See [bench list](../cli/bench-list.md).

## bench plugin

```
Usage: bench plugin [OPTIONS] COMMAND [ARGS]...

Commands:
  list     List installed plugins.
  init     Scaffold a new plugin package.
  install  Install a plugin from PyPI (Phase 1 stub).
  info     Show details for a specific plugin.
```

See [bench plugin](../cli/bench-plugin.md).

## bench publish

```
Usage: bench publish [OPTIONS] RUN_ID

Arguments:
  RUN_ID  Path to a signed envelope JSON. [required]

Options:
  --to TEXT                                hf | local. [default: hf]
  --workspace TEXT                         Local mirror root.
  --tag TEXT                               Optional tag for this publish.
  --org TEXT                               HF organisation. [default: yobitel-bench-results]
  --dry-run / --no-dry-run                 Plan without touching the network.
  --raw-traces PATH                        Optional parquet of per-request traces.
  --update-model-card                      Append a backlink entry to the source model card.
```

See [bench publish](../cli/bench-publish.md).

## bench replay

```
Usage: bench replay [OPTIONS] ENVELOPE_PATH

Arguments:
  ENVELOPE_PATH  Path to the source envelope JSON. [required]

Options:
  --base-url TEXT             Engine base URL for the replay. [required]
  --output TEXT               Output directory for the replay envelope.
  --signing-mode TEXT         dev | keyless. [default: dev]
  --dev-key TEXT              Path to local cosign signing key.
  --verify / --no-verify      Verify source envelope before replaying. [default: --verify]
```

See [bench replay](../cli/bench-replay.md).

## bench run

```
Usage: bench run [OPTIONS] SUITE_ID

Arguments:
  SUITE_ID  Suite identifier or benchmark id. [required]

Options:
  --model TEXT             Provider-prefixed model id.
  --engine TEXT            Inference engine. [default: vllm]
  --hardware TEXT          Hardware class. [default: h100]
  --quant TEXT             Quantization format. [default: fp16]
  --concurrency TEXT       Comma-separated concurrency levels. [default: 1]
  --rps FLOAT              Open-loop arrival rate (req/s). [default: 0.0]
  --sweep TEXT             Closed-loop concurrency points (one envelope each).
  --rps-sweep TEXT         Open-loop RPS points (one envelope each).
  --all-benchmarks         Run every spec the plugin exposes.
  --list                   Print bundled benchmark ids and exit.
  --dataset TEXT           Dataset id override.
  --duration INTEGER       Measurement duration in seconds. [default: 300]
  --slo-template TEXT      SLO template id. [default: llm.standard]
  --seed INTEGER           Random seed. [default: 42]
  --base-url TEXT          Engine base URL.
  --output TEXT            Output directory.
  --signing-mode TEXT      dev | keyless. [default: dev]
  --dev-key TEXT           Path to local cosign signing key.
  --strict                 Treat plugin.validate() warnings as fatal.
```

See [bench run](../cli/bench-run.md).

## bench summary

```
Usage: bench summary [OPTIONS] PATH

Arguments:
  PATH  Directory to scan recursively or a single envelope file. [required]

Options:
  --json  Emit JSON instead of Rich tables.
```

See [bench summary](../cli/bench-summary.md).

## bench verify

```
Usage: bench verify [OPTIONS] ENVELOPE_URI

Arguments:
  ENVELOPE_URI  Local file path. [required]

Options:
  --dev-public-key PATH  Path to ed25519 public key for dev-signed envelopes.
```

See [bench verify](../cli/bench-verify.md).

## See also

- [CLI overview](../cli/overview.md)
- [Envelope schema](envelope-schema.md)
