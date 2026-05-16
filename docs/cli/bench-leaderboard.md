# bench leaderboard

Render a static HTML+CSS leaderboard from a directory of signed envelopes. Phase 1 supports the local `--build` path; the hosted browse mode (fetching from `https://yobitelcomm.github.io/bench`) is deferred to Phase 2+.

## Synopsis

```bash
bench leaderboard --build --envelopes <DIR> --out <DIR> [--base-url /]
```

## Example: build a site from a sweep corpus

```bash
bench leaderboard \
  --build \
  --envelopes ./validation-runs/2026-05-16-cross-model-corpus/corpus/all \
  --out ./site \
  --base-url /
```

Expected output:

```
                Leaderboard render summary
 metric              value
 envelopes loaded    8
 envelopes skipped   0
 categories          llm.inference.chatbot-short
 output              /home/abishek/.../site
```

Open `./site/index.html` in a browser. The renderer emits a per-category page plus a top-level index.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--build` / `--no-build` | off | Required in Phase 1. Render a static site from `--envelopes` into `--out`. |
| `--envelopes` / `-i` | (required with `--build`) | Directory of signed envelope JSON files. Recursively scanned. |
| `--out` / `-o` | (required with `--build`) | Destination directory for the rendered site. Created if absent. |
| `--base-url` | `/` | URL prefix for generated links. Use `/bench/` (or similar) when deploying behind a sub-path. |
| `CATEGORY` (positional) | `""` | Reserved for the Phase 2+ hosted browse mode. Ignored in `--build` mode. |

## Requirements

The renderer lives in the `inferencebench-leaderboard` package. Install it alongside the CLI:

```bash
pip install -e ./cli -e ./envelope -e ./harness -e ./integrations/leaderboard
```

Without it, `bench leaderboard --build` exits `2` with an install hint.

## See also

- [Pareto frontiers](../concepts/pareto.md)
- [Recipes: concurrency sweep](../recipes/concurrency-sweep.md)
