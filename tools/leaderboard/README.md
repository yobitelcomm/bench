# inferencebench-leaderboard

Static-site renderer that turns a directory of signed InferenceBench envelope
JSONs into a plain HTML+CSS+JSON leaderboard suitable for GitHub Pages
(`https://yobitelcomm.github.io/bench`). Vendor-neutral, no JavaScript
frameworks; the only client-side code is a ~40-line vanilla sorter for the
tables.

## Install (workspace)

This package is a `uv` workspace member of the root `bench/` monorepo:

```toml
# bench/pyproject.toml
[tool.uv.workspace]
members = [..., "tools/leaderboard"]
```

Then `uv sync` from the repo root.

## Build a site

```
python -m inferencebench_leaderboard build envelopes/ site/
# or
inferencebench-leaderboard build envelopes/ site/ --base-url /bench/
```

`envelopes/` must contain `*.json` files that parse against the canonical
`inferencebench.envelope.Envelope` Pydantic model. Files that don't parse
are logged and skipped; the rest of the site still renders.

## Output layout

```
site/
  index.html                       — category index
  static/site.css                  — Hacker News-style table CSS
  static/sort.js                   — vanilla sort for tables
  envelopes/<file>.json            — verbatim copies for `bench verify`
  <suite_id>/index.html            — per-category table
  <suite_id>/<run_id>.html         — per-entry detail (verify snippet)
  data/leaderboard.json            — machine-readable index
```

## Public API

```python
from inferencebench_leaderboard import (
    render_site,           # main entry point
    SiteRenderResult,      # return type
    load_envelopes,        # directory -> [LoadedEnvelope]
    compute_pareto,        # Pareto-frontier classifier
    LoadedEnvelope,
    PARETO_DIRECTIONS,
)
```

`compute_pareto` accepts per-axis `direction={"min","max"}` so the same
function works for throughput-vs-latency, latency-vs-cost, etc.

## Tests

```
pytest tools/leaderboard/tests/
```

Covers: smoke render, Pareto math on synthetic data, schema-validation
skipping.
