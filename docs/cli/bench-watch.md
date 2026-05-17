# bench watch

Poll an envelopes directory and rebuild the static leaderboard whenever the set of `*.json` files (or any file's mtime) changes. Pairs naturally with a long-running `bench run --sweep` so the hosted leaderboard stays fresh without manual `bench leaderboard --build` invocations.

Requires the `inferencebench-leaderboard` package (`pip install inferencebench-leaderboard`); the command exits `2` with an install hint if it isn't on `PYTHONPATH`.

## Synopsis

```bash
bench watch <envelopes-dir> --out DIR [--interval-s SECONDS] [--base-url URL]
                             [--max-iterations N] [--quiet]
```

## Example: keep the leaderboard fresh during a sweep

```bash
bench watch ./results --out ./site --interval-s 5 --base-url /bench/
```

Expected output:

```
rebuilt site: 4 envelopes, 0 skipped, 1 categories at 14:02:11
no changes at 14:02:16
no changes at 14:02:21
rebuilt site: 8 envelopes, 0 skipped, 1 categories at 14:02:26
rebuilt site: 12 envelopes, 0 skipped, 1 categories at 14:02:31
```

Run it in a separate shell during a long sweep; new envelopes are picked up within `--interval-s` seconds and a fresh `index.html` is published under `--out`. Ctrl-C stops cleanly.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--out`, `-o` | required | Destination directory for the rendered static site. |
| `--interval-s` | `5.0` | Polling interval in seconds. |
| `--base-url` | `/` | Base URL prefix for generated links (e.g. `/bench/` for GitHub Pages). |
| `--max-iterations` | `0` (unlimited) | Stop after N polls. Primarily for tests / CI smoke runs. |
| `--quiet`, `-q` | off | Suppress `no changes` lines between rebuilds. |

## Behaviour

- The initial site renders on the first iteration even if the directory is empty.
- A change is any added, removed, or modified `*.json` file under the watched directory (recursive).
- `render_site` failures emit a yellow warning and the loop keeps polling; the command exits `1` only if at least one rebuild failed by the time you stop it.
- Polling is pure-Python and portable; no `inotify`/`fsevents` dependency.

## See also

- [bench leaderboard](bench-leaderboard.md) — one-shot site build
- [bench run](bench-run.md) — the producer this command rebuilds for
