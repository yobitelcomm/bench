# bench cache

Manage the local envelope fetch cache populated by [`bench fetch`](bench-fetch.md). Default root is `~/.cache/inferencebench/fetched/`; override with the `BENCH_CACHE_ROOT` environment variable.

## Synopsis

```bash
bench cache list
bench cache path
bench cache clear [--older-than DAYS] [--yes/--no-yes]
```

## Example: list cached envelopes

```bash
bench cache list
```

Expected output:

```
                                Cached envelopes (/home/bench/.cache/inferencebench/fetched)
 file                              size       age   content_hash  suite                         model
 60be8efd6d21.json                 18.0 KB    2h    60be8efd6d21  llm.inference.chatbot-short   meta-llama/Llama-3.1-8B-Instruct
 8d7ef1b17fb7.json                 17.8 KB    1d    8d7ef1b17fb7  llm.inference.chatbot-short   Qwen/Qwen2.5-7B-Instruct
 fed81eb00398.json                 18.2 KB    5d    fed81eb00398  llm.inference.chatbot-short   meta-llama/Llama-3.1-8B-Instruct
```

## Example: drop everything older than a week

```bash
bench cache clear --older-than 7 --yes
```

```
removed 4 cache file(s)
```

## Example: pipe the cache root into a shell script

```bash
ls -lh "$(bench cache path)"
```

```
/home/bench/.cache/inferencebench/fetched
```

## Subcommands

| Subcommand | Description |
|---|---|
| `list` | Print every cached envelope with size, age, content hash, suite and model. Envelopes that fail to parse render as `[red]invalid[/red]` rather than crashing the table. |
| `path` | Print the resolved cache root on a single line. |
| `clear` | Delete cache files. Default is everything; pass `--older-than N` to drop entries older than N days. Prompts unless `--yes` is supplied. |

## Flags (`clear`)

| Flag | Default | Description |
|---|---|---|
| `--older-than` | unset (delete all) | Only delete cache files older than N days. `--older-than 0` (with `--yes`) drops everything that exists. |
| `--yes` / `--no-yes` | off | Skip the confirmation prompt. |

## See also

- [bench fetch](bench-fetch.md) — the populator
