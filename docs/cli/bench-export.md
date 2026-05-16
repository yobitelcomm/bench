# bench export

Render an envelope as a markdown block, CSV, or a Slack/Discord-friendly fenced snippet. Useful when an envelope needs to land in a PR comment, a spreadsheet, or a chat channel without bespoke formatting.

## Synopsis

```bash
bench export <envelope.json> [--format markdown|csv|slack] [--out PATH] [--metric KEY]...
```

Only local envelope paths are accepted. Use [`bench fetch`](bench-fetch.md) to download a remote envelope first.

## Example: PR-comment markdown

```bash
bench export ./results/c16-60be8efd6d21.json --format markdown
```

Expected output (truncated):

```markdown
## InferenceBench result — `llm.inference.chatbot-short`

- **Model**: `meta-llama/Llama-3.1-8B-Instruct` (revision `main`)
- **Engine**: `vllm v0.21.0`
- **Hardware**: `NVIDIA H100 80GB HBM3` x `1`
- **Signed**: `cosign-dev` — verify with `bench verify 60be8efd6d21.json`

| metric | value |
|---|---|
| joules_per_token | 0.6997 |
| ok_rate | 1 |
| ttft_p50_ms | 41.69 |
| throughput_tok_per_s | 1384 |
```

## Flags

| Flag | Default | Description |
|---|---|---|
| `--format` | `markdown` | One of `markdown`, `csv`, `slack`. |
| `--out` | stdout | Write to a file instead of stdout. |
| `--metric` | (all) | Repeatable; restricts the output to the listed metric keys. |

## Format notes

- **`markdown`** — header block (suite, model, engine, hardware, dataset hash, signing method) plus a sorted `metric / value` table.
- **`csv`** — `# key=value` comment header rows followed by a `metric,value` CSV body.
- **`slack`** — fenced code block (triple-backtick) with paired p50/p99 latencies, percentages for `ok_rate` / `compliance_rate`, and a `verify:` footer line.

## See also

- [bench summary](bench-summary.md) — directory-level overview
- [bench publish](bench-publish.md) — publish the canonical envelope rather than a rendered copy
