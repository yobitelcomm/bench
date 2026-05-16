# bench list

List every benchmark across every installed plugin. Where [`bench plugins`](bench-plugin.md) enumerates plugin packages, `bench list` goes one level deeper and shows every `BenchmarkSpec` they expose.

## Synopsis

```bash
bench list [--plugin <name>] [--json]
```

## Example: every benchmark in the `llm.inference` plugin

```bash
bench list --plugin llm.inference
```

Expected output:

```
                                  Available benchmarks
 Plugin         Benchmark ID                   Modality  Kind   Driver       Dataset       Description
 llm.inference  llm.inference.chatbot-short    llm       perf   closed_loop  chatbot-short Closed-loop concurrency sweep for ...
 llm.inference  llm.inference.long-context     llm       perf   open_loop    long-context  Long-context Poisson arrival ...
 llm.inference  llm.inference.sharegpt-v3      llm       perf   open_loop    sharegpt-v3   Open-loop Poisson arrival throughp...
```

## Flags

| Flag | Default | Description |
|---|---|---|
| `--plugin` | `""` | Filter to a single plugin (e.g. `llm.inference`). Exits `1` if the name is unknown. |
| `--json` | off | Emit `{"plugins": {<name>: {"version": ..., "benchmarks": [...]}}}` on stdout for piping into jq. |

## Behaviour

- With zero plugins installed, the command prints `No plugins installed.` and exits `0`.
- A plugin whose `list_benchmarks()` raises is logged as a yellow warning and recorded under `"error"` in `--json` output; the rest of the table still renders.

## See also

- [bench plugin](bench-plugin.md) — manage plugin packages
- [Plugins overview](../plugins/overview.md)
