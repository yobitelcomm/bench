# bench plugin

Manage benchmark plugins. The core CLI ships with no plugins — install at least one (`inferencebench-llm`) before running benchmarks.

The shorthand `bench plugins` (note plural) is equivalent to `bench plugin list`.

## Synopsis

```bash
bench plugin {list|init|install|info} [ARGS]
```

## bench plugin list

```bash
bench plugin list
```

Expected output:

```
                                Installed plugins
 Name             Module                          Distribution
 llm.inference    inferencebench_llm.plugin:LlmInferencePlugin  inferencebench-llm
```

With no plugins installed:

```
No plugins installed.
Install one: pip install inferencebench-llm
```

## bench plugin init

Scaffold a new plugin package under `./plugins/<name>/` with a registered entry point, src package, a passing smoke test, and a README stub.

```bash
bench plugin init voice --kind both --modality voice
```

| Option | Default | Description |
|---|---|---|
| `--kind` | `both` | `perf`, `quality`, or `both`. |
| `--modality` | `""` | One of `llm`, `voice`, `video`, `3d`, etc. |

The plugin name must match `[a-z][a-z0-9-]*` (lowercase, no underscores).

## bench plugin install

```bash
bench plugin install llm
```

Phase 1 stub — until ticket 0028 lands, this command prints a hint and exits. Use `pip install inferencebench-<name>` directly.

## bench plugin info

```bash
bench plugin info llm.inference
```

Expected output:

```
llm.inference
  module:        inferencebench_llm.plugin:LlmInferencePlugin
  distribution:  inferencebench-llm 0.0.0
  summary:       InferenceBench plugin for LLM inference performance.
  homepage:      https://github.com/yobitelcomm/bench
```

## See also

- [bench list](bench-list.md) — see the benchmarks each plugin exposes
- [Plugins overview](../plugins/overview.md)
