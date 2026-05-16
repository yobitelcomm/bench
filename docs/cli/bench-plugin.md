# bench plugin

Manage benchmark plugins. The core CLI ships with no plugins; install at least one before running benchmarks.

```bash
bench plugin {list|init|install|info} [ARGS]
```

The shorthand `bench plugins` is equivalent to `bench plugin list`.

## bench plugin list

List installed plugins. Plugins register via Python entry points under `inferencebench.plugins`.

```bash
bench plugin list
```

Expected output (no plugins):

```
No plugins installed.
Install one: pip install inferencebench-llm
```

Expected output (one plugin):

```
Installed plugins
Name             Module                          Distribution
llm.inference    inferencebench_llm:plugin       inferencebench-llm
```

## bench plugin install

Install a plugin from PyPI.

```bash
bench plugin install llm
```

This is a convenience wrapper around `pip install inferencebench-<name>`.

## bench plugin init

Scaffold a new plugin package. Useful when you want to contribute a new modality.

```bash
bench plugin init voice --kind both --modality voice
```

Options:

| Option | Default | Description |
|---|---|---|
| `--kind` | `both` | `perf`, `quality`, or `both`. |
| `--modality` | `""` | One of `llm`, `voice`, `video`, `3d`, etc. |

## bench plugin info

Show details for a specific plugin: version, entry points, suites it registers, supported engines.

```bash
bench plugin info llm.inference
```

## Phase 1 status

`bench plugin list` works today. `init`, `install`, and `info` are stubs in v0.0.0 and land in the v0.1 release.

## See also

- [Plugins overview](../plugins/overview.md)
- [llm.inference plugin reference](../plugins/llm-inference.md)
