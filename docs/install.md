# Install

InferenceBench is a Python package. The `bench` binary and the `inferencebench` alias both land in your `PATH`.

## From PyPI (Phase 1 v0.1, coming soon)

```bash
pip install inferencebench
```

!!! note "Pre-release"
    The PyPI release lands with v0.1 (target: late 2026). Until then, install from source — see below.

## From source

```bash
git clone https://github.com/yobitelcomm/bench
cd bench
pip install -e ./cli -e ./envelope -e ./harness
```

Expected output:

```
Successfully installed inferencebench-0.0.0 inferencebench-envelope-0.0.0 inferencebench-harness-0.0.0
```

Verify the install:

```bash
bench --version
```

Expected output:

```
bench 0.0.0
```

## Plugins

The core CLI ships with no plugins. Install at least one to run benchmarks:

```bash
pip install inferencebench-llm
```

List installed plugins:

```bash
bench plugin list
```

Expected output (no plugins yet):

```
No plugins installed.
Install one: pip install inferencebench-llm
```

## Supported platforms

| Platform | Status |
|---|---|
| Linux x86_64 with NVIDIA H100 + driver 560.35.03+ | Supported (Phase 1) |
| Linux x86_64, other NVIDIA GPUs | Best-effort, not the Phase 1 target |
| Linux ARM (Grace, Ampere Altra) | Phase 2 |
| macOS (Apple Silicon, M5 Max) | Phase 2 |
| Windows | Not planned |

Phase 1 ships the `llm.inference` plugin against vLLM on Linux H100 only. Other hardware classes (MI300X, RTX 5090, M5 Max) are deferred.

## Python versions

Python 3.12 or newer. Older versions are not supported.

## Optional extras

| Extra | Adds |
|---|---|
| `[hf]` | Hugging Face Hub publishing dependencies |
| `[dev]` | Test + lint + type-check tools |

```bash
pip install "inferencebench[hf]"
```

## Uninstall

```bash
pip uninstall inferencebench inferencebench-envelope inferencebench-harness
```
