# bench ci

Two subcommands that wire `bench run` and `bench diff` into a GitHub Actions regression workflow:

- `bench ci init` writes a stock `.github/workflows/bench-regression.yml`.
- `bench ci validate` parses an existing workflow and confirms the required shape (right triggers, calls `bench run` + `bench diff --strict`).

## Synopsis

```bash
bench ci init     [--out PATH] [--suite ID] [--model ID] [--engine NAME]
                  [--baseline PATH] [--runner LABELS] [--tolerance FLOAT] [--force]
bench ci validate [PATH]
```

## Example: generate a workflow

```bash
bench ci init
```

Expected output:

```
                bench ci init
 workflow   .github/workflows/bench-regression.yml
 suite      llm.inference.sharegpt-v3
 model      meta-llama/Llama-3.1-8B-Instruct
 engine     vllm
 baseline   .bench/baseline.json
 runner     self-hosted,gpu
 tolerance  0.05
Wrote .github/workflows/bench-regression.yml. Commit it alongside a baseline envelope at .bench/baseline.json to start gating PRs on regressions.
```

The generated job:

1. Installs `inferencebench` + `inferencebench-llm` via `uv pip --system`.
2. Generates a dev signing key at `.bench/cosign.key` on first run.
3. Runs `bench run <suite> --model <id> --engine <kind> --base-url http://localhost:8000/v1 --duration 60 --output .bench/results`.
4. `bench diff <baseline> <new> --strict --tolerance <value>` — fails the build on any regression.
5. Uploads the envelope as an Actions artifact.

If `.bench/baseline.json` is missing on the first run the workflow promotes the current envelope to baseline and emits a `::warning::`.

## Example: validate an existing workflow

```bash
bench ci validate .github/workflows/bench-regression.yml
```

```
                       bench ci validate — .github/workflows/bench-regression.yml
 check                                                       status  fix
 triggers include pull_request or workflow_dispatch          PASS
 a step runs `bench run`                                     PASS
 a step runs `bench diff`                                    PASS
 `bench diff` uses `--strict`                                PASS
All checks passed.
```

Exit code is `0` on all-pass, `1` if any check fails, `2` if the file can't be parsed as YAML.

## Flags (`init`)

| Flag | Default | Description |
|---|---|---|
| `--out` | `.github/workflows/bench-regression.yml` | Destination workflow file. |
| `--suite` | `llm.inference.sharegpt-v3` | Benchmark id passed to `bench run`. |
| `--model` | `meta-llama/Llama-3.1-8B-Instruct` | Model id passed to `bench run --model`. |
| `--engine` | `vllm` | Inference engine kind. |
| `--baseline` | `.bench/baseline.json` | In-repo path to the baseline envelope. |
| `--runner` | `self-hosted,gpu` | Comma-separated `runs-on` labels. |
| `--tolerance` | `0.05` | Regression tolerance forwarded to `bench diff --tolerance`. |
| `--force` | off | Overwrite an existing workflow file. |

## See also

- [bench diff](bench-diff.md) — the regression detector under the hood
- [Recipes: CI integration](../recipes/ci-integration.md)
- [Recipes: regression check](../recipes/regression-check.md)
