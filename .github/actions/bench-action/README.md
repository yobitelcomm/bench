# `bench-action` — InferenceBench reusable GitHub Action

Run an InferenceBench suite inside any GitHub Actions workflow, verify the
signed envelope, and publish the result as an artifact (or to Hugging Face).

This is a **composite action** — it shells out to the `inferencebench` CLI on
the runner. No Docker required.

- **Apache 2.0**, same license as the rest of the suite.
- **Vendor-neutral**: works against any model/engine/hardware combo supported
  by an installed plugin.
- **Reproducibility-first**: every successful run drops a signed envelope.

---

## Quick start

```yaml
# .github/workflows/bench.yml in your repo
name: Bench
on:
  pull_request:
  workflow_dispatch:

jobs:
  llm-inference:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: yobitelcomm/bench-action@v1
        with:
          model: meta-llama/Llama-3.1-8B-Instruct
          engine: vllm
          hardware: cpu
          suite-id: llm.inference.sharegpt-v1
          duration: 60s
          concurrency: 4
          publish-to: artifact
```

The signed envelope will appear on the workflow run as the
`inferencebench-envelope` artifact.

---

## Inputs

| Name              | Required | Default                | Description                                                              |
|-------------------|----------|------------------------|--------------------------------------------------------------------------|
| `model`           | yes      | —                      | Model identifier passed to `bench run --model`.                          |
| `engine`          | yes      | `vllm`                 | Inference engine. Phase 1: `vllm` only.                                  |
| `hardware`        | yes      | —                      | Hardware profile tag (e.g. `h100-80gb-sxm`, `a10g`, `cpu`).              |
| `suite-id`        | yes      | —                      | Suite identifier (e.g. `llm.inference.sharegpt-v1`).                     |
| `dataset`         | no       | `""`                   | Dataset slug or local path.                                              |
| `concurrency`     | no       | `1`                    | Concurrent request count.                                                |
| `duration`        | no       | `60s`                  | Wall-clock duration. Accepts `60s`, `5m`, `1h`.                          |
| `slo-template`    | no       | `""`                   | SLO template (e.g. `interactive-chat`).                                  |
| `publish-to`      | no       | `artifact`             | One of `artifact`, `hf`, `none`.                                         |
| `cli-version`     | no       | `""` (latest stable)   | Pin `inferencebench` to a specific version.                              |
| `plugin-package`  | no       | `inferencebench-llm`   | Plugin pip name to install alongside the CLI.                            |
| `working-directory`| no      | `.`                    | Where to run the bench commands.                                         |

## Outputs

| Name                 | Description                                                                |
|----------------------|----------------------------------------------------------------------------|
| `envelope-path`      | Filesystem path to the produced signed envelope JSON file.                 |
| `signature-verified` | `"true"` if `bench verify` passed; the step fails otherwise.               |

---

## Consuming the envelope downstream

```yaml
- uses: yobitelcomm/bench-action@v1
  id: bench
  with:
    model: meta-llama/Llama-3.1-8B-Instruct
    engine: vllm
    hardware: a10g
    suite-id: llm.inference.sharegpt-v1

- name: Use the envelope
  run: |
    echo "Envelope produced at ${{ steps.bench.outputs.envelope-path }}"
    echo "Signature verified: ${{ steps.bench.outputs.signature-verified }}"
    jq '.metrics' "${{ steps.bench.outputs.envelope-path }}"
```

---

## Pinning a major version

The recommended pin is the major version tag:

```yaml
uses: yobitelcomm/bench-action@v1
```

Pin to a specific SHA for stricter supply-chain hygiene:

```yaml
uses: yobitelcomm/bench-action@<full-sha>
```

The action's CLI dependency can be pinned independently with `cli-version`.

---

## Hardware caveats (Phase 1)

- The action runs on whatever runner you target. For GPU runs, point your
  workflow at a self-hosted GPU runner (TestBM-style 8×H100 or similar).
- CPU-only runs are useful for smoke tests but `bench doctor --strict` may
  warn about missing NVML or unavailable engines — that's expected.

---

## Troubleshooting

- **`bench doctor --strict` fails** — check that the runner has CUDA drivers
  matching the engine version, and that `nvidia-smi` works.
- **Envelope verification fails** — usually clock-skew on self-hosted runners
  or a corrupted Sigstore TUF cache. Clear `~/.cache/sigstore-python` and rerun.
- **`bench publish --target hf` fails** — make sure `HF_TOKEN` is set in the
  workflow env and the token has write access to the target dataset/org.

---

## Source

This action lives at `.github/actions/bench-action/` in the
[yobitelcomm/bench](https://github.com/yobitelcomm/bench) monorepo. A
release-time job mirrors it to
[yobitelcomm/bench-action](https://github.com/yobitelcomm/bench-action) so
third-party callers can use the short `uses: yobitelcomm/bench-action@v1`
syntax.

> **Open item**: the mirror repo / release job is tracked separately; until
> it lands, third parties can still use the action with the long form
> `uses: yobitelcomm/bench/.github/actions/bench-action@<ref>`.
