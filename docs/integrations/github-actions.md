# GitHub Actions

Run InferenceBench in CI as a reusable GitHub Actions workflow. Use this to gate releases of your model or your inference stack on signed benchmark envelopes.

## Reusable workflow

```yaml
# .github/workflows/bench.yml
name: bench

on:
  pull_request:
  push:
    branches: [main]

jobs:
  bench:
    uses: yobitelcomm/bench-action/.github/workflows/run.yml@v1
    with:
      suite: llm.inference
      model: meta-llama/Llama-4-Maverick
      engine: vllm
      hardware: h100
      quant: fp8
      duration: 60
    secrets:
      HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

## Step action

If you prefer to compose your own job, the step action lets you call `bench` directly:

```yaml
- uses: yobitelcomm/bench-action@v1
  with:
    command: run
    args: >-
      llm.inference
      --model meta-llama/Llama-4-Maverick
      --engine vllm
      --hardware h100
      --quant fp8
      --duration 60

- uses: yobitelcomm/bench-action@v1
  with:
    command: publish
    args: ~/.cache/inferencebench/runs/latest --to hf
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

## Sigstore keyless OIDC

GitHub Actions provides an OIDC token that `bench` uses to sign envelopes via Sigstore keyless mode. The workflow needs `id-token: write` permission:

```yaml
permissions:
  id-token: write
  contents: read
```

The signature certificate identifies the workflow run and the repository, so a verifier can prove an envelope was produced by your CI.

## Self-hosted GPU runners

Phase 1 hardware is H100. Run `bench` on a self-hosted runner labelled appropriately:

```yaml
jobs:
  bench:
    runs-on: [self-hosted, gpu, h100]
```

Free GitHub-hosted runners do not have GPUs. The CPU-only diagnostic `bench doctor` still runs on a `ubuntu-latest` runner; benchmarks themselves need real hardware.

## Phase 1 status

The `yobitelcomm/bench-action` repository is published with the v0.1 release. Until then, install `inferencebench` directly in a `run:` step and call the CLI inline.

## See also

- [bench run](../cli/bench-run.md)
- [bench publish](../cli/bench-publish.md)
- [Hugging Face Hub integration](huggingface-hub.md)
