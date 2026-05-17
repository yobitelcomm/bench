# Recipe: CI integration

Drop a regression gate into a GitHub Actions repo in one command. `bench ci init` writes a workflow that captures a benchmark on a self-hosted GPU runner, compares it against a baseline envelope checked into the repo, and fails the build on any regression worse than the configured tolerance.

## 1. Scaffold the workflow

From the repo root:

```bash
bench ci init \
  --suite llm.inference.chatbot-short \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --engine vllm \
  --baseline .bench/baseline.json \
  --runner self-hosted,gpu \
  --tolerance 0.02
```

Expected output:

```
                bench ci init
 workflow   .github/workflows/bench-regression.yml
 suite      llm.inference.chatbot-short
 model      meta-llama/Llama-3.1-8B-Instruct
 engine     vllm
 baseline   .bench/baseline.json
 runner     self-hosted,gpu
 tolerance  0.02
Wrote .github/workflows/bench-regression.yml. Commit it alongside a baseline envelope at .bench/baseline.json to start gating PRs on regressions.
```

The workflow body:

- Triggers on `pull_request` to `main` and `workflow_dispatch`.
- Installs `inferencebench` + `inferencebench-llm` via `uv pip --system`.
- Generates a dev signing key at `.bench/cosign.key` on first run.
- Runs the benchmark with `--duration 60 --signing-mode dev --output .bench/results`.
- Calls `bench diff .bench/baseline.json <new> --strict --tolerance 0.02`.
- Uploads the envelope as a build artifact.

## 2. Seed the baseline

Capture one envelope locally and commit it as the gating baseline:

```bash
bench run llm.inference.chatbot-short \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --engine vllm --quant fp16 \
  --concurrency 16 --duration 60 \
  --base-url http://localhost:8000/v1 \
  --output .bench/results

cp .bench/results/c16-*.json .bench/baseline.json
git add .bench/baseline.json .github/workflows/bench-regression.yml
```

On the very first CI run (before a baseline exists) the workflow emits a `::warning::No baseline at .bench/baseline.json — promoting current run.` and copies the fresh envelope into place. After that, every PR is gated.

## 3. Validate the workflow shape

`bench ci validate` parses the YAML and confirms the required pieces:

```bash
bench ci validate .github/workflows/bench-regression.yml
```

Expected output:

```
                       bench ci validate — .github/workflows/bench-regression.yml
 check                                                       status  fix
 triggers include pull_request or workflow_dispatch          PASS
 a step runs `bench run`                                     PASS
 a step runs `bench diff`                                    PASS
 `bench diff` uses `--strict`                                PASS
All checks passed.
```

This is what protects the gate against well-meaning edits that accidentally drop `--strict` (turning a regression gate into a benign warning).

## 4. What a failing PR looks like

The build fails with a diff like:

```
                         Diff: .bench/baseline.json vs .bench/results/c16-fc41a902c8de.json
 metric                 baseline     new        Δ abs        Δ rel%       verdict
 throughput_tok_per_s   1384         1186       -198         -14.31%      regression
 ttft_p99_ms            64.71        82.15      +17.44        +26.97%     regression
 joules_per_token       0.700        0.812      +0.112        +16.00%     regression
 ok_rate                1.000        1.000      0.000          0.00%      no_change
Verdict: regression
```

Click into the Actions tab, download the envelope artifact, and run `bench diff` locally for the same view; the artifact is also useful for backfilling a regression-history dashboard later.

## Where to go next

- [bench ci reference](../cli/bench-ci.md) — every flag and the workflow template
- [Recipes: regression check](regression-check.md) — the manual loop the CI workflow automates
- [GitHub Actions integration](../integrations/github-actions.md)
