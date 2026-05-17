# 10-minute tour

This page walks a brand-new user from `git clone` to a signed envelope you can hand to anyone and have them verify. It's the same path `bench tour` runs, with each step spelled out so you can stop and inspect.

The numbers shown in the output blocks come from a real corpus captured on H100-80GB-HBM3 in May 2026 (`validation-runs/2026-05-16-cross-model-corpus/`). Your numbers will differ; the shape will not.

## 0–2 min: Install

```bash
git clone https://github.com/yobitelcomm/bench
cd bench
uv sync --all-packages --dev --prerelease=allow
uv run bench --version
```

Expected:

```
bench 0.0.2
```

If `uv` is not on your path, install it with `pipx install uv` first. We use `uv` workspace mode so a single `uv sync` resolves the CLI, the harness, the envelope library, and every plugin from one lock file.

!!! tip "What you learned"
    - The repo is a uv workspace; one sync installs every package in the monorepo.

## 2–4 min: Hardware check

```bash
uv run bench doctor
```

Expected (excerpt, on a healthy H100 node):

```
Hardware diagnostic
Check                Status   Detail
NVML available       PASS     8 GPUs visible
Driver version       PASS     560.35.03
ECC enabled          PASS     enabled on all GPUs
Persistence mode     PASS     enabled
Thermal headroom     PASS     all GPUs < 75 degC
Clock state          PASS     no throttling flags
OK — all checks passed.
```

Field by field:

- **NVML available** — `bench` reads telemetry via `pynvml`. No NVML, no envelope.
- **Driver version** — captured into the envelope's hardware fingerprint.
- **ECC enabled** — single-bit memory errors silently corrupt logits. Off = result rejected.
- **Persistence mode** — keeps the driver loaded between runs so cold-start latency does not pollute warm-up timings.
- **Thermal headroom** — anything above ~83 °C will trigger throttling on H100s and skew TTFT.
- **Clock state** — `bench` aborts if any throttling flag is set during the run.

`bench doctor` exits non-zero on a failure. `--strict` also fails on warnings.

!!! tip "What you learned"
    - Every check `doctor` runs corresponds to a field captured in the signed envelope's hardware fingerprint.

## 4–7 min: First signed envelope

A real `bench run` needs a model server. The realistic flow on an 8×H100 box is:

```bash
# In one terminal:
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0

# In another, once it's serving:
cosign generate-key-pair                # produces ./cosign.key + cosign.pub
uv run bench run llm.inference.chatbot-short \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --engine vllm --quant fp16 \
  --sweep 1,4 \
  --base-url http://localhost:8000/v1 \
  --signing-mode dev --dev-key ./cosign.key \
  --output ./corpus/tiny
```

The harness discards three warm-up runs, waits for the convergence gate (CoV < 5% over the last 30 requests), then drives Poisson-arrival load at each concurrency in the sweep. Output ends with the envelope path:

```
Envelope: ./corpus/tiny/c1-<id>.json
Signed:   sigstore-cosign (dev key)
```

Inspect the JSON directly:

```bash
cat ./corpus/tiny/c1-*.json | jq '{run_id, suite, model, metrics, signature: .signature.method}'
```

You'll see the `content_hash`, the hardware fingerprint, the dataset hash, the seed, the metrics, and a Sigstore signature block.

Verify it:

```bash
uv run bench verify ./corpus/tiny/c1-*.json
```

Expected:

```
OK  ./corpus/tiny/c1-<id>.json
  method:           sigstore-cosign-dev
  content_hash:     8b1a…e2c4
  suite:            llm.inference v1.0.0
```

Verification recomputes the content hash, checks the cosign signature against the bundled public key, and confirms every metric is internally consistent. Any mismatch is a hard failure.

!!! tip "What you learned"
    - An envelope is the unit of trust. If `bench verify` passes, you can hand the JSON to anyone and they can reproduce the claim.

## 7–9 min: Compare + leaderboard

Run a second model the same way (or copy two envelopes from `validation-runs/2026-05-16-cross-model-corpus/corpus/all/`). Diff them:

```bash
uv run bench diff \
  ./corpus/llama-3.1-8b/c16-*.json \
  ./corpus/qwen-2.5-7b/c16-*.json
```

You'll get a side-by-side metric table with deltas. The corpus shipped in the repo shows Llama-3.1-8B at 1384.2 tok/s and Qwen2.5-7B at 1362.3 tok/s at concurrency 16 — same hardware, same suite, ~1.6 % apart on throughput and ~1.4 % apart on J/tok.

Render a static leaderboard from any directory of envelopes:

```bash
uv run bench leaderboard --build \
  --envelopes validation-runs/2026-05-16-cross-model-corpus/corpus/all \
  --out ./site
open ./site/index.html
```

The output is a self-contained HTML site with Pareto plots — no JavaScript framework, no server.

!!! tip "What you learned"
    - `bench diff` and `bench leaderboard --build` are pure functions of a directory of envelopes; no network, no DB.

## 9–10 min: Share

Bundle a single envelope plus the public key plus the cosign certificate for offline-recipient verification:

```bash
uv run bench bundle create ./corpus/tiny/c1-*.json --out ./tiny.bundle.zip
```

A recipient can verify your run without the original repo by running `bench bundle extract` followed by `bench verify`.

Mirror a whole corpus to a local directory tree (a stand-in for a future hosted Studio mirror):

```bash
uv run bench publish ./corpus/tiny/c1-*.json --to local --workspace ./mirror
```

The mirror layout matches what `bench fetch` consumes, so collaborators can pull from a shared NFS path or an S3 bucket synced to local disk.

!!! tip "What you learned"
    - Envelopes are portable. Bundle for one-off sharing; publish to a workspace mirror for a team.

## Where to go next

- [Quickstart](quickstart.md) — the canonical 5-minute install + run.
- [Signed envelope](concepts/envelope.md) — what is in the JSON, and why every field is there.
- [Cross-model comparison recipe](recipes/cross-model.md) — the full Llama vs Qwen walkthrough with real numbers.
- [Plugin authoring](community/contributing.md#contributing-a-new-plugin) — scaffold your own benchmark with `bench plugin init`.
