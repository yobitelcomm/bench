# Hugging Face Hub

`bench publish --to hf` mirrors a signed envelope to Hugging Face Hub as a dataset repo. Every published run becomes a permanent, citable, verifiable record.

```bash
export HF_TOKEN=hf_xxx
bench publish ~/.cache/inferencebench/runs/latest --to hf
```

Expected output:

```
Verifying envelope... OK
Creating dataset repo... yobitel-bench-results/llama-4-maverick__llm-inference__01j7q5c6
Uploading envelope.json... done
Uploading traces.parquet... done
Rendering README.md... done
Published: https://huggingface.co/datasets/yobitel-bench-results/llama-4-maverick__llm-inference__01j7q5c6
```

## What gets published

| File | Content |
|---|---|
| `envelope.json` | The signed envelope. Source of truth. |
| `traces.parquet` | Per-request raw measurements, if present. |
| `README.md` | Auto-rendered metrics card. |
| YAML frontmatter | Dataset card metadata with key envelope fields for discoverability. |

The README includes the headline metrics, the run configuration, a Sigstore verification snippet, and a citation block.

## Authentication

Set `HF_TOKEN` in your environment. The token needs write access to the target organization:

- For your own publishing, your personal HF token writing to your user namespace works.
- For canonical results, the `yobitel-bench-publisher` org token writes to `yobitel-bench-results/*`. This token is held by the project maintainers.

## Repo naming

Repos are deterministically named from the envelope:

```
yobitel-bench-results/{model-slug}__{suite-slug}__{run-hash}
```

Where `run-hash` is the first 12 characters of `envelope.run_id`. UUIDv7 makes the hash naturally sortable by time.

## Dataset card metadata

The dataset card carries machine-readable envelope fields so that consumers (the static leaderboard, third-party tooling) can index without parsing the full envelope:

```yaml
---
license: cc-by-4.0
tags:
  - benchmark
  - inferencebench
  - llm
inferencebench:
  envelope_version: v1
  suite_id: llm.inference
  suite_version: 1.0.0
  model: meta-llama/Llama-4-Maverick
  engine: vllm
  hardware_class: h100
  fingerprint_sha256: 8b1a9c2f...
  signature_verified: true
  rekor_log_index: 12345
---
```

## Optional model card backlink

With `--update-model-card`, the publisher adds a non-intrusive metadata entry to the model card on Hugging Face Hub:

```yaml
inferencebench-verified:
  - url: https://huggingface.co/datasets/yobitel-bench-results/...
    suite: llm.inference
    date: 2026-11-15
    fingerprint_sha256: 8b1a9c2f...
```

The publisher never modifies visible model card content. If the model card author objects, there is a documented takedown path.

## Verifying a published envelope

Anyone, anywhere, can verify a published envelope without trusting our infrastructure:

```bash
pip install inferencebench
bench verify hf://datasets/yobitel-bench-results/llama-4-maverick__llm-inference__01j7q5c6/envelope.json
```

`bench verify` re-derives the content hash and checks the Sigstore signature plus the Rekor inclusion proof.

## Phase 1 status

The HF publishing flow is a stub in v0.0.0 and lands in the v0.1 release. The static leaderboard at `yobitelcomm.github.io/bench` will render from the published dataset corpus.

## See also

- [The signed envelope](../concepts/envelope.md)
- [bench publish](../cli/bench-publish.md)
- [bench verify](../cli/bench-verify.md)
