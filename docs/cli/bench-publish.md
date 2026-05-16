# bench publish

Publish a signed envelope to Hugging Face Hub or a local filesystem mirror. The local mirror writes a self-describing `index.json` so a directory of envelopes is replayable as a static catalogue.

## Synopsis

```bash
bench publish <envelope.json> [--to hf|local] [--tag TAG] [--org ORG] [--dry-run]
```

## Example: publish to HF Hub

```bash
export HF_TOKEN=hf_xxx
bench publish ./results/c16-60be8efd6d21.json \
  --to hf \
  --org yobitel-bench-results \
  --tag llama-3.1-8b-conc16
```

Expected output:

```
OK published llm.inference.chatbot-short run to https://huggingface.co/datasets/yobitel-bench-results/...
  tag:           llama-3.1-8b-conc16
  repo_id:       yobitel-bench-results/llama-3.1-8b__chatbot-short__abcdef123456
  files:         3
  verified:      True
```

## Example: local mirror with index

```bash
bench publish ./results/c16-60be8efd6d21.json --to local --workspace ./bench-mirror
```

Writes:

```
./bench-mirror/
  llm-inference-chatbot-short/
    60be8efd6d21.json
  index.json
```

The `index.json` carries one entry per published envelope (suite, model, content hash, signed flag, optional tag, timestamp), sorted by timestamp desc.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--to` | `hf` | `hf` (Hugging Face Hub) or `local` (filesystem mirror). `studio` is Phase 2+ and exits with a hint. |
| `--workspace` | `""` | Local mirror root (`--to local`). Defaults to `./bench-mirror`. |
| `--tag` | `""` | Optional tag recorded with the publish. |
| `--org` | `yobitel-bench-results` | HF organisation namespace (`--to hf`). |
| `--dry-run` / `--no-dry-run` | off | Plan the publish without touching the network or filesystem. |
| `--raw-traces` | none | Path to an optional parquet of per-request traces uploaded alongside. |
| `--update-model-card` | off | Best-effort append a backlink entry to the source model card. |

## Auth (Hugging Face)

`--to hf` requires `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) in the environment, or a prior `huggingface-cli login`. Without one, the command exits `2` unless `--dry-run` is set.

## Failure modes

| Error | Exit |
|---|---|
| Repo collision (envelope's `run_id` already exists in the target org) | `1` |
| HF rate-limit | `1` |
| Token missing without `--dry-run` | `2` |
| Envelope file not found / schema-invalid | `2` |

## See also

- [bench verify](bench-verify.md) — verify before publishing
- [bench fetch](bench-fetch.md) — round-trip a published envelope back to local
- [Hugging Face Hub integration](../integrations/huggingface-hub.md)
