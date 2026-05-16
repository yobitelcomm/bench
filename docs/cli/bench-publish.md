# bench publish

Mirror a signed envelope to Hugging Face Hub or a local archive.

```bash
bench publish <run-id-or-path> [--to hf|local] [--tag <tag>]
```

## Example

```bash
export HF_TOKEN=hf_xxx
bench publish ~/.cache/inferencebench/runs/latest --to hf --tag fp8-baseline
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

## Arguments

| Argument | Required | Description |
|---|---|---|
| `run_id` | yes | Run ID or envelope path. |

## Options

| Option | Default | Description |
|---|---|---|
| `--to` | `hf` | Target: `hf` (Hugging Face Hub), `local` (a `.bench` archive), or `studio` (Phase 2). |
| `--workspace` | `""` | Workspace id (Studio only, Phase 2). |
| `--tag` | `""` | Optional tag attached to the publish. |

## Auth

`bench publish --to hf` requires the `HF_TOKEN` environment variable. The token needs write access to the target organization (`yobitel-bench-results` for production; a personal user namespace also works).

## What gets uploaded

- `envelope.json` — the canonical signed envelope
- `traces.parquet` — per-request raw traces, if present
- `README.md` — auto-rendered metrics card with headline numbers and a Sigstore verification snippet
- Dataset card metadata — YAML frontmatter with envelope fields for discoverability

## Phase 1 status

`bench publish` is a stub in v0.0.0. The Hugging Face publishing integration wires in during the v0.1 release.

## See also

- [Hugging Face Hub integration](../integrations/huggingface-hub.md)
- [bench verify](bench-verify.md)
