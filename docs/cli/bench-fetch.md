# bench fetch

Download a signed envelope from a remote URI to the local cache. Supports `hf://datasets/...`, `https://...`, `http://...`, `file://...`, and plain local paths. The downloaded payload is validated against the `Envelope` schema before being declared OK.

## Synopsis

```bash
bench fetch <uri> [--out PATH] [--force]
```

## Example: pull an envelope from Hugging Face

```bash
bench fetch hf://datasets/Yobitel/llama-3.1-8b__chatbot-short__abcdef123456
```

Expected output:

```
OK  fetched hf://datasets/Yobitel/llama-3.1-8b__chatbot-short__abcdef123456
  local_path:       /home/bench/.cache/inferencebench/fetched/3f9c1a2b8e7d.json
  content_hash:     60be8efd6d21...
  suite_id:         llm.inference.chatbot-short
  model_id:         meta-llama/Llama-3.1-8B-Instruct
  signature:        cosign-dev
```

## URI schemes

| Scheme | Resolved via |
|---|---|
| `hf://datasets/<owner>/<repo>[/<file>]` | `huggingface_hub.hf_hub_download`. Default filename is `envelope.json`. |
| `https://...` / `http://...` | `urllib.request`. |
| `file://<path>` | Local copy. |
| plain path | Local copy. |

## Flags

| Flag | Default | Description |
|---|---|---|
| `--out` | `~/.cache/inferencebench/fetched/<sha256(uri)[:12]>.json` | Local destination path. |
| `--force` / `--no-force` | off | Re-download even if the cached file already exists. Without `--force`, an existing cache file prints `cache hit` and the validator runs against the cached copy. |

## Failure modes

- Schema-invalid payloads are left on disk so you can `cat` them for debugging; the command exits `2`.
- Unsupported URI schemes (`s3://`, custom) exit `2` with the offending scheme echoed.

## See also

- [bench verify](bench-verify.md) — verify the fetched envelope's signature
- [Hugging Face Hub integration](../integrations/huggingface-hub.md)
