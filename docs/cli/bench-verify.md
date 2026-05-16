# bench verify

Verify a signed envelope. Recomputes the content hash from the envelope minus the signature block, then validates the signature (Sigstore keyless or dev ed25519 key). Any non-OK is a hard failure — there are no warnings.

## Synopsis

```bash
bench verify <envelope-uri> [--dev-public-key <path>]
```

## Example: verify a local dev-signed envelope

```bash
bench verify ./results/c16-60be8efd6d21.json
```

Expected output (success):

```
OK  ./results/c16-60be8efd6d21.json
  method:           cosign-dev
  content_hash:     60be8efd6d2178b40c5e...
  suite:            llm.inference.chatbot-short v1.0.0
  model:            meta-llama/Llama-3.1-8B-Instruct
  engine:           vllm v0.21.0
```

Expected output (failure):

```
FAIL  ./results/c16-60be8efd6d21.json
  method:  cosign-dev
  reason:  content hash mismatch (stored=60be8efd6d21..., recomputed=9c2f0a14...)
```

Exit code is `0` on success, `1` on failure (`2` if the envelope can't be loaded at all).

## Flags

| Flag | Default | Description |
|---|---|---|
| `--dev-public-key` | none | Path to an ed25519 public key for dev-signed envelopes. Use only for local testing. |

## Argument

| Argument | Required | Description |
|---|---|---|
| `envelope-uri` | yes | Local file path. Phase 1 supports local paths only; use [`bench fetch`](bench-fetch.md) for remote URIs. |

## What the verification does

1. Loads the envelope JSON and validates the schema.
2. Recomputes the content hash from the envelope minus the signature block.
3. Verifies the signature against the bundled certificate (Sigstore) or supplied public key (dev mode).
4. For Sigstore keyless: walks the certificate chain to the Sigstore root and validates the Rekor inclusion proof.

## See also

- [Recipes: reproducibility](../recipes/reproducibility.md)
- [bench replay](bench-replay.md) — re-run a verified envelope on your own hardware
- [The signed envelope](../concepts/envelope.md)
