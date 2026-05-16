# bench verify

Verify a signed envelope. Checks the Sigstore signature, the certificate chain, the Rekor inclusion proof, and the content hash.

```bash
bench verify <envelope-uri> [--dev-public-key <path>]
```

## Example

```bash
bench verify ~/.cache/inferencebench/runs/latest/envelope.json
```

Expected output (success):

```
OK  ~/.cache/inferencebench/runs/latest/envelope.json
  method:           sigstore-cosign
  content_hash:     8b1a...e2c4
  suite:            llm.inference v1.0.0
  model:            meta-llama/Llama-4-Maverick
  engine:           vllm v0.7.2
  rekor_log_index:  12345
```

Expected output (failure):

```
FAIL  ~/.cache/inferencebench/runs/latest/envelope.json
  method:  sigstore-cosign
  reason:  content hash mismatch (stored=8b1a..., recomputed=9c2f...)
```

Exit code is `0` on success, `1` on failure.

## Arguments

| Argument | Required | Description |
|---|---|---|
| `envelope-uri` | yes | Local file path, `hf://datasets/...`, or `https://...` URL. |

## Options

| Option | Default | Description |
|---|---|---|
| `--dev-public-key` | none | Path to an ed25519 public key for dev-signed envelopes. Use only for local testing. |

## What the verification flow does

1. Fetches the envelope (local path; Phase 1 supports local paths only).
2. Parses and validates the schema version.
3. Recomputes the content hash from the envelope minus the signature block.
4. Verifies the Sigstore signature against the bundled certificate.
5. Walks the certificate chain to the Sigstore root.
6. Looks up the Rekor entry by log index and validates the inclusion proof.

Any non-OK is a hard failure. There are no warnings — the result either verifies or it does not.

## Phase 1 status

Local-path verification works against dev-key-signed envelopes today. Keyless OIDC verification and remote URIs (`hf://`, `https://`, `s3://`) land in v0.1.

## See also

- [The signed envelope](../concepts/envelope.md)
- [Envelope schema reference](../reference/envelope-schema.md)
