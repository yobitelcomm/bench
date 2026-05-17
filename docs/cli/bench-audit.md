# bench audit

Verify every envelope under a directory and report failures. For each `*.json` (recursive, skipping any `samples-*` traces) the command parses the JSON, recomputes the canonical `content_hash`, runs the signature check that matches the envelope's `signature.method`, and rejects placeholder hardware fingerprints. Pair with [`bench fetch`](bench-fetch.md) before trusting a third-party corpus.

## Synopsis

```bash
bench audit <path> [--dev-public-key <pem>] [--strict/--no-strict] [--report table|json]
```

`<path>` is either a directory or a single envelope JSON.

## Example: audit a downloaded corpus

```bash
bench audit ./validation-runs/2026-05-16-cross-model-corpus/corpus/all
```

Expected output:

```
                                Audit of ./validation-runs/.../corpus/all
 status  envelope                       model                              method     content_hash  reason
   ✓     c1-07b69e640395.json           Qwen/Qwen2.5-7B-Instruct           dev-key    07b69e640395
   ✓     c1-814953250c16.json           meta-llama/Llama-3.1-8B-Instruct   dev-key    814953250c16
   ✓     c4-4a7ac8857dbf.json           meta-llama/Llama-3.1-8B-Instruct   dev-key    4a7ac8857dbf
   ✓     c4-73219d1aa7f1.json           Qwen/Qwen2.5-7B-Instruct           dev-key    73219d1aa7f1
   ✓     c16-60be8efd6d21.json          meta-llama/Llama-3.1-8B-Instruct   dev-key    60be8efd6d21
   ✓     c16-8d7ef1b17fb7.json          Qwen/Qwen2.5-7B-Instruct           dev-key    8d7ef1b17fb7
   ✓     c64-4b9d631b5296.json          Qwen/Qwen2.5-7B-Instruct           dev-key    4b9d631b5296
   ✓     c64-fed81eb00398.json          meta-llama/Llama-3.1-8B-Instruct   dev-key    fed81eb00398
8 / 8 envelopes verified (0 failed)
```

## Flags

| Flag | Default | Description |
|---|---|---|
| `--dev-public-key` | none | Path to an ed25519 public key for dev-signed envelopes. Required to verify any dev-key envelope whose signing key isn't pre-baked. |
| `--strict` / `--no-strict` | `--strict` | Exit `1` if any envelope fails any check. With `--no-strict`, the table is rendered but the command always exits `0`. |
| `--report` | `table` | `table` (Rich) or `json` (machine-readable `inferencebench.audit.v1` payload). |

## Failure-case output

When the directory mixes a tampered envelope, an unsigned one, and a placeholder fingerprint, failures sort to the top:

```
   ✗   c16-tampered.json             meta-llama/Llama-3.1-8B-Instruct   dev-key   60be8efd6d21  signature does not match content_hash (tampered or wrong key)
   ✗   c1-unsigned.json              meta-llama/Llama-3.1-8B-Instruct   unsigned  814953250c16  no signature
   ✗   c4-fixture.json               (test fixture)                     dev-key   00000…        placeholder hardware_fingerprint
   ✓   c1-ok.json                    meta-llama/Llama-3.1-8B-Instruct   dev-key   07b69e640395
1 / 4 envelopes verified (3 failed)
```

Exit code is `0` only when every audited envelope passes; in strict mode any failure flips it to `1`.

## See also

- [bench verify](bench-verify.md) — single-envelope verifier
- [bench fetch](bench-fetch.md) — pulls a remote corpus into cache before auditing
- [Recipes: audit a published corpus](../recipes/audit.md)
