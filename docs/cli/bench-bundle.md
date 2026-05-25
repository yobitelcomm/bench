# bench bundle

Pack a signed envelope (plus optional sample trace and public key) into a single shareable `.bundle.zip`, or unpack one. Every bundle ships with a stdlib + `cryptography` verifier so the recipient does not need to `pip install inferencebench` to check the signature.

## Synopsis

```bash
bench bundle create <envelope.json> [--out PATH] [--include-samples/--no-include-samples]
                                    [--include-public-key <pem>]
bench bundle extract <bundle.zip>   [--out DIR]
```

## Example: create a bundle from a sweep envelope

```bash
bench bundle create ./results/c16-60be8efd6d21.json \
  --include-public-key ./cosign.pub
```

Expected output:

```
                Bundle created: 60be8efd6d21.bundle.zip
 field        value
 path         /home/bench/work/60be8efd6d21.bundle.zip
 size         18.4 KB
 files        envelope.json, signature_info.json, verify.py, README.txt, samples.jsonl, cosign.pub
 content_hash 60be8efd6d2178b40c5e...
 signature    dev-key
```

The recipient runs `python verify.py --pubkey cosign.pub` (or just `python verify.py` if the bundle includes the key) and gets:

```
OK
  content_hash: 60be8efd6d21...
  suite:        llm.inference.chatbot-short v1.0.0
  model:        meta-llama/Llama-3.1-8B-Instruct
  engine:       vllm
```

## Example: extract a bundle

```bash
bench bundle extract 60be8efd6d21.bundle.zip
```

Re-validates the inner envelope against the `Envelope` schema before printing:

```
              Bundle extracted: ./60be8efd6d21
 field        value
 out_dir      /home/bench/work/60be8efd6d21
 content_hash 60be8efd6d2178b40c5e...
 suite        llm.inference.chatbot-short v1.0.0
 model        meta-llama/Llama-3.1-8B-Instruct
 engine       vllm v0.21.0
 signature    dev-key
```

## Flags (`create`)

| Flag | Default | Description |
|---|---|---|
| `--out` | `<content_hash[:12]>.bundle.zip` in cwd | Destination zip path. |
| `--include-samples` / `--no-include-samples` | on | Pack `samples-*.jsonl` files sitting next to the envelope (matched by mtime within 5 minutes). |
| `--include-public-key` | none | Embed a PEM ed25519 public key so dev-key envelopes verify without a separate download. |

## Flags (`extract`)

| Flag | Default | Description |
|---|---|---|
| `--out` | `./<basename>/` (with `.bundle.zip`/`.zip` stripped) | Destination directory. |

## Bundle contents

| File | Purpose |
|---|---|
| `envelope.json` | Original signed envelope, exact bytes. |
| `signature_info.json` | At-a-glance summary: method, key id, content hash. |
| `verify.py` | Self-contained ed25519 verifier (Python 3.12 + `cryptography`). |
| `README.txt` | Three-line orientation for the recipient. |
| `samples.jsonl` | Optional concatenated per-sample trace (only if `--include-samples`). |
| `cosign.pub` | Optional public key (only if `--include-public-key`). |

## See also

- [bench verify](bench-verify.md) — verify an envelope without unpacking a bundle
- [The signed envelope](../concepts/envelope.md)
