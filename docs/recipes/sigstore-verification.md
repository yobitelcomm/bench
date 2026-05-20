# Sigstore-verified envelopes (keyless)

The 50 envelopes in the multi-vendor marathon were dev-key signed — the private key was generated on TestBM, used once, and destroyed. The trust anchor is published at [Yobitel/bench-trust-anchors](https://huggingface.co/datasets/Yobitel/bench-trust-anchors) so anyone can verify them.

For **v0.0.3+** releases the bench release pipeline produces **keyless-signed envelopes** via Sigstore. Instead of trusting a human-controlled key, you verify against the GitHub Actions OIDC subject — the signature ties the envelope to a specific workflow file at a specific commit.

## Why this matters

| Property | Dev-key signing | Sigstore keyless |
|---|---|---|
| Trust anchor | A public key file we host | The cryptographic identity of `yobitelcomm/bench`'s release workflow |
| Key lifecycle | Generated, used, destroyed | Per-signature ephemeral certificate (~10 min) issued by Sigstore Fulcio |
| Transparency | None | Every signature appears on the public [Rekor log](https://rekor.sigstore.dev) |
| Verifier needs | `cosign.pub` file | Internet access + sigstore-python |
| Failure mode if our infra is compromised | Attacker mints fake envelopes | Attacker would need GitHub repo write + a successful release workflow run, both publicly auditable |

## Try it

We've published one production-grade keyless envelope as the proof point:

[`Yobitel/keyless-signed-demo/keyless-43a8c771ef48.json`](https://huggingface.co/datasets/Yobitel/keyless-signed-demo)

Verify it from your local terminal:

```bash
bench fetch hf://datasets/Yobitel/keyless-signed-demo/keyless-43a8c771ef48.json
bench verify ~/.cache/inferencebench/fetched/<hash>.json \
  --require-issuer https://token.actions.githubusercontent.com \
  --require-identity-pattern 'github\.com/yobitelcomm/bench/'
```

Expected output:

```
OK
  method:           sigstore-cosign
  content_hash:     43a8c771ef48d66d38a1e673a79f3b315a9906d029a042a9b8078ae29ede2155
  suite:            llm.inference.chatbot-short v1.0.0
  model:            Qwen/Qwen2.5-7B-Instruct
  engine:           vllm v0.21.0
  rekor_log_index:  1575899993
  signer_identity:  https://github.com/yobitelcomm/bench/.github/workflows/keyless-sign-demo.yml@refs/heads/main
  signer_issuer:    https://token.actions.githubusercontent.com
```

The `rekor_log_index` is publicly queryable at https://search.sigstore.dev — anyone can audit the full chain of custody.

## What the policy flags do

| Flag | Effect |
|---|---|
| `--require-issuer <url>` | Reject if the OIDC issuer doesn't exactly equal this string. For GitHub Actions, that's always `https://token.actions.githubusercontent.com`. |
| `--require-identity-pattern <regex>` | Reject if the signer identity (the SAN value from the Fulcio cert) doesn't match this regex. Use this to scope acceptance to a specific repo and workflow file. |

Both are **caller-enforced** — `bench verify` will return `OK` from the cryptographic layer regardless of who signed; the flags decide whether to honour that OK or reject. This mirrors `cosign verify-blob`'s UX and lets organizations write their own trust policies on top of the same primitive.

## How to sign your own envelopes keyless

Two paths:

### 1. Inside GitHub Actions (recommended)

Add `permissions: id-token: write` to the job and call:

```python
from inferencebench.envelope import sign_envelope, SigningMode
signed = sign_envelope(envelope, mode=SigningMode.KEYLESS)
```

GitHub injects an OIDC token into the runner's environment; `sigstore-python` picks it up automatically, mints a Fulcio cert, signs the envelope's content hash, and writes a Sigstore bundle into `signature.bundle`.

See [`.github/workflows/keyless-sign-demo.yml`](https://github.com/yobitelcomm/bench/blob/main/.github/workflows/keyless-sign-demo.yml) and [`scripts/keyless_sign_envelope.py`](https://github.com/yobitelcomm/bench/blob/main/scripts/keyless_sign_envelope.py) for the working reference implementation.

### 2. Locally (interactive)

```bash
pip install 'inferencebench-envelope[keyless]'
# A browser window opens for OAuth — pick your GitHub / Google / Microsoft identity.
python -c "
from inferencebench.envelope import sign_envelope, SigningMode
from inferencebench.envelope.models import Envelope
import json
env = Envelope.model_validate(json.load(open('my-envelope.json')))
signed = sign_envelope(env, mode=SigningMode.KEYLESS)
open('my-envelope.signed.json', 'w').write(signed.model_dump_json(indent=2))
"
```

The interactive browser flow is for one-off local signing only. Production envelopes should always come from CI.

## Limitations + caveats

- **`UnsafeNoOp` verify policy at the library layer.** The bench verifier checks the cryptographic signature against the certificate without enforcing identity at the Sigstore policy level. Callers MUST supply `--require-identity-pattern` and `--require-issuer` (or equivalent) to make the verify meaningful. Bare `bench verify` without those flags accepts any Sigstore-signed envelope as valid.
- **TUF root rotation warnings** (`Key … failed to verify root`) are normal output from sigstore-python's transparency log validation — they don't indicate failure. They appear during routine TUF metadata rotation events.
- **Network required.** Keyless verify contacts Sigstore's transparency log and trust root services. Air-gapped verifiers should use the dev-key path with a pre-distributed trust anchor.
- **Phase 2 plan:** the bench release workflow will pin both `signer_identity` and `signer_issuer` to specific values in a wrapped policy module that the CLI exposes via `bench verify --bench-trust-policy` — saves users from having to remember the regex.
