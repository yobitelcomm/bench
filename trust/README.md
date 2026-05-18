# Trust anchors

Public keys that signed batches of published envelopes under
[Yobitel](https://huggingface.co/Yobitel) on Hugging Face Hub.

Mirror of <https://huggingface.co/datasets/Yobitel/bench-trust-anchors>.
Committed here so anyone cloning the repo can verify the published corpus
offline without a separate HF Hub fetch.

| File | Signed | Fingerprint | Provenance |
|---|---|---|---|
| `cosign-2026-05-18-marathon.pub` | 50 envelopes from 2026-05-18 multi-vendor marathon | `3479557f288500af` | Generated on TestBM (8×H100), private half destroyed at teardown |

## Verify any published envelope

```bash
bench fetch hf://datasets/Yobitel/<run-repo-id>
bench verify ~/.cache/inferencebench/fetched/<hash>.json \
  --dev-public-key trust/cosign-2026-05-18-marathon.pub
```

OK output ⇒ the envelope's content_hash matches the signed payload AND the
ed25519 signature verifies against this key.

## What this proves and doesn't

- **Proves**: bytes haven't changed since signing. The hardware fingerprint
  + dataset hash + metrics in the envelope are exactly what the signing
  process saw.
- **Doesn't prove**: any binding between this key and a specific identity
  on GitHub/PyPI. For identity-bound provenance, the v0.0.2 release pipeline
  will additionally sign via Sigstore keyless OIDC tied to the release.yml
  workflow.

License: Apache 2.0.
