# inferencebench-envelope

The canonical signed-envelope spec for InferenceBench results. This is the product's defensibility moat — every benchmark result is signed via Sigstore and verifiable by anyone with `bench verify`.

## Status

Phase 1 active development. Schema v1 in progress (ticket 0004).

## Concepts

- **Envelope**: Pydantic v2 model representing one benchmark run's full provenance
- **Hardware fingerprint**: SHA-256 of DMI UUID + GPU PCI IDs + serials + driver + BIOS state
- **Software provenance**: pip freeze hash + git commit + (optional) container image digest
- **Signature**: keyless OIDC via Sigstore cosign + Rekor transparency log entry

See [docs/concepts/envelope.md](../docs/concepts/envelope.md) for the full conceptual guide.
