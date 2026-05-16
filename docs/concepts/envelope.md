# The signed envelope

Every benchmark result is captured in a single JSON document called an envelope. The envelope is the unit of trust in InferenceBench. If a result is not in an envelope, it does not exist.

## Why an envelope

A bare metric like "1842 tokens per second" is meaningless without context. The envelope binds the metric to:

- The exact model and revision
- The exact engine and configuration hash
- The hardware fingerprint (GPU, CPU, memory, BIOS, drivers)
- The software provenance (image digest, pip freeze hash, git commit)
- The dataset hash
- The seed
- A Sigstore signature

Anyone with the envelope can re-derive the configuration, attempt to reproduce, and verify the signature against the Sigstore transparency log.

## What is in an envelope

```json
{
  "envelope_version": "v1",
  "suite_id": "llm.inference",
  "suite_version": "1.0.0",
  "run_id": "01J7Q5C6...",
  "timestamp": "2026-11-15T12:34:56Z",
  "model": {
    "id": "meta-llama/Llama-4-Maverick",
    "revision": "abc123...",
    "provider": "vllm-local",
    "endpoint_hash": "8f9a..."
  },
  "engine": {
    "name": "vllm",
    "version": "0.7.2",
    "config_hash": "8b1a...",
    "image_digest": "sha256:..."
  },
  "quantization": {"format": "fp8", "method": "..."},
  "hardware_fingerprint": { "...": "see below" },
  "software_provenance": {
    "image_digest": "sha256:...",
    "pip_freeze_hash": "...",
    "git_commit": "..."
  },
  "dataset": {"id": "sharegpt-v3", "hash": "..."},
  "seed": 42,
  "driver_options": {"...": "..."},
  "metrics": {
    "ttft_p50_ms": 142.0,
    "ttft_p99_ms": 280.3,
    "tpot_p50_ms": 18.5,
    "throughput_tok_per_s": 1842.1,
    "goodput_at_slo": 142.3,
    "joules_per_token": 0.32
  },
  "distributions": {
    "ttft_ms": "traces.parquet#ttft_ms",
    "tpot_ms": "traces.parquet#tpot_ms"
  },
  "slo_template": "llm.standard",
  "warnings": [],
  "signature": {
    "method": "sigstore-cosign",
    "certificate": "-----BEGIN CERTIFICATE-----...",
    "rekor_log_index": 12345,
    "bundle": "..."
  }
}
```

## Fields, top to bottom

| Field | Meaning |
|---|---|
| `envelope_version` | Schema version. v1 is the current schema. |
| `suite_id` | Suite identifier, e.g. `llm.inference`. |
| `suite_version` | SemVer of the plugin that produced the result. |
| `run_id` | UUIDv7. Sortable by creation time. |
| `timestamp` | RFC 3339 UTC. |
| `model` | Provider-prefixed id, revision, provider, endpoint hash. |
| `engine` | Engine name, version, config hash, optional image digest. |
| `quantization` | Quant format (`fp8`, `nvfp4`, `awq-int4`, ...) plus method notes. |
| `hardware_fingerprint` | DMI UUID, GPUs, CPU, memory, BIOS, NUMA, driver, CUDA, NCCL, and the SHA-256 of all of it. See [Hardware fingerprinting](fingerprinting.md). |
| `software_provenance` | OCI image digest (if containerized), `pip freeze` hash, git commit, `nvidia-smi -q` hash. |
| `dataset` | Dataset id and canonical SHA-256. |
| `seed` | Explicit integer used to seed every stochastic step. |
| `driver_options` | Driver settings (Poisson rate, concurrency, duration). |
| `metrics` | Per-suite metrics. Keys are defined by the plugin. |
| `distributions` | Pointers to distribution data (parquet column or inline histogram). |
| `slo_template` | SLO template id (`llm.standard`, `voice.realtime`, ...). |
| `warnings` | Non-fatal warnings emitted during the run. |
| `signature` | Sigstore signature bundle. |

## The content hash

The signature signs the SHA-256 of the canonical JSON body, excluding the `signature` block itself:

```python
body = envelope.model_dump(exclude={"signature"}, mode="json")
canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

Any byte change in the body invalidates the signature. `bench verify` recomputes the content hash before checking the Sigstore signature, so a tampered envelope is rejected even if the signature would otherwise verify.

## Signing modes

| Mode | When | Identity |
|---|---|---|
| `sigstore-cosign` | Production OSS use | GitHub Actions OIDC token (keyless) or local browser OIDC flow |
| `dev-key` | Local development and tests | Ed25519 key generated with `cosign generate-key-pair` |

The `dev-key` mode is for testing only. Published envelopes must be signed with `sigstore-cosign`.

## Schema versioning

The schema is versioned via the `envelope_version` field. Any breaking change bumps the version and ships a migration. Every previous-version fixture must still verify after a bump.

## See also

- [Envelope schema reference](../reference/envelope-schema.md)
- [Hardware fingerprinting](fingerprinting.md)
- [bench verify](../cli/bench-verify.md)
