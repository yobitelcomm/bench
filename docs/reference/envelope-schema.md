# Envelope schema (v1)

The canonical schema for the signed envelope. The machine-readable JSON Schema lives at `envelope/schema/envelope.v1.json` in the repository.

## Top-level fields

| Field | Type | Required | Description |
|---|---|---|---|
| `envelope_version` | string | yes | Literal `"v1"`. |
| `suite_id` | string | yes | Suite identifier, e.g. `llm.inference`. |
| `suite_version` | string | yes | Plugin SemVer (e.g. `1.0.0`). |
| `run_id` | string | yes | UUIDv7, sortable by timestamp. |
| `timestamp` | string | yes | RFC 3339 UTC. |
| `model` | object | yes | See `ModelConfig`. |
| `engine` | object | yes | See `EngineConfig`. |
| `quantization` | object \| null | no | See `Quantization`. |
| `hardware_fingerprint` | object | yes | See `HardwareFingerprint`. |
| `software_provenance` | object | yes | See `SoftwareProvenance`. |
| `dataset` | object | yes | See `DatasetSpec`. |
| `seed` | integer | yes | Explicit integer seed. |
| `driver_options` | object | no | Free-form driver config. |
| `metrics` | object | yes | At least one metric. Per-suite keys. |
| `distributions` | object | no | Pointers to distribution data. |
| `slo_template` | string | no | SLO template id. |
| `warnings` | array<string> | no | Non-fatal warnings. |
| `signature` | object \| null | no | Sigstore signature bundle. `null` for unsigned envelopes. |

## ModelConfig

| Field | Type | Description |
|---|---|---|
| `id` | string | Provider-prefixed model id (`meta-llama/Llama-4-Maverick`). |
| `revision` | string | Git SHA or HF revision (7–40 chars). |
| `provider` | string | `vllm-local`, `together`, `anthropic`, ... |
| `endpoint_hash` | string | SHA-256 of the endpoint configuration. |

## EngineConfig

| Field | Type | Description |
|---|---|---|
| `name` | string | `vllm`, `sglang`, `trtllm`, ... |
| `version` | string | Engine SemVer. |
| `config_hash` | string | SHA-256 of the engine config. |
| `image_digest` | string | OCI image digest if containerized; empty otherwise. |

## Quantization

| Field | Type | Description |
|---|---|---|
| `format` | string | `bf16`, `fp16`, `fp8`, `nvfp4`, `awq-int4`, `gptq-int4`, `gguf-q4_k_m`, ... |
| `method` | string | Free-form description of the quant method. |

## HardwareFingerprint

| Field | Type | Description |
|---|---|---|
| `fingerprint_sha256` | string | SHA-256 of the canonical body. |
| `dmi_uuid` | string | DMI product UUID. |
| `gpus` | array<GPU> | Per-GPU detail. |
| `cpu` | object | `{model, microcode}`. |
| `memory` | object | `{channels, speed_mts, ecc}`. |
| `bios` | object | `{version, resizable_bar, above_4g}`. |
| `numa` | object | Canonical NUMA topology. |
| `driver` | string | GPU driver version. |
| `cuda` | string | CUDA toolkit version. |
| `nccl` | string | NCCL version. |

### GPU

| Field | Type | Description |
|---|---|---|
| `model` | string | Marketing name, e.g. `H100-SXM5-80GB`. |
| `pci_id` | string | PCI bus id (`0000:1b:00.0`). |
| `serial` | string | GPU serial. |
| `vbios` | string | VBIOS version. |

## SoftwareProvenance

| Field | Type | Description |
|---|---|---|
| `image_digest` | string | OCI image digest; empty if running natively. |
| `pip_freeze_hash` | string | SHA-256 of `pip freeze` output. |
| `git_commit` | string | Git commit of the harness. |
| `nvidia_smi_q_hash` | string | SHA-256 of `nvidia-smi -q`; empty on non-NVIDIA. |

## DatasetSpec

| Field | Type | Description |
|---|---|---|
| `id` | string | Dataset id (e.g. `sharegpt-v3`). |
| `hash` | string | SHA-256 of the canonical-ordered dataset. |

## Signature

| Field | Type | Description |
|---|---|---|
| `method` | string | `sigstore-cosign` or `dev-key`. |
| `certificate` | string | PEM-encoded certificate. |
| `rekor_log_index` | integer | Rekor transparency log index; `-1` for dev mode. |
| `bundle` | string | Full Sigstore bundle (base64). |

## Content hash

The content hash is the SHA-256 of the canonical JSON body, excluding the `signature` block:

```python
body = envelope.model_dump(exclude={"signature"}, mode="json")
canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

This is the value Sigstore signs.

## Forward compatibility

The schema tolerates unknown fields (`extra="allow"` in the Pydantic models). Old clients can read newer envelopes, ignoring fields they do not understand. Breaking changes bump `envelope_version` and ship a migration.

## See also

- [The signed envelope](../concepts/envelope.md)
- [Hardware fingerprinting](../concepts/fingerprinting.md)
- [bench verify](../cli/bench-verify.md)
