"""Unit tests for envelope Pydantic models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from inferencebench.envelope import (
    BIOS,
    CPU,
    GPU,
    DatasetSpec,
    EngineConfig,
    Envelope,
    EnvelopeBuilder,
    HardwareFingerprint,
    Memory,
    ModelConfig,
    Quantization,
    SoftwareProvenance,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _gpu() -> GPU:
    return GPU(
        model="H100-SXM5-80GB",
        pci_id="0000:01:00.0",
        serial="1234567890",
        vbios="96.00.74.00.01",
    )


def _cpu() -> CPU:
    return CPU(model="Intel(R) Xeon(R) Platinum 8480C", microcode="0x2b000571")


def _memory() -> Memory:
    return Memory(channels=12, speed_mts=4800, ecc=True)


def _bios() -> BIOS:
    return BIOS(version="3.4a", resizable_bar=True, above_4g=True)


def _hardware_fp_unhashed_body() -> dict:
    return {
        "dmi_uuid": "11111111-2222-3333-4444-555555555555",
        "gpus": [_gpu()],
        "cpu": _cpu(),
        "memory": _memory(),
        "bios": _bios(),
        "driver": "560.35.03",
        "cuda": "12.6",
        "nccl": "2.22.3",
    }


def _hardware_fp() -> HardwareFingerprint:
    body = _hardware_fp_unhashed_body()
    # Build once with a placeholder, then recompute and rebuild
    placeholder = "0" * 64
    fp_first = HardwareFingerprint.model_construct(
        fingerprint_sha256=placeholder,
        numa={},
        **body,
    )
    real_sha = fp_first.compute_fingerprint_sha256()
    return HardwareFingerprint(fingerprint_sha256=real_sha, numa={}, **body)


def _software_provenance() -> SoftwareProvenance:
    return SoftwareProvenance(
        image_digest="sha256:" + "a" * 64,
        pip_freeze_hash="b" * 64,
        git_commit="deadbeef1234567",
        nvidia_smi_q_hash="c" * 64,
    )


def _model_config() -> ModelConfig:
    return ModelConfig(
        id="meta-llama/Llama-4-Maverick",
        revision="abc1234",
        provider="vllm-local",
        endpoint_hash="d" * 64,
    )


def _engine_config() -> EngineConfig:
    return EngineConfig(
        name="vllm",
        version="0.7.2",
        config_hash="e" * 64,
        image_digest="sha256:" + "f" * 64,
    )


def _dataset_spec() -> DatasetSpec:
    return DatasetSpec(id="sharegpt-v3", hash="1" * 64)


def _envelope(**overrides: object) -> Envelope:
    defaults = {
        "envelope_version": "v1",
        "suite_id": "llm.inference",
        "suite_version": "1.0.0",
        "run_id": "01934567-89ab-7000-8000-000000000000",
        "timestamp": datetime(2026, 5, 15, 10, 30, 0, tzinfo=UTC),
        "model": _model_config(),
        "engine": _engine_config(),
        "quantization": Quantization(format="fp8"),
        "hardware_fingerprint": _hardware_fp(),
        "software_provenance": _software_provenance(),
        "dataset": _dataset_spec(),
        "seed": 42,
        "metrics": {"ttft_p50_ms": 142.0, "throughput_tok_per_s": 1842.1},
        "slo_template": "llm.standard",
    }
    defaults.update(overrides)
    return Envelope(**defaults)


# --------------------------------------------------------------------------- #
# Basic model construction                                                    #
# --------------------------------------------------------------------------- #
def test_gpu_minimal() -> None:
    gpu = _gpu()
    assert gpu.model == "H100-SXM5-80GB"
    assert gpu.pci_id == "0000:01:00.0"


def test_hardware_fingerprint_self_consistent() -> None:
    fp = _hardware_fp()
    assert fp.fingerprint_sha256 == fp.compute_fingerprint_sha256()


def test_hardware_fingerprint_mismatch_rejected() -> None:
    body = _hardware_fp_unhashed_body()
    with pytest.raises(ValidationError, match="fingerprint_sha256 mismatch"):
        HardwareFingerprint(
            fingerprint_sha256="0" * 64,  # deliberately wrong
            numa={},
            **body,
        )


def test_envelope_requires_at_least_one_metric() -> None:
    with pytest.raises(ValidationError, match="metrics must contain"):
        _envelope(metrics={})


def test_envelope_timestamp_must_be_utc() -> None:
    with pytest.raises(ValidationError, match="timestamp must be timezone-aware"):
        _envelope(timestamp=datetime(2026, 5, 15, 10, 30, 0))  # naive


def test_envelope_version_locked_to_v1() -> None:
    with pytest.raises(ValidationError):
        _envelope(envelope_version="v2")  # type: ignore[arg-type]


def test_envelope_serializes_roundtrip() -> None:
    env = _envelope()
    raw = env.model_dump_json()
    parsed = Envelope.model_validate_json(raw)
    assert parsed.content_hash() == env.content_hash()


def test_envelope_content_hash_deterministic() -> None:
    e1 = _envelope()
    e2 = _envelope()
    # Same inputs (including run_id and timestamp) → same content_hash
    assert e1.content_hash() == e2.content_hash()


def test_envelope_content_hash_excludes_signature() -> None:
    """Adding/changing the signature field must not change the content_hash."""
    from inferencebench.envelope import Signature

    env_unsigned = _envelope()
    h_before = env_unsigned.content_hash()

    # Mutate signature
    env_signed = _envelope(
        signature=Signature(
            method="dev-key",
            certificate="-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----",
            rekor_log_index=-1,
        )
    )
    h_after = env_signed.content_hash()
    assert h_before == h_after, "content_hash must be invariant under signature changes"


def test_envelope_builder_assigns_uuid7_and_now() -> None:
    builder = EnvelopeBuilder(
        suite_id="llm.inference",
        suite_version="1.0.0",
        model=_model_config(),
        engine=_engine_config(),
        hardware_fingerprint=_hardware_fp(),
        software_provenance=_software_provenance(),
        dataset=_dataset_spec(),
        seed=42,
        metrics={"ttft_p50_ms": 100.0},
    )
    env = builder.build()
    # UUIDv7 has version nibble 7
    assert env.run_id[14] == "7", f"Expected UUIDv7, got run_id={env.run_id}"
    # Timestamp is UTC and roughly now
    assert env.timestamp.tzinfo is not None
    now_utc = datetime.now(UTC)
    delta_s = abs((env.timestamp - now_utc).total_seconds())
    assert delta_s < 5, f"Builder timestamp {env.timestamp} is {delta_s}s off from now"


def test_envelope_extra_fields_preserved() -> None:
    """Forward-compat: unknown fields survive a round trip."""
    env = _envelope()
    raw = env.model_dump_json()
    import json

    parsed_dict = json.loads(raw)
    parsed_dict["future_field"] = {"hint": "this might come in v1.1"}

    new = Envelope.model_validate(parsed_dict)
    re_serialized = new.model_dump_json()
    assert "future_field" in re_serialized


def test_canonical_json_is_sorted() -> None:
    env = _envelope()
    raw = env.to_canonical_json()
    # First top-level keys should be alphabetical
    # quick sanity: the substring "dataset" comes before "engine" before "envelope_version"
    assert raw.index('"dataset"') < raw.index('"engine"')
    assert raw.index('"engine"') < raw.index('"envelope_version"')
