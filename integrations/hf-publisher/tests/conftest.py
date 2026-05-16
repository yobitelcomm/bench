"""Shared fixtures for hf-publisher tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from inferencebench.envelope import (
    BIOS,
    CPU,
    GPU,
    DatasetSpec,
    EngineConfig,
    Envelope,
    HardwareFingerprint,
    Memory,
    ModelConfig,
    Quantization,
    Signature,
    SoftwareProvenance,
)


def _hardware_fp() -> HardwareFingerprint:
    body = {
        "dmi_uuid": "11111111-2222-3333-4444-555555555555",
        "gpus": [
            GPU(
                model="H100-SXM5-80GB",
                pci_id="0000:01:00.0",
                serial="1234567890",
                vbios="96.00.74.00.01",
            ),
        ],
        "cpu": CPU(model="Intel(R) Xeon(R) Platinum 8480C", microcode="0x2b000571"),
        "memory": Memory(channels=12, speed_mts=4800, ecc=True),
        "bios": BIOS(version="3.4a", resizable_bar=True, above_4g=True),
        "driver": "560.35.03",
        "cuda": "12.6",
        "nccl": "2.22.3",
    }
    placeholder = HardwareFingerprint.model_construct(
        fingerprint_sha256="0" * 64,
        numa={},
        **body,
    )
    real_sha = placeholder.compute_fingerprint_sha256()
    return HardwareFingerprint(fingerprint_sha256=real_sha, numa={}, **body)


def _envelope(**overrides: object) -> Envelope:
    defaults: dict[str, object] = {
        "envelope_version": "v1",
        "suite_id": "llm.inference",
        "suite_version": "1.0.0",
        "run_id": "01934567-89ab-7000-8000-000000000000",
        "timestamp": datetime(2026, 5, 15, 10, 30, 0, tzinfo=UTC),
        "model": ModelConfig(
            id="meta-llama/Llama-4-Maverick",
            revision="abc1234",
            provider="vllm-local",
            endpoint_hash="d" * 64,
        ),
        "engine": EngineConfig(
            name="vllm",
            version="0.7.2",
            config_hash="e" * 64,
            image_digest="sha256:" + "f" * 64,
        ),
        "quantization": Quantization(format="fp8"),
        "hardware_fingerprint": _hardware_fp(),
        "software_provenance": SoftwareProvenance(
            image_digest="sha256:" + "a" * 64,
            pip_freeze_hash="b" * 64,
            git_commit="deadbeef1234567",
            nvidia_smi_q_hash="c" * 64,
        ),
        "dataset": DatasetSpec(id="sharegpt-v3", hash="1" * 64),
        "seed": 42,
        "metrics": {
            "ttft_p50_ms": 142.0,
            "ttft_p99_ms": 280.0,
            "throughput_tok_s": 1842.1,
            "goodput_req_s": 142,
            "cost_per_million_tokens_usd": 0.18,
            "joules_per_token": 0.32,
        },
        "slo_template": "llm.standard",
    }
    defaults.update(overrides)
    return Envelope(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def envelope() -> Envelope:
    """Unsigned reference envelope used across publisher tests."""
    return _envelope()


@pytest.fixture
def signed_envelope() -> Envelope:
    """Envelope with a dev-key signature populated."""
    return _envelope(
        signature=Signature(
            method="dev-key",
            certificate="-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----",
            rekor_log_index=987654,
        ),
    )
