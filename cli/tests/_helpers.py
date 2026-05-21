"""Test helpers shared across CLI subcommand tests.

Lives in a dedicated module (not ``conftest.py``) so it can be imported
unambiguously even when other workspace test trees ship their own
``conftest.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    SigningMode,
    SoftwareProvenance,
    sign_envelope,
)


def _hardware_fp() -> HardwareFingerprint:
    body: dict[str, Any] = {
        "dmi_uuid": "11111111-2222-3333-4444-555555555555",
        "gpus": [
            GPU(
                model="H100-SXM5-80GB",
                pci_id="0000:01:00.0",
                serial="1234567890",
                vbios="96.00.74.00.01",
            )
        ],
        "cpu": CPU(model="Intel(R) Xeon(R) Platinum 8480C", microcode="0x2b000571"),
        "memory": Memory(channels=12, speed_mts=4800, ecc=True),
        "bios": BIOS(version="3.4a", resizable_bar=True, above_4g=True),
        "driver": "560.35.03",
        "cuda": "12.6",
        "nccl": "2.22.3",
    }
    placeholder = HardwareFingerprint.model_construct(fingerprint_sha256="0" * 64, numa={}, **body)
    real = placeholder.compute_fingerprint_sha256()
    return HardwareFingerprint(fingerprint_sha256=real, numa={}, **body)


def make_envelope(
    *,
    model_id: str,
    metrics: dict[str, float | int | str | None],
    run_id: str = "01934567-89ab-7000-8000-000000000000",
    suite_id: str = "llm.inference",
) -> Envelope:
    """Build an unsigned envelope with the given model + metrics."""
    return Envelope(
        envelope_version="v1",
        suite_id=suite_id,
        suite_version="1.0.0",
        run_id=run_id,
        timestamp=datetime(2026, 5, 15, 10, 30, 0, tzinfo=UTC),
        model=ModelConfig(
            id=model_id,
            revision="abc1234",
            provider="vllm-local",
            endpoint_hash="d" * 64,
        ),
        engine=EngineConfig(name="vllm", version="0.7.2", config_hash="e" * 64),
        quantization=Quantization(format="fp8"),
        hardware_fingerprint=_hardware_fp(),
        software_provenance=SoftwareProvenance(
            pip_freeze_hash="b" * 64,
            git_commit="deadbeef1234567",
        ),
        dataset=DatasetSpec(id="sharegpt-v3", hash="1" * 64),
        seed=42,
        metrics=metrics,
        slo_template="llm.standard",
    )


def write_envelope_json(path: Path, envelope: Envelope) -> Path:
    """Serialize ``envelope`` to ``path`` as canonical JSON."""
    path.write_text(
        json.dumps(envelope.model_dump(mode="json"), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return path


def write_signed_envelope_json(path: Path, envelope: Envelope, *, dev_key: Path) -> Path:
    """Sign ``envelope`` with a dev key and write the signed JSON to ``path``."""
    signed = sign_envelope(envelope, mode=SigningMode.DEV, dev_key_path=dev_key)
    return write_envelope_json(path, signed)
