"""Tests for ``bench doctor --show-slo``.

The flag adds a second Rich table after the diagnostic, showing the host's
detected hardware class and the resolved ``llm.standard`` SLO thresholds.
"""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from inferencebench.cli import app
from inferencebench.envelope import (
    BIOS,
    CPU,
    GPU,
    HardwareFingerprint,
    Memory,
)

runner = CliRunner()


def _h100_fp() -> HardwareFingerprint:
    body: dict[str, Any] = {
        "dmi_uuid": "11111111-2222-3333-4444-555555555555",
        "gpus": [
            GPU(
                model="NVIDIA H100 80GB HBM3",
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


def _rtx_4090_fp() -> HardwareFingerprint:
    body: dict[str, Any] = {
        "dmi_uuid": "22222222-3333-4444-5555-666666666666",
        "gpus": [
            GPU(
                model="NVIDIA GeForce RTX 4090",
                pci_id="0000:01:00.0",
                serial="0000000002",
                vbios="95.02.3c.00.91",
            )
        ],
        "cpu": CPU(model="AMD Ryzen 9 7950X", microcode="0x0a601203"),
        "memory": Memory(channels=2, speed_mts=6000, ecc=False),
        "bios": BIOS(version="F8", resizable_bar=True, above_4g=True),
        "driver": "550.54.15",
        "cuda": "12.4",
        "nccl": "",
    }
    placeholder = HardwareFingerprint.model_construct(fingerprint_sha256="0" * 64, numa={}, **body)
    real = placeholder.compute_fingerprint_sha256()
    return HardwareFingerprint(fingerprint_sha256=real, numa={}, **body)


def test_doctor_show_slo_prints_h100_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On an H100 the resolved thresholds equal the unscaled base."""
    monkeypatch.setattr(
        "inferencebench.commands.doctor.collect_hardware_fingerprint",
        lambda: _h100_fp(),
    )
    result = runner.invoke(app, ["doctor", "--show-slo"])
    out = result.stdout + (result.stderr or "")
    assert "SLO template" in out
    assert "h100" in out
    assert "ttft<200ms" in out
    assert "tpot<50ms" in out
    assert "total<3000ms" in out


def test_doctor_show_slo_prints_rtx4090_scaled_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RTX 4090 (1.8x multiplier) produces 360/90/5400 thresholds."""
    monkeypatch.setattr(
        "inferencebench.commands.doctor.collect_hardware_fingerprint",
        lambda: _rtx_4090_fp(),
    )
    result = runner.invoke(app, ["doctor", "--show-slo"])
    out = result.stdout + (result.stderr or "")
    assert "rtx-4090" in out
    assert "ttft<360ms" in out
    assert "tpot<90ms" in out
    assert "total<5400ms" in out
