"""Tests for :mod:`inferencebench_llm.slo_profiles`.

Each test builds a :class:`HardwareFingerprint` with a distinct GPU / CPU
combination and asserts :func:`classify` picks the right
:class:`HardwareClass`.
"""

from __future__ import annotations

from typing import Any

from inferencebench.envelope import (
    BIOS,
    CPU,
    GPU,
    HardwareFingerprint,
    Memory,
)
from inferencebench_llm.slo_profiles import (
    HARDWARE_CLASSES,
    HardwareClass,
    classify,
    format_resolved,
    scale_slos,
)


def _fp(
    *,
    gpus: list[GPU],
    cpu_model: str = "Intel(R) Xeon(R) Platinum 8480C",
) -> HardwareFingerprint:
    """Build a HardwareFingerprint with the given GPUs and CPU model."""
    body: dict[str, Any] = {
        "dmi_uuid": "11111111-2222-3333-4444-555555555555",
        "gpus": gpus,
        "cpu": CPU(model=cpu_model, microcode="0x2b000571"),
        "memory": Memory(channels=12, speed_mts=4800, ecc=True),
        "bios": BIOS(version="3.4a", resizable_bar=True, above_4g=True),
        "driver": "560.35.03",
        "cuda": "12.6",
        "nccl": "2.22.3",
    }
    placeholder = HardwareFingerprint.model_construct(
        fingerprint_sha256="0" * 64, numa={}, **body
    )
    real = placeholder.compute_fingerprint_sha256()
    return HardwareFingerprint(fingerprint_sha256=real, numa={}, **body)


def _gpu(model: str) -> GPU:
    return GPU(
        model=model,
        pci_id="0000:01:00.0",
        serial="1234567890",
        vbios="96.00.74.00.01",
    )


def _class(key: str) -> HardwareClass:
    for cls in HARDWARE_CLASSES:
        if cls.key == key:
            return cls
    msg = f"unknown class {key!r}"
    raise AssertionError(msg)


def test_classify_h100_fingerprint() -> None:
    fp = _fp(gpus=[_gpu("H100-SXM5-80GB")])
    assert classify(fp) == _class("h100")


def test_classify_h200_fingerprint() -> None:
    fp = _fp(gpus=[_gpu("H200-SXM-141GB")])
    assert classify(fp) == _class("h200")


def test_classify_a100_fingerprint() -> None:
    fp = _fp(gpus=[_gpu("A100-SXM4-80GB")])
    assert classify(fp) == _class("a100")


def test_classify_rtx_4090_fingerprint() -> None:
    fp = _fp(gpus=[_gpu("NVIDIA GeForce RTX 4090")])
    assert classify(fp) == _class("rtx-4090")


def test_classify_rtx_ada_laptop_fingerprint() -> None:
    """RTX 4000/5000/3000 Ada Laptop GPU — common in mobile workstations."""
    fp = _fp(gpus=[_gpu("NVIDIA RTX 4000 Ada Generation Laptop GPU")])
    assert classify(fp) == _class("rtx-ada-laptop")
    fp = _fp(gpus=[_gpu("NVIDIA RTX 5000 Ada Generation Laptop GPU")])
    assert classify(fp) == _class("rtx-ada-laptop")


def test_classify_rtx_ada_workstation_fingerprint() -> None:
    """RTX 5000/6000 Ada desktop workstation cards (not laptop variants)."""
    fp = _fp(gpus=[_gpu("NVIDIA RTX 6000 Ada Generation")])
    assert classify(fp) == _class("rtx-ada-workstation")
    fp = _fp(gpus=[_gpu("NVIDIA RTX 5000 Ada Generation")])
    assert classify(fp) == _class("rtx-ada-workstation")


def test_classify_rtx_4080_4070_consumer_ada() -> None:
    fp = _fp(gpus=[_gpu("NVIDIA GeForce RTX 4080")])
    assert classify(fp) == _class("rtx-4080")
    fp = _fp(gpus=[_gpu("NVIDIA GeForce RTX 4070")])
    assert classify(fp) == _class("rtx-4070")


def test_classify_no_gpu_apple_cpu_returns_m_series() -> None:
    fp = _fp(gpus=[], cpu_model="Apple M3 Pro")
    assert classify(fp) == _class("m-series")


def test_classify_no_gpu_intel_cpu_returns_cpu() -> None:
    fp = _fp(gpus=[], cpu_model="Intel(R) Xeon(R) Platinum 8480C")
    assert classify(fp) == _class("cpu")


def test_classify_unknown_gpu_falls_back_to_cpu() -> None:
    """An exotic / unrecognised GPU on an Intel host falls back to ``cpu``."""
    fp = _fp(gpus=[_gpu("Mystery Vendor GPU 9000")])
    assert classify(fp) == _class("cpu")


def test_scale_and_format_for_h100_matches_base() -> None:
    """H100 is the 1.0x anchor — resolved thresholds equal base thresholds."""
    from inferencebench.harness.metrics import SLOPredicate

    base = [
        SLOPredicate("ttft", "ttft_ms", "<", 200.0),
        SLOPredicate("tpot", "tpot_ms", "<", 50.0),
        SLOPredicate("total", "total_ms", "<", 3000.0),
    ]
    rescaled = scale_slos(base, _class("h100"))
    assert [s.value for s in rescaled] == [200.0, 50.0, 3000.0]
    assert format_resolved(rescaled) == "ttft<200ms, tpot<50ms, total<3000ms"


def test_scale_for_rtx_4090_uses_1_8x_multiplier() -> None:
    from inferencebench.harness.metrics import SLOPredicate

    base = [
        SLOPredicate("ttft", "ttft_ms", "<", 200.0),
        SLOPredicate("tpot", "tpot_ms", "<", 50.0),
        SLOPredicate("total", "total_ms", "<", 3000.0),
    ]
    rescaled = scale_slos(base, _class("rtx-4090"))
    assert [s.value for s in rescaled] == [360.0, 90.0, 5400.0]
