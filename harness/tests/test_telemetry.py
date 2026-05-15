"""Tests for telemetry samplers (NVML + RAPL)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from inferencebench.harness.telemetry import NVMLSampler, RAPLSampler


# --------------------------------------------------------------------------- #
# Base Sampler contract                                                       #
# --------------------------------------------------------------------------- #
def test_sampler_validates_interval() -> None:
    with pytest.raises(ValueError, match="interval_ms"):
        NVMLSampler(interval_ms=0)
    with pytest.raises(ValueError, match="interval_ms"):
        RAPLSampler(interval_ms=0)


def test_nvml_sampler_works_without_nvidia() -> None:
    """On a host without NVIDIA, NVMLSampler runs and returns 0 samples — never raises."""
    sampler = NVMLSampler(interval_ms=20)
    with sampler:
        time.sleep(0.1)
    snapshot = sampler.snapshot()
    assert isinstance(snapshot, list)
    # If NVIDIA isn't here: 0 samples. If it is, samples should accumulate.
    if snapshot:
        for s in snapshot:
            assert s.t_ms >= 0
            assert isinstance(s.devices, tuple)


def test_nvml_sampler_idempotent_start_stop() -> None:
    sampler = NVMLSampler(interval_ms=20)
    sampler.start()
    sampler.start()  # no-op
    sampler.stop()
    sampler.stop()  # no-op


# --------------------------------------------------------------------------- #
# RAPL                                                                        #
# --------------------------------------------------------------------------- #
def test_rapl_sampler_empty_root(tmp_path: Path) -> None:
    """RAPLSampler pointed at an empty filesystem returns no samples."""
    sampler = RAPLSampler(interval_ms=20, powercap_root=tmp_path / "missing")
    with sampler:
        time.sleep(0.1)
    assert sampler.snapshot() == []


def test_rapl_sampler_reads_synthetic_filesystem(tmp_path: Path) -> None:
    """A faked /sys/class/powercap is enumerated and sampled."""
    powercap = tmp_path / "powercap"
    package = powercap / "intel-rapl:0"
    package.mkdir(parents=True)
    (package / "name").write_text("package-0\n")
    (package / "energy_uj").write_text("12345678\n")

    dram = powercap / "intel-rapl:0:0"
    dram.mkdir(parents=True)
    (dram / "name").write_text("dram\n")
    (dram / "energy_uj").write_text("87654321\n")

    sampler = RAPLSampler(interval_ms=10, powercap_root=powercap)
    with sampler:
        time.sleep(0.05)
    samples = sampler.snapshot()
    assert samples  # at least one tick
    for s in samples:
        names = {d["name"] for d in s.domains}
        assert "package-0" in names
        assert "dram" in names


def test_rapl_sampler_skips_unreadable_domains(tmp_path: Path) -> None:
    """Directories without energy_uj or with no name file are skipped."""
    powercap = tmp_path / "powercap"
    bogus = powercap / "intel-rapl:99"
    bogus.mkdir(parents=True)
    # No name, no energy_uj
    sampler = RAPLSampler(interval_ms=10, powercap_root=powercap)
    with sampler:
        time.sleep(0.05)
    assert sampler.snapshot() == []
