"""Tests for hardware fingerprint collection."""

from __future__ import annotations

from pathlib import Path

import pytest

from inferencebench.envelope import HardwareFingerprint, SoftwareProvenance
from inferencebench.harness.fingerprint import (
    collect_hardware_fingerprint,
    collect_software_provenance,
)
from inferencebench.harness.fingerprint.cpu import collect_cpu
from inferencebench.harness.fingerprint.dmi import collect_bios, collect_dmi_uuid
from inferencebench.harness.fingerprint.memory import collect_memory
from inferencebench.harness.fingerprint.numa import collect_numa


# --------------------------------------------------------------------------- #
# DMI / BIOS                                                                  #
# --------------------------------------------------------------------------- #
def test_dmi_uuid_reads_product_uuid(tmp_path: Path) -> None:
    sysfs = tmp_path / "sys"
    dmi_dir = sysfs / "class" / "dmi" / "id"
    dmi_dir.mkdir(parents=True)
    (dmi_dir / "product_uuid").write_text("12345678-1234-1234-1234-123456789abc\n")
    assert collect_dmi_uuid(sysfs) == "12345678-1234-1234-1234-123456789abc"


def test_dmi_uuid_missing_returns_unknown(tmp_path: Path) -> None:
    sysfs = tmp_path / "sys"
    sysfs.mkdir()
    assert collect_dmi_uuid(sysfs) == "unknown"


def test_dmi_uuid_unreadable_returns_unknown(tmp_path: Path) -> None:
    sysfs = tmp_path / "sys"
    dmi_dir = sysfs / "class" / "dmi" / "id"
    dmi_dir.mkdir(parents=True)
    # Empty file
    (dmi_dir / "product_uuid").write_text("")
    assert collect_dmi_uuid(sysfs) == "unknown"


def test_bios_collects_version(tmp_path: Path) -> None:
    sysfs = tmp_path / "sys"
    dmi_dir = sysfs / "class" / "dmi" / "id"
    dmi_dir.mkdir(parents=True)
    (dmi_dir / "bios_version").write_text("3.4a\n")
    bios = collect_bios(sysfs)
    assert bios.version == "3.4a"
    assert bios.resizable_bar is False
    assert bios.above_4g is False


def test_bios_missing_yields_unknown(tmp_path: Path) -> None:
    sysfs = tmp_path / "sys"
    sysfs.mkdir()
    bios = collect_bios(sysfs)
    assert bios.version == "unknown"


# --------------------------------------------------------------------------- #
# CPU                                                                         #
# --------------------------------------------------------------------------- #
def test_cpu_parses_proc_cpuinfo(tmp_path: Path) -> None:
    proc = tmp_path / "cpuinfo"
    proc.write_text(
        "processor       : 0\n"
        "model name      : Intel(R) Xeon(R) Platinum 8480C\n"
        "microcode       : 0x2b000571\n"
        "cpu MHz         : 2000.0\n"
        "\n"
        "processor       : 1\n"
        "model name      : Intel(R) Xeon(R) Platinum 8480C\n"
        "microcode       : 0x2b000571\n"
    )
    cpu = collect_cpu(proc_cpuinfo=proc)
    assert "Xeon" in cpu.model
    assert cpu.microcode == "0x2b000571"


def test_cpu_missing_falls_back_to_platform(tmp_path: Path) -> None:
    missing = tmp_path / "no_cpuinfo"
    cpu = collect_cpu(proc_cpuinfo=missing)
    assert cpu.model  # non-empty (uses platform.processor or .machine)
    assert cpu.microcode == "unknown"


# --------------------------------------------------------------------------- #
# Memory                                                                      #
# --------------------------------------------------------------------------- #
def test_memory_parses_dmidecode() -> None:
    dmi_text = """# dmidecode 3.5
Handle 0x0023, DMI type 17, 92 bytes
Memory Device
        Array Handle: 0x0022
        Total Width: 72 bits
        Data Width: 64 bits
        Size: 32 GB
        Form Factor: DIMM
        Set: None
        Locator: DIMM_A1
        Speed: 4800 MT/s
        Configured Memory Speed: 4800 MT/s

Handle 0x0024, DMI type 17, 92 bytes
Memory Device
        Array Handle: 0x0022
        Total Width: 72 bits
        Data Width: 64 bits
        Size: 32 GB
        Speed: 4800 MT/s
        Configured Memory Speed: 4800 MT/s

Handle 0x0025, DMI type 17, 92 bytes
Memory Device
        Size: No Module Installed
"""
    mem = collect_memory(dmidecode_output=dmi_text)
    assert mem.channels == 2
    assert mem.speed_mts == 4800
    assert mem.ecc is True  # total_width > data_width


def test_memory_no_dmidecode_falls_back_to_minimums() -> None:
    mem = collect_memory(dmidecode_output="")
    assert mem.channels >= 1
    assert mem.speed_mts >= 1
    assert mem.ecc is False


# --------------------------------------------------------------------------- #
# NUMA                                                                        #
# --------------------------------------------------------------------------- #
def test_numa_empty_on_missing_sysfs(tmp_path: Path) -> None:
    assert collect_numa(sysfs_root=tmp_path / "nope") == {}


def test_numa_parses_node_topology(tmp_path: Path) -> None:
    sysfs = tmp_path / "sys"
    node_root = sysfs / "devices" / "system" / "node"
    node0 = node_root / "node0"
    node1 = node_root / "node1"
    node0.mkdir(parents=True)
    node1.mkdir(parents=True)
    (node0 / "cpulist").write_text("0-3,8-11\n")
    (node0 / "meminfo").write_text("Node 0 MemTotal:       65536000 kB\n")
    (node1 / "cpulist").write_text("4-7,12-15\n")
    (node1 / "meminfo").write_text("Node 1 MemTotal:       65536000 kB\n")

    numa = collect_numa(sysfs_root=sysfs)
    assert "nodes" in numa
    assert len(numa["nodes"]) == 2
    assert numa["nodes"][0]["id"] == 0
    assert numa["nodes"][0]["cpus"] == [0, 1, 2, 3, 8, 9, 10, 11]
    assert numa["nodes"][0]["memory_mb"] == 65536000 // 1024


# --------------------------------------------------------------------------- #
# Top-level integration                                                       #
# --------------------------------------------------------------------------- #
def test_collect_hardware_fingerprint_returns_valid_model(tmp_path: Path) -> None:
    """End-to-end: collect on a host that may have nothing useful, get a valid envelope."""
    sysfs = tmp_path / "sys"
    sysfs.mkdir()
    fp = collect_hardware_fingerprint(sysfs_root=sysfs)
    assert isinstance(fp, HardwareFingerprint)
    # fingerprint_sha256 is 64 hex chars and self-consistent
    assert len(fp.fingerprint_sha256) == 64
    assert fp.compute_fingerprint_sha256() == fp.fingerprint_sha256


def test_collect_hardware_fingerprint_deterministic(tmp_path: Path) -> None:
    """Same /sys contents → same fingerprint_sha256."""
    sysfs = tmp_path / "sys"
    dmi_dir = sysfs / "class" / "dmi" / "id"
    dmi_dir.mkdir(parents=True)
    (dmi_dir / "product_uuid").write_text("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\n")
    (dmi_dir / "bios_version").write_text("3.4a\n")
    fp1 = collect_hardware_fingerprint(sysfs_root=sysfs)
    fp2 = collect_hardware_fingerprint(sysfs_root=sysfs)
    assert fp1.fingerprint_sha256 == fp2.fingerprint_sha256


def test_collect_software_provenance_returns_valid_model() -> None:
    sp = collect_software_provenance()
    assert isinstance(sp, SoftwareProvenance)
    assert len(sp.pip_freeze_hash) == 64
    assert len(sp.git_commit) >= 7  # short SHA or pad
    # nvidia_smi_q_hash is "" or 64 hex chars
    assert sp.nvidia_smi_q_hash == "" or len(sp.nvidia_smi_q_hash) == 64


# --------------------------------------------------------------------------- #
# GPU — skipped unless pynvml + driver present                                #
# --------------------------------------------------------------------------- #
def test_gpu_collector_returns_list_even_without_nvidia() -> None:
    """On any host (with or without GPU), collect_gpus() returns a list, never raises."""
    from inferencebench.harness.fingerprint.gpu import collect_gpus

    gpus = collect_gpus()
    assert isinstance(gpus, list)


def test_nvidia_runtime_returns_triple() -> None:
    """collect_nvidia_runtime always returns a (str, str, str) tuple."""
    from inferencebench.harness.fingerprint.gpu import collect_nvidia_runtime

    driver, cuda, nccl = collect_nvidia_runtime()
    assert isinstance(driver, str)
    assert isinstance(cuda, str)
    assert isinstance(nccl, str)


@pytest.mark.gpu
def test_gpu_collector_on_nvidia_host() -> None:
    """When NVIDIA is present, at least one GPU is detected. Runs only with @gpu marker."""
    from inferencebench.harness.fingerprint.gpu import collect_gpus

    gpus = collect_gpus()
    assert len(gpus) >= 1
    for gpu in gpus:
        assert gpu.model
        assert gpu.pci_id
        assert gpu.serial
        assert gpu.vbios
