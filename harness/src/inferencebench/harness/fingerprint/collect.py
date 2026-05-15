"""Top-level fingerprint orchestration.

Calls per-domain collectors (DMI, CPU, memory, BIOS, GPU, NUMA, driver, CUDA,
NCCL) and assembles the result into an ``inferencebench.envelope.HardwareFingerprint``
model.

Each collector handles missing data gracefully: on CPU-only systems the GPU
collector returns an empty list and the driver/CUDA fields are empty.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from inferencebench.envelope import HardwareFingerprint, SoftwareProvenance
from inferencebench.harness.fingerprint.cpu import collect_cpu
from inferencebench.harness.fingerprint.dmi import collect_bios, collect_dmi_uuid
from inferencebench.harness.fingerprint.gpu import collect_gpus, collect_nvidia_runtime
from inferencebench.harness.fingerprint.memory import collect_memory
from inferencebench.harness.fingerprint.numa import collect_numa


def collect_hardware_fingerprint(*, sysfs_root: Path | None = None) -> HardwareFingerprint:
    """Probe the running host and assemble a HardwareFingerprint.

    Args:
        sysfs_root: Override ``/sys`` root for testing. Production calls leave it None.
    """
    sysfs = sysfs_root or Path("/sys")

    dmi_uuid = collect_dmi_uuid(sysfs)
    gpus = collect_gpus()
    cpu = collect_cpu()
    memory = collect_memory()
    bios = collect_bios(sysfs)
    numa = collect_numa()
    driver, cuda, nccl = collect_nvidia_runtime()

    # Construct explicitly (not via **kwargs) so mypy sees concrete types.
    placeholder = HardwareFingerprint.model_construct(
        fingerprint_sha256="0" * 64,
        dmi_uuid=dmi_uuid,
        gpus=gpus,
        cpu=cpu,
        memory=memory,
        bios=bios,
        numa=numa,
        driver=driver,
        cuda=cuda,
        nccl=nccl,
    )
    real_sha = placeholder.compute_fingerprint_sha256()
    return HardwareFingerprint(
        fingerprint_sha256=real_sha,
        dmi_uuid=dmi_uuid,
        gpus=gpus,
        cpu=cpu,
        memory=memory,
        bios=bios,
        numa=numa,
        driver=driver,
        cuda=cuda,
        nccl=nccl,
    )


def collect_software_provenance(*, project_root: Path | None = None) -> SoftwareProvenance:
    """Compute the SoftwareProvenance for the current Python env + git checkout.

    - ``image_digest``: empty in Phase 1 (we don't yet run in containers by default).
    - ``pip_freeze_hash``: SHA-256 over canonical-sorted ``pip freeze`` output.
    - ``git_commit``: current HEAD short SHA from ``project_root`` (or cwd).
    - ``nvidia_smi_q_hash``: SHA-256 over ``nvidia-smi -q`` if available, else "".
    """
    pip_freeze_hash = _sha256_of(_run_capture(["pip", "freeze"]) or "")
    nvsmi_hash = (
        _sha256_of(_run_capture(["nvidia-smi", "-q"]) or "") if _has_command("nvidia-smi") else ""
    )
    git_commit = _git_commit(project_root)

    return SoftwareProvenance(
        image_digest="",
        pip_freeze_hash=pip_freeze_hash,
        git_commit=git_commit,
        nvidia_smi_q_hash=nvsmi_hash,
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _sha256_of(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _has_command(cmd: str) -> bool:
    """Return True if ``cmd`` is executable on PATH."""
    from shutil import which

    return which(cmd) is not None


def _run_capture(argv: list[str], *, timeout: int = 10) -> str | None:
    """Run a subprocess, return stdout decoded — or None on failure.

    Failures (missing command, non-zero exit, timeout) are silent here; callers
    decide how to handle absent data (usually substituting an empty hash).
    """
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _git_commit(project_root: Path | None) -> str:
    """Return the current git HEAD SHA, padded to ≥7 chars. Falls back to all-zeros."""
    cwd = str(project_root) if project_root else None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            cwd=cwd,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "0" * 40
    if out.returncode != 0 or not out.stdout.strip():
        return "0" * 40
    sha = out.stdout.strip()
    return sha[:40] if len(sha) >= 7 else "0" * 40
