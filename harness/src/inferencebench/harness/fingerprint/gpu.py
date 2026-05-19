"""GPU + NVIDIA driver/CUDA/NCCL collector. Uses pynvml if NVIDIA is present."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from inferencebench.envelope import GPU

if TYPE_CHECKING:
    pass


def collect_gpus() -> list[GPU]:
    """Enumerate NVIDIA GPUs via pynvml. Returns [] on non-NVIDIA or driver missing."""
    try:
        import pynvml
    except ImportError:
        return []

    try:
        pynvml.nvmlInit()
    except Exception:
        return []

    try:
        count = pynvml.nvmlDeviceGetCount()
    except Exception:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return []

    gpus: list[GPU] = []
    for i in range(count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            model = _decode(pynvml.nvmlDeviceGetName(handle))
            pci = pynvml.nvmlDeviceGetPciInfo(handle)
            pci_id = _decode(pci.busId) if hasattr(pci, "busId") else ""
            try:
                serial = _decode(pynvml.nvmlDeviceGetSerial(handle))
            except Exception:
                serial = "unknown"
            try:
                vbios = _decode(pynvml.nvmlDeviceGetVbiosVersion(handle))
            except Exception:
                vbios = "unknown"
            gpus.append(
                GPU(
                    model=model or f"NVIDIA-GPU-{i}",
                    pci_id=pci_id or f"0000:00:{i:02x}.0",
                    serial=serial or "unknown",
                    vbios=vbios or "unknown",
                )
            )
        except Exception:
            continue

    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass

    return gpus


def collect_nvidia_runtime() -> tuple[str, str, str]:
    """Return (driver_version, cuda_version, nccl_version) as strings, empty if unavailable."""
    driver = ""
    cuda = ""
    nccl = ""

    try:
        import pynvml

        try:
            pynvml.nvmlInit()
        except Exception:
            # No NVML library / no NVIDIA driver — runtime info just stays empty.
            return "", "", os.environ.get("INFERENCEBENCH_NCCL_VERSION", "")
        try:
            driver = _decode(pynvml.nvmlSystemGetDriverVersion())
        except Exception:
            driver = ""
        try:
            cuda_int = pynvml.nvmlSystemGetCudaDriverVersion()
            # NVML reports CUDA as e.g. 12060 = 12.6.0; convert to dotted form
            major = cuda_int // 1000
            minor = (cuda_int % 1000) // 10
            cuda = f"{major}.{minor}"
        except Exception:
            cuda = ""
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except ImportError:
        pass

    # NCCL version is harder to read without runtime import. Phase 1: env var hint.
    nccl = os.environ.get("INFERENCEBENCH_NCCL_VERSION", "")

    return driver, cuda, nccl


def _decode(value: str | bytes) -> str:
    """NVML sometimes returns bytes (older pynvml) and sometimes str. Normalise."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
