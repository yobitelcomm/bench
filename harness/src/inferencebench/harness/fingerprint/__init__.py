"""Hardware fingerprint collection.

Reads DMI, GPU, CPU, memory, BIOS, driver, CUDA, NCCL info from the running
host and assembles them into an ``inferencebench.envelope.HardwareFingerprint``
model. The fingerprint's SHA-256 is computed from the canonical body and
embedded in the model for fast equality checks.

Phase 1 supports Linux only (reads ``/sys`` and ``/proc``). Apple Silicon and
Windows fall back to best-effort values; missing data becomes sensible defaults.

Public API::

    from inferencebench.harness.fingerprint import (
        collect_hardware_fingerprint,
        collect_software_provenance,
    )

    hw = collect_hardware_fingerprint()  # HardwareFingerprint model
    sw = collect_software_provenance()  # SoftwareProvenance model
"""

from inferencebench.harness.fingerprint.collect import (
    collect_hardware_fingerprint,
    collect_software_provenance,
)

__all__ = ["collect_hardware_fingerprint", "collect_software_provenance"]
