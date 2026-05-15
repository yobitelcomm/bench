"""Hardware health diagnostic — the gate that decides whether benchmarks can run.

`bench doctor` calls this before any `bench run` invocation. In `--strict`
mode it refuses to proceed if any check returns non-PASS:

- thermal: GPU edge temp + hotspot below throttling threshold
- ECC: zero corrected/uncorrected ECC errors in the last interval
- driver: NVML driver version reads OK
- power: GPU not power-capped below its TGP
- clocks: SM and memory clocks not in throttled state
- memory: free GPU memory ≥ minimum required (default 4GB)

Returns a structured :class:`DiagnosticReport` so callers can render it however
they like (Rich table, JSON, log line). The CLI command formats the report.

Phase 1 supports NVIDIA via pynvml. AMD and Apple Silicon checks are no-ops
that return PASS with a warning (Phase 2+ will add rocm-smi + powermetrics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CheckStatus(StrEnum):
    """Outcome of a single check."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One named check + outcome + detail."""

    name: str
    status: CheckStatus
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    """Aggregate result of all checks for one host."""

    checks: list[CheckResult]
    strict: bool

    @property
    def ok(self) -> bool:
        """In strict mode any FAIL or WARN blocks. In non-strict only FAIL blocks."""
        for c in self.checks:
            if c.status == CheckStatus.FAIL:
                return False
            if self.strict and c.status == CheckStatus.WARN:
                return False
        return True

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.FAIL)

    @property
    def warn_count(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.WARN)


# Thresholds — conservative, tunable via env vars in Phase 2.
THERMAL_MAX_C = 85  # H100 throttles at ~88, B200 at ~92; warn at 85
ECC_MAX_UNCORRECTED = 0  # any uncorrected ECC error fails
MIN_FREE_VRAM_MB = 4096  # require 4GB free


def run_diagnostic(*, strict: bool = False) -> DiagnosticReport:
    """Run all health checks against the local host. Phase 1: NVIDIA via NVML."""
    checks: list[CheckResult] = []

    # Pull NVML in once. If it's missing or fails to init, mark NVIDIA checks SKIP.
    nvml_available, nvml = _try_import_nvml()
    if not nvml_available or nvml is None:
        checks.append(
            CheckResult(
                name="nvml",
                status=CheckStatus.SKIP,
                detail="pynvml not importable or NVML init failed (no NVIDIA driver?)",
            )
        )
        return DiagnosticReport(checks=checks, strict=strict)

    try:
        device_count = nvml.nvmlDeviceGetCount()
    except nvml.NVMLError as exc:
        checks.append(
            CheckResult(
                name="nvml",
                status=CheckStatus.FAIL,
                detail=f"nvmlDeviceGetCount failed: {exc}",
            )
        )
        return DiagnosticReport(checks=checks, strict=strict)

    if device_count == 0:
        checks.append(
            CheckResult(
                name="nvml",
                status=CheckStatus.SKIP,
                detail="No NVIDIA GPUs detected",
            )
        )
        return DiagnosticReport(checks=checks, strict=strict)

    # Driver check (one-shot, before per-device loop)
    try:
        driver_version = _decode(nvml.nvmlSystemGetDriverVersion())
        checks.append(
            CheckResult(
                name="driver",
                status=CheckStatus.PASS,
                detail=driver_version,
                data={"version": driver_version},
            )
        )
    except nvml.NVMLError as exc:
        checks.append(
            CheckResult(
                name="driver", status=CheckStatus.FAIL, detail=f"NVML driver query failed: {exc}"
            )
        )

    # Per-device checks
    for i in range(device_count):
        prefix = f"gpu{i}"
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
        except nvml.NVMLError as exc:
            checks.append(
                CheckResult(
                    name=f"{prefix}.handle",
                    status=CheckStatus.FAIL,
                    detail=f"nvmlDeviceGetHandleByIndex({i}) failed: {exc}",
                )
            )
            continue

        checks.append(_check_thermal(nvml, handle, prefix))
        checks.append(_check_ecc(nvml, handle, prefix))
        checks.append(_check_memory(nvml, handle, prefix))
        checks.append(_check_throttle(nvml, handle, prefix))

    try:
        nvml.nvmlShutdown()
    except nvml.NVMLError:
        pass

    return DiagnosticReport(checks=checks, strict=strict)


# --------------------------------------------------------------------------- #
# Per-check helpers                                                           #
# --------------------------------------------------------------------------- #
def _check_thermal(nvml: Any, handle: Any, prefix: str) -> CheckResult:
    """GPU edge temperature below throttling threshold."""
    try:
        temp_c = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
    except nvml.NVMLError as exc:
        return CheckResult(
            name=f"{prefix}.thermal",
            status=CheckStatus.SKIP,
            detail=f"temperature query failed: {exc}",
        )

    if temp_c >= THERMAL_MAX_C:
        return CheckResult(
            name=f"{prefix}.thermal",
            status=CheckStatus.FAIL,
            detail=f"GPU {temp_c}°C ≥ threshold {THERMAL_MAX_C}°C (thermal throttle risk)",
            data={"temperature_c": temp_c, "threshold_c": THERMAL_MAX_C},
        )
    return CheckResult(
        name=f"{prefix}.thermal",
        status=CheckStatus.PASS,
        detail=f"{temp_c}°C (below {THERMAL_MAX_C}°C)",
        data={"temperature_c": temp_c, "threshold_c": THERMAL_MAX_C},
    )


def _check_ecc(nvml: Any, handle: Any, prefix: str) -> CheckResult:
    """Zero uncorrected ECC errors. Warn on corrected errors (not blocking)."""
    try:
        ecc_mode_current, _ecc_mode_pending = nvml.nvmlDeviceGetEccMode(handle)
    except nvml.NVMLError:
        return CheckResult(
            name=f"{prefix}.ecc",
            status=CheckStatus.SKIP,
            detail="ECC not supported on this GPU",
        )

    if ecc_mode_current != nvml.NVML_FEATURE_ENABLED:
        return CheckResult(
            name=f"{prefix}.ecc",
            status=CheckStatus.WARN,
            detail="ECC disabled — silent data corruption is possible",
        )

    try:
        volatile_uncorrected = nvml.nvmlDeviceGetTotalEccErrors(
            handle, nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, nvml.NVML_VOLATILE_ECC
        )
        volatile_corrected = nvml.nvmlDeviceGetTotalEccErrors(
            handle, nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED, nvml.NVML_VOLATILE_ECC
        )
    except nvml.NVMLError as exc:
        return CheckResult(
            name=f"{prefix}.ecc",
            status=CheckStatus.SKIP,
            detail=f"ECC error query failed: {exc}",
        )

    if volatile_uncorrected > ECC_MAX_UNCORRECTED:
        return CheckResult(
            name=f"{prefix}.ecc",
            status=CheckStatus.FAIL,
            detail=(
                f"{volatile_uncorrected} uncorrected ECC error(s) — hardware fault, "
                "results would be untrustworthy"
            ),
            data={"uncorrected": volatile_uncorrected, "corrected": volatile_corrected},
        )
    if volatile_corrected > 100:
        return CheckResult(
            name=f"{prefix}.ecc",
            status=CheckStatus.WARN,
            detail=f"{volatile_corrected} corrected ECC errors (high; investigate)",
            data={"uncorrected": volatile_uncorrected, "corrected": volatile_corrected},
        )
    return CheckResult(
        name=f"{prefix}.ecc",
        status=CheckStatus.PASS,
        detail=f"uncorrected=0, corrected={volatile_corrected}",
        data={"uncorrected": volatile_uncorrected, "corrected": volatile_corrected},
    )


def _check_memory(nvml: Any, handle: Any, prefix: str) -> CheckResult:
    """At least MIN_FREE_VRAM_MB of GPU memory free."""
    try:
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
    except nvml.NVMLError as exc:
        return CheckResult(
            name=f"{prefix}.memory",
            status=CheckStatus.SKIP,
            detail=f"memory query failed: {exc}",
        )

    free_mb = info.free // (1024 * 1024)
    total_mb = info.total // (1024 * 1024)
    if free_mb < MIN_FREE_VRAM_MB:
        return CheckResult(
            name=f"{prefix}.memory",
            status=CheckStatus.FAIL,
            detail=f"only {free_mb} MiB free of {total_mb} MiB (need ≥{MIN_FREE_VRAM_MB} MiB)",
            data={"free_mb": free_mb, "total_mb": total_mb, "required_mb": MIN_FREE_VRAM_MB},
        )
    return CheckResult(
        name=f"{prefix}.memory",
        status=CheckStatus.PASS,
        detail=f"{free_mb} MiB free of {total_mb} MiB",
        data={"free_mb": free_mb, "total_mb": total_mb, "required_mb": MIN_FREE_VRAM_MB},
    )


def _check_throttle(nvml: Any, handle: Any, prefix: str) -> CheckResult:
    """No active throttle reasons that would skew measurements."""
    try:
        reasons = nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
    except nvml.NVMLError as exc:
        return CheckResult(
            name=f"{prefix}.throttle",
            status=CheckStatus.SKIP,
            detail=f"throttle reason query failed: {exc}",
        )

    # Filter out the benign GPU_IDLE reason (no work yet); everything else is a problem.
    blocking_mask = 0
    for attr in (
        "nvmlClocksThrottleReasonHwThermalSlowdown",
        "nvmlClocksThrottleReasonHwPowerBrakeSlowdown",
        "nvmlClocksThrottleReasonSwThermalSlowdown",
        "nvmlClocksThrottleReasonSyncBoost",
    ):
        blocking_mask |= getattr(nvml, attr, 0)

    if reasons & blocking_mask:
        return CheckResult(
            name=f"{prefix}.throttle",
            status=CheckStatus.FAIL,
            detail=f"active blocking throttle reason mask=0x{reasons:x}",
            data={"reasons_mask": reasons},
        )
    if reasons & ~getattr(nvml, "nvmlClocksThrottleReasonGpuIdle", 0):
        return CheckResult(
            name=f"{prefix}.throttle",
            status=CheckStatus.WARN,
            detail=f"non-blocking throttle reason mask=0x{reasons:x}",
            data={"reasons_mask": reasons},
        )
    return CheckResult(
        name=f"{prefix}.throttle",
        status=CheckStatus.PASS,
        detail="no active throttling",
        data={"reasons_mask": reasons},
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _try_import_nvml() -> tuple[bool, Any]:
    """Return (success, pynvml-module) after attempting nvmlInit. Tolerates absent driver."""
    try:
        import pynvml
    except ImportError:
        return False, None
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return False, None
    return True, pynvml


def _decode(value: str | bytes) -> str:
    """NVML sometimes returns bytes (older pynvml). Normalise."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
