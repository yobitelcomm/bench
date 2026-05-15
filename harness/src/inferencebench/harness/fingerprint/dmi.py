"""DMI + BIOS collectors. Reads ``/sys/class/dmi/id/*`` on Linux."""

from __future__ import annotations

from pathlib import Path

from inferencebench.envelope import BIOS


def collect_dmi_uuid(sysfs_root: Path) -> str:
    """Return the system's DMI UUID, or a stable placeholder if unavailable.

    On most cloud VMs ``product_uuid`` is readable but may require root;
    on locked-down systems it returns "Not Specified" or similar. We accept
    any string but never blank — that satisfies the envelope schema's
    ``min_length=1`` constraint.
    """
    path = sysfs_root / "class" / "dmi" / "id" / "product_uuid"
    value = _read_first_line(path)
    if not value:
        return "unknown"
    return value


def collect_bios(sysfs_root: Path) -> BIOS:
    """Read BIOS version + the two flags we care about.

    ``resizable_bar`` and ``above_4g`` aren't directly exposed in ``/sys``
    on most systems; we infer from kernel cmdline + dmesg in Phase 2.
    For Phase 1 we mark them ``False`` unconditionally on CPU-only / cloud VMs
    and let the GPU detection step (Phase 2) refine the inference.
    """
    bios_root = sysfs_root / "class" / "dmi" / "id"
    version = _read_first_line(bios_root / "bios_version") or "unknown"
    return BIOS(
        version=version,
        resizable_bar=False,
        above_4g=False,
    )


def _read_first_line(path: Path) -> str:
    """Read the first line of ``path``, trimmed. Returns "" on any error."""
    try:
        return path.read_text(encoding="utf-8").splitlines()[0].strip()
    except (FileNotFoundError, PermissionError, IndexError, OSError, UnicodeDecodeError):
        return ""
