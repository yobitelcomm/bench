"""CPU collector. Parses /proc/cpuinfo on Linux; falls back to stdlib elsewhere."""

from __future__ import annotations

import platform
from pathlib import Path

from inferencebench.envelope import CPU


def collect_cpu(*, proc_cpuinfo: Path | None = None) -> CPU:
    """Return CPU model + microcode revision.

    Linux: reads /proc/cpuinfo for ``model name`` and ``microcode``.
    Other platforms: falls back to ``platform.processor()`` and microcode="unknown".
    """
    path = proc_cpuinfo or Path("/proc/cpuinfo")
    model = ""
    microcode = ""
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError, UnicodeDecodeError):
        text = ""

    for line in text.splitlines():
        if not line.strip():
            # First processor block done; cpuinfo repeats per logical core.
            if model and microcode:
                break
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "model name" and not model:
            model = value
        elif key == "microcode" and not microcode:
            microcode = value

    if not model:
        # Fallback for macOS / Windows / blocked /proc
        model = platform.processor() or platform.machine() or "unknown"
    if not microcode:
        microcode = "unknown"

    return CPU(model=model, microcode=microcode)
