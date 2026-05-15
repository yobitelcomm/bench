"""Memory collector. Reads /proc/meminfo + parses dmidecode if available."""

from __future__ import annotations

import re
import subprocess

from inferencebench.envelope import Memory


def collect_memory(*, dmidecode_output: str | None = None) -> Memory:
    """Return memory configuration: channels, speed in MT/s, ECC flag.

    Best-effort. ``dmidecode -t memory`` (root) is the only reliable way to
    enumerate channels and speeds. If not available we set channels=1 and
    speed_mts=1 (schema minimums) so the envelope still validates — the
    fingerprint_sha256 simply won't distinguish memory configurations on those
    hosts.

    Args:
        dmidecode_output: Override for testing. Production calls leave it None.
    """
    text = dmidecode_output if dmidecode_output is not None else _run_dmidecode()
    if not text:
        return Memory(channels=1, speed_mts=1, ecc=False)

    speeds: list[int] = []
    populated_count = 0
    ecc_seen = False

    # Each Memory Device block is delimited by blank lines.
    blocks = re.split(r"\n\s*\n", text)
    for block in blocks:
        if "Memory Device" not in block and "Size:" not in block:
            continue
        if re.search(r"Size:\s*(No Module Installed|None)", block):
            continue
        # Populated DIMM
        m_size = re.search(r"^\s*Size:\s+(\d+)\s*(MB|GB)", block, flags=re.MULTILINE)
        if not m_size:
            continue
        populated_count += 1
        m_speed = re.search(
            r"^\s*Configured Memory Speed:\s+(\d+)\s*MT/s", block, flags=re.MULTILINE
        )
        if not m_speed:
            m_speed = re.search(r"^\s*Speed:\s+(\d+)\s*MT/s", block, flags=re.MULTILINE)
        if m_speed:
            speeds.append(int(m_speed.group(1)))
        if re.search(r"^\s*Total Width:\s+(\d+)\s*bits", block, flags=re.MULTILINE):
            tw = int(re.search(r"Total Width:\s+(\d+)", block).group(1))  # type: ignore[union-attr]
            dw_match = re.search(r"^\s*Data Width:\s+(\d+)\s*bits", block, flags=re.MULTILINE)
            if dw_match:
                dw = int(dw_match.group(1))
                if tw > dw:
                    ecc_seen = True

    channels = max(populated_count, 1)
    speed = max(max(speeds), 1) if speeds else 1
    return Memory(channels=channels, speed_mts=speed, ecc=ecc_seen)


def _run_dmidecode() -> str | None:
    """Run dmidecode -t memory if present + readable. Otherwise None."""
    try:
        result = subprocess.run(
            ["dmidecode", "-t", "memory"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout
