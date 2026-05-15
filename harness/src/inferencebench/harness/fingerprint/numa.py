"""NUMA topology collector. Reads /sys/devices/system/node/."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def collect_numa(*, sysfs_root: Path | None = None) -> dict[str, Any]:
    """Return a canonical NUMA topology dict.

    Form: ``{"nodes": [{"id": int, "cpus": [int, ...], "memory_mb": int}, ...]}``.
    Empty dict on non-NUMA systems or when /sys is unreadable.
    """
    root = (sysfs_root or Path("/sys")) / "devices" / "system" / "node"
    if not root.exists():
        return {}

    nodes: list[dict[str, Any]] = []
    try:
        node_dirs = sorted(
            (d for d in root.iterdir() if d.is_dir() and d.name.startswith("node")),
            key=lambda d: d.name,
        )
    except (PermissionError, OSError):
        return {}

    for node_dir in node_dirs:
        try:
            node_id = int(node_dir.name[len("node") :])
        except ValueError:
            continue
        cpus = _parse_cpulist(_safe_read(node_dir / "cpulist"))
        meminfo = _parse_meminfo_total(_safe_read(node_dir / "meminfo"))
        nodes.append({"id": node_id, "cpus": cpus, "memory_mb": meminfo})

    if not nodes:
        return {}
    return {"nodes": nodes}


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError, UnicodeDecodeError):
        return ""


def _parse_cpulist(s: str) -> list[int]:
    """Parse a Linux cpulist string like "0-3,8-11" into [0,1,2,3,8,9,10,11]."""
    s = s.strip()
    if not s:
        return []
    result: list[int] = []
    for chunk in s.split(","):
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            try:
                result.extend(range(int(lo), int(hi) + 1))
            except ValueError:
                continue
        else:
            try:
                result.append(int(chunk))
            except ValueError:
                continue
    return sorted(set(result))


def _parse_meminfo_total(s: str) -> int:
    """Parse "Node N MemTotal: xxx kB" from /sys node/meminfo. Returns MB."""
    for line in s.splitlines():
        if "MemTotal" in line:
            # e.g. "Node 0 MemTotal:       65536000 kB"
            parts = line.split()
            try:
                idx = parts.index("MemTotal:")
                kb = int(parts[idx + 1])
                return kb // 1024
            except (ValueError, IndexError):
                continue
    return 0
