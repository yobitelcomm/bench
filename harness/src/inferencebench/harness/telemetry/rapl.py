"""Intel/AMD RAPL energy telemetry via ``/sys/class/powercap``.

RAPL exposes monotonically-increasing energy counters per power domain
(``package`` = whole CPU socket, ``dram`` = DDR controller, ``core`` = cores,
``uncore`` = LLC + memory controller, depending on platform).

We sample the raw counters at a fixed interval; downstream code computes
energy/power by differencing successive samples.

CAP_SYS_RAWIO or RAPL group membership is typically required to read these
files on modern kernels. If reads fail, the sampler silently returns no
samples — telemetry is best-effort, never blocking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from inferencebench.harness.telemetry.base import Sampler, TelemetrySample


@dataclass(frozen=True, slots=True)
class RAPLSample(TelemetrySample):
    """One RAPL reading across all readable domains."""

    domains: tuple[dict[str, str | int], ...]
    # Each entry: name (str, e.g. "package-0", "dram"), energy_uj (int microjoules).


class RAPLSampler(Sampler):
    """Poll ``/sys/class/powercap/intel-rapl:*`` energy counters.

    Args:
        interval_ms: Polling period. RAPL counters are cumulative so the
            interval mostly affects resolution; 100 ms is a sensible default.
        powercap_root: Filesystem root for tests. Production uses /sys.
    """

    def __init__(self, interval_ms: int = 100, *, powercap_root: Path | None = None) -> None:
        super().__init__(interval_ms=interval_ms)
        self._root = powercap_root or Path("/sys/class/powercap")
        self._domain_paths: list[tuple[str, Path]] = []

    def _setup(self) -> None:
        """Enumerate readable RAPL domains."""
        if not self._root.exists():
            return
        try:
            entries = sorted(self._root.iterdir())
        except (PermissionError, OSError):
            return
        for entry in entries:
            if not entry.is_dir():
                continue
            if not entry.name.startswith("intel-rapl"):
                continue
            name_file = entry / "name"
            energy_file = entry / "energy_uj"
            if not energy_file.exists() or not name_file.exists():
                continue
            try:
                domain_name = name_file.read_text(encoding="utf-8").strip()
            except (PermissionError, OSError, UnicodeDecodeError):
                continue
            # Probe read once to verify access (energy_uj returns EACCES on locked-down hosts)
            try:
                energy_file.read_text(encoding="utf-8")
            except (PermissionError, OSError):
                continue
            self._domain_paths.append((domain_name, energy_file))

    def _one_sample(self, t_ms: float) -> RAPLSample | None:
        if not self._domain_paths:
            return None
        domains: list[dict[str, str | int]] = []
        for name, path in self._domain_paths:
            try:
                raw = path.read_text(encoding="utf-8").strip()
                energy_uj = int(raw)
            except (FileNotFoundError, PermissionError, OSError, ValueError):
                continue
            domains.append({"name": name, "energy_uj": energy_uj})
        if not domains:
            return None
        return RAPLSample(t_ms=t_ms, domains=tuple(domains))
