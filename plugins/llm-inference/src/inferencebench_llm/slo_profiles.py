"""Hardware-class detection + per-class SLO multipliers.

A 7B model on an H100 should be held to a stricter SLO than the same model on
an M2 Pro. This module exposes a coarse hardware taxonomy and a
:func:`classify` function that maps a :class:`HardwareFingerprint` to a
:class:`HardwareClass` so the plugin can scale its SLO thresholds accordingly.

The multipliers are anchored to the H100 (1.0x); a class with multiplier
``1.5`` gets 1.5x more lenient thresholds, etc.
"""

from __future__ import annotations

from dataclasses import dataclass

from inferencebench.envelope import HardwareFingerprint


@dataclass(frozen=True)
class HardwareClass:
    """A coarse category used to pick SLO thresholds."""

    key: str  # "h100", "a100", "rtx-4090", "m-series", "cpu"
    description: str
    # Multipliers applied to the *base* SLO numbers; higher = more lenient.
    ttft_mult: float
    tpot_mult: float
    total_mult: float


# Order matters: first match wins. Substring matching is case-insensitive.
HARDWARE_CLASSES: tuple[HardwareClass, ...] = (
    HardwareClass("h200", "NVIDIA H200 (data center)", 0.6, 0.6, 0.6),
    HardwareClass("h100", "NVIDIA H100 (data center)", 1.0, 1.0, 1.0),  # the anchor
    HardwareClass("a100", "NVIDIA A100 (data center)", 1.5, 1.5, 1.5),
    HardwareClass("l4", "NVIDIA L4 (server inference)", 2.5, 2.5, 2.5),
    HardwareClass("rtx-5090", "NVIDIA RTX 5090 (consumer 2026)", 1.2, 1.2, 1.2),
    HardwareClass("rtx-4090", "NVIDIA RTX 4090 (consumer)", 1.8, 1.8, 1.8),
    HardwareClass("rtx-4080", "NVIDIA RTX 4080 (consumer Ada)", 2.2, 2.2, 2.2),
    HardwareClass("rtx-4070", "NVIDIA RTX 4070 (consumer Ada)", 2.8, 2.8, 2.8),
    HardwareClass("rtx-3090", "NVIDIA RTX 3090 (consumer)", 3.0, 3.0, 3.0),
    HardwareClass(
        "rtx-ada-workstation",
        "NVIDIA RTX Ada Workstation (5000/6000 desktop)",
        2.0,
        2.0,
        2.0,
    ),
    HardwareClass(
        "rtx-ada-laptop",
        "NVIDIA RTX Ada Laptop (3000/4000/5000 mobile)",
        5.0,
        5.0,
        5.0,
    ),
    HardwareClass("mi300x", "AMD MI300X (data center)", 1.2, 1.2, 1.2),
    HardwareClass("m-series", "Apple Silicon (M1/M2/M3/M4)", 5.0, 5.0, 5.0),
    HardwareClass("cpu", "CPU-only", 20.0, 20.0, 20.0),
)


# Map class keys to the substrings we look for in the GPU model name.
# Stored as a tuple of (key, substrings_to_match). Substrings are matched
# case-insensitively. The first class whose substrings appear wins.
_GPU_MATCHERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("h200", ("h200",)),
    ("h100", ("h100",)),
    ("a100", ("a100",)),
    ("l4", ("l4",)),
    ("rtx-5090", ("rtx 5090", "rtx-5090", "rtx5090")),
    ("rtx-4090", ("rtx 4090", "rtx-4090", "rtx4090")),
    ("rtx-4080", ("rtx 4080", "rtx-4080", "rtx4080")),
    ("rtx-4070", ("rtx 4070", "rtx-4070", "rtx4070")),
    ("rtx-3090", ("rtx 3090", "rtx-3090", "rtx3090")),
    # Ada Laptop GPUs report names like "RTX 4000 Ada Generation Laptop GPU".
    # Match the "ada" + "laptop" combo first so it wins over the workstation rule.
    ("rtx-ada-laptop", ("ada generation laptop", "ada laptop")),
    (
        "rtx-ada-workstation",
        (
            "rtx 5000 ada",
            "rtx 6000 ada",
            "rtx 4500 ada",
            "rtx 4000 ada",
            "ada generation",
        ),
    ),
    ("mi300x", ("mi300x",)),
)


def _class_by_key(key: str) -> HardwareClass:
    """Look up a :class:`HardwareClass` by its ``key``; raises if unknown."""
    for cls in HARDWARE_CLASSES:
        if cls.key == key:
            return cls
    msg = f"unknown hardware class key: {key!r}"
    raise KeyError(msg)


def _match_gpu(model: str) -> HardwareClass | None:
    """Return the matching :class:`HardwareClass` for a GPU model string."""
    lowered = model.lower()
    for key, needles in _GPU_MATCHERS:
        for needle in needles:
            if needle in lowered:
                return _class_by_key(key)
    return None


def classify(fingerprint: HardwareFingerprint) -> HardwareClass:
    """Classify a :class:`HardwareFingerprint` into a :class:`HardwareClass`.

    Inspects ``fingerprint.gpus[0].model`` (case-insensitive substring match)
    first. If there is no GPU, or the GPU model string is empty/unknown, falls
    back to checking the CPU model — Apple Silicon (``"Apple M"``) maps to
    ``m-series`` and anything else maps to ``cpu``.
    """
    if fingerprint.gpus:
        primary = fingerprint.gpus[0].model.strip()
        if primary:
            matched = _match_gpu(primary)
            if matched is not None:
                return matched
    # Fall back to CPU inspection.
    cpu_model = fingerprint.cpu.model.strip()
    if "apple m" in cpu_model.lower():
        return _class_by_key("m-series")
    return _class_by_key("cpu")


def scale_slos(
    base: list,  # type: ignore[type-arg]
    hw_class: HardwareClass,
) -> list:  # type: ignore[type-arg]
    """Return a new list of :class:`SLOPredicate` with thresholds rescaled.

    Imported lazily to avoid a hard dependency on :mod:`inferencebench.harness`
    at module-import time.
    """
    from inferencebench.harness.metrics import SLOPredicate

    mult_for: dict[str, float] = {
        "ttft": hw_class.ttft_mult,
        "tpot": hw_class.tpot_mult,
        "total": hw_class.total_mult,
    }
    rescaled: list[SLOPredicate] = []
    for slo in base:
        mult = mult_for.get(slo.name, 1.0)
        rescaled.append(
            SLOPredicate(
                name=slo.name,
                field=slo.field,
                op=slo.op,
                value=slo.value * mult,
            )
        )
    return rescaled


def format_resolved(slos: list) -> str:  # type: ignore[type-arg]
    """Render a list of :class:`SLOPredicate` as a compact human-readable string.

    Example: ``"ttft<300ms, tpot<75ms, total<4500ms"``. Whole-number thresholds
    are emitted without a trailing ``.0`` so the formatting matches the values
    a user would type by hand.
    """
    parts: list[str] = []
    for slo in slos:
        v = slo.value
        if float(v).is_integer():
            num = f"{int(v)}"
        else:
            num = f"{v:g}"
        # Field names end with ``_ms`` for the latency SLOs; surface the unit.
        unit = "ms" if str(slo.field).endswith("_ms") else ""
        parts.append(f"{slo.name}{slo.op}{num}{unit}")
    return ", ".join(parts)
