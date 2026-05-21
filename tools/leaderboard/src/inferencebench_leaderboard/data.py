"""Envelope loading and Pareto-frontier math for the static leaderboard.

Both helpers are pure: they do not touch disk except via the directory passed
in to :func:`load_envelopes`. Malformed JSON files and JSON blobs that fail
Pydantic validation are *skipped*, never crashed, because the leaderboard is
populated from community contributions and one bad envelope must not break
the whole site.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from inferencebench.envelope import Envelope

logger = logging.getLogger(__name__)

Direction = Literal["min", "max"]


# Convention for the leaderboard's headline metrics.  Lower is better for
# latency and price, higher is better for throughput.  The renderer uses this
# map to decide which axis direction to feed :func:`compute_pareto`.
PARETO_DIRECTIONS: dict[str, Direction] = {
    "ttft_p50_ms": "min",
    "ttft_p99_ms": "min",
    "itl_p50_ms": "min",
    "itl_p99_ms": "min",
    "throughput_tok_per_s": "max",
    "goodput_tok_per_s": "max",
    "cost_per_m_tokens_usd": "min",
    "joules_per_token": "min",
    "energy_per_token_j": "min",
}


@dataclass(frozen=True, slots=True)
class LoadedEnvelope:
    """A parsed envelope and the source filename it came from.

    The filename is kept so the rendered site can link to a stable JSON path
    (``/envelopes/<filename>``) for ``bench verify`` to re-download and check.
    """

    source_filename: str
    envelope: Envelope


def load_envelopes(envelopes_dir: Path) -> list[LoadedEnvelope]:
    """Load every ``*.json`` under ``envelopes_dir`` that parses as an Envelope.

    Files that are not valid JSON, or that parse as JSON but fail Pydantic
    validation against :class:`inferencebench.envelope.Envelope`, are logged
    at WARNING level and skipped.  This is intentional: the leaderboard
    aggregates community-submitted envelopes and one malformed file must not
    take down the rest of the site.

    Args:
        envelopes_dir: Directory containing envelope JSON files.

    Returns:
        Sorted list of successfully parsed envelopes (by source filename).
    """
    if not envelopes_dir.exists() or not envelopes_dir.is_dir():
        logger.warning("envelopes_dir does not exist or is not a directory: %s", envelopes_dir)
        return []

    loaded: list[LoadedEnvelope] = []
    for path in sorted(envelopes_dir.glob("*.json")):
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("could not read envelope file %s: %s", path, exc)
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("skipping non-JSON file %s: %s", path, exc)
            continue
        try:
            envelope = Envelope.model_validate(data)
        except (ValueError, TypeError) as exc:
            logger.warning("skipping invalid envelope %s: %s", path, exc)
            continue
        loaded.append(LoadedEnvelope(source_filename=path.name, envelope=envelope))
    return loaded


def compute_pareto(
    entries: list[tuple[float | None, float | None]],
    *,
    x_direction: Direction = "max",
    y_direction: Direction = "min",
) -> list[bool]:
    """Classify each ``(x, y)`` pair as on the Pareto frontier or dominated.

    A point ``p`` is *dominated* by another point ``q`` when ``q`` is at least
    as good as ``p`` on both axes and strictly better on at least one.  The
    direction parameters say whether higher or lower is "better" per axis:

    - ``x_direction="max", y_direction="min"`` (default): higher x and lower y
      are both improvements.  Matches the canonical
      throughput-vs-latency plot.
    - ``x_direction="min", y_direction="min"``: both axes minimize (e.g.
      latency vs. cost).
    - Any other combination is supported analogously.

    Entries containing ``None`` on either axis are never on the frontier
    (treated as missing data, marked ``False``).

    Args:
        entries: List of ``(x, y)`` coordinate tuples; ``None`` denotes missing.
        x_direction: ``"max"`` if higher x is better, ``"min"`` if lower.
        y_direction: ``"max"`` if higher y is better, ``"min"`` if lower.

    Returns:
        A list of booleans, one per input, ``True`` iff the entry is on the
        Pareto frontier.
    """

    def better_or_equal(a: float, b: float, direction: Direction) -> bool:
        return a >= b if direction == "max" else a <= b

    def strictly_better(a: float, b: float, direction: Direction) -> bool:
        return a > b if direction == "max" else a < b

    result = [False] * len(entries)
    for i, (xi, yi) in enumerate(entries):
        if xi is None or yi is None:
            continue
        dominated = False
        for j, (xj, yj) in enumerate(entries):
            if i == j or xj is None or yj is None:
                continue
            if (
                better_or_equal(xj, xi, x_direction)
                and better_or_equal(yj, yi, y_direction)
                and (strictly_better(xj, xi, x_direction) or strictly_better(yj, yi, y_direction))
            ):
                dominated = True
                break
        result[i] = not dominated
    return result
