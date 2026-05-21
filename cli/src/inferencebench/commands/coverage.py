"""``bench coverage`` — report metric completeness per envelope.

Each plugin declares an ``EXPECTED_METRICS`` tuple at its module root: the
metric names a healthy run is supposed to populate. ``bench coverage`` walks
one envelope file or a directory of envelopes, looks up the producing plugin
via the ``inferencebench.plugins`` entry-point group, and prints a Rich
table showing — for every envelope — how many of the plugin's expected
metrics actually landed (non-``None``).

The headline use case is catching silent failures: NVML samples failed →
``power_avg_w`` never landed → coverage drops below 100% even though the
envelope itself validates. ``--threshold`` turns this into a CI gate.
"""

from __future__ import annotations

import importlib
import json
from importlib import metadata
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from inferencebench.envelope import Envelope

console = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Command                                                                     #
# --------------------------------------------------------------------------- #
def coverage(
    path: Annotated[
        Path,
        typer.Argument(
            help="Envelope JSON file, or a directory to recursively scan for *.json envelopes.",
        ),
    ],
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit a JSON document on stdout instead of a Rich table (for piping into jq).",
        ),
    ] = False,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help=(
                "Minimum acceptable coverage fraction (0.0-1.0). When any envelope "
                "falls below this, the command exits 1 — useful as a CI gate."
            ),
            min=0.0,
            max=1.0,
        ),
    ] = 0.8,
) -> None:
    """Report metric completeness for one or more envelopes.

    Loads every envelope under ``path`` (or a single file), resolves the
    expected metric set from the producing plugin's ``EXPECTED_METRICS``
    constant, and reports the fraction of expected metrics that are present
    and non-``None`` in ``envelope.metrics``.
    """
    if not path.exists():
        err_console.print(f"[red]Path not found:[/red] {path}")
        raise typer.Exit(code=2)

    candidates = _collect_json_files(path)
    plugin_modules = _load_plugin_modules()

    rows = _build_rows(candidates, plugin_modules)
    # Sort worst coverage first — that's where users will look.
    rows.sort(key=lambda r: (r["coverage_pct"], r["filename"]))

    if json_output:
        _emit_json(rows, threshold=threshold)
    else:
        _emit_table(rows)

    below = [r for r in rows if r["coverage_pct"] < threshold * 100.0]
    if below:
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# Plugin discovery                                                            #
# --------------------------------------------------------------------------- #
def _load_plugin_modules() -> dict[str, tuple[str, tuple[str, ...]]]:
    """Map entry-point name → (top-level module, EXPECTED_METRICS).

    The top-level module is parsed out of the ``module:attr`` entry-point
    value so we can ``importlib.import_module(top_pkg)`` and read its
    ``EXPECTED_METRICS`` constant. An entry point whose module lacks the
    constant collapses to an empty tuple — coverage degrades gracefully
    rather than crashing.
    """
    try:
        eps = metadata.entry_points(group="inferencebench.plugins")
    except TypeError:  # pre-3.10 fallback shape
        eps = metadata.entry_points().get(  # type: ignore[attr-defined]
            "inferencebench.plugins", []
        )

    out: dict[str, tuple[str, tuple[str, ...]]] = {}
    for ep in eps:
        # ep.value is "inferencebench_llm.plugin:LLMInferencePlugin"; the
        # top package is everything before the first dot.
        module_path = ep.value.split(":", 1)[0]
        top_pkg = module_path.split(".", 1)[0]
        try:
            mod = importlib.import_module(top_pkg)
        except ImportError:
            continue
        expected = getattr(mod, "EXPECTED_METRICS", ())
        if not isinstance(expected, tuple):
            expected = tuple(expected)
        out[ep.name] = (top_pkg, expected)
    return out


def _plugin_for_suite_id(
    suite_id: str,
    plugin_modules: dict[str, tuple[str, tuple[str, ...]]],
) -> tuple[str, tuple[str, ...]] | None:
    """Resolve a plugin by matching the envelope's ``suite_id`` against entry-point names.

    ``envelope.suite_id`` is the per-benchmark id (``llm.inference.rag-style``);
    entry-point names are the plugin suite id (``llm.inference``). We match
    via prefix so the longest entry-point name wins (avoids ``llm.mt`` ever
    being matched as a prefix of ``llm.mt-extended.foo`` by accident).
    """
    matches = sorted(
        (name for name in plugin_modules if suite_id == name or suite_id.startswith(name + ".")),
        key=len,
        reverse=True,
    )
    if not matches:
        return None
    name = matches[0]
    return plugin_modules[name]


# --------------------------------------------------------------------------- #
# Envelope loading                                                            #
# --------------------------------------------------------------------------- #
def _collect_json_files(path: Path) -> list[Path]:
    """Return every ``*.json`` file under ``path`` (recursive) or just ``[path]``."""
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.json") if p.is_file())


def _load_envelope(path: Path) -> Envelope | None:
    """Try to load an envelope; return ``None`` if the file isn't a valid envelope."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Envelope.model_validate(raw)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Row construction                                                            #
# --------------------------------------------------------------------------- #
def _build_rows(
    candidates: list[Path],
    plugin_modules: dict[str, tuple[str, tuple[str, ...]]],
) -> list[dict[str, Any]]:
    """Compute one coverage row per envelope."""
    rows: list[dict[str, Any]] = []
    for path in candidates:
        env = _load_envelope(path)
        if env is None:
            continue
        resolved = _plugin_for_suite_id(env.suite_id, plugin_modules)
        if resolved is None:
            expected: tuple[str, ...] = ()
        else:
            _, expected = resolved
        found, missing = _split_present_missing(env, expected)
        coverage_pct = 100.0 * len(found) / len(expected) if expected else 100.0
        rows.append(
            {
                "filename": path.name,
                "path": str(path),
                "suite": env.suite_id,
                "expected": list(expected),
                "expected_count": len(expected),
                "found": found,
                "found_count": len(found),
                "missing": missing,
                "coverage_pct": coverage_pct,
            }
        )
    return rows


def _split_present_missing(env: Envelope, expected: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Bucket ``expected`` into (present-and-non-None, missing-or-None)."""
    found: list[str] = []
    missing: list[str] = []
    for name in expected:
        value = env.metrics.get(name)
        if value is None:
            missing.append(name)
        else:
            found.append(name)
    return found, missing


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def _coverage_cell(coverage_pct: float) -> str:
    """Colour the coverage cell red/yellow/green by band."""
    if coverage_pct >= 99.999:
        return f"[green]{coverage_pct:.1f}%[/green]"
    if coverage_pct >= 80.0:
        return f"[yellow]{coverage_pct:.1f}%[/yellow]"
    return f"[red]{coverage_pct:.1f}%[/red]"


def _emit_table(rows: list[dict[str, Any]]) -> None:
    """Render the coverage rows as a Rich table sorted worst-first."""
    table = Table(
        title="Envelope metric coverage",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Filename")
    table.add_column("Suite")
    table.add_column("Expected", justify="right")
    table.add_column("Found", justify="right")
    table.add_column("Missing")
    table.add_column("Coverage", justify="right")

    for row in rows:
        missing_label = ", ".join(row["missing"]) if row["missing"] else "-"
        table.add_row(
            row["filename"],
            row["suite"],
            str(row["expected_count"]),
            str(row["found_count"]),
            missing_label,
            _coverage_cell(row["coverage_pct"]),
        )

    console.print(table)


def _emit_json(rows: list[dict[str, Any]], *, threshold: float) -> None:
    """Emit the coverage report as a JSON document on stdout."""
    payload = {
        "threshold": threshold,
        "envelopes": [
            {
                "filename": r["filename"],
                "path": r["path"],
                "suite": r["suite"],
                "expected": r["expected"],
                "expected_count": r["expected_count"],
                "found": r["found"],
                "found_count": r["found_count"],
                "missing": r["missing"],
                "coverage_pct": r["coverage_pct"],
            }
            for r in rows
        ],
    }
    console.print_json(data=payload)
