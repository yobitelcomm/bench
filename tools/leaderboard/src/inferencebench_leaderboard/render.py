"""Render a static leaderboard site from a directory of signed envelopes.

The output layout is::

    out_dir/
      index.html                  — top-level category index
      static/site.css             — shared stylesheet
      static/sort.js              — tiny vanilla-JS table sorter (~40 LOC)
      envelopes/<filename>.json   — verbatim copies of input envelopes
      <suite_id>/index.html       — per-category table
      <suite_id>/<run_id>.html    — per-entry detail page
      data/leaderboard.json       — machine-readable index of all entries

The renderer intentionally produces *only* HTML, CSS, JSON, and a small
plain-JS sorter; no frameworks, no bundler, no build step on the GitHub Pages
side.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from inferencebench.envelope import Envelope
from inferencebench_leaderboard.data import (
    PARETO_DIRECTIONS,
    LoadedEnvelope,
    compute_pareto,
    load_envelopes,
)

logger = logging.getLogger(__name__)


# Metric keys we show in the per-category table.  Order is the column order.
# Cells are omitted (rendered as "—") for any metric the envelope didn't set,
# so adding rows here is safe even if older envelopes pre-date the metric.
HEADLINE_METRICS: list[tuple[str, str]] = [
    ("ttft_p50_ms", "TTFT P50 (ms)"),
    ("ttft_p99_ms", "TTFT P99 (ms)"),
    ("throughput_tok_per_s", "Throughput (tok/s)"),
    ("cost_per_m_tokens_usd", "$/M tokens"),
    ("joules_per_token", "J/token"),
    # NVML telemetry (added 2026-05-25). Present on any plugin that wraps
    # its run loop with NVMLSampler+RAPLSampler — currently llm-inference
    # and voice-transcription.
    ("power_avg_w", "Power avg (W)"),
    ("power_peak_w", "Power peak (W)"),
    # ASR-specific (voice-transcription only).
    ("wer_mean", "WER mean"),
    ("joules_per_audio_second", "J / audio s"),
]

# Pareto axes used to flag frontier entries in the per-category table.
PARETO_X_METRIC = "throughput_tok_per_s"
PARETO_Y_METRIC = "ttft_p50_ms"


@dataclass(frozen=True, slots=True)
class SiteRenderResult:
    """Summary of one :func:`render_site` invocation.

    Attributes:
        out_dir: Directory the site was written to.
        envelopes_loaded: Number of envelopes that parsed successfully.
        envelopes_skipped: Number of files that failed parse/validation.
        categories: Mapping of ``suite_id -> entry count`` in the rendered site.
        pages_written: Total HTML/JSON pages emitted (informational).
    """

    out_dir: Path
    envelopes_loaded: int
    envelopes_skipped: int
    categories: dict[str, int] = field(default_factory=dict)
    pages_written: int = 0


def _hardware_class(env: Envelope) -> str:
    """Compact label for the hardware row in the table."""
    gpus = env.hardware_fingerprint.gpus
    if not gpus:
        return env.hardware_fingerprint.cpu.model
    models = [g.model for g in gpus]
    if all(m == models[0] for m in models):
        return f"{len(models)}x {models[0]}"
    return ", ".join(models)


def _entry_payload(loaded: LoadedEnvelope, pareto: bool) -> dict[str, Any]:
    """Reduce an envelope to the dict the templates iterate over."""
    env = loaded.envelope
    return {
        "run_id": env.run_id,
        "suite_id": env.suite_id,
        "model_id": env.model.id,
        "model_revision": env.model.revision,
        "model_provider": env.model.provider,
        "engine_name": env.engine.name,
        "engine_version": env.engine.version,
        "hardware_class": _hardware_class(env),
        "quantization": env.quantization.format if env.quantization else "",
        "metrics": {key: env.metrics.get(key) for key, _ in HEADLINE_METRICS},
        "all_metrics": dict(env.metrics),
        "envelope_filename": loaded.source_filename,
        "signed": env.signature is not None,
        "timestamp": env.timestamp.isoformat(),
        "warnings": list(env.warnings),
        "on_pareto": pareto,
    }


def _format_metric(value: float | int | str | None) -> str:
    """Human-readable metric cell. Strings pass through unchanged."""
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return f"{value:,}"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:,.2f}"
    return f"{value:.4f}"


def _make_env() -> Environment:
    env = Environment(
        loader=PackageLoader("inferencebench_leaderboard", "templates"),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["fmt_metric"] = _format_metric
    return env


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _copy_static_assets(package_dir: Path, out_dir: Path) -> int:
    """Copy bundled static CSS + JS into ``out_dir/static/``. Returns file count."""
    src = package_dir / "static"
    dst = out_dir / "static"
    dst.mkdir(parents=True, exist_ok=True)
    written = 0
    if src.exists():
        for entry in src.iterdir():
            if entry.is_file():
                shutil.copy2(entry, dst / entry.name)
                written += 1
    return written


def _copy_raw_envelopes(loaded: list[LoadedEnvelope], out_dir: Path) -> None:
    """Copy raw envelope JSONs so ``bench verify`` can fetch them by URL."""
    dst = out_dir / "envelopes"
    dst.mkdir(parents=True, exist_ok=True)
    for item in loaded:
        payload = item.envelope.model_dump(mode="json")
        (dst / item.source_filename).write_text(
            json.dumps(payload, sort_keys=True, indent=2),
            encoding="utf-8",
        )


def _normalize_base_url(base_url: str) -> str:
    """Ensure base_url starts and ends with exactly one slash."""
    if not base_url:
        return "/"
    if not base_url.startswith("/") and "://" not in base_url:
        base_url = "/" + base_url
    if not base_url.endswith("/"):
        base_url = base_url + "/"
    return base_url


def render_site(
    envelopes_dir: Path,
    out_dir: Path,
    *,
    base_url: str = "/",
) -> SiteRenderResult:
    """Render a static leaderboard site.

    Args:
        envelopes_dir: Directory holding canonical envelope JSON files.
        out_dir: Destination directory for the generated site.  Created if
            absent; existing files in matching paths are overwritten.
        base_url: URL prefix the site will be served from (``"/"`` for
            ``yobitelcomm.github.io/bench`` root; ``"/bench/"`` if served
            under a subpath).

    Returns:
        :class:`SiteRenderResult` summarizing the render.
    """
    envelopes_dir = Path(envelopes_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = _normalize_base_url(base_url)

    # Count attempted files vs. parseable to compute the "skipped" delta.
    file_count = sum(1 for _ in envelopes_dir.glob("*.json")) if envelopes_dir.is_dir() else 0
    loaded = load_envelopes(envelopes_dir)
    skipped = file_count - len(loaded)

    # Group by suite_id (the "category").
    by_category: dict[str, list[LoadedEnvelope]] = {}
    for item in loaded:
        by_category.setdefault(item.envelope.suite_id, []).append(item)

    template_env = _make_env()
    package_dir = Path(__file__).parent
    static_written = _copy_static_assets(package_dir, out_dir)
    _copy_raw_envelopes(loaded, out_dir)
    pages_written = static_written

    # --- per-category and per-entry pages --------------------------------- #
    categories_payload: list[dict[str, Any]] = []
    headline_keys = [k for k, _ in HEADLINE_METRICS]
    headline_labels = dict(HEADLINE_METRICS)

    for suite_id in sorted(by_category):
        items = by_category[suite_id]
        # Pareto frontier on (throughput, latency) where data permits.
        pareto_pairs: list[tuple[float | None, float | None]] = [
            (
                _safe_float(it.envelope.metrics.get(PARETO_X_METRIC)),
                _safe_float(it.envelope.metrics.get(PARETO_Y_METRIC)),
            )
            for it in items
        ]
        x_dir = PARETO_DIRECTIONS.get(PARETO_X_METRIC, "max")
        y_dir = PARETO_DIRECTIONS.get(PARETO_Y_METRIC, "min")
        pareto_flags = compute_pareto(
            pareto_pairs,
            x_direction=x_dir,
            y_direction=y_dir,
        )
        entries = [
            _entry_payload(item, flag) for item, flag in zip(items, pareto_flags, strict=True)
        ]

        category_ctx = {
            "base_url": base_url,
            "suite_id": suite_id,
            "entries": entries,
            "headline_keys": headline_keys,
            "headline_labels": headline_labels,
            "pareto_x_metric": PARETO_X_METRIC,
            "pareto_y_metric": PARETO_Y_METRIC,
        }
        _write(
            out_dir / suite_id / "index.html",
            template_env.get_template("category.html").render(**category_ctx),
        )
        pages_written += 1

        for entry, item in zip(entries, items, strict=True):
            entry_ctx = {
                "base_url": base_url,
                "entry": entry,
                "envelope": item.envelope,
                "headline_keys": headline_keys,
                "headline_labels": headline_labels,
                "verify_snippet": _verify_snippet(base_url, item.source_filename),
            }
            _write(
                out_dir / suite_id / f"{entry['run_id']}.html",
                template_env.get_template("entry.html").render(**entry_ctx),
            )
            pages_written += 1

        categories_payload.append({"suite_id": suite_id, "count": len(entries)})

    # --- top-level index --------------------------------------------------- #
    _write(
        out_dir / "index.html",
        template_env.get_template("index.html").render(
            base_url=base_url,
            categories=categories_payload,
            total_envelopes=len(loaded),
            total_skipped=skipped,
        ),
    )
    pages_written += 1

    # --- machine-readable data dump --------------------------------------- #
    data_dump = {
        "schema": "inferencebench-leaderboard.v1",
        "base_url": base_url,
        "categories": [
            {
                "suite_id": suite_id,
                "entries": [
                    {
                        "run_id": it.envelope.run_id,
                        "model_id": it.envelope.model.id,
                        "engine": it.envelope.engine.name,
                        "engine_version": it.envelope.engine.version,
                        "hardware_class": _hardware_class(it.envelope),
                        "metrics": dict(it.envelope.metrics),
                        "envelope_url": f"{base_url}envelopes/{it.source_filename}",
                        "signed": it.envelope.signature is not None,
                    }
                    for it in by_category[suite_id]
                ],
            }
            for suite_id in sorted(by_category)
        ],
    }
    _write(
        out_dir / "data" / "leaderboard.json",
        json.dumps(data_dump, sort_keys=True, indent=2),
    )
    pages_written += 1

    return SiteRenderResult(
        out_dir=out_dir,
        envelopes_loaded=len(loaded),
        envelopes_skipped=skipped,
        categories={c["suite_id"]: c["count"] for c in categories_payload},
        pages_written=pages_written,
    )


def _safe_float(value: float | int | str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _verify_snippet(base_url: str, filename: str) -> str:
    """Return the recommended ``bench verify`` snippet shown on detail pages."""
    return f"bench verify {base_url}envelopes/{filename}"
