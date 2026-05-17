"""``bench run`` — execute a benchmark and produce a signed envelope.

Phase 1 implementation (ticket 0025). Discovers plugins via the
``inferencebench.plugins`` entry-point group, instantiates the requested
plugin, validates the spec, runs the benchmark, and writes the signed
envelope JSON to ``<output_dir>/<content_hash[:12]>.json``.

This module is intentionally plugin-agnostic — it never imports a specific
plugin module by name. It resolves plugin-defined types (``RunContext``,
``EngineKind``) from the plugin's top-level package at call time so any
plugin following the same convention works.
"""

from __future__ import annotations

import importlib
import os
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console
from rich.status import Status
from rich.table import Table

from inferencebench.envelope import Envelope

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

console = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Plugin discovery + schema resolution                                        #
# --------------------------------------------------------------------------- #
def _entry_points() -> list[EntryPoint]:
    """Return every registered ``inferencebench.plugins`` entry point."""
    from importlib import metadata

    try:
        return list(metadata.entry_points(group="inferencebench.plugins"))
    except TypeError:  # pragma: no cover — pre-3.10 compat path
        return list(metadata.entry_points().get("inferencebench.plugins", []))  # type: ignore[attr-defined]


def _split_suite_id(suite_id: str) -> tuple[str, str | None]:
    """Split ``llm.inference.sharegpt-v3`` → (``llm.inference``, ``llm.inference.sharegpt-v3``).

    Entry-point names are exactly the plugin-id (``llm.inference``). A fully
    qualified benchmark id (``llm.inference.<suffix>``) prefixes the plugin
    name. Returns the (plugin-name, full-benchmark-id-or-None) tuple.
    """
    eps = _entry_points()
    names = {ep.name for ep in eps}
    if suite_id in names:
        return suite_id, None
    parts = suite_id.split(".")
    for i in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in names:
            return candidate, suite_id
    return suite_id, None


def _resolve_plugin_schemas(ep: EntryPoint) -> tuple[type[Any], type[Any]]:
    """Return (``RunContext``, ``EngineKind``) classes from the plugin package.

    Plugins are expected to re-export both at their top-level package
    (see ``plugins/llm-inference/src/inferencebench_llm/__init__.py``).
    """
    module_path = ep.value.split(":")[0]
    top_pkg = module_path.split(".")[0]
    pkg = importlib.import_module(top_pkg)
    try:
        return pkg.RunContext, pkg.EngineKind
    except AttributeError as exc:
        msg = (
            f"Plugin '{ep.name}' (package '{top_pkg}') does not expose "
            "RunContext and EngineKind at its top level."
        )
        raise RuntimeError(msg) from exc


def _find_entry_point(eps: list[EntryPoint], plugin_name: str, suite_id: str) -> EntryPoint:
    matched = [ep for ep in eps if ep.name == plugin_name]
    if matched:
        return matched[0]
    err_console.print(f"[red]No plugin registered for suite:[/red] [bold]{suite_id}[/bold]")
    if eps:
        err_console.print("Installed plugins:")
        for ep in eps:
            err_console.print(f"  • [cyan]{ep.name}[/cyan]   → {ep.value}")
    else:
        err_console.print("[yellow]No plugins installed.[/yellow]")
        err_console.print("Install one: [bold]pip install inferencebench-llm[/bold]")
    raise typer.Exit(code=1)


def _select_spec(specs: list[Any], full_id: str | None, plugin_name: str) -> Any:  # noqa: ANN401
    if full_id is None:
        return specs[0]
    spec = next((s for s in specs if s.benchmark_id == full_id), None)
    if spec is None:
        err_console.print(
            f"[red]benchmark_id not found in plugin '{plugin_name}':[/red] {full_id}"
        )
        err_console.print("Available:")
        for s in specs:
            err_console.print(f"  • [cyan]{s.benchmark_id}[/cyan]")
        raise typer.Exit(code=1)
    return spec


def _print_benchmark_list(plugin_name: str, specs: list[Any]) -> None:
    table = Table(title=f"Benchmarks in '{plugin_name}'")
    table.add_column("benchmark_id", style="cyan")
    table.add_column("description", style="dim")
    for s in specs:
        desc = (s.description or "").strip()
        first_line = desc.splitlines()[0] if desc else ""
        table.add_row(s.benchmark_id, first_line)
    console.print(table)


def _merge_driver_overrides(
    extra: dict[str, str | int | float | bool],
    *,
    concurrency: str,
    duration_s: int,
    rps: float,
) -> None:
    """Translate CLI driver flags into ``RunContext.extra`` override keys.

    Phase 1 only forwards the first entry from a comma-separated list — the
    full sweep is a Phase 2 enhancement (one envelope per point on the curve).
    """
    if duration_s:
        extra["duration_s"] = int(duration_s)
    if rps > 0:
        extra["rps"] = float(rps)
        extra["driver_type"] = "open_loop"
    if concurrency and concurrency != "1":
        first = concurrency.split(",")[0].strip()
        if first:
            try:
                extra["concurrency"] = int(first)
                extra["driver_type"] = "closed_loop"
            except ValueError:
                err_console.print(
                    f"[yellow]warning: ignoring non-integer --concurrency {concurrency!r}[/yellow]"
                )


def _apply_prices_file(
    extra: dict[str, str | int | float | bool], prices_file: str
) -> None:
    """Validate ``--prices-file`` and stash the resolved path on ``extra``.

    Plugins read ``extra['prices_file']`` to override the bundled pricing
    registry in the cost-fallback path; see
    :func:`inferencebench_llm.plugin.LLMInferencePlugin._custom_pricing_registry`.
    """
    if not prices_file:
        return
    prices_path = Path(prices_file)
    if not prices_path.is_file():
        err_console.print(f"[red]--prices-file not found:[/red] {prices_path}")
        raise typer.Exit(code=2)
    extra["prices_file"] = str(prices_path.resolve())


def _apply_judge_overrides(
    extra: dict[str, str | int | float | bool],
    *,
    judge_model: str,
    judge_max_questions: int,
    judge_rps: float,
) -> None:
    """Stash the LLM-as-judge CLI overrides on ``extra``.

    Only used when the spec selects ``scoring: judge_llm`` — the
    llm.quality plugin reads these keys to construct the judge ModelClient
    and to cap how many questions get judged. Empty / zero values are
    skipped so the plugin's own defaults apply.
    """
    if judge_model:
        extra["judge_model"] = judge_model
    if judge_max_questions:
        extra["judge_max_questions"] = int(judge_max_questions)
    if judge_rps > 0:
        extra["judge_rps"] = float(judge_rps)


def _build_signing_extra(
    signing_mode: str, dev_key: str
) -> dict[str, str | int | float | bool]:
    extra: dict[str, str | int | float | bool] = {"signing_mode": signing_mode}
    if signing_mode == "dev":
        key_path = Path(dev_key) if dev_key else Path("cosign.key")
        if not key_path.exists():
            err_console.print(
                "[red]Dev signing mode requires --dev-key (or ./cosign.key) to exist.[/red] "
                f"Tried: {key_path}"
            )
            raise typer.Exit(code=1)
        extra["dev_key_path"] = str(key_path)
    elif signing_mode != "keyless":
        err_console.print(
            f"[red]Unknown --signing-mode:[/red] {signing_mode} (expected 'dev' or 'keyless')"
        )
        raise typer.Exit(code=1)
    return extra


def _write_envelope(
    envelope: Envelope, output_dir: Path, *, prefix: str = ""
) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    content_hash = envelope.content_hash()
    fname = f"{prefix}-{content_hash[:12]}.json" if prefix else f"{content_hash[:12]}.json"
    out_path = output_dir / fname
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(envelope.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp_path, out_path)
    return out_path, content_hash


def _parse_sweep_ints(raw: str, flag: str) -> list[int]:
    """Parse ``--sweep`` into a list of positive ints; raise typer.Exit on bad input."""
    out: list[int] = []
    for tok in (t.strip() for t in raw.split(",") if t.strip()):
        try:
            value = int(tok)
        except ValueError as exc:
            err_console.print(
                f"[red]Invalid {flag} value:[/red] {tok!r} is not an integer."
            )
            raise typer.Exit(code=1) from exc
        if value <= 0:
            err_console.print(
                f"[red]Invalid {flag} value:[/red] {value} must be > 0."
            )
            raise typer.Exit(code=1)
        out.append(value)
    if not out:
        err_console.print(f"[red]{flag} was empty after parsing.[/red]")
        raise typer.Exit(code=1)
    return out


def _parse_sweep_floats(raw: str, flag: str) -> list[float]:
    """Parse ``--rps-sweep`` into a list of positive floats; raise typer.Exit on bad input."""
    out: list[float] = []
    for tok in (t.strip() for t in raw.split(",") if t.strip()):
        try:
            value = float(tok)
        except ValueError as exc:
            err_console.print(
                f"[red]Invalid {flag} value:[/red] {tok!r} is not a number."
            )
            raise typer.Exit(code=1) from exc
        if value <= 0:
            err_console.print(
                f"[red]Invalid {flag} value:[/red] {value} must be > 0."
            )
            raise typer.Exit(code=1)
        out.append(value)
    if not out:
        err_console.print(f"[red]{flag} was empty after parsing.[/red]")
        raise typer.Exit(code=1)
    return out


def _resolve_sweep_flags(
    *,
    sweep: str,
    rps_sweep: str,
    concurrency: str,
    rps: float,
) -> tuple[list[int] | list[float] | None, str | None]:
    """Validate sweep flags and return (points, sweep_kind) — both ``None`` if single-point."""
    sweep_set = bool(sweep)
    rps_sweep_set = bool(rps_sweep)
    if sweep_set and rps_sweep_set:
        err_console.print(
            "[red]--sweep and --rps-sweep are mutually exclusive[/red] "
            "(a run is either closed-loop or open-loop)."
        )
        raise typer.Exit(code=1)
    if sweep_set and concurrency != "1":
        err_console.print(
            "[red]--sweep and --concurrency are mutually exclusive[/red] "
            "(use --sweep alone for a multi-point closed-loop run)."
        )
        raise typer.Exit(code=1)
    if rps_sweep_set and rps > 0:
        err_console.print(
            "[red]--rps-sweep and --rps are mutually exclusive[/red] "
            "(use --rps-sweep alone for a multi-point open-loop run)."
        )
        raise typer.Exit(code=1)
    if sweep_set:
        return _parse_sweep_ints(sweep, "--sweep"), "concurrency"
    if rps_sweep_set:
        return _parse_sweep_floats(rps_sweep, "--rps-sweep"), "rps"
    return None, None


def _format_rps_point(value: float) -> str:
    """Render an RPS point compactly for filenames + table rows."""
    if value == int(value):
        return f"rps{int(value)}"
    return f"rps{value:g}".replace(".", "p")


def _run_sweep(
    *,
    plugin: Any,  # noqa: ANN401
    spec: Any,  # noqa: ANN401
    run_context_cls: type[Any],
    engine_kind: Any,  # noqa: ANN401
    model: str,
    base_url: str,
    quant: str,
    hardware: str,
    output_dir: Path,
    signing_extra: dict[str, str | int | float | bool],
    duration_s: int,
    sweep_kind: str,
    points: list[int] | list[float],
    strict: bool,
) -> None:
    """Run a closed-loop or open-loop sweep, one envelope per point.

    ``sweep_kind`` is ``"concurrency"`` (closed-loop) or ``"rps"`` (open-loop).
    Exits 0 iff every point completed with ok_rate >= 0.95.
    """
    summary_rows: list[dict[str, Any]] = []
    any_failure = False
    validated = False

    for idx, point in enumerate(points):
        extra: dict[str, str | int | float | bool] = dict(signing_extra)
        if duration_s:
            extra["duration_s"] = int(duration_s)
        if sweep_kind == "concurrency":
            extra["concurrency"] = int(point)
            extra["driver_type"] = "closed_loop"
            point_label = str(int(point))
            file_prefix = f"c{int(point)}"
        else:
            extra["rps"] = float(point)
            extra["driver_type"] = "open_loop"
            point_label = f"{float(point):g}"
            file_prefix = _format_rps_point(float(point))

        try:
            ctx = run_context_cls(
                model_id=model,
                engine_kind=engine_kind,
                base_url=base_url,
                quantization_format=quant,
                hardware_class=hardware,
                output_dir=output_dir,
                extra=extra,
            )
        except Exception as exc:
            err_console.print(f"[red]Invalid run context for point {point_label}:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        if not validated:
            warnings = list(plugin.validate(spec, ctx) or [])
            for w in warnings:
                err_console.print(f"[yellow]warning:[/yellow] {w}")
            if warnings and strict:
                err_console.print("[red]Refusing to run: --strict was set.[/red]")
                raise typer.Exit(code=1)
            validated = True

        status_label = (
            f"[bold]Sweep {idx + 1}/{len(points)}[/bold] — "
            f"{sweep_kind}={point_label}"
        )
        try:
            with Status(status_label, console=err_console):
                envelope = plugin.run(spec, ctx)
        except Exception as exc:
            err_console.print(
                f"[red]Sweep point {point_label} failed:[/red] {exc}"
            )
            err_console.print("[red]" + traceback.format_exc() + "[/red]")
            any_failure = True
            summary_rows.append(
                {
                    "point": point_label,
                    "envelope_path": "-",
                    "error": str(exc),
                    "ok_rate": None,
                    "metrics": {},
                    "model_id": model or "-",
                }
            )
            continue

        out_path, _content_hash = _write_envelope(envelope, output_dir, prefix=file_prefix)
        ok_rate = envelope.metrics.get("ok_rate")
        throughput = envelope.metrics.get("throughput_tok_per_s")
        tput_str = (
            f"{throughput:.4g}" if isinstance(throughput, int | float) else "-"
        )
        ok_str = (
            f"{ok_rate:.3f}" if isinstance(ok_rate, int | float) else "-"
        )
        console.print(
            f"[green]ok[/green] {sweep_kind}={point_label}  "
            f"model={envelope.model.id}  "
            f"throughput_tok_per_s={tput_str}  "
            f"ok_rate={ok_str}  "
            f"→ {out_path.name}"
        )
        if not (isinstance(ok_rate, int | float) and ok_rate >= 0.95):
            any_failure = True
        summary_rows.append(
            {
                "point": point_label,
                "envelope_path": str(out_path),
                "error": None,
                "ok_rate": ok_rate,
                "metrics": dict(envelope.metrics),
                "model_id": envelope.model.id,
            }
        )

    _print_sweep_table(sweep_kind, summary_rows)
    if any_failure:
        raise typer.Exit(code=1)


def _print_sweep_table(sweep_kind: str, rows: list[dict[str, Any]]) -> None:
    point_col = "concurrency" if sweep_kind == "concurrency" else "rps"
    table = Table(title=f"Sweep results ({sweep_kind})")
    table.add_column(point_col, style="cyan", no_wrap=True)
    table.add_column("throughput_tok_per_s", justify="right")
    table.add_column("ttft_p50_ms", justify="right")
    table.add_column("ttft_p99_ms", justify="right")
    table.add_column("tpot_p50_ms", justify="right")
    table.add_column("total_p50_ms", justify="right")
    table.add_column("ok_rate", justify="right")
    table.add_column("compliance_rate", justify="right")
    table.add_column("power_avg_w", justify="right")
    table.add_column("joules_per_token", justify="right")
    table.add_column("envelope", style="dim")

    def _fmt(metrics: dict[str, Any], key: str) -> str:
        v = metrics.get(key)
        if isinstance(v, int | float):
            return f"{v:.4g}"
        return "-"

    for row in rows:
        metrics = row["metrics"]
        if row["error"]:
            table.add_row(
                row["point"],
                "[red]ERROR[/red]",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                row["error"][:40],
            )
            continue
        ok_rate = row["ok_rate"]
        ok_cell = (
            f"{ok_rate:.3f}"
            if isinstance(ok_rate, int | float)
            else "-"
        )
        if isinstance(ok_rate, int | float) and ok_rate < 0.95:
            ok_cell = f"[red]{ok_cell}[/red]"
        table.add_row(
            row["point"],
            _fmt(metrics, "throughput_tok_per_s"),
            _fmt(metrics, "ttft_p50_ms"),
            _fmt(metrics, "ttft_p99_ms"),
            _fmt(metrics, "tpot_p50_ms"),
            _fmt(metrics, "total_p50_ms"),
            ok_cell,
            _fmt(metrics, "compliance_rate"),
            _fmt(metrics, "power_avg_w"),
            _fmt(metrics, "joules_per_token"),
            Path(row["envelope_path"]).name if row["envelope_path"] != "-" else "-",
        )
    console.print(table)


def _print_summary(envelope: Envelope, out_path: Path, content_hash: str) -> None:
    table = Table(title="Benchmark complete")
    table.add_column("key", style="cyan")
    table.add_column("value", style="bold")
    table.add_row("envelope", str(out_path))
    table.add_row("suite_id", str(envelope.suite_id))
    table.add_row("model", str(envelope.model.id))
    engine_version = envelope.engine.version or "unknown"
    engine_display = (
        f"{envelope.engine.name} {engine_version}"
        if engine_version == "unknown"
        else f"{envelope.engine.name} v{engine_version}"
    )
    table.add_row("engine", engine_display)
    table.add_row("content_hash", content_hash)
    ok_rate = envelope.metrics.get("ok_rate")
    if isinstance(ok_rate, int | float):
        marker = "[bold green]" if ok_rate >= 0.99 else "[bold red]"
        table.add_row("ok_rate", f"{marker}{ok_rate:.3f}[/]")
    headline_keys = (
        "throughput_tok_per_s",
        "ttft_p50_ms",
        "ttft_p99_ms",
        "tpot_p50_ms",
        "total_p50_ms",
        "req_per_s_passing",
        "compliance_rate",
        "power_avg_w",
        "joules_per_token",
        "cost_usd_per_million_tokens",
    )
    for key in headline_keys:
        val = envelope.metrics.get(key)
        if isinstance(val, int | float):
            table.add_row(key, f"{val:.4g}")
    other = [k for k in sorted(envelope.metrics) if k not in {*headline_keys, "ok_rate"}]
    if other:
        table.add_row("(other metrics)", ", ".join(other))
    console.print(table)
    if isinstance(ok_rate, int | float) and ok_rate < 0.99:
        err_console.print(
            f"[bold red]WARNING[/bold red] ok_rate={ok_rate:.3f} — most requests failed. "
            "Re-run with [bold]--verbose[/bold] and check the engine log."
        )


def _resolve_engine_kind(ep: EntryPoint, engine: str) -> tuple[type[Any], Any]:
    """Resolve (RunContext class, engine_kind value) from a plugin + CLI engine flag."""
    try:
        run_context_cls, engine_kind_cls = _resolve_plugin_schemas(ep)
    except RuntimeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    try:
        engine_kind = engine_kind_cls(engine)
    except ValueError as exc:
        err_console.print(f"[red]Unknown engine:[/red] {engine}")
        raise typer.Exit(code=1) from exc
    return run_context_cls, engine_kind


def _validate_all_benchmarks_flags(
    *,
    suite_id: str,
    full_id: str | None,
    list_: bool,
    sweep_points: list[int] | list[float] | None,
) -> None:
    """Raise typer.Exit if --all-benchmarks is combined with an incompatible flag."""
    if list_:
        err_console.print(
            "[red]--all-benchmarks and --list are mutually exclusive.[/red]"
        )
        raise typer.Exit(code=1)
    if sweep_points is not None:
        err_console.print(
            "[red]--all-benchmarks is mutually exclusive with --sweep and --rps-sweep.[/red]"
        )
        raise typer.Exit(code=1)
    if full_id is not None:
        err_console.print(
            "[red]--all-benchmarks requires a plugin id (e.g. 'llm.inference'), "
            f"not a fully-qualified benchmark id:[/red] {suite_id}"
        )
        raise typer.Exit(code=1)


def _run_all_benchmarks(
    *,
    plugin: Any,  # noqa: ANN401
    specs: list[Any],
    run_context_cls: type[Any],
    engine_kind: Any,  # noqa: ANN401
    model: str,
    base_url: str,
    quant: str,
    hardware: str,
    output_dir: Path,
    signing_extra: dict[str, str | int | float | bool],
    concurrency: str,
    duration_s: int,
    rps: float,
    strict: bool,
) -> None:
    """Run every benchmark spec the plugin exposes, one envelope per spec.

    Per-benchmark failures are logged in yellow and the loop continues; exit
    code is 0 iff at least one benchmark completed with ok_rate >= 0.95.
    """
    summary_rows: list[dict[str, Any]] = []
    any_pass = False

    for idx, spec in enumerate(specs):
        extra: dict[str, str | int | float | bool] = dict(signing_extra)
        _merge_driver_overrides(
            extra, concurrency=concurrency, duration_s=duration_s, rps=rps
        )

        try:
            ctx = run_context_cls(
                model_id=model,
                engine_kind=engine_kind,
                base_url=base_url,
                quantization_format=quant,
                hardware_class=hardware,
                output_dir=output_dir,
                extra=extra,
            )
        except Exception as exc:
            err_console.print(
                f"[yellow]warning:[/yellow] invalid run context for "
                f"{spec.benchmark_id}: {exc} — skipping."
            )
            summary_rows.append(
                {
                    "benchmark_id": spec.benchmark_id,
                    "envelope_path": "-",
                    "error": str(exc),
                    "ok_rate": None,
                    "metrics": {},
                    "model_id": model or "-",
                }
            )
            continue

        warnings = list(plugin.validate(spec, ctx) or [])
        for w in warnings:
            err_console.print(f"[yellow]warning:[/yellow] {w}")
        if warnings and strict:
            err_console.print(
                f"[yellow]warning:[/yellow] --strict — skipping {spec.benchmark_id}."
            )
            summary_rows.append(
                {
                    "benchmark_id": spec.benchmark_id,
                    "envelope_path": "-",
                    "error": "strict-mode validate warnings",
                    "ok_rate": None,
                    "metrics": {},
                    "model_id": model or "-",
                }
            )
            continue

        status_label = (
            f"[bold]Benchmark {idx + 1}/{len(specs)}[/bold] — "
            f"{spec.benchmark_id}"
        )
        try:
            with Status(status_label, console=err_console):
                envelope = plugin.run(spec, ctx)
        except Exception as exc:
            err_console.print(
                f"[yellow]warning:[/yellow] benchmark {spec.benchmark_id} failed: "
                f"{exc} — continuing."
            )
            summary_rows.append(
                {
                    "benchmark_id": spec.benchmark_id,
                    "envelope_path": "-",
                    "error": str(exc),
                    "ok_rate": None,
                    "metrics": {},
                    "model_id": model or "-",
                }
            )
            continue

        bench_slug = spec.benchmark_id.replace(".", "-")
        out_path, _content_hash = _write_envelope(envelope, output_dir, prefix=bench_slug)
        ok_rate = envelope.metrics.get("ok_rate")
        throughput = envelope.metrics.get("throughput_tok_per_s")
        tput_str = (
            f"{throughput:.4g}" if isinstance(throughput, int | float) else "-"
        )
        ok_str = (
            f"{ok_rate:.3f}" if isinstance(ok_rate, int | float) else "-"
        )
        console.print(
            f"[green]ok[/green] {spec.benchmark_id}  "
            f"model={envelope.model.id}  "
            f"throughput_tok_per_s={tput_str}  "
            f"ok_rate={ok_str}  "
            f"→ {out_path.name}"
        )
        if isinstance(ok_rate, int | float) and ok_rate >= 0.95:
            any_pass = True
        summary_rows.append(
            {
                "benchmark_id": spec.benchmark_id,
                "envelope_path": str(out_path),
                "error": None,
                "ok_rate": ok_rate,
                "metrics": dict(envelope.metrics),
                "model_id": envelope.model.id,
            }
        )

    _print_all_benchmarks_table(summary_rows)
    if not any_pass:
        raise typer.Exit(code=1)


def _print_all_benchmarks_table(rows: list[dict[str, Any]]) -> None:
    table = Table(title="All-benchmarks results")
    table.add_column("benchmark_id", style="cyan", no_wrap=True)
    table.add_column("throughput_tok_per_s", justify="right")
    table.add_column("ttft_p50_ms", justify="right")
    table.add_column("ttft_p99_ms", justify="right")
    table.add_column("tpot_p50_ms", justify="right")
    table.add_column("total_p50_ms", justify="right")
    table.add_column("ok_rate", justify="right")
    table.add_column("envelope", style="dim")

    def _fmt(metrics: dict[str, Any], key: str) -> str:
        v = metrics.get(key)
        if isinstance(v, int | float):
            return f"{v:.4g}"
        return "-"

    for row in rows:
        if row["error"]:
            table.add_row(
                row["benchmark_id"],
                "[red]ERROR[/red]",
                "-",
                "-",
                "-",
                "-",
                "-",
                row["error"][:40],
            )
            continue
        metrics = row["metrics"]
        ok_rate = row["ok_rate"]
        ok_cell = (
            f"{ok_rate:.3f}" if isinstance(ok_rate, int | float) else "-"
        )
        if isinstance(ok_rate, int | float) and ok_rate < 0.95:
            ok_cell = f"[red]{ok_cell}[/red]"
        table.add_row(
            row["benchmark_id"],
            _fmt(metrics, "throughput_tok_per_s"),
            _fmt(metrics, "ttft_p50_ms"),
            _fmt(metrics, "ttft_p99_ms"),
            _fmt(metrics, "tpot_p50_ms"),
            _fmt(metrics, "total_p50_ms"),
            ok_cell,
            Path(row["envelope_path"]).name if row["envelope_path"] != "-" else "-",
        )
    console.print(table)


# --------------------------------------------------------------------------- #
# CLI command                                                                 #
# --------------------------------------------------------------------------- #
def run(
    suite_id: Annotated[str, typer.Argument(help="Suite identifier, e.g. 'llm.inference'.")],
    model: Annotated[str, typer.Option("--model", help="Model id (provider-prefixed).")] = "",
    engine: Annotated[
        str, typer.Option("--engine", help="Inference engine (vllm, sglang, ...).")
    ] = "vllm",
    hardware: Annotated[
        str, typer.Option("--hardware", help="Hardware class (h100, h200, ...).")
    ] = "h100",
    quant: Annotated[
        str, typer.Option("--quant", help="Quantization format (fp16, fp8, nvfp4, ...).")
    ] = "fp16",
    concurrency: Annotated[
        str,
        typer.Option(
            "--concurrency",
            help="Comma-separated concurrency levels (e.g. '1,4,16,64').",
        ),
    ] = "1",
    dataset: Annotated[str, typer.Option("--dataset", help="Dataset id (e.g. sharegpt-v3).")] = "",
    duration: Annotated[
        int, typer.Option("--duration", help="Measurement duration in seconds.")
    ] = 300,
    rps: Annotated[
        float,
        typer.Option(
            "--rps",
            help="Open-loop arrival rate (req/s). Overrides the spec's first rps entry.",
        ),
    ] = 0.0,
    slo_template: Annotated[
        str,
        typer.Option(
            "--slo-template",
            help="SLO template (llm.standard, voice.realtime, ...).",
        ),
    ] = "llm.standard",
    seed: Annotated[int, typer.Option("--seed", help="Random seed for reproducibility.")] = 42,
    output: Annotated[
        str, typer.Option("--output", help="Output directory for the signed envelope.")
    ] = "",
    base_url: Annotated[
        str,
        typer.Option("--base-url", help="Engine base URL (e.g. http://localhost:8000/v1)."),
    ] = "",
    signing_mode: Annotated[
        str,
        typer.Option(
            "--signing-mode",
            help="Envelope signing mode: 'dev' (local cosign key) or 'keyless' (Sigstore OIDC).",
        ),
    ] = "dev",
    dev_key: Annotated[
        str,
        typer.Option(
            "--dev-key",
            help="Path to local cosign signing key (used when --signing-mode=dev).",
        ),
    ] = "",
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Treat plugin.validate() warnings as fatal.",
        ),
    ] = False,
    list_: Annotated[
        bool,
        typer.Option(
            "--list",
            help="List available benchmark_ids for the suite and exit.",
        ),
    ] = False,
    sweep: Annotated[
        str,
        typer.Option(
            "--sweep",
            help=(
                "Comma-separated closed-loop concurrency points (e.g. '1,4,16,64'). "
                "One signed envelope per point. Mutually exclusive with --concurrency "
                "and --rps-sweep."
            ),
        ),
    ] = "",
    rps_sweep: Annotated[
        str,
        typer.Option(
            "--rps-sweep",
            help=(
                "Comma-separated open-loop RPS points (e.g. '1,4,16'). One signed "
                "envelope per point. Mutually exclusive with --rps and --sweep."
            ),
        ),
    ] = "",
    all_benchmarks: Annotated[
        bool,
        typer.Option(
            "--all-benchmarks",
            help=(
                "Run every benchmark exposed by the plugin (one envelope per "
                "spec). Mutually exclusive with --list, --sweep, --rps-sweep, "
                "and a fully-qualified suite_id."
            ),
        ),
    ] = False,
    prices_file: Annotated[
        str,
        typer.Option(
            "--prices-file",
            help=(
                "Path to a custom prices YAML used by the plugin's "
                "registry-cost fallback (when LiteLLM doesn't report a "
                "provider cost). Forwarded as RunContext.extra['prices_file']."
            ),
        ),
    ] = "",
    judge_model: Annotated[
        str,
        typer.Option(
            "--judge-model",
            help=(
                "LLM-as-judge model id (only used when the spec selects "
                "scoring: judge_llm). Forwarded as RunContext.extra['judge_model']."
            ),
        ),
    ] = "",
    judge_max_questions: Annotated[
        int,
        typer.Option(
            "--judge-max-questions",
            help=(
                "Cap on the number of questions sent to the judge (0 = no cap). "
                "Only the judged questions contribute to the accuracy metric. "
                "Forwarded as RunContext.extra['judge_max_questions']."
            ),
        ),
    ] = 0,
    judge_rps: Annotated[
        float,
        typer.Option(
            "--judge-rps",
            help=(
                "Cap judge API calls at this rate (req/s). 0 = unlimited. "
                "Forwarded as RunContext.extra['judge_rps']; the llm.quality "
                "plugin sleeps to keep at most 1/rps seconds between calls."
            ),
        ),
    ] = 0.0,
) -> None:
    """Run a benchmark from the named suite and emit a signed envelope."""
    sweep_points, sweep_kind = _resolve_sweep_flags(
        sweep=sweep,
        rps_sweep=rps_sweep,
        concurrency=concurrency,
        rps=rps,
    )

    eps = _entry_points()
    plugin_name, full_id = _split_suite_id(suite_id)

    if all_benchmarks:
        _validate_all_benchmarks_flags(
            suite_id=suite_id,
            full_id=full_id,
            list_=list_,
            sweep_points=sweep_points,
        )

    ep = _find_entry_point(eps, plugin_name, suite_id)

    try:
        plugin_cls = ep.load()
    except Exception as exc:  # pragma: no cover - defensive
        err_console.print(f"[red]Failed to load plugin '{ep.name}':[/red] {exc}")
        raise typer.Exit(code=1) from exc
    plugin = plugin_cls()

    specs = list(plugin.list_benchmarks())
    if not specs:
        err_console.print(f"[red]Plugin '{ep.name}' exposes no benchmarks.[/red]")
        raise typer.Exit(code=1)

    if list_:
        _print_benchmark_list(ep.name, specs)
        raise typer.Exit(code=0)

    spec = _select_spec(specs, full_id, ep.name)

    run_context_cls, engine_kind = _resolve_engine_kind(ep, engine)

    output_dir = Path(output) if output else Path.cwd() / "results"
    signing_extra = _build_signing_extra(signing_mode, dev_key)

    _apply_prices_file(signing_extra, prices_file)
    _apply_judge_overrides(
        signing_extra,
        judge_model=judge_model,
        judge_max_questions=judge_max_questions,
        judge_rps=judge_rps,
    )

    if sweep_points is not None and sweep_kind is not None:
        _run_sweep(
            plugin=plugin,
            spec=spec,
            run_context_cls=run_context_cls,
            engine_kind=engine_kind,
            model=model,
            base_url=base_url,
            quant=quant,
            hardware=hardware,
            output_dir=output_dir,
            signing_extra=signing_extra,
            duration_s=duration,
            sweep_kind=sweep_kind,
            points=sweep_points,
            strict=strict,
        )
        return

    if all_benchmarks:
        _run_all_benchmarks(
            plugin=plugin,
            specs=specs,
            run_context_cls=run_context_cls,
            engine_kind=engine_kind,
            model=model,
            base_url=base_url,
            quant=quant,
            hardware=hardware,
            output_dir=output_dir,
            signing_extra=signing_extra,
            concurrency=concurrency,
            duration_s=duration,
            rps=rps,
            strict=strict,
        )
        return

    extra = dict(signing_extra)
    _merge_driver_overrides(extra, concurrency=concurrency, duration_s=duration, rps=rps)

    try:
        ctx = run_context_cls(
            model_id=model,
            engine_kind=engine_kind,
            base_url=base_url,
            quantization_format=quant,
            hardware_class=hardware,
            output_dir=output_dir,
            extra=extra,
        )
    except Exception as exc:
        err_console.print(f"[red]Invalid run context:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    warnings = list(plugin.validate(spec, ctx) or [])
    if warnings:
        for w in warnings:
            err_console.print(f"[yellow]warning:[/yellow] {w}")
        if strict:
            err_console.print("[red]Refusing to run: --strict was set.[/red]")
            raise typer.Exit(code=1)

    try:
        with Status("[bold]Running benchmark…[/bold]", console=err_console):
            envelope = plugin.run(spec, ctx)
    except Exception as exc:
        err_console.print(f"[red]Benchmark failed:[/red] {exc}")
        err_console.print("[red]" + traceback.format_exc() + "[/red]")
        raise typer.Exit(code=1) from exc

    out_path, content_hash = _write_envelope(envelope, output_dir)
    _print_summary(envelope, out_path, content_hash)
