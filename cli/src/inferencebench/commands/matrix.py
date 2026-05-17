"""``bench matrix`` — run one benchmark across multiple endpoints.

The matrix command takes a YAML config that lists N target endpoints
(model + engine + base URL, optionally an API key env var) and a single
``suite_id`` to drive against every target. For each target, the command
runs the benchmark once per concurrency sweep point and writes a signed
envelope per (target, point) pair to ``--output``.

The expected use case is multi-vendor comparison runs: ``bench run`` already
handles a single endpoint; ``bench matrix`` automates the "run-it-N-times"
shape so you can compare e.g. vLLM Llama vs vLLM Qwen vs OpenAI gpt-4o-mini
from one command + one config.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from inferencebench.commands.run import (
    _build_signing_extra,
    _entry_points,
    _resolve_plugin_schemas,
    _write_envelope,
)

console = Console()
err_console = Console(stderr=True)


MATRIX_SCHEMA = "inferencebench.matrix.v1"


# --------------------------------------------------------------------------- #
# YAML parsing + validation                                                   #
# --------------------------------------------------------------------------- #
def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        err_console.print(f"[red]Matrix config not found:[/red] {path}")
        raise typer.Exit(code=2)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        err_console.print(f"[red]Failed to parse YAML:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    if not isinstance(raw, dict):
        err_console.print(
            f"[red]Matrix config must be a mapping at the top level, got "
            f"{type(raw).__name__}.[/red]"
        )
        raise typer.Exit(code=2)
    return raw


def _validate_duration(value: Any, errors: list[str]) -> int:  # noqa: ANN401
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append("duration_s: must be an integer (seconds)")
        return 60
    if value <= 0:
        errors.append("duration_s: must be > 0")
        return 60
    return value


def _validate_sweep(raw: Any, errors: list[str]) -> list[int]:  # noqa: ANN401
    sweep: list[int] = []
    if not isinstance(raw, list) or not raw:
        errors.append("sweep: must be a non-empty list of positive integers")
        return sweep
    for i, point in enumerate(raw):
        if not isinstance(point, int) or isinstance(point, bool) or point <= 0:
            errors.append(f"sweep[{i}]: must be a positive integer (got {point!r})")
        else:
            sweep.append(point)
    return sweep


def _validate_target(
    i: int,
    t: Any,  # noqa: ANN401
    seen_names: set[str],
    errors: list[str],
) -> dict[str, Any] | None:
    if not isinstance(t, dict):
        errors.append(f"targets[{i}]: must be a mapping")
        return None
    name = t.get("name")
    model = t.get("model")
    engine = t.get("engine")
    if not isinstance(name, str) or not name.strip():
        errors.append(f"targets[{i}].name: required, must be non-empty string")
    elif name in seen_names:
        errors.append(f"targets[{i}].name: duplicate name {name!r}")
    else:
        seen_names.add(name)
    if not isinstance(model, str) or not model.strip():
        errors.append(f"targets[{i}].model: required, must be non-empty string")
    if not isinstance(engine, str) or not engine.strip():
        errors.append(f"targets[{i}].engine: required, must be non-empty string")
    extra = t.get("extra", {})
    if extra is not None and not isinstance(extra, dict):
        errors.append(f"targets[{i}].extra: must be a mapping if provided")
        extra = {}
    return {
        "name": name if isinstance(name, str) else f"target-{i}",
        "model": model if isinstance(model, str) else "",
        "engine": engine if isinstance(engine, str) else "",
        "base_url": t.get("base_url", "") or "",
        "quant": t.get("quant", "") or "",
        "api_key_env": t.get("api_key_env") or None,
        "extra": dict(extra) if isinstance(extra, dict) else {},
    }


def _validate_targets(raw: Any, errors: list[str]) -> list[dict[str, Any]]:  # noqa: ANN401
    targets: list[dict[str, Any]] = []
    if not isinstance(raw, list) or not raw:
        errors.append("targets: required, must be a non-empty list")
        return targets
    seen_names: set[str] = set()
    for i, t in enumerate(raw):
        normalized = _validate_target(i, t, seen_names, errors)
        if normalized is not None:
            targets.append(normalized)
    return targets


def _validate_matrix_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate the parsed config and return a normalised dict.

    On any schema problem prints a red error listing the invalid fields and
    exits with code 2.
    """
    errors: list[str] = []

    schema = cfg.get("schema")
    if schema is not None and schema != MATRIX_SCHEMA:
        errors.append(f"schema: expected '{MATRIX_SCHEMA}', got {schema!r}")

    suite_id = cfg.get("suite_id")
    if not isinstance(suite_id, str) or not suite_id.strip():
        errors.append("suite_id: required, must be a non-empty string")

    duration_s = _validate_duration(cfg.get("duration_s", 60), errors)
    sweep = _validate_sweep(cfg.get("sweep", [1]), errors)
    targets = _validate_targets(cfg.get("targets"), errors)

    if errors:
        err_console.print("[red]Invalid matrix config:[/red]")
        for e in errors:
            err_console.print(f"  - {e}")
        raise typer.Exit(code=2)

    return {
        "schema": schema or MATRIX_SCHEMA,
        "suite_id": str(suite_id),
        "duration_s": duration_s,
        "sweep": sweep,
        "targets": targets,
    }


# --------------------------------------------------------------------------- #
# Run plan execution                                                          #
# --------------------------------------------------------------------------- #
def _resolve_api_key(
    target: dict[str, Any],
) -> tuple[str, str | None]:
    """Return (api_key, missing_env_var_name).

    If ``api_key_env`` is set and the env var is absent, return
    ``("", env_var_name)``. Otherwise the caller can proceed.
    """
    env_var = target.get("api_key_env")
    if not env_var:
        return "", None
    value = os.environ.get(env_var)
    if value is None or value == "":
        return "", env_var
    return value, None


def _plugin_for_suite(suite_id: str) -> tuple[Any, Any, type[Any], Any]:
    """Resolve (plugin, spec, RunContext_cls, EngineKind_cls) for a suite_id.

    Mirrors the dispatch logic in ``commands.run`` but without the per-flag
    dance — matrix YAML is the single source of truth, so we don't need
    sweep/concurrency flag negotiation here.
    """
    from inferencebench.commands.run import _find_entry_point, _select_spec, _split_suite_id

    eps = _entry_points()
    plugin_name, full_id = _split_suite_id(suite_id)
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

    spec = _select_spec(specs, full_id, ep.name)

    try:
        run_context_cls, engine_kind_cls = _resolve_plugin_schemas(ep)
    except RuntimeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    return plugin, spec, run_context_cls, engine_kind_cls


def _print_summary_table(rows: list[dict[str, Any]]) -> None:
    table = Table(title="Matrix results")
    table.add_column("target", style="cyan", no_wrap=True)
    table.add_column("point", justify="right")
    table.add_column("throughput_tok_per_s", justify="right")
    table.add_column("ttft_p50_ms", justify="right")
    table.add_column("ok_rate", justify="right")
    table.add_column("envelope", style="dim")
    table.add_column("status", justify="center")

    def _fmt(metrics: dict[str, Any], key: str) -> str:
        v = metrics.get(key)
        if isinstance(v, int | float):
            return f"{v:.4g}"
        return "-"

    for row in rows:
        status = row["status"]
        if status == "ok":
            status_cell = "[green]✓[/green]"
        elif status == "skip":
            status_cell = "[yellow]skip[/yellow]"
        else:
            status_cell = "[red]✗[/red]"
        metrics = row.get("metrics", {})
        ok_rate = metrics.get("ok_rate")
        ok_cell = (
            f"{ok_rate:.3f}" if isinstance(ok_rate, int | float) else "-"
        )
        if isinstance(ok_rate, int | float) and ok_rate < 0.95:
            ok_cell = f"[red]{ok_cell}[/red]"
        envelope_cell = row.get("envelope_name") or "-"
        table.add_row(
            row["target"],
            str(row["point"]),
            _fmt(metrics, "throughput_tok_per_s") if status == "ok" else "-",
            _fmt(metrics, "ttft_p50_ms") if status == "ok" else "-",
            ok_cell if status == "ok" else "-",
            envelope_cell,
            status_cell,
        )
    console.print(table)


def _build_run_extra(
    signing_extra: dict[str, str | int | float | bool],
    duration_s: int,
    point: int,
    target_extra: dict[str, Any] | None,
) -> dict[str, str | int | float | bool]:
    extra: dict[str, str | int | float | bool] = dict(signing_extra)
    extra["duration_s"] = int(duration_s)
    extra["concurrency"] = int(point)
    extra["driver_type"] = "closed_loop"
    for k, v in (target_extra or {}).items():
        if isinstance(v, str | int | float | bool):
            extra[k] = v
    return extra


def _make_summary_row(
    target_name: str, point: int, status: str
) -> dict[str, Any]:
    return {
        "target": target_name,
        "point": point,
        "status": status,
        "metrics": {},
        "envelope_name": None,
    }


def _execute_pair(
    *,
    target: dict[str, Any],
    point: int,
    plugin: Any,  # noqa: ANN401
    spec: Any,  # noqa: ANN401
    run_context_cls: type[Any],
    engine_kind_cls: Any,  # noqa: ANN401
    output_dir: Path,
    signing_extra: dict[str, str | int | float | bool],
    duration_s: int,
) -> tuple[str, dict[str, Any]]:
    """Execute one (target, point) pair.

    Returns (status, summary_row). status is one of: ``"ok"``, ``"skip"``,
    ``"error"``. The summary row is ready to append to the matrix summary.
    """
    target_name: str = target["name"]

    api_key, missing_env = _resolve_api_key(target)
    if missing_env is not None:
        err_console.print(
            f"[yellow]warning:[/yellow] target {target_name!r}: env "
            f"var {missing_env} is not set - skipping."
        )
        return "skip", _make_summary_row(target_name, point, "skip")

    try:
        engine_kind = engine_kind_cls(target["engine"])
    except ValueError as exc:
        err_console.print(
            f"[red]target {target_name!r}: unknown engine "
            f"{target['engine']!r}: {exc}[/red]"
        )
        return "error", _make_summary_row(target_name, point, "error")

    extra = _build_run_extra(signing_extra, duration_s, point, target.get("extra"))

    try:
        ctx = run_context_cls(
            model_id=target["model"],
            engine_kind=engine_kind,
            base_url=target.get("base_url", "") or "",
            api_key=api_key,
            quantization_format=target.get("quant", "") or "",
            output_dir=output_dir,
            extra=extra,
        )
    except Exception as exc:
        err_console.print(
            f"[red]target {target_name!r}: invalid run context:[/red] {exc}"
        )
        return "error", _make_summary_row(target_name, point, "error")

    try:
        envelope = plugin.run(spec, ctx)
    except Exception as exc:
        err_console.print(
            f"[red]target {target_name!r} c{point} failed:[/red] {exc}"
        )
        err_console.print("[red]" + traceback.format_exc() + "[/red]")
        return "error", _make_summary_row(target_name, point, "error")

    prefix = f"{target_name}-c{point}"
    out_path, _content_hash = _write_envelope(envelope, output_dir, prefix=prefix)
    row: dict[str, Any] = {
        "target": target_name,
        "point": point,
        "status": "ok",
        "metrics": dict(envelope.metrics),
        "envelope_name": out_path.name,
    }
    return "ok", row


# --------------------------------------------------------------------------- #
# CLI command                                                                 #
# --------------------------------------------------------------------------- #
def matrix(
    config_path: Annotated[
        Path,
        typer.Argument(
            help="Path to a matrix YAML config (schema: inferencebench.matrix.v1).",
        ),
    ],
    output: Annotated[
        str,
        typer.Option(
            "--output",
            help="Output directory for produced envelopes.",
        ),
    ],
    signing_mode: Annotated[
        str,
        typer.Option(
            "--signing-mode",
            help="Envelope signing mode: 'dev' (local cosign key) or 'keyless' (Sigstore).",
        ),
    ] = "dev",
    dev_key: Annotated[
        str,
        typer.Option(
            "--dev-key",
            help="Path to local cosign signing key (used when --signing-mode=dev).",
        ),
    ] = "./cosign.key",
    continue_on_error: Annotated[
        bool,
        typer.Option(
            "--continue-on-error/--no-continue-on-error",
            help=(
                "If set (default), keep going past failed targets. If "
                "--no-continue-on-error, stop the matrix on first failure."
            ),
        ),
    ] = True,
) -> None:
    """Run one benchmark across multiple endpoints (multi-vendor matrix)."""
    if not output:
        err_console.print("[red]--output is required for matrix runs.[/red]")
        raise typer.Exit(code=2)

    raw = _load_yaml(Path(config_path))
    cfg = _validate_matrix_config(raw)

    suite_id: str = cfg["suite_id"]
    duration_s: int = cfg["duration_s"]
    sweep: list[int] = cfg["sweep"]
    targets: list[dict[str, Any]] = cfg["targets"]

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    signing_extra = _build_signing_extra(signing_mode, dev_key)

    plugin, spec, run_context_cls, engine_kind_cls = _plugin_for_suite(suite_id)

    pairs: list[tuple[dict[str, Any], int]] = [(t, p) for t in targets for p in sweep]
    summary_rows: list[dict[str, Any]] = []
    any_error = False
    any_envelope = False

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=err_console,
        transient=False,
    ) as progress:
        task_ids: dict[tuple[str, int], TaskID] = {}
        for target, point in pairs:
            label = f"{target['name']} c{point}"
            task_ids[(target["name"], point)] = progress.add_task(label, total=1)

        for target, point in pairs:
            target_name = target["name"]
            tid = task_ids[(target_name, point)]
            progress.update(tid, description=f"{target_name} c{point} running")

            status, row = _execute_pair(
                target=target,
                point=point,
                plugin=plugin,
                spec=spec,
                run_context_cls=run_context_cls,
                engine_kind_cls=engine_kind_cls,
                output_dir=output_dir,
                signing_extra=signing_extra,
                duration_s=duration_s,
            )
            summary_rows.append(row)

            if status == "skip":
                progress.update(tid, completed=1, description=f"{target_name} c{point} skip")
                continue
            if status == "error":
                any_error = True
                progress.update(tid, completed=1, description=f"{target_name} c{point} fail")
                if not continue_on_error:
                    _print_summary_table(summary_rows)
                    raise typer.Exit(code=1)
                continue

            any_envelope = True
            progress.update(
                tid, completed=1, description=f"{target_name} c{point} ok"
            )

    _print_summary_table(summary_rows)

    if not any_envelope:
        err_console.print("[red]Matrix produced no envelopes.[/red]")
        raise typer.Exit(code=1)
    if not continue_on_error and any_error:
        raise typer.Exit(code=1)
