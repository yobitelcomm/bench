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


def _write_envelope(envelope: Envelope, output_dir: Path) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    content_hash = envelope.content_hash()
    out_path = output_dir / f"{content_hash[:12]}.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(envelope.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp_path, out_path)
    return out_path, content_hash


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
) -> None:
    """Run a benchmark from the named suite and emit a signed envelope."""
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

    if list_:
        _print_benchmark_list(ep.name, specs)
        raise typer.Exit(code=0)

    spec = _select_spec(specs, full_id, ep.name)

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

    output_dir = Path(output) if output else Path.cwd() / "results"
    extra = _build_signing_extra(signing_mode, dev_key)
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
