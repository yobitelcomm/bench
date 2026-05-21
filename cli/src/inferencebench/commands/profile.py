"""``bench profile`` — re-run a benchmark with high-frequency telemetry.

Where :mod:`inferencebench.commands.replay` is tuned for reproducibility (same
config, signed envelope diff against the source), ``bench profile`` is tuned
for diagnostic detail: the same envelope-driven re-run, but with NVML and RAPL
samplers cranked up to 10 ms / 25 ms intervals respectively. The default
``--duration`` is short (30 s) because profiling is for inspection, not for
steady-state metrics.

After the run completes, in addition to the side-by-side metric table we emit
a profiling breakdown — % time on host, GPU-vs-CPU/DRAM energy ratio, average
power under load, and the raw NVML/RAPL sample counts — so the user can
quickly answer "where did my throughput go?".
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.status import Status
from rich.table import Table

from inferencebench.commands.run import (
    _build_signing_extra,
    _entry_points,
    _find_entry_point,
    _resolve_plugin_schemas,
    _split_suite_id,
    _write_envelope,
)
from inferencebench.envelope import Envelope, verify_envelope

console = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Envelope loader                                                             #
# --------------------------------------------------------------------------- #
def _load_envelope(uri: str) -> Envelope:
    """Load an envelope from a local path. Phase 1: local paths only."""
    if uri.startswith(("hf://", "https://", "s3://")):
        err_console.print(
            f"[red]URI scheme not yet supported in v0.0.0:[/red] {uri.split('://')[0]}://"
        )
        err_console.print("Phase 1 supports local file paths only. Download the envelope first.")
        raise typer.Exit(code=2)

    path = Path(uri)
    if not path.exists():
        err_console.print(f"[red]Envelope not found:[/red] {path}")
        raise typer.Exit(code=2)

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        err_console.print(f"[red]Invalid JSON in envelope:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    try:
        return Envelope.model_validate(raw)
    except Exception as exc:
        err_console.print(f"[red]Envelope schema validation failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc


# --------------------------------------------------------------------------- #
# Summary table                                                               #
# --------------------------------------------------------------------------- #
_HEADLINE_METRICS: tuple[str, ...] = (
    "throughput_tok_per_s",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "tpot_p50_ms",
    "total_p50_ms",
    "ok_rate",
    "compliance_rate",
    "req_per_s_passing",
    "cost_usd_per_million_tokens",
    "power_avg_w",
    "joules_per_token",
)


def _fmt_metric(value: Any) -> str:  # noqa: ANN401
    if isinstance(value, int | float):
        return f"{value:.4g}"
    return "-"


def _print_profile_summary(
    source: Envelope,
    source_uri: str,
    profile: Envelope,
    profile_path: Path,
) -> None:
    """Render a side-by-side Rich summary so the user can eyeball reproducibility."""
    header = Table(title="Profile summary", show_header=True, header_style="bold")
    header.add_column("field", style="cyan")
    header.add_column("source", style="bold")
    header.add_column("profile", style="bold")
    header.add_column("match", justify="center")

    def _row(label: str, a: str, b: str) -> None:
        ok = a == b
        marker = "[green]yes[/green]" if ok else "[red]no[/red]"
        header.add_row(label, a, b, marker)

    _row("envelope", source_uri, str(profile_path))
    _row("suite_id", source.suite_id, profile.suite_id)
    _row("model.id", source.model.id, profile.model.id)
    _row(
        "engine.name",
        source.engine.name,
        profile.engine.name,
    )
    src_quant = source.quantization.format if source.quantization else ""
    rep_quant = profile.quantization.format if profile.quantization else ""
    _row("quantization", src_quant or "-", rep_quant or "-")
    _row("dataset.id", source.dataset.id, profile.dataset.id)
    _row("seed", str(source.seed), str(profile.seed))
    _row("slo_template", source.slo_template, profile.slo_template)
    console.print(header)

    metrics_table = Table(
        title="Headline metrics (source vs profile)",
        show_header=True,
        header_style="bold",
    )
    metrics_table.add_column("metric", style="cyan")
    metrics_table.add_column("source", justify="right")
    metrics_table.add_column("profile", justify="right")

    keys: list[str] = []
    for key in _HEADLINE_METRICS:
        if key in source.metrics or key in profile.metrics:
            keys.append(key)
    for key in keys:
        metrics_table.add_row(
            key,
            _fmt_metric(source.metrics.get(key)),
            _fmt_metric(profile.metrics.get(key)),
        )
    console.print(metrics_table)


# --------------------------------------------------------------------------- #
# Profiling breakdown                                                         #
# --------------------------------------------------------------------------- #
def _avg_gpu_util_pct(envelope: Envelope) -> float | None:
    """Average GPU util_gpu_pct across every NVML sample x every device.

    Envelopes don't currently carry raw NVML samples, so we read the aggregated
    metric ``gpu_util_avg_pct`` if it exists. When absent we return ``None``.
    """
    for key in ("gpu_util_avg_pct", "util_gpu_pct_avg", "util_avg_pct"):
        v = envelope.metrics.get(key)
        if isinstance(v, int | float):
            return float(v)
    return None


def _print_profiling_breakdown(
    profile: Envelope,
    *,
    nvml_interval_ms: int,
    rapl_interval_ms: int,
) -> None:
    """Emit the profile-specific diagnostic table."""
    metrics = profile.metrics

    table = Table(
        title="Profiling breakdown",
        show_header=True,
        header_style="bold",
    )
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right", style="bold")
    table.add_column("note", style="dim")

    util_avg = _avg_gpu_util_pct(profile)
    if util_avg is not None:
        host_pct = max(0.0, 100.0 - float(util_avg))
        table.add_row(
            "% time on host",
            f"{host_pct:.2f}%",
            f"100 - avg GPU util ({util_avg:.2f}%)",
        )
    else:
        table.add_row(
            "% time on host",
            "-",
            "no gpu_util_avg_pct in envelope",
        )

    gpu_e = metrics.get("energy_joules_gpu")
    cpu_e = metrics.get("energy_joules_cpu_dram")
    if not isinstance(gpu_e, int | float):
        gpu_e = metrics.get("energy_joules_total")
        cpu_e = None
    if isinstance(gpu_e, int | float) and isinstance(cpu_e, int | float) and cpu_e > 0:
        ratio = float(gpu_e) / float(cpu_e)
        table.add_row(
            "Energy GPU vs CPU+DRAM",
            f"{ratio:.3f}",
            f"{gpu_e:.4g} J / {cpu_e:.4g} J",
        )
    elif isinstance(gpu_e, int | float):
        table.add_row(
            "Energy GPU vs CPU+DRAM",
            "-",
            f"GPU={gpu_e:.4g} J, CPU+DRAM unavailable",
        )
    else:
        table.add_row(
            "Energy GPU vs CPU+DRAM",
            "-",
            "no energy_joules_* in envelope",
        )

    load_power = metrics.get("power_avg_w_under_load")
    if isinstance(load_power, int | float):
        table.add_row(
            "Avg power under load",
            f"{load_power:.2f} W",
            "samples where util_gpu > 50%",
        )
    else:
        fallback = metrics.get("power_avg_w")
        if isinstance(fallback, int | float):
            table.add_row(
                "Avg power under load",
                f"{fallback:.2f} W",
                "fallback: power_avg_w (no load filter)",
            )
        else:
            table.add_row(
                "Avg power under load",
                "-",
                "no power_avg_w_under_load in envelope",
            )

    nvml_count = metrics.get("nvml_sample_count")
    if isinstance(nvml_count, int | float):
        table.add_row(
            "NVML sample count",
            f"{int(nvml_count):d}",
            f"interval={nvml_interval_ms} ms",
        )
    else:
        table.add_row(
            "NVML sample count",
            "-",
            f"interval={nvml_interval_ms} ms; count not aggregated into envelope",
        )

    rapl_count = metrics.get("rapl_sample_count")
    if isinstance(rapl_count, int | float):
        table.add_row(
            "RAPL sample count",
            f"{int(rapl_count):d}",
            f"interval={rapl_interval_ms} ms",
        )
    else:
        table.add_row(
            "RAPL sample count",
            "-",
            f"interval={rapl_interval_ms} ms; count not aggregated into envelope",
        )

    console.print(table)
    console.print(
        "[dim]Profiling overrides telemetry intervals — metrics here are for "
        "inspection, not steady-state comparisons.[/dim]"
    )


# --------------------------------------------------------------------------- #
# CLI command                                                                 #
# --------------------------------------------------------------------------- #
def profile(
    envelope_path: Annotated[
        str,
        typer.Argument(help="Path to the source envelope JSON to profile."),
    ],
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help=(
                "Engine base URL for the profile run (e.g. http://localhost:8000/v1). "
                "Required: envelopes are deliberately host-agnostic and do not "
                "store live endpoints."
            ),
        ),
    ] = "",
    duration: Annotated[
        int,
        typer.Option(
            "--duration",
            help=(
                "Measurement duration in seconds. Defaults to 30 s — profiling "
                "is for inspection, not steady-state metrics."
            ),
        ),
    ] = 30,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            help="Output directory for the profile envelope.",
        ),
    ] = "",
    signing_mode: Annotated[
        str,
        typer.Option(
            "--signing-mode",
            help="Envelope signing mode: 'dev' (local cosign key) or 'keyless'.",
        ),
    ] = "dev",
    dev_key: Annotated[
        str,
        typer.Option(
            "--dev-key",
            help="Path to local cosign signing key (used when --signing-mode=dev).",
        ),
    ] = "",
    verify: Annotated[
        bool,
        typer.Option(
            "--verify/--no-verify",
            help=(
                "Verify the source envelope's signature before profiling. "
                "A bad envelope shouldn't seed a profile."
            ),
        ),
    ] = True,
) -> None:
    """Re-run a benchmark at high telemetry frequency to diagnose throughput.

    Same plumbing as ``bench replay`` but sets NVML to 10 ms / RAPL to 25 ms
    and a short default duration. Useful for answering "where did my
    throughput go?" — i.e. is the bottleneck the GPU, the host, or energy.
    """
    source = _load_envelope(envelope_path)

    if verify:
        result = verify_envelope(source)
        if not result.ok:
            err_console.print(
                f"[bold red]FAIL[/bold red] source envelope failed verification: {result.reason}"
            )
            err_console.print(
                "[red]Refusing to profile an unverified envelope.[/red] "
                "Pass [bold]--no-verify[/bold] to bypass (e.g. for unsigned local fixtures)."
            )
            raise typer.Exit(code=1)

    if not base_url:
        err_console.print(
            "[red]--base-url is required for bench profile.[/red] "
            "Envelopes are host-agnostic and do not store the live engine URL — "
            "you must point this profile at a running engine "
            "(e.g. [bold]--base-url http://localhost:8000/v1[/bold])."
        )
        raise typer.Exit(code=2)

    # Plugin discovery via the envelope's suite_id.
    eps = _entry_points()
    plugin_name, _full_id = _split_suite_id(source.suite_id)
    ep = _find_entry_point(eps, plugin_name, source.suite_id)

    try:
        plugin_cls = ep.load()
    except Exception as exc:  # pragma: no cover - defensive
        err_console.print(f"[red]Failed to load plugin '{ep.name}':[/red] {exc}")
        raise typer.Exit(code=1) from exc
    plugin = plugin_cls()

    try:
        spec = plugin.get_benchmark(source.suite_id)
    except KeyError:
        err_console.print(
            f"[red]Plugin '{ep.name}' no longer ships benchmark id "
            f"'{source.suite_id}'.[/red] "
            "The plugin may have been upgraded and dropped this benchmark. "
            "Pin to the plugin version recorded in the source envelope to profile."
        )
        raise typer.Exit(code=1) from None
    except Exception as exc:  # pragma: no cover - defensive
        err_console.print(f"[red]Plugin error while resolving spec:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        run_context_cls, engine_kind_cls = _resolve_plugin_schemas(ep)
    except RuntimeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    try:
        engine_kind = engine_kind_cls(source.engine.name)
    except ValueError as exc:
        err_console.print(f"[red]Unknown engine in source envelope:[/red] {source.engine.name}")
        raise typer.Exit(code=1) from exc

    output_dir = Path(output) if output else Path.cwd() / "profile-results"
    try:
        signing_extra = _build_signing_extra(signing_mode, dev_key)
    except typer.Exit:
        raise

    quant_fmt = source.quantization.format if source.quantization else ""

    # Profile-specific telemetry overrides — much tighter than `bench run`.
    nvml_interval_ms = 10
    rapl_interval_ms = 25

    extra: dict[str, str | int | float | bool] = dict(signing_extra)
    extra["duration_s"] = int(duration)
    extra["nvml_interval_ms"] = nvml_interval_ms
    extra["rapl_interval_ms"] = rapl_interval_ms

    try:
        ctx = run_context_cls(
            model_id=source.model.id,
            engine_kind=engine_kind,
            base_url=base_url,
            quantization_format=quant_fmt,
            output_dir=output_dir,
            extra=extra,
        )
    except Exception as exc:
        err_console.print(f"[red]Invalid run context:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        with Status("[bold]Profiling benchmark…[/bold]", console=err_console):
            new_envelope = plugin.run(spec, ctx)
    except Exception as exc:
        err_console.print(f"[red]Profile run failed:[/red] {exc}")
        err_console.print("[red]" + traceback.format_exc() + "[/red]")
        raise typer.Exit(code=1) from exc

    out_path, _content_hash = _write_envelope(new_envelope, output_dir, prefix="profile")
    _print_profile_summary(source, envelope_path, new_envelope, out_path)
    _print_profiling_breakdown(
        new_envelope,
        nvml_interval_ms=nvml_interval_ms,
        rapl_interval_ms=rapl_interval_ms,
    )
