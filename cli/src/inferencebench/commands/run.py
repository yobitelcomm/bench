"""``bench run`` — execute a benchmark and produce a signed envelope.

Phase 1 stub. Real implementation lands in ticket 0025.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

err_console = Console(stderr=True)


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
    slo_template: Annotated[
        str,
        typer.Option(
            "--slo-template",
            help="SLO template (llm.standard, voice.realtime, ...).",
        ),
    ] = "llm.standard",
    seed: Annotated[int, typer.Option("--seed", help="Random seed for reproducibility.")] = 42,
    output: Annotated[
        str, typer.Option("--output", help="Output path for the signed envelope.")
    ] = "",
) -> None:
    """Run a benchmark from the named suite.

    Phase 1 stub — emits a "[stub]" message and exits 0. Ticket 0025 wires the harness.
    """
    err_console.print(
        f"[yellow][stub][/yellow] bench run [bold]{suite_id}[/bold] "
        f"--model {model or '<none>'} --engine {engine} --hardware {hardware} "
        f"--quant {quant} --concurrency {concurrency} --duration {duration}s — "
        "not yet implemented in v0.0.0 (ticket 0025)."
    )
