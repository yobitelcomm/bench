"""``bench tour`` — interactive end-to-end install-validation walkthrough.

After ``pip install inferencebench`` a new user typically wants a quick
"prove it works" command that exercises the major capabilities against
bundled fixtures. ``bench tour`` is that command.

The tour runs every step locally — no GPU, no running engine, no network
access. It generates a dev keypair, builds a fake (but schema-valid)
envelope, signs it, verifies it, exports / bundles / leaderboards / audits
it, and prints a final Rich table summarising each step. The user gets an
unambiguous "everything works" signal in under a minute.

Implementation notes:

* The tour must NEVER subprocess back to ``bench`` — it imports and calls
  the underlying functions directly. That keeps the tour fast, makes
  failures debuggable, and lets the tests run without the entry-point
  console-script being installed.
* The fake envelope is built via :class:`EnvelopeBuilder` with a
  deterministic-but-non-placeholder hardware fingerprint (the placeholder
  ``0`` * 64 fingerprint is the canonical "fake hardware" marker the
  ``bench audit`` command rejects).
"""

from __future__ import annotations

import json
import tempfile
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.status import Status
from rich.table import Table

from inferencebench.envelope import (
    BIOS,
    CPU,
    GPU,
    DatasetSpec,
    EngineConfig,
    EnvelopeBuilder,
    HardwareFingerprint,
    Memory,
    ModelConfig,
    SigningMode,
    SoftwareProvenance,
    generate_dev_keypair,
    sign_envelope,
    verify_envelope,
)

console = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Deterministic fake-envelope construction                                    #
# --------------------------------------------------------------------------- #
def _tour_hardware_fingerprint() -> HardwareFingerprint:
    """Build a deterministic but non-placeholder hardware fingerprint.

    The placeholder ``0`` * 64 sha is the canonical "fake hardware" marker
    that ``bench audit`` rejects — using real values here means the
    tour-produced envelope passes the audit step.
    """
    body: dict[str, Any] = {
        "dmi_uuid": "tour-uuid-deadbeef-deadbeef-deadbeef",
        "gpus": [
            GPU(
                model="H100-SXM5-80GB",
                pci_id="0000:01:00.0",
                serial="tour-serial-0001",
                vbios="96.00.74.00.01",
            )
        ],
        "cpu": CPU(
            model="Intel(R) Xeon(R) Platinum 8480C", microcode="0x2b000571"
        ),
        "memory": Memory(channels=12, speed_mts=4800, ecc=True),
        "bios": BIOS(version="3.4a", resizable_bar=True, above_4g=True),
        "driver": "560.35.03",
        "cuda": "12.6",
        "nccl": "2.22.3",
    }
    placeholder = HardwareFingerprint.model_construct(
        fingerprint_sha256="0" * 64, numa={}, **body
    )
    real = placeholder.compute_fingerprint_sha256()
    return HardwareFingerprint(fingerprint_sha256=real, numa={}, **body)


def _build_tour_envelope() -> Any:  # noqa: ANN401 — Envelope; avoid extra import-cycle risk
    """Build (don't sign yet) an envelope filled with plausible tour values."""
    builder = EnvelopeBuilder(
        suite_id="llm.inference",
        suite_version="1.0.0",
        model=ModelConfig(
            id="tour-demo/llm-tour-model",
            revision="tour001",
            provider="tour-mock",
            endpoint_hash="a" * 64,
        ),
        engine=EngineConfig(name="vllm", version="0.7.2", config_hash="b" * 64),
        hardware_fingerprint=_tour_hardware_fingerprint(),
        software_provenance=SoftwareProvenance(
            pip_freeze_hash="c" * 64,
            git_commit="tour001",
        ),
        dataset=DatasetSpec(id="tour-dataset", hash="d" * 64),
        seed=42,
        metrics={
            "throughput_tok_per_s": 1500.0,
            "ttft_p50_ms": 80.0,
            "ttft_p99_ms": 240.0,
            "tpot_p50_ms": 12.0,
            "ok_rate": 1.0,
            "power_avg_w": 720.0,
            "joules_per_token": 0.85,
            "cost_usd_per_million_tokens": 0.42,
        },
        slo_template="llm.standard",
    )
    return builder.build()


# --------------------------------------------------------------------------- #
# Step wrappers                                                               #
# --------------------------------------------------------------------------- #
class _StepResult:
    """One row in the tour summary table."""

    __slots__ = ("detail", "name", "ok")

    def __init__(self, name: str, ok: bool, detail: str = "") -> None:
        self.name = name
        self.ok = ok
        self.detail = detail


def _step_list_plugins() -> _StepResult:
    """Verify plugins are discoverable via entry points."""
    # Import the helper lazily so a fresh interpreter would still work — and
    # to keep the import surface of this module narrow.
    from inferencebench.commands.run import _entry_points

    eps = _entry_points()
    names = sorted(ep.name for ep in eps)
    detail = ", ".join(names) if names else "<none>"
    return _StepResult("bench list (plugins)", ok=True, detail=detail)


def _step_plugin_init() -> _StepResult:
    """Scaffold a throwaway plugin into a temp dir to confirm ``plugin init`` works."""
    from inferencebench.commands.plugin import init_plugin

    with tempfile.TemporaryDirectory(prefix="bench-tour-plugin-") as tmp:
        tmp_path = Path(tmp)
        cwd = Path.cwd()
        try:
            # init_plugin scaffolds under ./plugins/<name>; cd into the tmp dir.
            import os

            os.chdir(tmp_path)
            init_plugin("bench-tour-demo", kind="perf", modality="llm")
            created = tmp_path / "plugins" / "bench-tour-demo" / "pyproject.toml"
            if not created.exists():
                return _StepResult(
                    "bench plugin init",
                    ok=False,
                    detail="scaffolded pyproject.toml missing",
                )
            return _StepResult(
                "bench plugin init", ok=True, detail="bench-tour-demo scaffolded"
            )
        finally:
            os.chdir(cwd)


def _step_generate_keypair(dev_key: Path) -> _StepResult:
    """Generate a dev keypair if one doesn't already exist."""
    pub_path = dev_key.with_suffix(".pub")
    if dev_key.exists() and pub_path.exists():
        return _StepResult(
            "generate dev keypair", ok=True, detail=f"reused {dev_key.name}"
        )
    generate_dev_keypair(dev_key, force=dev_key.exists() or pub_path.exists())
    return _StepResult(
        "generate dev keypair", ok=True, detail=f"wrote {dev_key.name} + .pub"
    )


def _step_build_envelope(out_dir: Path, dev_key: Path) -> tuple[_StepResult, Path]:
    """Build + sign a fake envelope and write it to ``<out>/tour-envelope.json``."""
    envelope = _build_tour_envelope()
    signed = sign_envelope(envelope, mode=SigningMode.DEV, dev_key_path=dev_key)
    out_path = out_dir / "tour-envelope.json"
    out_path.write_text(signed.model_dump_json(indent=2), encoding="utf-8")
    return (
        _StepResult(
            "build + sign fake envelope",
            ok=True,
            detail=f"tour-envelope.json ({signed.content_hash()[:12]})",
        ),
        out_path,
    )


def _step_verify_envelope(envelope_path: Path, dev_key: Path) -> _StepResult:
    """Verify the signed envelope using the dev public key."""
    pub_path = dev_key.with_suffix(".pub")
    raw = json.loads(envelope_path.read_text(encoding="utf-8"))
    # Re-import Envelope locally to avoid an additional top-level import.
    from inferencebench.envelope import Envelope

    envelope = Envelope.model_validate(raw)
    result = verify_envelope(envelope, dev_public_key_path=pub_path)
    if not result.ok:
        return _StepResult(
            "bench verify", ok=False, detail=result.reason or "verification failed"
        )
    return _StepResult(
        "bench verify", ok=True, detail=f"method={result.method}"
    )


def _step_summary(out_dir: Path) -> _StepResult:
    """Run the summary command on the envelope directory.

    We call the underlying helpers directly rather than subprocess'ing
    back into ``bench summary`` so failures surface as Python tracebacks.
    """
    from inferencebench.commands.summary import (
        _collect_json_files,
        _group_by_suite,
        _load_envelopes,
    )

    candidates = _collect_json_files(out_dir)
    envelopes, skipped = _load_envelopes(candidates)
    suites = _group_by_suite(envelopes)
    return _StepResult(
        "bench summary",
        ok=True,
        detail=(
            f"{len(envelopes)} envelopes, {len(suites)} suite(s), "
            f"{skipped} skipped"
        ),
    )


def _step_export_markdown(envelope_path: Path, out_dir: Path) -> _StepResult:
    """Export the envelope as markdown to ``<out>/tour-envelope.md``."""
    from inferencebench.commands.export import _filter_metrics, _render_markdown
    from inferencebench.envelope import Envelope

    envelope = Envelope.model_validate(
        json.loads(envelope_path.read_text(encoding="utf-8"))
    )
    metrics = _filter_metrics(envelope.metrics, keep=None)
    rendered = _render_markdown(envelope, metrics)
    md_path = out_dir / "tour-envelope.md"
    md_path.write_text(rendered, encoding="utf-8")
    return _StepResult(
        "bench export --format markdown", ok=True, detail=f"→ {md_path.name}"
    )


def _step_bundle_create(envelope_path: Path, out_dir: Path, dev_key: Path) -> _StepResult:
    """Build a ``.bundle.zip`` containing the envelope + public key."""
    # Import the underlying create function directly. Typer wraps the
    # callable but the wrapped function is still importable.
    from inferencebench.commands.bundle import bundle_create

    pub_path = dev_key.with_suffix(".pub")
    bundle_path = out_dir / "tour.bundle.zip"
    bundle_create(
        envelope_path=envelope_path,
        out=bundle_path,
        include_samples=False,
        include_public_key=pub_path,
    )
    return _StepResult(
        "bench bundle create",
        ok=True,
        detail=f"→ {bundle_path.name}",
    )


def _step_leaderboard(out_dir: Path) -> _StepResult:
    """Render the static leaderboard site from the tour envelope directory."""
    try:
        from inferencebench_leaderboard import render_site
    except ImportError:
        return _StepResult(
            "bench leaderboard --build",
            ok=False,
            detail="inferencebench-leaderboard not installed",
        )
    site_dir = out_dir / "site"
    site_dir.mkdir(parents=True, exist_ok=True)
    result = render_site(out_dir, site_dir, base_url="/")
    return _StepResult(
        "bench leaderboard --build",
        ok=True,
        detail=(
            f"{result.envelopes_loaded} envelopes, "
            f"{len(result.categories)} categor(y/ies)"
        ),
    )


def _step_audit(out_dir: Path, dev_key: Path) -> _StepResult:
    """Audit every envelope in the tour output directory."""
    from inferencebench.commands.audit import _audit_one

    pub_path = dev_key.with_suffix(".pub")
    # Only audit the envelope file itself — the bundle zip and rendered md
    # would otherwise be picked up as junk JSON.
    target = out_dir / "tour-envelope.json"
    row = _audit_one(target, pub_path)
    if not row["ok"]:
        return _StepResult(
            "bench audit", ok=False, detail=row.get("reason", "audit failure")
        )
    return _StepResult(
        "bench audit", ok=True, detail=f"verified {row['content_hash_short']}"
    )


# --------------------------------------------------------------------------- #
# Tour command                                                                #
# --------------------------------------------------------------------------- #
def tour(
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Scratch directory for envelopes, bundles, and site output.",
        ),
    ] = Path("./bench-tour-output"),
    dev_key: Annotated[
        Path | None,
        typer.Option(
            "--dev-key",
            help="Path to a dev signing key (generated if absent).",
        ),
    ] = None,
    with_vllm: Annotated[
        bool,
        typer.Option(
            "--with-vllm/--no-with-vllm",
            help=(
                "Reserved: when set, the tour would invoke a real `bench run` "
                "against --base-url. v0.0.0 always uses the mock-engine path."
            ),
        ),
    ] = False,
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help="Engine base URL (used only when --with-vllm is set).",
        ),
    ] = "",
) -> None:
    """Run a curated end-to-end walkthrough exercising every major capability.

    The tour produces a signed envelope, verifies it, exports / bundles /
    leaderboards / audits it, and prints a Rich table summarising each
    step. Designed to be the first thing a new user runs after ``pip
    install inferencebench`` — exit 0 means "your install works".
    """
    _ = with_vllm  # reserved for Phase 2+; surface it in --help for now
    _ = base_url

    out.mkdir(parents=True, exist_ok=True)
    if dev_key is None:
        dev_key = out / "cosign.key"

    results: list[_StepResult] = []

    def _run(step_name: str, fn: Callable[[], _StepResult]) -> _StepResult:
        """Wrap a step in a Rich Status; capture failures into the summary table."""
        try:
            with Status(
                f"[bold]{step_name}[/bold] running…", console=err_console
            ):
                return fn()
        except Exception as exc:
            err_console.print(
                f"[red]{step_name} failed:[/red] {exc}\n"
                + traceback.format_exc()
            )
            return _StepResult(step_name, ok=False, detail=str(exc)[:80])

    # 1. bench list
    results.append(_run("bench list", _step_list_plugins))

    # 2. bench plugin init bench-tour-demo
    results.append(_run("bench plugin init", _step_plugin_init))

    # 3. generate dev keypair
    results.append(_run("generate dev keypair", lambda: _step_generate_keypair(dev_key)))

    # 4. build + sign fake envelope
    envelope_path: Path | None = None
    try:
        with Status("[bold]build + sign envelope[/bold] running…", console=err_console):
            step, envelope_path = _step_build_envelope(out, dev_key)
        results.append(step)
    except Exception as exc:
        err_console.print(
            f"[red]build + sign envelope failed:[/red] {exc}\n"
            + traceback.format_exc()
        )
        results.append(
            _StepResult("build + sign fake envelope", ok=False, detail=str(exc)[:80])
        )

    # 5. bench verify
    if envelope_path is not None:
        # Bind the non-None path to a local so the lambda's type is unambiguous.
        env_path_verify = envelope_path
        results.append(
            _run("bench verify", lambda: _step_verify_envelope(env_path_verify, dev_key))
        )
    else:
        results.append(_StepResult("bench verify", ok=False, detail="no envelope"))

    # 6. bench summary
    results.append(_run("bench summary", lambda: _step_summary(out)))

    # 7. bench export --format markdown
    if envelope_path is not None:
        env_path_export = envelope_path
        results.append(
            _run(
                "bench export --format markdown",
                lambda: _step_export_markdown(env_path_export, out),
            )
        )
    else:
        results.append(
            _StepResult("bench export --format markdown", ok=False, detail="no envelope")
        )

    # 8. bench bundle create
    if envelope_path is not None:
        env_path_bundle = envelope_path
        results.append(
            _run(
                "bench bundle create",
                lambda: _step_bundle_create(env_path_bundle, out, dev_key),
            )
        )
    else:
        results.append(
            _StepResult("bench bundle create", ok=False, detail="no envelope")
        )

    # 9. bench leaderboard --build
    results.append(_run("bench leaderboard --build", lambda: _step_leaderboard(out)))

    # 10. bench audit
    results.append(_run("bench audit", lambda: _step_audit(out, dev_key)))

    _print_summary_table(results, out)
    n_fail = sum(1 for r in results if not r.ok)
    if n_fail:
        raise typer.Exit(code=1)


def _print_summary_table(results: list[_StepResult], out_dir: Path) -> None:
    """Print the final Rich table summarising every tour step."""
    table = Table(title="bench tour — summary")
    table.add_column("#", justify="right", style="dim")
    table.add_column("step", style="cyan")
    table.add_column("status", justify="center")
    table.add_column("detail", style="dim")
    for idx, r in enumerate(results, start=1):
        marker = "[bold green]✓[/]" if r.ok else "[bold red]✗[/]"
        table.add_row(str(idx), r.name, marker, r.detail)
    console.print(table)
    n_ok = sum(1 for r in results if r.ok)
    console.print(
        f"[bold]{n_ok}[/bold] / {len(results)} steps passed. "
        f"Outputs in [cyan]{out_dir.resolve()}[/cyan]"
    )
    console.print(
        "Next: open the markdown summary, "
        "share the bundle zip, or browse the rendered site."
    )
    # Timestamp helps users correlate tour runs with later artefacts.
    console.print(f"[dim]tour completed at {datetime.now(UTC).isoformat()}[/dim]")
