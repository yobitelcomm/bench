"""``bench cluster`` — runner-side coordinator for distributed envelopes.

``bench server`` (see ``commands/server.py``) is the central envelope receiver.
``bench cluster`` is the runner side of that contract: it reads a matrix-style
config, dispatches each ``(target, sweep-point)`` pair as a benchmark run via
the existing ``bench matrix`` execution path, then optionally POSTs each
resulting envelope to a configured ``bench server`` endpoint.

This is a Phase 2 unlock. The Phase 1 skeleton shipped here runs locally one
pair at a time (true parallel dispatch is deferred to Phase 3), reuses the
``bench matrix`` helpers, and talks to the server via stdlib ``urllib`` so we
don't pick up an HTTP client dependency.

Subcommands:
    - ``run`` — execute a matrix config, optionally POST each envelope.
    - ``status`` — GET ``<server>/envelopes`` and tabulate what the server has.
    - ``sync`` — pull every envelope from a server down to a local directory.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from inferencebench.commands.matrix import (
    _execute_pair,
    _load_yaml,
    _plugin_for_suite,
    _print_summary_table,
    _validate_matrix_config,
)
from inferencebench.commands.run import _build_signing_extra, _write_envelope
from inferencebench.envelope import Envelope

app = typer.Typer(no_args_is_help=True)
console = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# HTTP helpers (stdlib-only)                                                  #
# --------------------------------------------------------------------------- #
def _post_envelope(server_url: str, envelope: Envelope) -> tuple[int, str]:
    """POST a single envelope to ``<server_url>/envelopes``.

    Returns ``(status_code, body_text)``. On a transport-level failure the
    status is 0 and the body carries the error message.
    """
    payload = json.dumps(envelope.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    url = server_url.rstrip("/") + "/envelopes"
    req = urllib.request.Request(  # noqa: S310 — bench-server HTTP base URL is operator-supplied
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except OSError:
            pass
        return int(exc.code), body
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return 0, str(exc)


def _get_json(url: str) -> tuple[int, Any]:
    """GET a URL expected to return JSON. Returns ``(status, parsed_or_error)``."""
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            raw = resp.read()
            status = int(resp.status)
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read()
        except OSError:
            raw = b""
        return int(exc.code), raw
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return 0, str(exc)

    try:
        return status, json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return status, f"invalid JSON from {url}: {exc}"


# --------------------------------------------------------------------------- #
# bench cluster run                                                           #
# --------------------------------------------------------------------------- #
@app.command("run")
def cluster_run(
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
    ] = "./envelopes/",
    server_url: Annotated[
        str,
        typer.Option(
            "--server-url",
            help=(
                "Optional ``bench server`` base URL. If set, each successful "
                "envelope is POSTed to ``<server-url>/envelopes`` after being "
                "written to ``--output`` on disk."
            ),
        ),
    ] = "",
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
    """Run each (target, sweep-point) sequentially; optionally POST envelopes."""
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
    posted = 0
    post_failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=err_console,
        transient=False,
    ) as progress:
        task_ids = {
            (t["name"], p): progress.add_task(f"{t['name']} c{p}", total=1) for t, p in pairs
        }
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

            if server_url:
                envelope_path = output_dir / row["envelope_name"]
                try:
                    envelope = Envelope.model_validate_json(
                        envelope_path.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError) as exc:
                    err_console.print(
                        f"[yellow]warning:[/yellow] could not reload "
                        f"{envelope_path.name} for POST: {exc}"
                    )
                    post_failed += 1
                else:
                    code, body = _post_envelope(server_url, envelope)
                    if code == 201:
                        posted += 1
                    else:
                        err_console.print(
                            f"[yellow]warning:[/yellow] POST {envelope_path.name} "
                            f"-> {code or 'connection-error'}: {body[:200]}"
                        )
                        post_failed += 1

            progress.update(tid, completed=1, description=f"{target_name} c{point} ok")

    _print_summary_table(summary_rows)
    if server_url:
        console.print(f"[bold]POST summary:[/bold] {posted} succeeded, {post_failed} failed.")

    if not any_envelope:
        err_console.print("[red]Cluster run produced no envelopes.[/red]")
        raise typer.Exit(code=1)
    if not continue_on_error and any_error:
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# bench cluster status                                                        #
# --------------------------------------------------------------------------- #
@app.command("status")
def cluster_status(
    server_url: Annotated[
        str,
        typer.Option("--server-url", help="``bench server`` base URL to query."),
    ],
) -> None:
    """List envelopes currently held by a ``bench server`` instance."""
    if not server_url:
        err_console.print("[red]--server-url is required.[/red]")
        raise typer.Exit(code=1)

    url = server_url.rstrip("/") + "/envelopes"
    status, payload = _get_json(url)
    if status == 0:
        err_console.print(f"[red]Cannot reach {url}:[/red] {payload}")
        raise typer.Exit(code=1)
    if status != 200 or not isinstance(payload, dict):
        err_console.print(
            f"[red]Unexpected response from {url}:[/red] status={status} body={payload!r}"
        )
        raise typer.Exit(code=1)

    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        err_console.print(f"[red]Malformed response from {url}:[/red] 'entries' is not a list.")
        raise typer.Exit(code=1)

    table = Table(title=f"Envelopes @ {server_url}")
    table.add_column("content_hash", style="cyan", no_wrap=True)
    table.add_column("suite_id", style="green")
    table.add_column("model_id")
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        table.add_row(
            str(entry.get("content_hash", ""))[:16],
            str(entry.get("suite_id", "")),
            str(entry.get("model_id", "")),
        )
    console.print(table)
    console.print(f"[dim]{len(entries)} envelope(s) on server.[/dim]")


# --------------------------------------------------------------------------- #
# bench cluster sync                                                          #
# --------------------------------------------------------------------------- #
@app.command("sync")
def cluster_sync(
    server_url: Annotated[
        str,
        typer.Option("--server-url", help="``bench server`` base URL to pull from."),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Destination directory for synced envelopes (created if missing).",
        ),
    ],
) -> None:
    """Pull every envelope from a server into a local directory."""
    if not server_url:
        err_console.print("[red]--server-url is required.[/red]")
        raise typer.Exit(code=1)

    out.mkdir(parents=True, exist_ok=True)

    list_url = server_url.rstrip("/") + "/envelopes"
    status, payload = _get_json(list_url)
    if status == 0:
        err_console.print(f"[red]Cannot reach {list_url}:[/red] {payload}")
        raise typer.Exit(code=1)
    if status != 200 or not isinstance(payload, dict):
        err_console.print(f"[red]Unexpected response from {list_url}:[/red] status={status}")
        raise typer.Exit(code=1)

    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        err_console.print(
            f"[red]Malformed response from {list_url}:[/red] 'entries' is not a list."
        )
        raise typer.Exit(code=1)

    synced = 0
    skipped = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=err_console,
        transient=False,
    ) as progress:
        task_id = progress.add_task("syncing envelopes", total=len(entries))
        for entry in entries:
            if not isinstance(entry, dict):
                failed += 1
                progress.update(task_id, advance=1)
                continue
            content_hash = str(entry.get("content_hash", ""))
            if not content_hash:
                failed += 1
                progress.update(task_id, advance=1)
                continue

            target_path = out / f"{content_hash[:12]}.json"
            if target_path.exists():
                skipped += 1
                progress.update(task_id, advance=1)
                continue

            get_url = server_url.rstrip("/") + f"/envelopes/{content_hash}"
            status_one, body = _get_json(get_url)
            if status_one == 0:
                err_console.print(f"[yellow]warning:[/yellow] could not reach {get_url}: {body}")
                failed += 1
                progress.update(task_id, advance=1)
                continue
            if status_one != 200 or not isinstance(body, dict):
                err_console.print(f"[yellow]warning:[/yellow] {get_url} -> {status_one}")
                failed += 1
                progress.update(task_id, advance=1)
                continue
            try:
                envelope = Envelope.model_validate(body)
            except ValueError as exc:
                err_console.print(
                    f"[yellow]warning:[/yellow] {content_hash[:12]} did not parse: {exc}"
                )
                failed += 1
                progress.update(task_id, advance=1)
                continue
            _write_envelope(envelope, out)
            synced += 1
            progress.update(task_id, advance=1)

    table = Table(title="cluster sync results")
    table.add_column("synced", justify="right", style="green")
    table.add_column("skipped (already present)", justify="right", style="yellow")
    table.add_column("failed", justify="right", style="red")
    table.add_row(str(synced), str(skipped), str(failed))
    console.print(table)
