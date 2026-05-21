"""``bench fetch`` — download a signed envelope from a remote URI to a local cache.

Supported URI schemes (Phase 1):

- ``hf://datasets/<owner>/<repo>[/<file>]`` — Hugging Face Hub dataset repo,
  fetched via :func:`huggingface_hub.hf_hub_download`. The ``<file>`` part is
  optional; if absent, the canonical filename ``envelope.json`` is assumed
  (which is what the publisher writes).
- ``https://...`` / ``http://...`` — plain HTTPS download via :mod:`urllib.request`.
- ``file://<path>`` or a plain local path — local copy.

The downloaded payload is validated as an :class:`Envelope`. If validation fails,
the file is left in place so the user can ``cat`` it for debugging.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from inferencebench.envelope import Envelope

console = Console()
err_console = Console(stderr=True)

_DEFAULT_HF_FILENAME = "envelope.json"


def fetch(
    uri: Annotated[
        str,
        typer.Argument(
            help=(
                "Envelope URI: hf://datasets/OWNER/REPO (optional /FILE), "
                "https://..., file://..., or a local path."
            ),
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Local path to write the envelope to. "
            "Defaults to ~/.cache/inferencebench/fetched/<sha256-of-uri:12>.json.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force/--no-force",
            help="Re-download even if a cached copy already exists.",
        ),
    ] = False,
) -> None:
    """Download a signed envelope from a remote source to a local cache directory."""
    dest = out if out is not None else _default_cache_path(uri)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        console.print(f"[yellow]cache hit[/yellow]  {dest}")
    else:
        try:
            _download(uri, dest)
        except FetchError as exc:
            err_console.print(f"[red]fetch failed:[/red] {exc}")
            raise typer.Exit(code=2) from exc

    envelope = _validate_envelope(dest, uri)
    _print_summary(uri, dest, envelope)
    raise typer.Exit(code=0)


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #
class FetchError(Exception):
    """Raised by the per-scheme downloaders when the source is unreachable."""


def _default_cache_path(uri: str) -> Path:
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:12]
    return Path.home() / ".cache" / "inferencebench" / "fetched" / f"{digest}.json"


def _download(uri: str, dest: Path) -> None:
    """Dispatch to the right scheme-specific downloader."""
    if uri.startswith("hf://"):
        _download_hf(uri, dest)
    elif uri.startswith(("https://", "http://")):
        _download_https(uri, dest)
    elif uri.startswith("file://"):
        _copy_local(uri[len("file://") :], dest)
    elif "://" in uri:
        scheme = uri.split("://", 1)[0]
        msg = f"unsupported URI scheme: {scheme}://"
        raise FetchError(msg)
    else:
        _copy_local(uri, dest)


def _download_hf(uri: str, dest: Path) -> None:
    """Resolve an ``hf://datasets/<owner>/<repo>[/<file>]`` URI via huggingface_hub."""
    repo_id, filename = _parse_hf_uri(uri)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        msg = "huggingface_hub is not installed (pip install huggingface_hub)"
        raise FetchError(msg) from exc

    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
        )
    except Exception as exc:
        msg = f"huggingface_hub download failed: {exc}"
        raise FetchError(msg) from exc

    shutil.copyfile(local, dest)


def _parse_hf_uri(uri: str) -> tuple[str, str]:
    """Parse ``hf://datasets/<owner>/<repo>[/<file>]`` into ``(repo_id, filename)``.

    The ``<file>`` part is optional. Default filename is ``envelope.json``.
    """
    rest = uri[len("hf://") :]
    if not rest.startswith("datasets/"):
        msg = f"only hf://datasets/<owner>/<repo>[/<file>] is supported, got: {uri}"
        raise FetchError(msg)
    parts = rest[len("datasets/") :].split("/")
    if len(parts) < 2:
        msg = f"hf URI must include owner and repo: {uri}"
        raise FetchError(msg)
    owner, repo = parts[0], parts[1]
    if not owner or not repo:
        msg = f"hf URI has empty owner or repo: {uri}"
        raise FetchError(msg)
    filename = "/".join(parts[2:]) if len(parts) > 2 else _DEFAULT_HF_FILENAME
    return f"{owner}/{repo}", filename


def _download_https(uri: str, dest: Path) -> None:
    """Plain HTTP(S) download via :mod:`urllib.request`."""
    try:
        # Soft-validate the URL to fail fast on garbage input.
        parsed = urllib.parse.urlparse(uri)
        if not parsed.netloc:
            msg = f"invalid HTTPS URL: {uri}"
            raise FetchError(msg)
        with urllib.request.urlopen(uri) as resp:  # noqa: S310 — scheme-checked above
            payload = resp.read()
    except FetchError:
        raise
    except Exception as exc:
        msg = f"https download failed: {exc}"
        raise FetchError(msg) from exc

    dest.write_bytes(payload)


def _copy_local(path_str: str, dest: Path) -> None:
    """Resolve a ``file://`` URI or plain path into a local copy."""
    src = Path(path_str)
    if not src.exists():
        msg = f"source not found: {src}"
        raise FetchError(msg)
    if src.resolve() == dest.resolve():
        return
    shutil.copyfile(src, dest)


def _validate_envelope(path: Path, uri: str) -> Envelope:
    """Parse and validate the downloaded JSON as an :class:`Envelope`.

    Leaves the file in place on failure so the user can inspect it.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        err_console.print(f"[red]invalid JSON in fetched payload:[/red] {exc}")
        err_console.print(f"  uri:    {uri}")
        err_console.print(f"  saved:  {path}")
        raise typer.Exit(code=2) from exc

    try:
        return Envelope.model_validate(raw)
    except Exception as exc:
        err_console.print(f"[red]envelope schema validation failed:[/red] {exc}")
        err_console.print(f"  uri:    {uri}")
        err_console.print(f"  saved:  {path}")
        raise typer.Exit(code=2) from exc


def _print_summary(uri: str, dest: Path, envelope: Envelope) -> None:
    signature_method = envelope.signature.method if envelope.signature else "unsigned"
    console.print(f"[bold green]OK[/bold green]  fetched {uri}")
    console.print(f"  local_path:       {dest}")
    console.print(f"  content_hash:     {envelope.content_hash()}")
    console.print(f"  suite_id:         {envelope.suite_id}")
    console.print(f"  model_id:         {envelope.model.id}")
    console.print(f"  signature:        {signature_method}")
