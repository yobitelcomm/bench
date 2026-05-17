"""``bench bundle`` — pack/unpack a single-file shareable envelope artifact.

A *bundle* is a plain ``.zip`` containing:

* ``envelope.json`` — the original signed envelope (exact bytes).
* ``samples.jsonl`` — optional concatenated per-sample trace.
* ``signature_info.json`` — a quick at-a-glance summary of the signature block.
* ``verify.py`` — a tiny, dependency-free verifier (stdlib + ``cryptography``)
  the recipient can run with nothing installed from this project.
* ``cosign.pub`` — optional, for dev-key envelopes when the sender includes it.
* ``README.txt`` — three-line orientation.

The product motivation is simple: sharing a benchmark result should be one
file. Recipients should not need to ``pip install inferencebench`` to verify
it; they should only need Python and the ubiquitous ``cryptography`` package.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from inferencebench.envelope import Envelope

app = typer.Typer(no_args_is_help=True)
console = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Standalone verify.py — shipped INSIDE every bundle.                         #
#                                                                             #
# This script must work on Python 3.12 + cryptography ONLY. No imports from   #
# inferencebench or pydantic. Keep it small and well-commented; this is the   #
# face we present to recipients of a bundle.                                  #
# --------------------------------------------------------------------------- #
_VERIFY_SCRIPT = '''\
"""Standalone envelope verifier shipped inside an InferenceBench bundle.

Verifies that ``envelope.json`` (next to this script) has not been tampered
with since it was signed. Requires Python 3.12 and the ``cryptography``
package only — no InferenceBench install needed.

For dev-key envelopes, supply the signer's public key via ``--pubkey``
(defaults to ``./cosign.pub`` if present). For keyless (Sigstore) envelopes,
this script defers to ``bench verify`` / ``cosign verify-blob``, since the
keyless path needs the Sigstore transparency-log machinery that is too heavy
to inline here.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


def canonical_content_hash(envelope: dict) -> str:
    """Recompute the canonical content hash that the signer signed.

    Mirrors ``Envelope.content_hash()`` exactly: drop the ``signature`` field,
    json.dumps with sort_keys=True and separators=(",", ":"), UTF-8 encode,
    sha256.

    Example (with the fixture envelope shipped by the test suite):
        >>> # content_hash is 64 lowercase hex chars
        >>> # e.g. "3f9c1d7a4e8b2cf6..."
    """
    body = {k: v for k, v in envelope.items() if k != "signature"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_dev_key(envelope: dict, pubkey_path: Path) -> tuple[bool, str]:
    """Verify a ``dev-key`` envelope against a PEM ed25519 public key."""
    sig = envelope.get("signature")
    if not sig:
        return False, "envelope has no signature block"
    if sig.get("method") != "dev-key":
        return False, f"unexpected signature method: {sig.get('method')!r}"

    if not pubkey_path.exists():
        return False, f"public key not found: {pubkey_path}"

    try:
        public_key = serialization.load_pem_public_key(pubkey_path.read_bytes())
    except ValueError as exc:
        return False, f"failed to load public key: {exc}"
    if not isinstance(public_key, Ed25519PublicKey):
        return False, f"expected ed25519 public key, got {type(public_key).__name__}"

    try:
        sig_bytes = base64.b64decode(sig.get("bundle", ""), validate=True)
    except Exception as exc:  # noqa: BLE001 — base64 raises many subclasses
        return False, f"invalid base64 in signature bundle: {exc}"

    content_hash = canonical_content_hash(envelope).encode("utf-8")
    try:
        public_key.verify(sig_bytes, content_hash)
    except InvalidSignature:
        return False, "signature does not match content_hash (tampered or wrong key)"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pubkey",
        type=Path,
        default=Path(__file__).parent / "cosign.pub",
        help="PEM-encoded ed25519 public key (default: ./cosign.pub).",
    )
    parser.add_argument(
        "--envelope",
        type=Path,
        default=Path(__file__).parent / "envelope.json",
        help="Path to envelope.json (default: alongside this script).",
    )
    args = parser.parse_args()

    if not args.envelope.exists():
        print(f"FAIL: envelope not found: {args.envelope}", file=sys.stderr)
        return 1

    try:
        envelope = json.loads(args.envelope.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"FAIL: invalid JSON in envelope: {exc}", file=sys.stderr)
        return 1

    sig = envelope.get("signature") or {}
    method = sig.get("method", "<none>")

    if method == "sigstore-cosign":
        print(
            "Keyless (Sigstore) envelopes require Sigstore tooling to verify.\\n"
            "Run one of:\\n"
            "  bench verify envelope.json\\n"
            "  cosign verify-blob --bundle <bundle.json> envelope.json",
            file=sys.stderr,
        )
        return 2

    if method != "dev-key":
        print(f"FAIL: unsupported signature method: {method!r}", file=sys.stderr)
        return 1

    ok, reason = verify_dev_key(envelope, args.pubkey)
    if not ok:
        print(f"FAIL: {reason}", file=sys.stderr)
        return 1

    print("OK")
    print(f"  content_hash:  {canonical_content_hash(envelope)}")
    print(f"  suite:         {envelope.get('suite_id')} v{envelope.get('suite_version')}")
    print(f"  model:         {(envelope.get('model') or {}).get('id')}")
    print(f"  engine:        {(envelope.get('engine') or {}).get('name')}")
    print(f"  run_id:        {envelope.get('run_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_README_TEXT = """\
InferenceBench bundle — one-file shareable benchmark result.

To verify:  python verify.py --pubkey cosign.pub
Docs:       https://github.com/yobitelcomm/bench
"""


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _load_envelope(path: Path) -> tuple[Envelope, bytes]:
    """Load and validate an envelope; return the parsed model + raw bytes.

    The raw bytes are preserved so the bundle ships the exact envelope the
    user produced (including their chosen indentation / key ordering). The
    recipient's verifier re-canonicalises before hashing, so byte preservation
    is purely cosmetic — but it makes diffs against the original file work.
    """
    if not path.exists():
        err_console.print(f"[red]Envelope not found:[/red] {path}")
        raise typer.Exit(code=2)
    raw = path.read_bytes()
    try:
        Envelope.model_validate(json.loads(raw.decode("utf-8")))
    except Exception as exc:
        err_console.print(f"[red]Envelope schema validation failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    envelope = Envelope.model_validate(json.loads(raw.decode("utf-8")))
    return envelope, raw


def _matching_samples(envelope_path: Path, window_seconds: float = 300.0) -> list[Path]:
    """Find ``samples-*.jsonl`` files in the same dir whose mtime is close to the envelope's.

    Best-effort — pulls in nothing if no match.
    """
    parent = envelope_path.parent
    if not parent.exists():
        return []
    env_mtime = envelope_path.stat().st_mtime
    matches: list[Path] = []
    for candidate in parent.iterdir():
        if not candidate.is_file():
            continue
        if not candidate.name.startswith("samples-"):
            continue
        if candidate.suffix != ".jsonl":
            continue
        if abs(candidate.stat().st_mtime - env_mtime) <= window_seconds:
            matches.append(candidate)
    matches.sort()
    return matches


def _signature_info(envelope: Envelope) -> dict[str, Any]:
    """Build the signature_info.json payload from the envelope's signature block."""
    sig = envelope.signature
    if sig is None:
        return {
            "method": "none",
            "key_id": None,
            "bundle_present": False,
            "content_hash": envelope.content_hash(),
        }
    # For dev-key envelopes, the certificate is the PEM public key — fingerprint
    # it as a short key_id so recipients can eyeball which key signed without
    # parsing PEM themselves.
    cert = sig.certificate or ""
    import hashlib as _hashlib

    key_id = _hashlib.sha256(cert.encode("utf-8")).hexdigest()[:16] if cert else None
    return {
        "method": sig.method,
        "key_id": key_id,
        "bundle_present": bool(sig.bundle),
        "content_hash": envelope.content_hash(),
    }


def _format_size(num_bytes: int) -> str:
    """Render a byte count as a short human-readable string."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


# --------------------------------------------------------------------------- #
# Commands                                                                    #
# --------------------------------------------------------------------------- #
@app.command("create")
def bundle_create(
    envelope_path: Annotated[
        Path,
        typer.Argument(help="Path to the envelope.json to bundle."),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Destination zip path. Defaults to <hash>.bundle.zip in cwd."),
    ] = None,
    include_samples: Annotated[
        bool,
        typer.Option(
            "--include-samples/--no-include-samples",
            help="Also pack samples-*.jsonl files alongside the envelope.",
        ),
    ] = True,
    include_public_key: Annotated[
        Path | None,
        typer.Option(
            "--include-public-key",
            help="Optional public key to embed (dev-key envelopes only).",
        ),
    ] = None,
) -> None:
    """Pack an envelope + verifier into a single shareable ``.bundle.zip``."""
    envelope, raw = _load_envelope(envelope_path)

    content_hash = envelope.content_hash()
    bundle_path = out if out is not None else Path.cwd() / f"{content_hash[:12]}.bundle.zip"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    sample_files: list[Path] = (
        _matching_samples(envelope_path) if include_samples else []
    )

    sig_info = _signature_info(envelope)

    files_in_zip: list[str] = []
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("envelope.json", raw)
        files_in_zip.append("envelope.json")

        zf.writestr(
            "signature_info.json",
            json.dumps(sig_info, sort_keys=True, indent=2) + "\n",
        )
        files_in_zip.append("signature_info.json")

        zf.writestr("verify.py", _VERIFY_SCRIPT)
        files_in_zip.append("verify.py")

        zf.writestr("README.txt", _README_TEXT)
        files_in_zip.append("README.txt")

        if sample_files:
            buf: list[str] = []
            for sample_path in sample_files:
                buf.append(sample_path.read_text(encoding="utf-8").rstrip("\n"))
            zf.writestr("samples.jsonl", "\n".join(buf) + "\n")
            files_in_zip.append("samples.jsonl")

        if include_public_key is not None:
            if not include_public_key.exists():
                err_console.print(
                    f"[red]Public key not found:[/red] {include_public_key}"
                )
                raise typer.Exit(code=2)
            zf.writestr("cosign.pub", include_public_key.read_bytes())
            files_in_zip.append("cosign.pub")

    size = bundle_path.stat().st_size

    table = Table(title=f"Bundle created: {bundle_path.name}")
    table.add_column("field", style="cyan")
    table.add_column("value")
    table.add_row("path", str(bundle_path))
    table.add_row("size", _format_size(size))
    table.add_row("files", ", ".join(files_in_zip))
    table.add_row("content_hash", content_hash)
    table.add_row("signature", str(sig_info["method"]))
    console.print(table)


@app.command("extract")
def bundle_extract(
    bundle_path: Annotated[
        Path,
        typer.Argument(help="Path to the .bundle.zip to extract."),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Destination directory. Defaults to ./<basename>/."),
    ] = None,
) -> None:
    """Extract a bundle and re-validate the envelope inside it."""
    if not bundle_path.exists():
        err_console.print(f"[red]Bundle not found:[/red] {bundle_path}")
        raise typer.Exit(code=2)

    if out is None:
        # Strip ".bundle.zip" → preferred; otherwise strip ".zip".
        name = bundle_path.name
        if name.endswith(".bundle.zip"):
            stem = name[: -len(".bundle.zip")]
        elif name.endswith(".zip"):
            stem = name[: -len(".zip")]
        else:
            stem = name
        out = Path.cwd() / stem

    out.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(bundle_path, "r") as zf:
            zf.extractall(out)
    except zipfile.BadZipFile as exc:
        err_console.print(f"[red]Invalid zip:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    envelope_path = out / "envelope.json"
    if not envelope_path.exists():
        err_console.print(f"[red]Bundle missing envelope.json:[/red] {bundle_path}")
        raise typer.Exit(code=2)

    try:
        envelope = Envelope.model_validate(
            json.loads(envelope_path.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        err_console.print(f"[red]Envelope schema validation failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    sig_method = envelope.signature.method if envelope.signature else "none"

    table = Table(title=f"Bundle extracted: {out}")
    table.add_column("field", style="cyan")
    table.add_column("value")
    table.add_row("out_dir", str(out))
    table.add_row("content_hash", envelope.content_hash())
    table.add_row("suite", f"{envelope.suite_id} v{envelope.suite_version}")
    table.add_row("model", envelope.model.id)
    table.add_row("engine", f"{envelope.engine.name} v{envelope.engine.version}")
    table.add_row("signature", sig_method)
    console.print(table)
