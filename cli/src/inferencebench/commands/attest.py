"""``bench attest`` — render a third-party-readable attestation slip from an envelope.

The envelope itself is a dense JSON blob suited for machine consumption. For
auditors, contract attachments, and PR-comment reviewers we want a one-page
human summary that:

* Quotes the hardware fingerprint hash, software provenance hash, dataset hash.
* States the signature method + key fingerprint (NOT the key itself).
* Lists the headline metrics + units inferred from the suffix.
* Includes a verification stanza any human can follow.

Markdown is the default (paste into a contract / PR / docs). A sibling
``--format json`` form is provided for machine-readable archives.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from inferencebench.envelope import Envelope

console = Console()
err_console = Console(stderr=True)


_FORMATS: frozenset[str] = frozenset({"markdown", "json"})
_DOCS_URL = "https://yobitelcomm.github.io/bench"
_SPEC_URL = "https://github.com/yobitelcomm/bench/blob/main/envelope/SPEC.md"


# --------------------------------------------------------------------------- #
# Command                                                                     #
# --------------------------------------------------------------------------- #
def attest(
    envelope_path: Annotated[
        Path,
        typer.Argument(help="Path to the envelope JSON file (local file only)."),
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help=(
                "Destination path. Defaults to "
                "``<content_hash[:12]>.attestation.{md,json}`` in cwd."
            ),
        ),
    ] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: markdown (default) or json.",
        ),
    ] = "markdown",
    organization: Annotated[
        str,
        typer.Option(
            "--organization",
            help="When set, included in the attestation header (``Issued for: ...``).",
        ),
    ] = "",
) -> None:
    """Render a third-party-readable attestation slip for an envelope."""
    if format not in _FORMATS:
        err_console.print(
            f"[red]Unknown --format value:[/red] {format} "
            "(expected one of: markdown, json)"
        )
        raise typer.Exit(code=2)

    envelope = _load_envelope(envelope_path)
    content_hash = envelope.content_hash()

    if envelope.signature is None:
        err_console.print(
            "[yellow]Warning:[/yellow] envelope is unsigned. "
            "The attestation will mark the signature method as ``unsigned``."
        )

    payload = _build_payload(
        envelope=envelope,
        envelope_path=envelope_path,
        organization=organization,
    )

    if out is None:
        suffix = "md" if format == "markdown" else "json"
        out = Path.cwd() / f"{content_hash[:12]}.attestation.{suffix}"

    out.parent.mkdir(parents=True, exist_ok=True)

    if format == "markdown":
        rendered = _render_markdown(payload)
    else:
        rendered = json.dumps(payload, sort_keys=True, indent=2) + "\n"

    out.write_text(rendered, encoding="utf-8")

    console.print(f"[bold green]Wrote attestation:[/bold green] {out}")
    console.print(f"  content_hash:  {content_hash}")
    console.print(f"  format:        {format}")
    if organization:
        console.print(f"  organization:  {organization}")


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def _load_envelope(path: Path) -> Envelope:
    """Load an envelope from a local file path."""
    if not path.exists():
        err_console.print(f"[red]Envelope not found:[/red] {path}")
        raise typer.Exit(code=2)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        err_console.print(f"[red]Invalid JSON in envelope:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    try:
        return Envelope.model_validate(raw)
    except Exception as exc:
        err_console.print(f"[red]Envelope schema validation failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc


# --------------------------------------------------------------------------- #
# Payload construction                                                        #
# --------------------------------------------------------------------------- #
def _build_payload(
    *,
    envelope: Envelope,
    envelope_path: Path,
    organization: str,
) -> dict[str, Any]:
    """Build the canonical attestation payload (shared by markdown + json)."""
    content_hash = envelope.content_hash()
    issued_at = datetime.now(UTC).isoformat(timespec="seconds")

    header: dict[str, Any] = {
        "title": "InferenceBench attestation",
        "content_hash": content_hash,
        "issued_at": issued_at,
        "run_id": envelope.run_id,
        "organization": organization,
    }

    hw_fp = envelope.hardware_fingerprint.fingerprint_sha256
    subject: dict[str, Any] = {
        "model_id": envelope.model.id,
        "model_revision": envelope.model.revision,
        "engine": envelope.engine.name,
        "engine_version": envelope.engine.version,
        "hardware_fingerprint_sha256": hw_fp,
        "hardware_fingerprint_sha256_short": hw_fp[:12],
        "software_provenance_pip_freeze_hash": (
            envelope.software_provenance.pip_freeze_hash
        ),
        "software_provenance_pip_freeze_hash_short": (
            envelope.software_provenance.pip_freeze_hash[:12]
        ),
        "software_provenance_git_commit": envelope.software_provenance.git_commit,
        "dataset_id": envelope.dataset.id,
        "dataset_hash_sha256": envelope.dataset.hash,
        "dataset_hash_sha256_short": envelope.dataset.hash[:12],
        "seed": envelope.seed,
    }

    metrics = _build_metrics_list(envelope.metrics)
    signature = _build_signature_block(envelope)
    verification = _build_verification_block(
        envelope=envelope,
        envelope_path=envelope_path,
    )

    return {
        "schema": "inferencebench.attestation.v1",
        "header": header,
        "subject": subject,
        "metrics": metrics,
        "signature": signature,
        "verification": verification,
        "footer": {
            "docs_url": _DOCS_URL,
            "spec_url": _SPEC_URL,
        },
    }


# --------------------------------------------------------------------------- #
# Metric inference                                                            #
# --------------------------------------------------------------------------- #
# Unit suffixes recognised on metric keys. Mapping is suffix → human-readable
# unit string. Order matters: longer / more specific suffixes first so e.g.
# ``joules_per_token`` matches before ``per_token``.
_UNIT_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_tok_per_s", "tokens/s"),
    ("_tokens_per_s", "tokens/s"),
    ("_req_per_s", "requests/s"),
    ("_per_token", "per token"),
    ("_per_request", "per request"),
    ("_ms", "ms"),
    ("_us", "us"),
    ("_ns", "ns"),
    ("_seconds", "s"),
    ("_bytes", "B"),
    ("_mb", "MB"),
    ("_gb", "GB"),
    ("_w", "W"),
    ("_joules", "J"),
    ("_rate", "ratio"),
    ("_pct", "%"),
    ("_percent", "%"),
)


def _infer_unit(metric_key: str) -> str:
    """Infer a human-readable unit from a metric-key suffix."""
    key_lower = metric_key.lower()
    for suffix, unit in _UNIT_SUFFIXES:
        if key_lower.endswith(suffix):
            return unit
    return ""


def _format_metric_value(value: float | int | str | None) -> str:
    """Format a metric value for display."""
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return f"{float(value):.4g}"


def _build_metrics_list(
    metrics: dict[str, float | int | str | None],
) -> list[dict[str, Any]]:
    """Return a list of ``{key, value, unit}`` sorted alphabetically by key."""
    rows: list[dict[str, Any]] = []
    for key in sorted(metrics.keys()):
        raw_value = metrics[key]
        rows.append(
            {
                "key": key,
                "value": raw_value,
                "value_str": _format_metric_value(raw_value),
                "unit": _infer_unit(key),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Signature block                                                             #
# --------------------------------------------------------------------------- #
def _build_signature_block(envelope: Envelope) -> dict[str, Any]:
    """Build the ``signature`` section of the attestation payload."""
    sig = envelope.signature
    if sig is None:
        return {
            "method": "unsigned",
            "key_id": None,
            "rekor_log_index": None,
            "warning": (
                "This envelope is unsigned. Anyone could have produced or edited it; "
                "do not rely on it for compliance attestation until it is signed."
            ),
        }

    cert = sig.certificate or ""
    key_id = (
        hashlib.sha256(cert.encode("utf-8")).hexdigest()[:16] if cert else None
    )
    block: dict[str, Any] = {
        "method": sig.method,
        "key_id": key_id,
        "rekor_log_index": None,
    }
    if sig.method == "sigstore-cosign" and sig.rekor_log_index >= 0:
        block["rekor_log_index"] = sig.rekor_log_index
    return block


# --------------------------------------------------------------------------- #
# Verification block                                                          #
# --------------------------------------------------------------------------- #
def _build_verification_block(
    *,
    envelope: Envelope,
    envelope_path: Path,
) -> dict[str, Any]:
    """Build the ``verification`` section: instructions + the verify command."""
    sig = envelope.signature
    if sig is not None and sig.method == "dev-key":
        command = (
            f"bench verify {envelope_path.name} --dev-public-key cosign.pub"
        )
        notes = (
            "Save this envelope file next to the signer's ``cosign.pub`` public key, "
            "install ``inferencebench``, and run the command above. A zero exit "
            "code means the signature matches the canonical content hash above."
        )
    elif sig is not None and sig.method == "sigstore-cosign":
        command = f"bench verify {envelope_path.name}"
        notes = (
            "Save this envelope file, install ``inferencebench``, and run the "
            "command above. Verification consults the Sigstore Rekor "
            "transparency log; a zero exit code means the signature matches the "
            "canonical content hash and the log entry is intact."
        )
    else:
        command = f"bench verify {envelope_path.name}"
        notes = (
            "The envelope is unsigned. ``bench verify`` will exit non-zero "
            "until a signature is attached."
        )
    return {
        "command": command,
        "envelope_filename": envelope_path.name,
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# Markdown renderer                                                           #
# --------------------------------------------------------------------------- #
def _render_markdown(payload: dict[str, Any]) -> str:
    """Render the attestation payload as one-page markdown."""
    header = payload["header"]
    subject = payload["subject"]
    metrics = payload["metrics"]
    signature = payload["signature"]
    verification = payload["verification"]
    footer = payload["footer"]

    lines: list[str] = []

    # 1. Header
    lines.append("# InferenceBench attestation")
    lines.append("")
    if header.get("organization"):
        lines.append(f"**Issued for:** {header['organization']}")
        lines.append("")
    lines.append(f"- **Content hash**: `{header['content_hash']}`")
    lines.append(f"- **Issued at**: `{header['issued_at']}`")
    lines.append(f"- **Run ID**: `{header['run_id']}`")
    lines.append("")

    # 2. What this is
    lines.append("## What this is")
    lines.append("")
    lines.append(
        "This attestation summarises a benchmark run signed by InferenceBench. "
        f"The full envelope is at `{verification['envelope_filename']}`; "
        "verify it independently with `bench verify`."
    )
    lines.append("")

    # 3. Subject
    lines.append("## Subject")
    lines.append("")
    lines.append(
        f"- **Model**: `{subject['model_id']}` "
        f"(revision `{subject['model_revision']}`)"
    )
    lines.append(
        f"- **Engine**: `{subject['engine']} v{subject['engine_version']}`"
    )
    lines.append(
        "- **Hardware fingerprint (sha256, short)**: "
        f"`{subject['hardware_fingerprint_sha256_short']}`"
    )
    lines.append(
        f"- **Dataset**: `{subject['dataset_id']}` "
        f"(sha256 `{subject['dataset_hash_sha256_short']}`)"
    )
    lines.append(f"- **Seed**: `{subject['seed']}`")
    lines.append("")

    # 4. Metrics
    lines.append("## Metrics")
    lines.append("")
    lines.append("| metric | value | unit | interpretation |")
    lines.append("|---|---|---|---|")
    for row in metrics:
        unit = row["unit"] or "-"
        interpretation = _metric_interpretation(row["key"])
        lines.append(
            f"| `{row['key']}` | {row['value_str']} | {unit} | {interpretation} |"
        )
    lines.append("")

    # 5. Signature
    lines.append("## Signature")
    lines.append("")
    method = signature["method"]
    if method == "unsigned":
        lines.append(":warning: **This envelope is unsigned.**")
        lines.append("")
        lines.append(signature.get("warning", ""))
    else:
        lines.append(f"- **Method**: `{method}`")
        if signature.get("key_id"):
            lines.append(f"- **Key fingerprint (key_id)**: `{signature['key_id']}`")
        if signature.get("rekor_log_index") is not None:
            lines.append(
                f"- **Rekor log index**: `{signature['rekor_log_index']}`"
            )
    lines.append("")

    # 6. Verification
    lines.append("## Verification")
    lines.append("")
    lines.append(verification["notes"])
    lines.append("")
    lines.append("```sh")
    lines.append(verification["command"])
    lines.append("```")
    lines.append("")

    # 7. Footer
    lines.append("## About")
    lines.append("")
    lines.append(f"- Docs: <{footer['docs_url']}>")
    lines.append(f"- InferenceBench envelope spec: <{footer['spec_url']}>")
    lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Interpretation copy                                                         #
# --------------------------------------------------------------------------- #
_INTERPRETATIONS: dict[str, str] = {
    "throughput_tok_per_s": "tokens generated per wall-clock second (higher is better)",
    "ttft_p50_ms": "median time-to-first-token (lower is better)",
    "ttft_p99_ms": "p99 time-to-first-token (lower is better)",
    "tpot_p50_ms": "median time-per-output-token (lower is better)",
    "tpot_p99_ms": "p99 time-per-output-token (lower is better)",
    "total_p50_ms": "median end-to-end request latency (lower is better)",
    "total_p99_ms": "p99 end-to-end request latency (lower is better)",
    "ok_rate": "fraction of requests that succeeded (closer to 1 is better)",
    "compliance_rate": "fraction of requests meeting the SLO (closer to 1 is better)",
    "power_avg_w": "average power draw during the run",
    "power_peak_w": "peak power draw during the run",
    "joules_per_token": "energy per generated token (lower is better)",
    "cost_per_1k_tokens": "USD cost per 1,000 generated tokens (lower is better)",
}


def _metric_interpretation(key: str) -> str:
    """Return a short interpretation string for a metric, or a sensible default."""
    if key in _INTERPRETATIONS:
        return _INTERPRETATIONS[key]
    if key.endswith("_ms"):
        return "lower is better"
    if key.endswith("_tok_per_s") or key.endswith("_req_per_s"):
        return "higher is better"
    if key.endswith("_rate"):
        return "ratio in [0, 1]"
    if key.endswith("_w") or key.endswith("_joules") or key.endswith("_per_token"):
        return "lower is better"
    return "see plugin docs"
