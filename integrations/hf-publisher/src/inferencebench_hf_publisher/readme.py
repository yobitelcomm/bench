"""Render a dataset-repo README.md from a signed envelope.

The README is the human-facing landing page for every published benchmark
run on Hugging Face Hub. It carries the headline metrics, the run
configuration, a verification snippet, and a citation block.

Public surface:

    render_envelope_readme(envelope) -> str
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inferencebench.envelope import Envelope


# Pretty labels + unit hints for known metric keys. Unknown keys fall through
# with a humanised version of the raw key and an empty unit.
_METRIC_DISPLAY: dict[str, tuple[str, str]] = {
    "ttft_p50_ms": ("TTFT P50", "ms"),
    "ttft_p90_ms": ("TTFT P90", "ms"),
    "ttft_p95_ms": ("TTFT P95", "ms"),
    "ttft_p99_ms": ("TTFT P99", "ms"),
    "tpot_p50_ms": ("TPOT P50", "ms"),
    "tpot_p99_ms": ("TPOT P99", "ms"),
    "throughput_tok_s": ("Throughput", "tok/s"),
    "goodput_req_s": ("Goodput at SLO", "req/s"),
    "cost_per_million_tokens_usd": ("Cost per million tokens", "USD"),
    "joules_per_token": ("Joules per token", "J"),
    "energy_per_token_j": ("Joules per token", "J"),
}


def _humanise_metric(key: str) -> tuple[str, str]:
    """Return (label, unit) for a metric key, falling back to a humanised key."""
    if key in _METRIC_DISPLAY:
        return _METRIC_DISPLAY[key]
    label = key.replace("_", " ").strip().title()
    return label, ""


def _format_value(value: float | int | str | None) -> str:
    """Format a metric value for display in the README table.

    Strings (qualitative tags like ``cost_source = "registry:groq"``) pass
    through unchanged; numeric values get the original formatting.
    """
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return f"{value}"
    # Keep at most 4 significant fractional digits, drop trailing zeros.
    formatted = f"{value:.4f}".rstrip("0").rstrip(".")
    return formatted or "0"


def _frontmatter_tags(envelope: Envelope) -> list[str]:
    """Compute the YAML frontmatter `tags:` list."""
    modality = envelope.suite_id.split(".", 1)[0] if envelope.suite_id else "unknown"
    model_family = _model_family(envelope.model.id)
    hardware_class = _hardware_class(envelope)
    return [
        "benchmark",
        "inferencebench",
        modality,
        model_family,
        hardware_class,
    ]


def _model_family(model_id: str) -> str:
    """Best-effort model family tag from the HF-style provider/model id."""
    if "/" in model_id:
        owner, _, _name = model_id.partition("/")
        return owner.lower().replace(" ", "-")
    return model_id.lower().replace(" ", "-")


def _hardware_class(envelope: Envelope) -> str:
    """Coarse hardware class tag derived from the first GPU model, or 'cpu'."""
    gpus = envelope.hardware_fingerprint.gpus
    if not gpus:
        return "cpu"
    raw = gpus[0].model.lower()
    if "h100" in raw:
        return "h100"
    if "h200" in raw:
        return "h200"
    if "a100" in raw:
        return "a100"
    if "mi300" in raw:
        return "mi300x"
    if "rtx" in raw:
        return "rtx-consumer"
    return raw.replace(" ", "-")


def _frontmatter(envelope: Envelope, *, signature_verified: bool) -> str:
    """Render the YAML frontmatter block (without surrounding ``---`` markers)."""
    rekor_log_index = envelope.signature.rekor_log_index if envelope.signature is not None else -1
    tags = _frontmatter_tags(envelope)
    tag_lines = "\n".join(f"- {tag}" for tag in tags)
    return (
        "license: cc-by-4.0\n"
        "language:\n"
        "- en\n"
        "size_categories:\n"
        "- n<1K\n"
        "task_categories:\n"
        "- text-generation\n"
        "tags:\n"
        f"{tag_lines}\n"
        "inferencebench:\n"
        f"  envelope_version: {envelope.envelope_version}\n"
        f"  suite_id: {envelope.suite_id}\n"
        f"  suite_version: {envelope.suite_version}\n"
        f"  model: {envelope.model.id}\n"
        f"  engine: {envelope.engine.name}\n"
        f"  hardware_class: {_hardware_class(envelope)}\n"
        f"  fingerprint_sha256: {envelope.hardware_fingerprint.fingerprint_sha256}\n"
        f"  signature_verified: {str(signature_verified).lower()}\n"
        f"  rekor_log_index: {rekor_log_index}\n"
    )


def render_envelope_readme(envelope: Envelope) -> str:
    """Render the dataset-repo README.md for a published envelope.

    The output includes YAML frontmatter (so the HF Hub dataset card picks
    up the metadata) and human-readable sections for headline metrics, run
    configuration, verification, methodology, and citation.

    Args:
        envelope: The signed (or unsigned) envelope to render.

    Returns:
        Markdown source ready to be uploaded as ``README.md`` to the dataset repo.
    """
    signature_verified = envelope.signature is not None and envelope.signature.rekor_log_index >= 0
    rekor_log_index = envelope.signature.rekor_log_index if envelope.signature is not None else -1

    hw = envelope.hardware_fingerprint
    gpu_model = hw.gpus[0].model if hw.gpus else "cpu-only"
    quantization_format = (
        envelope.quantization.format if envelope.quantization is not None else "n/a"
    )
    run_hash = envelope.run_id.replace("-", "")[:12]
    repo_id = _repo_id_for(envelope)

    metric_rows: list[str] = []
    for key, value in envelope.metrics.items():
        label, unit = _humanise_metric(key)
        metric_rows.append(f"| {label} | {_format_value(value)} | {unit} |")
    if not metric_rows:
        metric_rows.append("| (no metrics emitted) | — | — |")

    suite_methodology_url = f"https://yobitelcomm.github.io/bench/suites/{envelope.suite_id}"

    frontmatter = _frontmatter(envelope, signature_verified=signature_verified)

    lines = [
        "---",
        frontmatter.rstrip("\n"),
        "---",
        "",
        f"# {envelope.model.id} on {envelope.suite_id} ({gpu_model})",
        "",
        f"[Back to leaderboard](https://huggingface.co/spaces/yobitel/leaderboard-{envelope.suite_id})",
        "",
        "## Headline metrics",
        "",
        "| Metric | Value | Unit |",
        "|---|---|---|",
        *metric_rows,
        "",
        "## Run configuration",
        "",
        f"- **Model**: {envelope.model.id} @ {envelope.model.revision}",
        f"- **Engine**: {envelope.engine.name} v{envelope.engine.version}",
        f"- **Quantization**: {quantization_format}",
        f"- **Hardware**: {gpu_model}",
        f"- **Driver**: {hw.driver or 'n/a'}",
        f"- **CUDA**: {hw.cuda or 'n/a'}",
        f"- **Run date**: {envelope.timestamp.isoformat()}",
        f"- **Seed**: {envelope.seed}",
        "",
        "## Verification",
        "",
        "This result is Sigstore-signed and Rekor-logged. Verify:",
        "",
        "```bash",
        "pip install inferencebench",
        f"bench verify hf://datasets/{repo_id}/envelope.json",
        "```",
        "",
        (
            f"[Rekor entry: log index {rekor_log_index}]"
            f"(https://search.sigstore.dev/?logIndex={rekor_log_index})"
        ),
        "",
        "## Methodology",
        "",
        f"See [the suite methodology page]({suite_methodology_url}).",
        "",
        "## Citation",
        "",
        "```bibtex",
        f"@misc{{inferencebench_{run_hash},",
        f"  title = {{ {envelope.model.id} on {envelope.suite_id} }},",
        "  author = { {InferenceBench community} },",
        f"  year = {{ {envelope.timestamp.year} }},",
        f"  url = {{ https://huggingface.co/datasets/{repo_id} }},",
        "}",
        "```",
        "",
        "---",
        "",
        (
            "*Published via [InferenceBench](https://github.com/yobitelcomm/bench)"
            " — vendor-neutral AI benchmarks.*"
        ),
        "",
    ]
    return "\n".join(lines)


def _repo_id_for(envelope: Envelope) -> str:
    """Helper for README rendering only — mirrors ``publish.compute_repo_id``.

    Re-imported in ``publish.py`` to avoid a circular import; the canonical
    implementation lives there but we duplicate slugification rules here so
    the README is self-contained and renderable without the publisher.
    """
    from inferencebench_hf_publisher.publish import compute_repo_id

    return compute_repo_id(envelope)
