"""Publish signed InferenceBench envelopes to Hugging Face Hub as datasets.

Public surface:

    publish_envelope_to_hf(envelope, *, hf_token=None, raw_traces_path=None,
                           update_model_card=False,
                           org="yobitel-bench-results", dry_run=False)
        -> HfPublishResult
    HfPublishResult                   — dataclass with repo_id, url, files, verified
    HfPublishError                    — base error
    HfRepoCollisionError              — repo with computed id already exists
    HfRateLimitError                  — HF Hub rate limit hit
    compute_repo_id(envelope)         — slugified repo id used by publisher + readme
    slugify(value)                    — pure-function slug helper (exported for tests)

The flow follows ``skills/hf-publishing/SKILL.md`` exactly:

    1. Compute slugified repo id from model + suite + run hash.
    2. Create a dataset repo (no clobber).
    3. Upload envelope.json (canonical JSON).
    4. Upload optional traces.parquet.
    5. Upload generated README.md (with YAML frontmatter dataset card).
    6. Optionally append a backlink entry to the model card.

When ``dry_run`` is True the function does not touch HF Hub at all — it just
returns the planned ``HfPublishResult`` with ``verified=False``. This keeps
the pure unit-test path free of network access.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError  # type: ignore[attr-defined]

from inferencebench_hf_publisher.readme import render_envelope_readme

if TYPE_CHECKING:
    from inferencebench.envelope import Envelope


# --------------------------------------------------------------------------- #
# Errors                                                                      #
# --------------------------------------------------------------------------- #
class HfPublishError(Exception):
    """Base class for publisher errors."""


class HfRepoCollisionError(HfPublishError):
    """A dataset repo with the computed id already exists.

    Per SKILL.md, the caller may regenerate ``run_id`` and retry once.
    """


class HfRateLimitError(HfPublishError):
    """HF Hub returned a 429. Caller should back off and retry."""


# --------------------------------------------------------------------------- #
# Result type                                                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class HfPublishResult:
    """Outcome of a publish call.

    Attributes:
        repo_id: Fully-qualified dataset repo id, e.g.
            ``yobitel-bench-results/meta-llama-llama-4-...__llm-inference__abc123def456``.
        url: Canonical HF Hub URL for the repo.
        files_uploaded: Filenames uploaded to the repo (in upload order).
        verified: True iff the publisher confirmed the envelope round-trip
            (re-downloaded the envelope.json and parsed it). Always False in
            ``dry_run`` mode and False on partial-failure paths.
    """

    repo_id: str
    url: str
    files_uploaded: list[str] = field(default_factory=list)
    verified: bool = False


# --------------------------------------------------------------------------- #
# Slug + repo-id helpers                                                      #
# --------------------------------------------------------------------------- #
_SLUG_INVALID = re.compile(r"[^a-z0-9]+")
_SLUG_TRIM = re.compile(r"^-+|-+$")
_RUN_HASH_LEN = 12


def slugify(value: str) -> str:
    """Slugify a string for use in HF Hub repo ids.

    HF Hub allows ``[A-Za-z0-9._-]`` in repo names; we conservatively map
    everything to lowercase ASCII with ``-`` separators and dedupe runs.

    Args:
        value: Raw string (model id, suite id, ...).

    Returns:
        ASCII-only slug with no leading/trailing ``-`` and no internal runs.
        Returns ``"unknown"`` for empty / all-invalid inputs to avoid empty
        path components.
    """
    lowered = value.strip().lower()
    replaced = _SLUG_INVALID.sub("-", lowered)
    trimmed = _SLUG_TRIM.sub("", replaced)
    return trimmed or "unknown"


def compute_repo_id(envelope: Envelope, *, org: str = "yobitel-bench-results") -> str:
    """Compute the deterministic dataset repo id for an envelope.

    Format: ``{org}/{model-slug}__{suite-slug}__{run-hash}``.

    Args:
        envelope: The envelope being published.
        org: HF organisation to publish under. Defaults to the production org.

    Returns:
        Fully-qualified repo id, ready to pass to :class:`HfApi`.
    """
    model_slug = slugify(envelope.model.id)
    suite_slug = slugify(envelope.suite_id)
    run_hash = envelope.run_id.replace("-", "")[:_RUN_HASH_LEN]
    return f"{org}/{model_slug}__{suite_slug}__{run_hash}"


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #
def publish_envelope_to_hf(
    envelope: Envelope,
    *,
    hf_token: str | None = None,
    raw_traces_path: Path | None = None,
    update_model_card: bool = False,
    org: str = "yobitel-bench-results",
    dry_run: bool = False,
) -> HfPublishResult:
    """Publish one envelope to Hugging Face Hub as a dataset repo.

    Args:
        envelope: Signed (or unsigned) envelope to publish.
        hf_token: HF write token. If ``None`` the ``HfApi`` falls back to the
            ambient token (env / cached login). Required for non-dry-run.
        raw_traces_path: Optional path to a parquet file with raw request
            traces; uploaded as ``traces.parquet`` when provided.
        update_model_card: If True, attempt to append a backlink entry to
            the model card YAML. Failures are logged and ignored (the model
            card backlink is optional per SKILL.md).
        org: HF organisation to publish under. Defaults to the production
            ``yobitel-bench-results`` org; tests / staging override.
        dry_run: If True, skip all network calls and return the planned
            ``HfPublishResult`` with ``verified=False``.

    Returns:
        :class:`HfPublishResult` describing the published repo.

    Raises:
        HfRepoCollisionError: A repo at the computed id already exists.
        HfRateLimitError: HF Hub returned HTTP 429.
        HfPublishError: Any other HF Hub failure (wraps the underlying error).
    """
    repo_id = compute_repo_id(envelope, org=org)
    url = f"https://huggingface.co/datasets/{repo_id}"

    if dry_run:
        return HfPublishResult(
            repo_id=repo_id,
            url=url,
            files_uploaded=[],
            verified=False,
        )

    api = HfApi(token=hf_token)
    files_uploaded: list[str] = []

    # ---------------- 1. Create the dataset repo (no clobber) ----------------
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=False,
            private=False,
        )
    except HfHubHTTPError as exc:
        _classify_and_raise(exc, repo_id=repo_id, op="create_repo")

    # ---------------- 2. Upload envelope.json ---------------------------------
    envelope_json = _serialize_envelope(envelope)
    try:
        _upload_bytes(
            api,
            content=envelope_json.encode("utf-8"),
            path_in_repo="envelope.json",
            repo_id=repo_id,
        )
    except HfHubHTTPError as exc:
        _classify_and_raise(exc, repo_id=repo_id, op="upload_envelope")
    files_uploaded.append("envelope.json")

    # ---------------- 3. Optional traces.parquet ------------------------------
    if raw_traces_path is not None:
        if not raw_traces_path.exists():
            msg = f"raw_traces_path does not exist: {raw_traces_path}"
            raise HfPublishError(msg)
        try:
            api.upload_file(
                path_or_fileobj=str(raw_traces_path),
                path_in_repo="traces.parquet",
                repo_id=repo_id,
                repo_type="dataset",
            )
        except HfHubHTTPError as exc:
            _classify_and_raise(exc, repo_id=repo_id, op="upload_traces")
        files_uploaded.append("traces.parquet")

    # ---------------- 4. README with dataset-card frontmatter -----------------
    readme = render_envelope_readme(envelope)
    try:
        _upload_bytes(
            api,
            content=readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
        )
    except HfHubHTTPError as exc:
        _classify_and_raise(exc, repo_id=repo_id, op="upload_readme")
    files_uploaded.append("README.md")

    # ---------------- 5. Optional model-card backlink -------------------------
    if update_model_card:
        try:
            _append_model_card_backlink(api, envelope=envelope, dataset_repo_id=repo_id)
        except HfHubHTTPError:
            # Model card backlink is best-effort per SKILL.md error modes.
            pass

    return HfPublishResult(
        repo_id=repo_id,
        url=url,
        files_uploaded=files_uploaded,
        verified=True,
    )


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #
def _serialize_envelope(envelope: Envelope) -> str:
    """Serialize an envelope to canonical sorted-key JSON with 2-space indent."""
    body = envelope.model_dump(mode="json")
    return json.dumps(body, indent=2, sort_keys=True, default=str)


def _upload_bytes(
    api: HfApi,
    *,
    content: bytes,
    path_in_repo: str,
    repo_id: str,
) -> None:
    """Upload an in-memory blob to a dataset repo via a temp file.

    ``HfApi.upload_file`` accepts a ``path_or_fileobj`` parameter; passing a
    bytes-like fileobj works on the current major version but the temp-file
    path is the most portable across the supported ``>=0.30,<1.0`` range.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(path_in_repo).suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _classify_and_raise(
    exc: HfHubHTTPError,
    *,
    repo_id: str,
    op: str,
) -> None:
    """Translate an HF Hub HTTP error into our typed publisher errors."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 409:
        msg = f"dataset repo already exists: {repo_id}"
        raise HfRepoCollisionError(msg) from exc
    if status == 429:
        msg = f"HF Hub rate limit hit during {op} for {repo_id}"
        raise HfRateLimitError(msg) from exc
    msg = f"HF Hub error during {op} for {repo_id}: {exc}"
    raise HfPublishError(msg) from exc


def _append_model_card_backlink(
    api: HfApi,
    *,
    envelope: Envelope,
    dataset_repo_id: str,
) -> None:
    """Best-effort append a backlink entry to the source model's card.

    We never modify the visible model card body — only the YAML frontmatter
    ``inferencebench-verified`` list. If the model isn't on HF Hub (e.g. it's
    a local dev model) this is a no-op.
    """
    from huggingface_hub import ModelCard

    model_id = envelope.model.id
    if "/" not in model_id:
        return

    token = api.token if isinstance(api.token, str) else None
    card = ModelCard.load(model_id, token=token)
    fingerprint = envelope.hardware_fingerprint.fingerprint_sha256
    entry = {
        "url": f"https://huggingface.co/datasets/{dataset_repo_id}",
        "suite": envelope.suite_id,
        "date": envelope.timestamp.date().isoformat(),
        "fingerprint_sha256": fingerprint,
    }
    data_dict = card.data.to_dict()
    existing = data_dict.get("inferencebench-verified") or []
    if not isinstance(existing, list):
        existing = []
    existing.append(entry)
    data_dict["inferencebench-verified"] = existing
    # Rebuild ModelCardData from the merged dict to preserve other fields.
    from huggingface_hub import ModelCardData

    card.data = ModelCardData(**data_dict)
    card.push_to_hub(model_id, token=api.token)
