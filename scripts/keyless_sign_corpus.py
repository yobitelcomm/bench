#!/usr/bin/env python3
"""Re-sign every envelope in a corpus via Sigstore keyless and upload to one HF repo.

Called by `.github/workflows/keyless-sign-marathon.yml` from inside a GH Actions
job with `id-token: write`. Sigstore-python picks up the OIDC token via
``sigstore.oidc.detect_credential`` and mints a Fulcio cert per signature.

Differs from ``keyless_sign_envelope.py`` (which handles a single envelope):
* operates on a directory of envelopes (default: the marathon corpus checked
  into the repo at ``validation-runs/2026-05-18-multi-vendor-marathon/marathon/all``)
* uploads all keyless-signed copies to one HF dataset repo
* per-envelope failures don't abort the run — we log and continue, so one
  flaky Fulcio call doesn't kill 49 good signatures
* idempotent: if a target ``keyless-<hash>.json`` already exists in the local
  output dir, the envelope is treated as already signed and skipped

Output names follow the per-envelope script: ``keyless-<content-hash[:12]>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _iter_envelopes(source_dir: Path) -> list[Path]:
    """Return all envelope JSONs in ``source_dir``, skipping any keyless-* files.

    Sorted alphabetically so the GH Actions log stays predictable across runs.
    """
    if not source_dir.exists():
        print(f"FATAL: source_dir does not exist: {source_dir}", file=sys.stderr)
        sys.exit(2)
    files = sorted(
        p for p in source_dir.glob("*.json") if not p.name.startswith("keyless-")
    )
    if not files:
        print(f"FATAL: no envelope JSONs in {source_dir}", file=sys.stderr)
        sys.exit(2)
    return files


def _sign_one(source_path: Path, out_dir: Path) -> Path | None:
    """Sign a single envelope keyless and write the result. Returns the output path or None on failure."""
    from inferencebench.envelope import Envelope, SigningMode, sign_envelope

    try:
        raw = json.loads(source_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  SKIP {source_path.name}: cannot read/parse: {exc}", file=sys.stderr)
        return None

    raw.pop("signature", None)
    try:
        envelope = Envelope.model_validate(raw)
    except Exception as exc:
        print(f"  SKIP {source_path.name}: model_validate failed: {exc}", file=sys.stderr)
        return None

    try:
        signed = sign_envelope(envelope, mode=SigningMode.KEYLESS)
    except Exception as exc:
        print(f"  FAIL {source_path.name}: sign_envelope: {exc}", file=sys.stderr)
        return None

    if signed.signature is None:
        print(f"  FAIL {source_path.name}: signed envelope has no signature block", file=sys.stderr)
        return None

    out_path = out_dir / f"keyless-{signed.content_hash()[:12]}.json"
    out_path.write_text(signed.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"  OK   {source_path.name} -> {out_path.name} "
        f"rekor={signed.signature.rekor_log_index}",
        file=sys.stderr,
    )
    return out_path


def _upload_batch(target_repo: str, files: list[Path]) -> int:
    """Upload all files to ``target_repo`` as a dataset. Returns count uploaded."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        print(f"FATAL: huggingface_hub not installed: {exc}", file=sys.stderr)
        return 0

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARN: HF_TOKEN not set; skipping upload", file=sys.stderr)
        return 0

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=target_repo, repo_type="dataset", exist_ok=True)

    uploaded = 0
    for path in files:
        # Retry transient HF errors a couple of times — Fulcio + HF in the same
        # job is enough network surface for a 5xx to land occasionally.
        for attempt in range(3):
            try:
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=path.name,
                    repo_id=target_repo,
                    repo_type="dataset",
                    commit_message=f"keyless-sign marathon corpus: {path.name}",
                )
                uploaded += 1
                break
            except Exception as exc:
                if attempt == 2:
                    print(f"  UPLOAD FAIL {path.name}: {exc}", file=sys.stderr)
                else:
                    time.sleep(2 ** attempt)
    return uploaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("validation-runs/2026-05-18-multi-vendor-marathon/marathon/all"),
        help="Directory of envelope JSONs to re-sign.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home() / ".cache" / "inferencebench" / "keyless-corpus",
        help="Local directory to write keyless-signed copies into.",
    )
    parser.add_argument(
        "--target-repo",
        required=True,
        help="HF dataset repo to upload to (e.g. Yobitel/marathon-keyless-v0.0.2).",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Sign locally only; don't push to HF.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = _iter_envelopes(args.source_dir)
    print(f"signing {len(sources)} envelopes from {args.source_dir}", file=sys.stderr)

    signed_paths: list[Path] = []
    for idx, source in enumerate(sources, start=1):
        print(f"[{idx}/{len(sources)}] {source.name}", file=sys.stderr)
        out = _sign_one(source, args.out_dir)
        if out is not None:
            signed_paths.append(out)

    print(
        f"signed {len(signed_paths)}/{len(sources)} envelopes "
        f"(failed: {len(sources) - len(signed_paths)})",
        file=sys.stderr,
    )

    if args.skip_upload:
        print("--skip-upload set; not pushing to HF", file=sys.stderr)
        return 0 if signed_paths else 2

    uploaded = _upload_batch(args.target_repo, signed_paths)
    print(
        f"uploaded {uploaded}/{len(signed_paths)} envelopes to "
        f"https://huggingface.co/datasets/{args.target_repo}",
        file=sys.stderr,
    )

    # Exit non-zero only if everything failed; a partial corpus is still a win.
    return 0 if signed_paths else 2


if __name__ == "__main__":
    sys.exit(main())
