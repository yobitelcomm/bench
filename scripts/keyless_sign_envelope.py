#!/usr/bin/env python3
"""Re-sign one envelope via Sigstore keyless OIDC and upload it to HF.

Called by `.github/workflows/keyless-sign-demo.yml` from inside a GH Actions
job that has `id-token: write` permission. The OIDC token is auto-injected
into ``SIGSTORE_ID_TOKEN`` (or detected via ``sigstore.oidc.detect_credential``)
by sigstore-python.

Steps:
1. Resolve the source envelope from the local fetch cache (already-downloaded
   by ``bench fetch`` in the previous workflow step).
2. Strip the existing dev-key signature so we can re-sign.
3. Call ``sign_envelope(..., mode=SigningMode.KEYLESS)`` — sigstore-python
   mints a Fulcio cert via OIDC and signs the canonical content hash.
4. Write the re-signed envelope to ``keyless-<hash>.json`` next to the
   source in the fetch cache.
5. Upload to the target HF dataset repo (created if missing).

The script is intentionally side-effect-only — it doesn't print envelope
contents. The workflow's downstream `bench verify` step is the test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _resolve_cache_path(source_uri: str) -> Path:
    """Find the local cache file `bench fetch` wrote earlier in the workflow."""
    cache_root = Path.home() / ".cache" / "inferencebench" / "fetched"
    if not cache_root.exists():
        print(f"FATAL: fetch cache missing at {cache_root}", file=sys.stderr)
        sys.exit(2)
    # Pick the most-recently-modified envelope file in the cache.
    candidates = sorted(
        (p for p in cache_root.glob("*.json") if not p.name.startswith("keyless-")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print(f"FATAL: no envelope JSONs in {cache_root}", file=sys.stderr)
        print(f"hint: run `bench fetch {source_uri}` before this script", file=sys.stderr)
        sys.exit(2)
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-uri",
        required=True,
        help="HF or file:// URI of the envelope to re-sign (used for log lines only).",
    )
    parser.add_argument(
        "--target-repo",
        required=True,
        help=(
            "HF dataset repo to upload the keyless-signed envelope to "
            "(e.g. Yobitel/keyless-signed-demo)."
        ),
    )
    args = parser.parse_args()

    from inferencebench.envelope import Envelope, SigningMode, sign_envelope

    source_path = _resolve_cache_path(args.source_uri)
    print(f"source envelope: {source_path}", file=sys.stderr)

    raw = json.loads(source_path.read_text("utf-8"))
    # Strip the existing signature so sign_envelope doesn't raise
    # EnvelopeAlreadySignedError on the re-sign step.
    raw.pop("signature", None)
    envelope = Envelope.model_validate(raw)

    print("signing keyless (sigstore-cosign)...", file=sys.stderr)
    signed = sign_envelope(envelope, mode=SigningMode.KEYLESS)
    if signed.signature is None:
        print("FATAL: sign_envelope returned envelope with no signature", file=sys.stderr)
        return 2
    print(
        f"signed: method={signed.signature.method} "
        f"rekor_log_index={signed.signature.rekor_log_index}",
        file=sys.stderr,
    )

    out_path = source_path.parent / f"keyless-{signed.content_hash()[:12]}.json"
    out_path.write_text(signed.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote: {out_path}", file=sys.stderr)

    # Upload to HF
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        print(f"FATAL: huggingface_hub not installed: {exc}", file=sys.stderr)
        return 2

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARN: HF_TOKEN not set; skipping upload", file=sys.stderr)
        return 0

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=args.target_repo, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=out_path.name,
        repo_id=args.target_repo,
        repo_type="dataset",
        commit_message=(
            f"keyless-signed via GH Actions for {args.source_uri}; "
            f"rekor_log_index={signed.signature.rekor_log_index}"
        ),
    )
    print(
        f"uploaded: https://huggingface.co/datasets/{args.target_repo}/blob/main/{out_path.name}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
