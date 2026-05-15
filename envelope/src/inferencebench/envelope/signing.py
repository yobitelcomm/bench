"""Envelope signing.

Two modes:

* ``SigningMode.DEV`` — pure-Python ed25519 keypair (via ``cryptography``).
  Deterministic, testable, no external services. Used in CI and local dev.

* ``SigningMode.KEYLESS`` — Sigstore OIDC.
  Uses ``sigstore-python`` with a GitHub Actions identity token (CI) or an
  interactive browser flow (local). Bundle includes a Rekor transparency-log
  entry. Production path for OSS users; the verifiability is the moat.

Public API:

    from inferencebench.envelope.signing import sign_envelope, SigningMode
    from inferencebench.envelope.signing import generate_dev_keypair, SigningError

    # First-time setup (dev only)
    generate_dev_keypair(Path("./cosign.key"))   # writes cosign.key + cosign.pub

    # Sign
    signed = sign_envelope(envelope, mode=SigningMode.DEV, dev_key_path=Path("./cosign.key"))

    # Or keyless in CI
    signed = sign_envelope(envelope, mode=SigningMode.KEYLESS)

Signing is idempotent only in that re-signing produces a fresh signature; an
already-signed envelope is rejected (raise ``EnvelopeAlreadySignedError``).
"""

from __future__ import annotations

import base64
import os
from enum import StrEnum
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from inferencebench.envelope.models import Envelope, Signature


class SigningMode(StrEnum):
    """Signing backend."""

    DEV = "dev"
    KEYLESS = "keyless"


class SigningError(Exception):
    """Raised when signing cannot proceed (missing key, no OIDC token, etc.)."""


class EnvelopeAlreadySignedError(SigningError):
    """Raised when ``sign_envelope`` is called on an already-signed envelope."""


# --------------------------------------------------------------------------- #
# Dev-key keypair management                                                  #
# --------------------------------------------------------------------------- #
def generate_dev_keypair(private_key_path: Path, *, force: bool = False) -> tuple[Path, Path]:
    """Generate an ed25519 keypair for dev-mode signing.

    Writes the private key (PEM, unencrypted) to ``private_key_path`` and the
    public key alongside as ``<name>.pub``. Refuses to overwrite unless
    ``force=True``.

    Args:
        private_key_path: Destination for the private key. The corresponding
            public key is written to the same path with a ``.pub`` suffix
            appended (or the ``.key`` extension replaced).
        force: If True, overwrite existing files.

    Returns:
        (private_key_path, public_key_path)

    Raises:
        FileExistsError: If either file exists and ``force=False``.
    """
    public_key_path = _public_path_for(private_key_path)

    if not force:
        if private_key_path.exists():
            msg = f"Refusing to overwrite existing key: {private_key_path}"
            raise FileExistsError(msg)
        if public_key_path.exists():
            msg = f"Refusing to overwrite existing key: {public_key_path}"
            raise FileExistsError(msg)

    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    private_key_path.parent.mkdir(parents=True, exist_ok=True)
    private_key_path.write_bytes(private_pem)
    private_key_path.chmod(0o600)
    public_key_path.write_bytes(public_pem)
    public_key_path.chmod(0o644)

    return private_key_path, public_key_path


def _public_path_for(private_path: Path) -> Path:
    """Derive the public-key path from a private-key path."""
    if private_path.suffix == ".key":
        return private_path.with_suffix(".pub")
    return private_path.with_name(private_path.name + ".pub")


# --------------------------------------------------------------------------- #
# Public signing entry point                                                  #
# --------------------------------------------------------------------------- #
def sign_envelope(
    envelope: Envelope,
    mode: SigningMode = SigningMode.KEYLESS,
    *,
    dev_key_path: Path | None = None,
) -> Envelope:
    """Sign an envelope; return a new Envelope with ``.signature`` populated.

    Args:
        envelope: The unsigned envelope to sign.
        mode: ``DEV`` (local ed25519) or ``KEYLESS`` (Sigstore OIDC).
        dev_key_path: Path to a PEM-encoded ed25519 private key. Required
            for ``DEV`` mode. Ignored in ``KEYLESS``.

    Returns:
        A new :class:`Envelope` with ``.signature`` set. The original is unchanged.

    Raises:
        EnvelopeAlreadySignedError: If the input envelope already has a signature.
        SigningError: Mode-specific errors (missing key, no OIDC token, network failure).
    """
    if envelope.signature is not None:
        msg = (
            "Envelope already has a signature; refusing to re-sign. "
            "Strip .signature first if you really want to re-sign."
        )
        raise EnvelopeAlreadySignedError(msg)

    content_hash = envelope.content_hash()

    if mode == SigningMode.DEV:
        if dev_key_path is None:
            msg = "SigningMode.DEV requires dev_key_path"
            raise SigningError(msg)
        signature = _sign_dev(content_hash, dev_key_path)
    else:
        # SigningMode.KEYLESS is the only remaining enum value; if we ever add more,
        # mypy will flag this exhaustiveness via the StrEnum.
        signature = _sign_keyless(content_hash)

    return envelope.model_copy(update={"signature": signature})


# --------------------------------------------------------------------------- #
# Dev-key signing                                                             #
# --------------------------------------------------------------------------- #
def _sign_dev(content_hash: str, key_path: Path) -> Signature:
    """Sign the content_hash with a local ed25519 private key."""
    if not key_path.exists():
        msg = f"Dev signing key not found: {key_path}. Run generate_dev_keypair() first."
        raise SigningError(msg)

    private_pem = key_path.read_bytes()
    try:
        private_key = serialization.load_pem_private_key(private_pem, password=None)
    except ValueError as exc:
        msg = f"Failed to load private key from {key_path}: {exc}"
        raise SigningError(msg) from exc

    if not isinstance(private_key, Ed25519PrivateKey):
        msg = f"Expected ed25519 private key, got {type(private_key).__name__}"
        raise SigningError(msg)

    sig_bytes = private_key.sign(content_hash.encode("utf-8"))
    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )

    return Signature(
        method="dev-key",
        certificate=public_pem,
        rekor_log_index=-1,
        bundle=base64.b64encode(sig_bytes).decode("ascii"),
    )


# --------------------------------------------------------------------------- #
# Keyless signing (Sigstore OIDC)                                             #
# --------------------------------------------------------------------------- #
def _sign_keyless(content_hash: str) -> Signature:
    """Sign via Sigstore OIDC keyless flow.

    Requires either:
    - ``SIGSTORE_ID_TOKEN`` env var (CI path — GH Actions provides this)
    - Interactive browser flow (local path — sigstore-python opens a browser)

    Raises ``SigningError`` if neither is available or the Sigstore call fails.
    """
    try:
        from sigstore.oidc import IdentityToken, detect_credential
        from sigstore.sign import SigningContext
    except ImportError as exc:
        msg = (
            "sigstore-python is required for keyless signing. "
            "Install with: pip install 'inferencebench-envelope[keyless]'"
        )
        raise SigningError(msg) from exc

    token_str = os.environ.get("SIGSTORE_ID_TOKEN")
    if not token_str:
        try:
            token_str = detect_credential()
        except Exception as exc:
            msg = (
                f"No OIDC token available for keyless signing: {exc}. "
                "Set SIGSTORE_ID_TOKEN or run from a GHA workflow with id-token: write."
            )
            raise SigningError(msg) from exc
        if not token_str:
            msg = (
                "No OIDC token available for keyless signing. "
                "Set SIGSTORE_ID_TOKEN or run from a GHA workflow with id-token: write."
            )
            raise SigningError(msg)

    try:
        identity = IdentityToken(token_str)
    except Exception as exc:
        msg = f"Invalid OIDC token: {exc}"
        raise SigningError(msg) from exc

    try:
        ctx = SigningContext.production()
        with ctx.signer(identity) as signer:
            # sigstore-python >= 3.5 returns a `Bundle` directly from `sign_artifact`.
            bundle = signer.sign_artifact(content_hash.encode("utf-8"))
    except Exception as exc:
        msg = f"Sigstore signing failed: {exc}"
        raise SigningError(msg) from exc

    cert_pem = bundle.signing_certificate.public_bytes(
        encoding=serialization.Encoding.PEM,
    ).decode("utf-8")
    bundle_json: str = bundle.to_json()

    return Signature(
        method="sigstore-cosign",
        certificate=cert_pem,
        rekor_log_index=bundle.log_entry.log_index,
        bundle=base64.b64encode(bundle_json.encode("utf-8")).decode("ascii"),
    )
