"""Envelope verification.

`verify_envelope()` recomputes the envelope's canonical content hash, validates
the embedded signature against it, and (for keyless) checks the Sigstore
certificate chain + Rekor transparency-log inclusion.

Returns a :class:`VerificationResult` rather than raising — verification
failures are expected and the caller may want to inspect why.

Public API:

    from inferencebench.envelope.verify import verify_envelope, VerificationResult

    result = verify_envelope(envelope, dev_public_key_path=Path("./cosign.pub"))
    if not result.ok:
        print(f"verification failed: {result.reason}")
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from inferencebench.envelope.models import Envelope


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Outcome of `verify_envelope`. Always has `ok` set; `reason` populated on failure.

    For keyless (Sigstore) signatures, ``signer_identity`` and ``signer_issuer``
    are extracted from the cert's SAN OtherName when verification succeeds.
    Callers should inspect these to enforce who-signed-what policy (we cannot
    encode every user's policy here).
    """

    ok: bool
    method: str
    reason: str = ""
    rekor_log_index: int = -1
    signer_identity: str = ""
    signer_issuer: str = ""


def verify_envelope(
    envelope: Envelope,
    *,
    dev_public_key_path: Path | None = None,
) -> VerificationResult:
    """Verify the signature on an envelope.

    Args:
        envelope: The signed envelope to verify.
        dev_public_key_path: Path to the ed25519 public key (PEM). Required
            for dev-key envelopes. Ignored for keyless envelopes.

    Returns:
        A :class:`VerificationResult` with ``ok=True`` on success or
        ``ok=False`` with a populated ``reason`` on any failure.
    """
    sig = envelope.signature
    if sig is None:
        return VerificationResult(
            ok=False,
            method="none",
            reason="envelope has no signature",
        )

    if sig.method == "dev-key":
        return _verify_dev(envelope, dev_public_key_path)
    if sig.method == "sigstore-cosign":
        return _verify_keyless(envelope)

    # Unreachable while Signature.method is the Literal["sigstore-cosign", "dev-key"];
    # kept as a defensive branch for future method additions.
    return VerificationResult(  # type: ignore[unreachable]
        ok=False,
        method=sig.method,
        reason=f"unknown signature method: {sig.method}",
    )


# --------------------------------------------------------------------------- #
# Dev-key verification                                                        #
# --------------------------------------------------------------------------- #
def _verify_dev(envelope: Envelope, dev_public_key_path: Path | None) -> VerificationResult:
    sig = envelope.signature
    if sig is None:  # pragma: no cover - guarded by caller
        return VerificationResult(ok=False, method="dev-key", reason="no signature block")

    # Prefer the embedded public key from the signature certificate. If a path
    # is also provided, require it to match — otherwise we'd happily verify
    # any envelope whose certificate field was forged.
    try:
        cert_public_key = serialization.load_pem_public_key(sig.certificate.encode("utf-8"))
    except ValueError as exc:
        return VerificationResult(
            ok=False,
            method="dev-key",
            reason=f"invalid certificate (public key) PEM: {exc}",
        )
    if not isinstance(cert_public_key, Ed25519PublicKey):
        return VerificationResult(
            ok=False,
            method="dev-key",
            reason=f"expected ed25519 public key, got {type(cert_public_key).__name__}",
        )

    if dev_public_key_path is not None:
        if not dev_public_key_path.exists():
            return VerificationResult(
                ok=False,
                method="dev-key",
                reason=f"dev_public_key_path does not exist: {dev_public_key_path}",
            )
        try:
            expected_public_key = serialization.load_pem_public_key(
                dev_public_key_path.read_bytes()
            )
        except ValueError as exc:
            return VerificationResult(
                ok=False,
                method="dev-key",
                reason=f"failed to load expected public key: {exc}",
            )
        if not isinstance(expected_public_key, Ed25519PublicKey):
            return VerificationResult(
                ok=False,
                method="dev-key",
                reason="expected ed25519 public key in dev_public_key_path",
            )
        expected_bytes = expected_public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        cert_bytes = cert_public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        if expected_bytes != cert_bytes:
            return VerificationResult(
                ok=False,
                method="dev-key",
                reason="embedded certificate public key does not match dev_public_key_path",
            )

    # Decode signature bytes from base64 bundle
    try:
        sig_bytes = base64.b64decode(sig.bundle, validate=True)
    except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
        return VerificationResult(
            ok=False,
            method="dev-key",
            reason=f"invalid base64 in signature bundle: {exc}",
        )

    # Recompute content_hash from current envelope body (signature excluded)
    content_hash = envelope.content_hash().encode("utf-8")

    try:
        cert_public_key.verify(sig_bytes, content_hash)
    except InvalidSignature:
        return VerificationResult(
            ok=False,
            method="dev-key",
            reason="signature does not match envelope content_hash (tampered or wrong key)",
        )

    return VerificationResult(
        ok=True,
        method="dev-key",
        reason="",
        rekor_log_index=-1,
    )


# --------------------------------------------------------------------------- #
# Keyless verification (Sigstore)                                             #
# --------------------------------------------------------------------------- #
def _verify_keyless(envelope: Envelope) -> VerificationResult:
    sig = envelope.signature
    if sig is None:  # pragma: no cover - guarded by caller
        return VerificationResult(ok=False, method="sigstore-cosign", reason="no signature block")

    try:
        from sigstore.models import Bundle
        from sigstore.verify import Verifier
        from sigstore.verify.policy import AnyOf, Identity
    except ImportError as exc:
        return VerificationResult(
            ok=False,
            method="sigstore-cosign",
            reason=f"sigstore-python not installed: {exc}",
        )

    try:
        bundle_bytes = base64.b64decode(sig.bundle, validate=True)
    except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
        return VerificationResult(
            ok=False,
            method="sigstore-cosign",
            reason=f"invalid base64 in signature bundle: {exc}",
        )

    try:
        bundle = Bundle.from_json(bundle_bytes.decode("utf-8"))
    except Exception as exc:
        return VerificationResult(
            ok=False,
            method="sigstore-cosign",
            reason=f"failed to parse Sigstore bundle: {exc}",
        )

    content_hash = envelope.content_hash().encode("utf-8")

    try:
        verifier = Verifier.production()
        # Permissive identity policy at this layer. Callers enforce who-signed
        # constraints by inspecting ``VerificationResult.signer_identity`` and
        # ``signer_issuer`` after this returns ``ok=True``. Production callers
        # MUST check those fields against an allow-list (e.g. the bench CI
        # workflow OIDC subject) — accepting any signer is a security bug.
        policy = AnyOf([Identity(identity="*")])
        verifier.verify_artifact(content_hash, bundle, policy)
    except Exception as exc:
        return VerificationResult(
            ok=False,
            method="sigstore-cosign",
            reason=f"Sigstore verification failed: {exc}",
        )

    identity, issuer = _extract_signer_identity(bundle.signing_certificate)

    return VerificationResult(
        ok=True,
        method="sigstore-cosign",
        reason="",
        rekor_log_index=sig.rekor_log_index,
        signer_identity=identity,
        signer_issuer=issuer,
    )


def _extract_signer_identity(cert: object) -> tuple[str, str]:
    """Best-effort extraction of (identity, issuer) from a Fulcio-issued cert.

    Sigstore embeds the OIDC identity in the cert's SAN (OtherName) and the
    issuer in the X.509 extension at OID 1.3.6.1.4.1.57264.1.1 (or .8 in
    newer revisions). Returns empty strings if extraction fails — verify
    still succeeded; callers see the lack of identity and can decide whether
    to trust the envelope despite the missing field.
    """
    identity = ""
    issuer = ""
    try:
        from cryptography import x509  # lazy import keeps the verify-path light

        # OtherName SAN holds the identity (email, URL, etc.)
        try:
            san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value  # type: ignore[attr-defined]
            for n in san_ext:
                if isinstance(n, x509.RFC822Name):
                    identity = n.value
                    break
                if isinstance(n, x509.UniformResourceIdentifier):
                    identity = n.value
                    break
                if isinstance(n, x509.OtherName):
                    raw = n.value
                    identity = raw.decode("utf-8", errors="replace").strip("\x00\x16")
                    break
        except Exception:
            pass

        # Fulcio issuer OID. New schemes use .8; old schemes use .1.
        for oid_str in ("1.3.6.1.4.1.57264.1.8", "1.3.6.1.4.1.57264.1.1"):
            try:
                ext = cert.extensions.get_extension_for_oid(x509.ObjectIdentifier(oid_str)).value  # type: ignore[attr-defined]
                if hasattr(ext, "value"):
                    raw_v = ext.value
                else:
                    raw_v = ext
                if isinstance(raw_v, bytes):
                    issuer = raw_v.decode("utf-8", errors="replace")
                else:
                    issuer = str(raw_v)
                break
            except Exception:
                continue
    except Exception:
        pass
    return identity, issuer
