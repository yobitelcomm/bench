"""Canonical signed-envelope spec for InferenceBench.

The Envelope is the product's defensibility moat. Every benchmark run produces
one signed Envelope that any third party can verify with `bench verify`.

Public API:

    from inferencebench.envelope import Envelope, EnvelopeBuilder
    from inferencebench.envelope import HardwareFingerprint, Signature
    from inferencebench.envelope import SCHEMA_VERSION
    from inferencebench.envelope import sign_envelope, verify_envelope, SigningMode

    builder = EnvelopeBuilder(...)
    envelope = builder.build()                                          # unsigned
    signed = sign_envelope(envelope, mode=SigningMode.DEV,
                           dev_key_path=Path("./cosign.key"))           # signed
    result = verify_envelope(signed,
                             dev_public_key_path=Path("./cosign.pub"))  # verified
    assert result.ok
"""

from inferencebench.envelope.models import (
    BIOS,
    CPU,
    GPU,
    SCHEMA_VERSION,
    DatasetSpec,
    EngineConfig,
    Envelope,
    EnvelopeBuilder,
    HardwareFingerprint,
    Memory,
    ModelConfig,
    Quantization,
    Signature,
    SoftwareProvenance,
)
from inferencebench.envelope.signing import (
    EnvelopeAlreadySignedError,
    SigningError,
    SigningMode,
    generate_dev_keypair,
    sign_envelope,
)
from inferencebench.envelope.verify import VerificationResult, verify_envelope

__all__ = [
    "BIOS",
    "CPU",
    "GPU",
    "SCHEMA_VERSION",
    "DatasetSpec",
    "EngineConfig",
    "Envelope",
    "EnvelopeAlreadySignedError",
    "EnvelopeBuilder",
    "HardwareFingerprint",
    "Memory",
    "ModelConfig",
    "Quantization",
    "Signature",
    "SigningError",
    "SigningMode",
    "SoftwareProvenance",
    "VerificationResult",
    "generate_dev_keypair",
    "sign_envelope",
    "verify_envelope",
]
