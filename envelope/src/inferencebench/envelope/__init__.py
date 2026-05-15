"""Canonical signed-envelope spec for InferenceBench.

The Envelope is the product's defensibility moat. Every benchmark run produces
one signed Envelope that any third party can verify with `bench verify`.

Public API:

    from inferencebench.envelope import Envelope, EnvelopeBuilder
    from inferencebench.envelope import HardwareFingerprint, Signature
    from inferencebench.envelope import SCHEMA_VERSION

    builder = EnvelopeBuilder(...)
    envelope = builder.build()           # unsigned
    h = envelope.content_hash()          # SHA-256 of canonical body
    # signing is ticket 0005

See `skills/envelope-signing/SKILL.md` for the full signing/verification flow.
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

__all__ = [
    "BIOS",
    "CPU",
    "GPU",
    "SCHEMA_VERSION",
    "DatasetSpec",
    "EngineConfig",
    "Envelope",
    "EnvelopeBuilder",
    "HardwareFingerprint",
    "Memory",
    "ModelConfig",
    "Quantization",
    "Signature",
    "SoftwareProvenance",
]
