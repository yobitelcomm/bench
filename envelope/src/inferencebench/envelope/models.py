"""Pydantic v2 models for the canonical signed envelope.

The Envelope is the product's defensibility moat. Every benchmark result is
captured as one Envelope, content-hashed, and Sigstore-signed.

Public surface:

    Envelope             — top-level model, signed result of one benchmark run
    EnvelopeBuilder      — builder for assembling an unsigned envelope from raw bits
    HardwareFingerprint  — DMI + GPUs + CPU + memory + BIOS + drivers
    SoftwareProvenance   — pip freeze hash, container image digest, git commit
    EngineConfig         — inference engine identity + config
    ModelConfig          — model identity + revision + endpoint
    Quantization         — precision/quant format
    DatasetSpec          — dataset id + canonical hash
    Metrics              — measured numbers (free-form dict; per-suite keys)
    Signature            — Sigstore signature bundle

See `skills/envelope-signing/SKILL.md` for the full design.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "v1"


# --------------------------------------------------------------------------- #
# Base configuration shared by every envelope sub-model.                      #
# --------------------------------------------------------------------------- #
class _Base(BaseModel):
    """Pydantic config shared by all envelope models."""

    model_config = ConfigDict(
        extra="allow",  # forward-compat: tolerate unknown fields, preserve them
        frozen=False,
        str_strip_whitespace=True,
        populate_by_name=True,
    )


# --------------------------------------------------------------------------- #
# Sub-models                                                                  #
# --------------------------------------------------------------------------- #
class GPU(_Base):
    """A single GPU in the hardware fingerprint."""

    model: Annotated[
        str, Field(min_length=1, description="Marketing model name, e.g. 'H100-SXM5-80GB'.")
    ]
    pci_id: Annotated[
        str, Field(min_length=1, description="PCI bus identifier, e.g. '0000:01:00.0'.")
    ]
    serial: Annotated[str, Field(min_length=1, description="GPU serial number.")]
    vbios: Annotated[str, Field(min_length=1, description="VBIOS version string.")]


class CPU(_Base):
    """CPU info in the hardware fingerprint."""

    model: Annotated[str, Field(min_length=1, description="CPU model name from /proc/cpuinfo.")]
    microcode: Annotated[str, Field(min_length=1, description="Microcode revision (hex).")]


class Memory(_Base):
    """RAM configuration."""

    channels: Annotated[int, Field(ge=1, le=16, description="DDR channels populated.")]
    speed_mts: Annotated[int, Field(ge=1, description="Memory speed in MT/s.")]
    ecc: bool = Field(default=False, description="ECC enabled?")


class BIOS(_Base):
    """BIOS settings relevant to inference perf."""

    version: Annotated[str, Field(min_length=1)]
    resizable_bar: bool
    above_4g: bool


class HardwareFingerprint(_Base):
    """Composite hardware fingerprint, SHA-256'd for fast equality checks.

    The `fingerprint_sha256` field is computed from the canonical JSON
    representation of the other fields (sorted keys, no whitespace).
    """

    fingerprint_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    dmi_uuid: Annotated[str, Field(min_length=1)]
    gpus: Annotated[list[GPU], Field(default_factory=list)]
    cpu: CPU
    memory: Memory
    bios: BIOS
    numa: dict[str, Any] = Field(default_factory=dict, description="NUMA topology.")
    driver: Annotated[str, Field(min_length=1, description="GPU driver version.")]
    cuda: Annotated[str, Field(min_length=1, description="CUDA toolkit version.")]
    nccl: str = Field(default="", description="NCCL version (empty if not applicable).")

    @model_validator(mode="after")
    def _verify_fingerprint(self) -> HardwareFingerprint:
        """Ensure the embedded `fingerprint_sha256` matches the canonical body."""
        recomputed = self.compute_fingerprint_sha256()
        if recomputed != self.fingerprint_sha256:
            msg = (
                f"HardwareFingerprint.fingerprint_sha256 mismatch: "
                f"stored={self.fingerprint_sha256!r}, computed={recomputed!r}"
            )
            raise ValueError(msg)
        return self

    def compute_fingerprint_sha256(self) -> str:
        """Compute the SHA-256 over the canonical fingerprint body."""
        body = {
            "dmi_uuid": self.dmi_uuid,
            "gpus": sorted(
                [
                    {
                        "model": g.model,
                        "pci_id": g.pci_id,
                        "serial": g.serial,
                        "vbios": g.vbios,
                    }
                    for g in self.gpus
                ],
                key=lambda g: g["pci_id"],
            ),
            "cpu": {"model": self.cpu.model, "microcode": self.cpu.microcode},
            "memory": {
                "channels": self.memory.channels,
                "speed_mts": self.memory.speed_mts,
                "ecc": self.memory.ecc,
            },
            "bios": {
                "version": self.bios.version,
                "resizable_bar": self.bios.resizable_bar,
                "above_4g": self.bios.above_4g,
            },
            "numa": self.numa,
            "driver": self.driver,
            "cuda": self.cuda,
            "nccl": self.nccl,
        }
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SoftwareProvenance(_Base):
    """Identifies the exact software stack that produced the result."""

    image_digest: Annotated[
        str,
        Field(
            description="OCI image digest of the engine container, or empty if running natively.",
        ),
    ] = ""
    pip_freeze_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    git_commit: Annotated[str, Field(min_length=7, max_length=40)]
    nvidia_smi_q_hash: str = Field(
        default="",
        description="SHA-256 of `nvidia-smi -q` output, empty if not on NVIDIA.",
    )


class ModelConfig(_Base):
    """The model under test."""

    id: Annotated[
        str,
        Field(
            min_length=1, description="Provider-prefixed id, e.g. 'meta-llama/Llama-4-Maverick'."
        ),
    ]
    revision: Annotated[
        str, Field(min_length=7, max_length=40, description="Git SHA or HF revision.")
    ]
    provider: Annotated[
        str, Field(min_length=1, description="vllm-local, together, anthropic, ...")
    ]
    endpoint_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class EngineConfig(_Base):
    """The inference engine."""

    name: Annotated[str, Field(min_length=1, description="vllm, sglang, trtllm, ...")]
    version: Annotated[str, Field(min_length=1)]
    config_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    image_digest: str = Field(default="", description="OCI image digest if containerized.")


class Quantization(_Base):
    """Precision / quantization format."""

    format: Annotated[
        str,
        Field(description="bf16, fp16, fp8, nvfp4, awq-int4, gptq-int4, gguf-q4_k_m, ..."),
    ]
    method: str = Field(default="", description="Free-form description of the quant method.")


class DatasetSpec(_Base):
    """Dataset under evaluation."""

    id: Annotated[str, Field(min_length=1)]
    hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Signature(_Base):
    """Sigstore signature bundle."""

    method: Literal["sigstore-cosign", "dev-key"]
    certificate: Annotated[str, Field(min_length=1)]
    rekor_log_index: int = Field(
        default=-1, description="Rekor transparency log index; -1 for dev mode."
    )
    bundle: str = Field(default="", description="Full Sigstore bundle (base64).")


# --------------------------------------------------------------------------- #
# Top-level Envelope                                                          #
# --------------------------------------------------------------------------- #
class Envelope(_Base):
    """The canonical signed result envelope for one benchmark run."""

    envelope_version: Literal["v1"] = Field(default="v1", description="Schema version.")
    suite_id: Annotated[str, Field(min_length=1, description="e.g. 'llm.inference'.")]
    suite_version: Annotated[
        str,
        Field(pattern=r"^\d+\.\d+\.\d+(-[\w.]+)?$", description="Plugin SemVer."),
    ]
    run_id: Annotated[str, Field(description="UUIDv7 (sortable by timestamp).")]
    timestamp: datetime = Field(description="UTC RFC 3339 timestamp.")
    model: ModelConfig
    engine: EngineConfig
    quantization: Quantization | None = None
    hardware_fingerprint: HardwareFingerprint
    software_provenance: SoftwareProvenance
    dataset: DatasetSpec
    seed: int
    driver_options: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float | int | None] = Field(
        default_factory=dict,
        description="Per-suite metrics, free-form. Validated by plugin's render_leaderboard.",
    )
    distributions: dict[str, str] = Field(
        default_factory=dict,
        description="Pointers to distribution data (path-to-parquet or inline histogram).",
    )
    slo_template: str = Field(
        default="", description="SLO template id (llm.standard, voice.realtime, ...)."
    )
    warnings: list[str] = Field(default_factory=list)
    signature: Signature | None = Field(
        default=None,
        description="Sigstore signature. None = unsigned; signed envelopes always populate this.",
    )

    @model_validator(mode="after")
    def _basic_consistency(self) -> Envelope:
        """Cross-field validation."""
        if not self.metrics:
            msg = "Envelope.metrics must contain at least one metric."
            raise ValueError(msg)
        # timestamp must be UTC
        if self.timestamp.tzinfo is None:
            msg = "Envelope.timestamp must be timezone-aware (UTC)."
            raise ValueError(msg)
        return self

    def content_hash(self) -> str:
        """SHA-256 over the canonical body (everything except the signature block).

        Sigstore signs this value. Recomputing it during `bench verify` must
        match the signed digest exactly — any mutation between sign and verify
        invalidates the signature.
        """
        body = self.model_dump(exclude={"signature"}, mode="json")
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_canonical_json(self) -> str:
        """Return the canonical JSON representation used for content_hash."""
        body = self.model_dump(exclude={"signature"}, mode="json")
        return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)


# --------------------------------------------------------------------------- #
# Builder                                                                     #
# --------------------------------------------------------------------------- #
class EnvelopeBuilder:
    """Build an unsigned Envelope from raw measurement bits + context."""

    def __init__(
        self,
        *,
        suite_id: str,
        suite_version: str,
        model: ModelConfig,
        engine: EngineConfig,
        hardware_fingerprint: HardwareFingerprint,
        software_provenance: SoftwareProvenance,
        dataset: DatasetSpec,
        metrics: dict[str, float | int | None],
        seed: int,
        quantization: Quantization | None = None,
        slo_template: str = "",
        driver_options: dict[str, Any] | None = None,
        distributions: dict[str, str] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self._suite_id = suite_id
        self._suite_version = suite_version
        self._model = model
        self._engine = engine
        self._quantization = quantization
        self._hardware_fingerprint = hardware_fingerprint
        self._software_provenance = software_provenance
        self._dataset = dataset
        self._seed = seed
        self._metrics = metrics
        self._slo_template = slo_template
        self._driver_options = driver_options or {}
        self._distributions = distributions or {}
        self._warnings = warnings or []

    def build(self) -> Envelope:
        """Produce an unsigned envelope. Signature is added later by AttestService.

        Returns:
            An :class:`Envelope` with a freshly-minted UUIDv7 run_id and UTC timestamp.
        """
        return Envelope(
            envelope_version="v1",
            suite_id=self._suite_id,
            suite_version=self._suite_version,
            run_id=str(_uuid7()),
            timestamp=datetime.now(UTC),
            model=self._model,
            engine=self._engine,
            quantization=self._quantization,
            hardware_fingerprint=self._hardware_fingerprint,
            software_provenance=self._software_provenance,
            dataset=self._dataset,
            seed=self._seed,
            driver_options=self._driver_options,
            metrics=self._metrics,
            distributions=self._distributions,
            slo_template=self._slo_template,
            warnings=self._warnings,
            signature=None,
        )


# --------------------------------------------------------------------------- #
# UUIDv7 helper                                                               #
# --------------------------------------------------------------------------- #
def _uuid7() -> uuid.UUID:
    """Minimal UUIDv7 generator. Sortable by creation time.

    Avoids a third-party dep; if we add ``uuid7`` later we can swap to that.
    """
    import os
    import time

    ts_ms = int(time.time() * 1000)
    # 48 bits timestamp + 12 bits random + 62 bits random + version/variant nibbles
    rand_a = int.from_bytes(os.urandom(2), "big") & 0x0FFF
    rand_b = int.from_bytes(os.urandom(8), "big") & 0x3FFFFFFFFFFFFFFF
    uuid_int = (ts_ms & 0xFFFFFFFFFFFF) << 80 | 0x7 << 76 | rand_a << 64 | 0b10 << 62 | rand_b
    return uuid.UUID(int=uuid_int)
