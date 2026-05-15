"""Tests for `Envelope.content_hash()` — determinism + tamper detection.

These are critical because Sigstore signs the `content_hash`. If the hash
changes between sign and verify, every signature is invalid.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from inferencebench.envelope import (
    BIOS,
    CPU,
    GPU,
    DatasetSpec,
    EngineConfig,
    Envelope,
    HardwareFingerprint,
    Memory,
    ModelConfig,
    Quantization,
    SoftwareProvenance,
)


def _hw_fp() -> HardwareFingerprint:
    body = {
        "dmi_uuid": "11111111-2222-3333-4444-555555555555",
        "gpus": [
            GPU(
                model="H100-SXM5-80GB",
                pci_id="0000:01:00.0",
                serial="1234567890",
                vbios="96.00.74.00.01",
            ),
        ],
        "cpu": CPU(model="Intel(R) Xeon(R) 8480C", microcode="0x2b000571"),
        "memory": Memory(channels=12, speed_mts=4800, ecc=True),
        "bios": BIOS(version="3.4a", resizable_bar=True, above_4g=True),
        "driver": "560.35.03",
        "cuda": "12.6",
        "nccl": "2.22.3",
    }
    placeholder = HardwareFingerprint.model_construct(fingerprint_sha256="0" * 64, numa={}, **body)
    return HardwareFingerprint(
        fingerprint_sha256=placeholder.compute_fingerprint_sha256(), numa={}, **body
    )


def _envelope() -> Envelope:
    return Envelope(
        envelope_version="v1",
        suite_id="llm.inference",
        suite_version="1.0.0",
        run_id="01934567-89ab-7000-8000-000000000000",
        timestamp=datetime(2026, 5, 15, 10, 30, 0, tzinfo=UTC),
        model=ModelConfig(
            id="meta-llama/Llama-4-Maverick",
            revision="abc1234",
            provider="vllm-local",
            endpoint_hash="d" * 64,
        ),
        engine=EngineConfig(name="vllm", version="0.7.2", config_hash="e" * 64),
        quantization=Quantization(format="fp8"),
        hardware_fingerprint=_hw_fp(),
        software_provenance=SoftwareProvenance(
            pip_freeze_hash="b" * 64,
            git_commit="deadbeef1234567",
        ),
        dataset=DatasetSpec(id="sharegpt-v3", hash="1" * 64),
        seed=42,
        metrics={"ttft_p50_ms": 142.0, "throughput_tok_per_s": 1842.1},
        slo_template="llm.standard",
    )


def test_content_hash_is_64_hex_chars() -> None:
    h = _envelope().content_hash()
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_content_hash_stable_across_field_order() -> None:
    """The canonical JSON sorts keys, so re-ordering inputs shouldn't change the hash."""
    env_a = _envelope()
    # Reconstruct from JSON in a different order
    raw = env_a.model_dump_json()
    parsed = json.loads(raw)
    # Shuffle top-level keys by reversing
    shuffled = dict(reversed(list(parsed.items())))
    env_b = Envelope.model_validate(shuffled)
    assert env_a.content_hash() == env_b.content_hash()


def test_content_hash_changes_on_metric_mutation() -> None:
    env_a = _envelope()
    h_a = env_a.content_hash()

    env_b = env_a.model_copy(update={"metrics": {"ttft_p50_ms": 999.9}})
    h_b = env_b.content_hash()
    assert h_a != h_b, "Metric change must alter content_hash"


def test_content_hash_changes_on_seed_mutation() -> None:
    env_a = _envelope()
    env_b = env_a.model_copy(update={"seed": 999})
    assert env_a.content_hash() != env_b.content_hash()


def test_content_hash_invariant_under_signature_attach() -> None:
    """Attaching a signature is the LAST step — it must not alter content_hash."""
    from inferencebench.envelope import Signature

    env_unsigned = _envelope()
    h_before = env_unsigned.content_hash()

    sig = Signature(
        method="dev-key",
        certificate="dummy",
        rekor_log_index=42,
        bundle="base64bundle",
    )
    env_signed = env_unsigned.model_copy(update={"signature": sig})
    h_after = env_signed.content_hash()

    assert h_before == h_after, (
        "content_hash MUST exclude the signature block — otherwise signing "
        "would change the value it's signing, which is undefined."
    )


def test_canonical_json_no_whitespace() -> None:
    """Canonical JSON uses (',', ':') separators — no extra whitespace."""
    raw = _envelope().to_canonical_json()
    assert ", " not in raw, "Canonical JSON should not contain ', ' (use ',')"
    assert ": " not in raw, "Canonical JSON should not contain ': ' (use ':')"


def test_canonical_json_sorted_keys() -> None:
    """Canonical JSON sorts keys at every nesting level."""
    raw = _envelope().to_canonical_json()
    parsed = json.loads(raw)
    # Check top-level keys are sorted
    top_keys = list(parsed.keys())
    assert top_keys == sorted(top_keys), f"Top keys not sorted: {top_keys}"
    # Check a nested dict is sorted
    hw_keys = list(parsed["hardware_fingerprint"].keys())
    assert hw_keys == sorted(hw_keys), f"hardware_fingerprint keys not sorted: {hw_keys}"


@pytest.mark.parametrize(
    "tamper_field,new_value",
    [
        ("suite_id", "llm.different"),
        ("suite_version", "2.0.0"),
        ("seed", 9999),
    ],
)
def test_content_hash_changes_under_tampering(tamper_field: str, new_value: object) -> None:
    """Any meaningful field change should produce a different content_hash."""
    env = _envelope()
    h_before = env.content_hash()
    tampered = env.model_copy(update={tamper_field: new_value})
    h_after = tampered.content_hash()
    assert h_before != h_after, f"Tampering with {tamper_field} did not change content_hash"
