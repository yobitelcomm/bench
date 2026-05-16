"""Pytest fixtures for the leaderboard tests.

Builds tiny in-memory envelopes via the canonical Pydantic constructor so the
tests don't have to maintain a parallel JSON shape.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


def _hardware_fp() -> HardwareFingerprint:
    body: dict[str, Any] = {
        "dmi_uuid": "11111111-2222-3333-4444-555555555555",
        "gpus": [
            GPU(
                model="H100-SXM5-80GB",
                pci_id="0000:01:00.0",
                serial="1234567890",
                vbios="96.00.74.00.01",
            )
        ],
        "cpu": CPU(model="Intel(R) Xeon(R) Platinum 8480C", microcode="0x2b000571"),
        "memory": Memory(channels=12, speed_mts=4800, ecc=True),
        "bios": BIOS(version="3.4a", resizable_bar=True, above_4g=True),
        "driver": "560.35.03",
        "cuda": "12.6",
        "nccl": "2.22.3",
    }
    placeholder = HardwareFingerprint.model_construct(
        fingerprint_sha256="0" * 64, numa={}, **body
    )
    real = placeholder.compute_fingerprint_sha256()
    return HardwareFingerprint(fingerprint_sha256=real, numa={}, **body)


def _make_envelope(
    *,
    model_id: str,
    suite_id: str = "llm.inference",
    metrics: dict[str, float | int | None] | None = None,
    run_id: str = "01934567-89ab-7000-8000-000000000000",
) -> Envelope:
    return Envelope(
        envelope_version="v1",
        suite_id=suite_id,
        suite_version="1.0.0",
        run_id=run_id,
        timestamp=datetime(2026, 5, 15, 10, 30, 0, tzinfo=UTC),
        model=ModelConfig(
            id=model_id,
            revision="abc1234",
            provider="vllm-local",
            endpoint_hash="d" * 64,
        ),
        engine=EngineConfig(
            name="vllm",
            version="0.7.2",
            config_hash="e" * 64,
            image_digest="sha256:" + "f" * 64,
        ),
        quantization=Quantization(format="fp8"),
        hardware_fingerprint=_hardware_fp(),
        software_provenance=SoftwareProvenance(
            image_digest="sha256:" + "a" * 64,
            pip_freeze_hash="b" * 64,
            git_commit="deadbeef1234567",
            nvidia_smi_q_hash="c" * 64,
        ),
        dataset=DatasetSpec(id="sharegpt-v3", hash="1" * 64),
        seed=42,
        metrics=metrics
        or {
            "ttft_p50_ms": 142.0,
            "ttft_p99_ms": 421.0,
            "throughput_tok_per_s": 1842.1,
            "cost_per_m_tokens_usd": 0.45,
            "joules_per_token": 1.8,
        },
        slo_template="llm.standard",
    )


def _write_envelope(directory: Path, filename: str, envelope: Envelope) -> Path:
    path = directory / filename
    path.write_text(
        json.dumps(envelope.model_dump(mode="json"), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def envelope_corpus(tmp_path: Path) -> Path:
    """Three valid envelopes — two ``llm.inference``, one ``embeddings.retrieval``."""
    env_dir = tmp_path / "envelopes"
    env_dir.mkdir()

    e1 = _make_envelope(
        model_id="meta-llama/Llama-4-Maverick",
        run_id="01934567-89ab-7000-8000-000000000001",
        metrics={
            "ttft_p50_ms": 142.0,
            "ttft_p99_ms": 421.0,
            "throughput_tok_per_s": 1842.1,
            "cost_per_m_tokens_usd": 0.45,
            "joules_per_token": 1.8,
        },
    )
    e2 = _make_envelope(
        model_id="mistralai/Mistral-Large",
        run_id="01934567-89ab-7000-8000-000000000002",
        metrics={
            "ttft_p50_ms": 95.0,
            "ttft_p99_ms": 310.0,
            "throughput_tok_per_s": 2200.0,
            "cost_per_m_tokens_usd": 0.62,
            "joules_per_token": 2.1,
        },
    )
    e3 = _make_envelope(
        model_id="BAAI/bge-large-en-v1.5",
        run_id="01934567-89ab-7000-8000-000000000003",
        suite_id="embeddings.retrieval",
        metrics={
            "ttft_p50_ms": 4.2,
            "throughput_tok_per_s": 45000.0,
        },
    )
    _write_envelope(env_dir, "01-llama.json", e1)
    _write_envelope(env_dir, "02-mistral.json", e2)
    _write_envelope(env_dir, "03-bge.json", e3)
    return env_dir


@pytest.fixture
def corpus_with_garbage(envelope_corpus: Path) -> Path:
    """Same corpus, plus one non-JSON file and one validation-failure file."""
    (envelope_corpus / "broken-syntax.json").write_text(
        "{ this is not json", encoding="utf-8"
    )
    (envelope_corpus / "broken-schema.json").write_text(
        json.dumps({"envelope_version": "v1", "suite_id": "x"}),
        encoding="utf-8",
    )
    return envelope_corpus
