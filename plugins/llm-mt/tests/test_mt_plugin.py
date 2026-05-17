"""Tests for the llm-mt plugin scaffold + scoring pipeline."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inferencebench.cli import app
from inferencebench.envelope import generate_dev_keypair
from inferencebench_mt import (
    BenchmarkSpec,
    EngineKind,
    LLMMTPlugin,
    RunContext,
)


# --------------------------------------------------------------------------- #
# Plugin contract                                                             #
# --------------------------------------------------------------------------- #
def test_plugin_metadata() -> None:
    plugin = LLMMTPlugin()
    assert plugin.suite_id == "llm.mt"
    assert plugin.version == "0.0.2"
    assert plugin.description


def test_plugin_lists_two_bundled_benchmarks() -> None:
    plugin = LLMMTPlugin()
    specs = plugin.list_benchmarks()
    assert len(specs) == 2
    ids = {s.benchmark_id for s in specs}
    assert ids == {
        "llm.mt.flores-200-mini-en-fr",
        "llm.mt.flores-200-mini-en-de",
    }


def test_plugin_get_benchmark_en_fr() -> None:
    plugin = LLMMTPlugin()
    spec = plugin.get_benchmark("llm.mt.flores-200-mini-en-fr")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.modality == "llm"
    assert spec.kind == "translation"
    assert spec.scoring == "chrf"
    assert spec.source_lang == "en"
    assert spec.target_lang == "fr"
    assert spec.dataset.path == "flores-mini-en-fr.jsonl"


def test_plugin_get_benchmark_en_de_target_lang() -> None:
    plugin = LLMMTPlugin()
    spec = plugin.get_benchmark("llm.mt.flores-200-mini-en-de")
    assert spec.target_lang == "de"
    assert spec.dataset.path == "flores-mini-en-de.jsonl"


def test_plugin_get_benchmark_missing_id_raises_keyerror() -> None:
    plugin = LLMMTPlugin()
    with pytest.raises(KeyError):
        plugin.get_benchmark("nonexistent.benchmark")


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def test_validate_warns_when_self_hosted_base_url_missing() -> None:
    plugin = LLMMTPlugin()
    spec = plugin.get_benchmark("llm.mt.flores-200-mini-en-fr")
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.VLLM,
        base_url="",
        output_dir=Path("/tmp/bench"),
    )
    warnings = plugin.validate(spec, ctx)
    assert any("base_url" in w.lower() for w in warnings)


def test_validate_cohere_engine_does_not_require_base_url() -> None:
    plugin = LLMMTPlugin()
    spec = plugin.get_benchmark("llm.mt.flores-200-mini-en-fr")
    ctx = RunContext(
        model_id="cohere/command-r",
        engine_kind=EngineKind.COHERE,
        base_url="",
        output_dir=Path("/tmp/bench"),
    )
    warnings = plugin.validate(spec, ctx)
    assert not any("base_url" in w.lower() for w in warnings)


# --------------------------------------------------------------------------- #
# End-to-end run (mocked client)                                              #
# --------------------------------------------------------------------------- #
def _expected_fr_translations() -> dict[str, str]:
    """The canonical French references from the bundled en-fr fixture."""
    return {
        "Hello, how are you?": "Bonjour, comment allez-vous ?",
        "Good morning, my friend.": "Bonjour, mon ami.",
        "The president signed the new climate agreement yesterday.": (
            "Le président a signé le nouvel accord sur le climat hier."
        ),
        "Stock markets fell sharply after the central bank announcement.": (
            "Les marchés boursiers ont fortement chuté après l'annonce de la banque centrale."
        ),
        "The transformer architecture uses self-attention layers.": (
            "L'architecture transformeur utilise des couches d'auto-attention."
        ),
        "Please restart the server after the update completes.": (
            "Veuillez redémarrer le serveur une fois la mise à jour terminée."
        ),
        "I would like a coffee with milk and sugar, please.": (
            "Je voudrais un café avec du lait et du sucre, s'il vous plaît."
        ),
        "Where is the nearest train station?": (
            "Où se trouve la gare la plus proche ?"
        ),
    }


def _responder_for_fr() -> Callable[[str], str]:
    """Build a prompt → reference responder for the en-fr fixture.

    The plugin's prompt wraps the source string in a fixed prefix/suffix.
    We recover the source by splitting on the marker tokens.
    """
    answers = _expected_fr_translations()

    def responder(prompt: str) -> str:
        # Prompt format: "Translate from en to fr:\n\n{source}\n\nTranslation:"
        # Pull the source string out by stripping prefix + suffix.
        try:
            after_prefix = prompt.split("\n\n", 1)[1]
            source = after_prefix.rsplit("\n\nTranslation:", 1)[0]
        except (IndexError, ValueError):
            return ""
        return answers.get(source, "")

    return responder


def test_run_produces_signed_envelope_with_perfect_chrf(
    make_mock_modelclient, tmp_path: Path
) -> None:
    """Mock returns the canonical reference for every prompt → chrF = 1.0."""
    make_mock_modelclient(_responder_for_fr())

    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)

    plugin = LLMMTPlugin()
    spec = plugin.get_benchmark("llm.mt.flores-200-mini-en-fr")
    ctx = RunContext(
        model_id="openai/mock-model",
        model_revision="abc1234",
        engine_kind=EngineKind.OPENAI,
        output_dir=tmp_path / "out",
        extra={
            "signing_mode": "dev",
            "dev_key_path": str(private_key_path),
        },
    )

    envelope = plugin.run(spec, ctx)

    # Signature is real
    assert envelope.signature is not None
    assert envelope.signature.method == "dev-key"
    assert envelope.signature.bundle  # non-empty base64 blob

    # chrF should be 1.0 when the mock returns the reference verbatim.
    chrf_mean = envelope.metrics.get("chrf_mean")
    assert chrf_mean is not None
    assert isinstance(chrf_mean, (int, float))
    assert float(chrf_mean) == pytest.approx(1.0)

    # Supplementary metrics are present.
    assert envelope.metrics.get("chrf_p50") is not None
    assert envelope.metrics.get("chrf_p95") is not None
    assert envelope.metrics.get("ok_rate") == 1.0
    assert envelope.metrics.get("n_samples") == 8.0
    assert envelope.metrics.get("ttft_p50_ms") is not None


def test_run_with_wrong_translations_yields_low_chrf(
    make_mock_modelclient, tmp_path: Path
) -> None:
    """Mock returns gibberish → chrF stays in [0, 1) but envelope still valid."""
    make_mock_modelclient(lambda _prompt: "xxx yyy zzz")

    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)

    plugin = LLMMTPlugin()
    spec = plugin.get_benchmark("llm.mt.flores-200-mini-en-fr")
    ctx = RunContext(
        model_id="openai/mock-model",
        model_revision="abc1234",
        engine_kind=EngineKind.OPENAI,
        output_dir=tmp_path / "out",
        extra={
            "signing_mode": "dev",
            "dev_key_path": str(private_key_path),
        },
    )

    envelope = plugin.run(spec, ctx)
    assert envelope.signature is not None
    chrf_mean = envelope.metrics.get("chrf_mean")
    assert chrf_mean is not None
    assert 0.0 <= float(chrf_mean) < 0.5
    assert envelope.metrics.get("ok_rate") == 1.0  # all calls succeeded, just wrong


def test_run_writes_samples_jsonl_alongside_envelope(
    make_mock_modelclient, tmp_path: Path
) -> None:
    """The diagnostic samples-<ts>.jsonl is written to output_dir."""
    make_mock_modelclient(lambda _prompt: "Bonjour")

    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)

    plugin = LLMMTPlugin()
    spec = plugin.get_benchmark("llm.mt.flores-200-mini-en-fr")
    out_dir = tmp_path / "out"
    ctx = RunContext(
        model_id="openai/mock-model",
        engine_kind=EngineKind.OPENAI,
        output_dir=out_dir,
        extra={
            "signing_mode": "dev",
            "dev_key_path": str(private_key_path),
        },
    )

    plugin.run(spec, ctx)
    samples_files = list(out_dir.glob("samples-*.jsonl"))
    assert len(samples_files) == 1
    lines = samples_files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8  # one per fixture row


def test_run_envelope_accepted_by_bench_summary(
    make_mock_modelclient, tmp_path: Path
) -> None:
    """End-to-end: write the envelope JSON to a tmp dir, run ``bench summary``."""
    make_mock_modelclient(_responder_for_fr())

    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)

    plugin = LLMMTPlugin()
    spec = plugin.get_benchmark("llm.mt.flores-200-mini-en-fr")
    out_dir = tmp_path / "envelopes"
    out_dir.mkdir()
    ctx = RunContext(
        model_id="openai/mock-model",
        engine_kind=EngineKind.OPENAI,
        output_dir=tmp_path / "out",
        extra={
            "signing_mode": "dev",
            "dev_key_path": str(private_key_path),
        },
    )

    envelope = plugin.run(spec, ctx)
    env_path = out_dir / "mt-run.json"
    env_path.write_text(
        json.dumps(envelope.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    runner = CliRunner(env={"COLUMNS": "240"})
    result = runner.invoke(app, ["summary", str(out_dir)])
    assert result.exit_code == 0, result.output
    # The MT suite_id appears in the rendered output.
    assert "llm.mt.flores-200-mini-en-fr" in result.output
