"""End-to-end tests for the multi-turn persona-consistency benchmark.

These exercise :class:`LLMQualityPlugin` against a mocked
:class:`~inferencebench.harness.client.ModelClient`. No real provider is
ever contacted.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from inferencebench.envelope import generate_dev_keypair
from inferencebench.harness.client import CompletionResult, ModelClient
from inferencebench_quality import (
    BenchmarkSpec,
    EngineKind,
    LLMQualityPlugin,
    RunContext,
)


# --------------------------------------------------------------------------- #
# Spec resolution                                                             #
# --------------------------------------------------------------------------- #
def test_persona_consistency_mini_spec_loads_with_multi_turn_flag() -> None:
    plugin = LLMQualityPlugin()
    spec = plugin.get_benchmark("llm.quality.persona-consistency-mini")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.multi_turn is True
    assert spec.scoring == "persona_consistency"
    assert spec.dataset.path == "persona-consistency-mini.jsonl"


def test_persona_consistency_mini_listed_alongside_existing_benchmarks() -> None:
    plugin = LLMQualityPlugin()
    ids = {s.benchmark_id for s in plugin.list_benchmarks()}
    # Existing single-shot benchmarks must keep working.
    assert "llm.quality.factual-mini" in ids
    assert "llm.quality.persona-consistency-mini" in ids


# --------------------------------------------------------------------------- #
# End-to-end multi-turn run                                                   #
# --------------------------------------------------------------------------- #
def _persona_consistent_then_breaks(
    turn_count: int = 5,
) -> Callable[[str, str | None], str]:
    """Build a responder that keeps the case's persona for ``turn_count`` turns then breaks.

    The mock inspects the ``system`` kwarg to pick a stock reply that fits
    THAT case's markers, then switches to a plain reply once the turn index
    exceeds the budget. Turn index = number of ``Assistant:`` lines already
    present in the prompt body (0 on the first turn).
    """
    # Map: substring fragment unique to each case's system_prompt -> a
    # marker-rich reply for that case.
    persona_replies: list[tuple[str, str]] = [
        ("pirate", "Arr matey! On the sea we sail, ship captain ahoy."),
        ("formal academic", "Indeed, furthermore one might observe the answer."),
        ("terse minimalist", "yes."),
        ("Shakespeare", "Verily, thou hast spoken; 'tis a fine question."),
        ("Gen Z", "lowkey bussin, no cap, slay the vibe ngl."),
    ]

    def responder(prompt: str, system: str | None) -> str:
        # Count completed prior turns: each prior turn appears as
        # "Assistant: <text>\n" in the prompt; the trailing "Assistant:" with
        # no newline after is the current-turn placeholder, so we subtract it.
        raw = prompt.count("Assistant:")
        turn_idx = max(raw - 1, 0) if prompt.endswith("Assistant:") else raw
        if turn_idx >= turn_count:
            return f"Plain non-persona reply on turn {turn_idx}."
        sys_text = system or ""
        for fragment, reply in persona_replies:
            if fragment in sys_text:
                return reply
        return "Generic in-persona reply."

    return responder


def _install_mock_complete(
    monkeypatch: Any,
    responder: Callable[[str, str | None], str],
) -> None:
    def _fake_complete(
        self: ModelClient,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stream: bool = True,
        system: str | None = None,
        **extra: Any,
    ) -> CompletionResult:
        return CompletionResult(
            text=responder(prompt, system),
            tokens_in=16,
            tokens_out=16,
            ttft_ms=5.0,
            total_ms=25.0,
            tpot_ms=1.33,
            cost_usd=0.0,
            finish_reason="stop",
            token_source="mock",
        )

    monkeypatch.setattr(ModelClient, "complete", _fake_complete)


def test_run_persona_mini_persona_kept_for_two_turns_then_drifts(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Mock keeps persona for first 2 turns of EVERY case, then drifts.

    Each case has 5 turns. With persona present on turns 0 & 1 and absent
    from turns 2..4, per-case score = 2/5 = 0.4. Mean across 5 identical
    cases is 0.4. Every case drifts at turn 2 → drift_rate = 1.0,
    mean_drift_turn = 2.0.
    """
    _install_mock_complete(monkeypatch, _persona_consistent_then_breaks(turn_count=2))

    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)

    plugin = LLMQualityPlugin()
    spec = plugin.get_benchmark("llm.quality.persona-consistency-mini")
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
    assert envelope.signature.method == "dev-key"

    mean = envelope.metrics.get("persona_consistency_mean")
    assert mean is not None
    assert float(mean) == 0.4
    # Aliased for downstream tooling that ranks on `accuracy`.
    assert float(envelope.metrics["accuracy"]) == 0.4

    assert float(envelope.metrics["drift_rate"]) == 1.0
    assert float(envelope.metrics["mean_drift_turn"]) == 2.0
    assert envelope.metrics["n_samples"] == 5.0
    assert envelope.metrics["ok_rate"] == 1.0


def test_run_persona_mini_full_persona_no_drift(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Mock keeps persona for ALL turns of every case → score 1.0, no drift."""
    _install_mock_complete(monkeypatch, _persona_consistent_then_breaks(turn_count=10))

    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)

    plugin = LLMQualityPlugin()
    spec = plugin.get_benchmark("llm.quality.persona-consistency-mini")
    ctx = RunContext(
        model_id="openai/mock",
        engine_kind=EngineKind.OPENAI,
        output_dir=tmp_path / "out",
        extra={
            "signing_mode": "dev",
            "dev_key_path": str(private_key_path),
        },
    )

    envelope = plugin.run(spec, ctx)
    mean = float(envelope.metrics["persona_consistency_mean"])
    assert mean == 1.0
    assert float(envelope.metrics["drift_rate"]) == 0.0
    assert "mean_drift_turn" not in envelope.metrics


def test_run_judge_llm_persona_returns_seven_tenths(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """End-to-end judge_llm_persona path: judge always replies '7' → score 0.7."""
    # The benchmark spec we load uses persona_consistency; copy it and flip to
    # the judge variant so we exercise the judge path without adding another
    # YAML to the plugin.
    plugin = LLMQualityPlugin()
    base_spec = plugin.get_benchmark("llm.quality.persona-consistency-mini")
    judge_spec = base_spec.model_copy(
        update={"scoring": "judge_llm_persona", "judge_model": "openai/mock-judge"}
    )

    call_idx = {"n": 0}

    def _fake_complete(
        self: ModelClient,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stream: bool = True,
        system: str | None = None,
        **extra: Any,
    ) -> CompletionResult:
        # Two clients are constructed: the model under test and the judge.
        # The judge prompt starts with "You are evaluating whether...".
        is_judge_prompt = "You are evaluating" in prompt
        call_idx["n"] += 1
        text = "7" if is_judge_prompt else "Arr matey reply"
        return CompletionResult(
            text=text,
            tokens_in=16,
            tokens_out=4,
            ttft_ms=5.0,
            total_ms=25.0,
            tpot_ms=6.0,
            cost_usd=0.0,
            finish_reason="stop",
            token_source="mock",
        )

    monkeypatch.setattr(ModelClient, "complete", _fake_complete)

    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)
    ctx = RunContext(
        model_id="openai/mock-model",
        engine_kind=EngineKind.OPENAI,
        output_dir=tmp_path / "out",
        extra={
            "signing_mode": "dev",
            "dev_key_path": str(private_key_path),
        },
    )

    envelope = plugin.run(judge_spec, ctx)
    mean = float(envelope.metrics["persona_consistency_mean"])
    assert mean == 0.7
    assert envelope.metrics["n_judged"] == 5.0
    assert envelope.metrics["judge_errors"] == 0.0


def test_run_persona_writes_samples_jsonl(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Confirms the diagnostic samples-<ts>.jsonl is written for multi-turn runs."""
    _install_mock_complete(monkeypatch, lambda _p: "Arr matey reply on the sea.")
    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)

    plugin = LLMQualityPlugin()
    spec = plugin.get_benchmark("llm.quality.persona-consistency-mini")
    out_dir = tmp_path / "out"
    ctx = RunContext(
        model_id="openai/mock",
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
    assert len(lines) == 5  # one row per case
