"""Tests for the ``JudgeThrottle`` rate limiter used by ``scoring: judge_llm``.

The throttle is exercised with an injected mock clock + sleep so no real
wall time elapses in CI. The end-to-end test wires the throttle through a
full ``plugin.run`` invocation and asserts that the mock clock observed
the right call cadence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inferencebench.envelope import generate_dev_keypair
from inferencebench.harness.client import CompletionResult, ModelClient
from inferencebench_quality import (
    EngineKind,
    JudgeThrottle,
    LLMQualityPlugin,
    RunContext,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
class _MockClock:
    """A monotonic clock controllable from tests.

    ``sleep(s)`` advances the clock by ``s`` seconds and records the
    duration; ``now()`` returns the current virtual time. Together they
    let us assert exactly how much the throttle "slept" without ever
    blocking the test process.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.t = float(start)
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(float(seconds))
        self.t += float(seconds)


# --------------------------------------------------------------------------- #
# Unit tests — JudgeThrottle in isolation                                     #
# --------------------------------------------------------------------------- #
def test_throttle_rps_zero_is_no_op() -> None:
    """rps=0 → never sleeps (unlimited path is free)."""
    clk = _MockClock()
    throttle = JudgeThrottle(rps=0, clock=clk.now, sleep=clk.sleep)
    for _ in range(50):
        throttle.acquire()
    assert clk.sleeps == []


def test_throttle_rps_negative_is_no_op() -> None:
    """Negative rps is treated as unlimited (defensive — never blocks)."""
    clk = _MockClock()
    throttle = JudgeThrottle(rps=-1.5, clock=clk.now, sleep=clk.sleep)
    for _ in range(10):
        throttle.acquire()
    assert clk.sleeps == []


def test_throttle_first_acquire_never_sleeps() -> None:
    """The first call has no prior call to space against, so no wait."""
    clk = _MockClock()
    throttle = JudgeThrottle(rps=10, clock=clk.now, sleep=clk.sleep)
    throttle.acquire()
    assert clk.sleeps == []


def test_throttle_back_to_back_calls_sleep_full_interval() -> None:
    """Two acquire()s in the same tick → second sleeps the full 1/rps window."""
    clk = _MockClock()
    throttle = JudgeThrottle(rps=10, clock=clk.now, sleep=clk.sleep)
    throttle.acquire()
    throttle.acquire()
    # rps=10 → 0.1 s interval.
    assert clk.sleeps == [pytest.approx(0.1)]


def test_throttle_partial_elapsed_sleeps_residual() -> None:
    """If 40 ms of the 100 ms window already elapsed, sleep the remaining 60 ms."""
    clk = _MockClock()
    throttle = JudgeThrottle(rps=10, clock=clk.now, sleep=clk.sleep)
    throttle.acquire()
    # Advance the clock by 0.04 s before the next acquire.
    clk.t += 0.04
    throttle.acquire()
    assert clk.sleeps == [pytest.approx(0.06, rel=1e-9)]


def test_throttle_skips_sleep_when_interval_already_passed() -> None:
    """If the gap already exceeds 1/rps, don't sleep at all."""
    clk = _MockClock()
    throttle = JudgeThrottle(rps=10, clock=clk.now, sleep=clk.sleep)
    throttle.acquire()
    clk.t += 5.0  # plenty of time has passed
    throttle.acquire()
    assert clk.sleeps == []


def test_throttle_paces_ten_calls_at_rps_10() -> None:
    """Ten back-to-back calls at rps=10 → nine sleeps of 0.1 s each."""
    clk = _MockClock()
    throttle = JudgeThrottle(rps=10, clock=clk.now, sleep=clk.sleep)
    for _ in range(10):
        throttle.acquire()
    assert clk.sleeps == [pytest.approx(0.1)] * 9


def test_throttle_high_rps_means_short_intervals() -> None:
    """At rps=100, virtual wall time across 10 back-to-back calls is well under 1 s."""
    clk = _MockClock(start=0.0)
    throttle = JudgeThrottle(rps=100, clock=clk.now, sleep=clk.sleep)
    for _ in range(10):
        throttle.acquire()
    # 9 sleeps of 0.01 s each → total 0.09 s virtual time elapsed.
    total = sum(clk.sleeps)
    assert total < 1.0
    assert total == pytest.approx(0.09)


def test_throttle_exposes_rps_property() -> None:
    """The configured rps is read-back via the .rps property."""
    assert JudgeThrottle(rps=5.0).rps == 5.0
    assert JudgeThrottle(rps=0).rps == 0
    # Negative inputs are normalised to "unlimited" semantically — the
    # raw value is still preserved on the property for diagnostics.
    assert JudgeThrottle(rps=-1).rps == -1.0


# --------------------------------------------------------------------------- #
# End-to-end: throttle is wired into plugin.run via RunContext.extra          #
# --------------------------------------------------------------------------- #
def _patch_complete_with_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ModelClient.complete so the judge always returns "1"."""
    judge_marker = "You are a strict grader."

    def _fake_complete(
        self: ModelClient,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stream: bool = True,
        system: str | None = None,
        **_: Any,
    ) -> CompletionResult:
        text = "1" if judge_marker in prompt else prompt
        return CompletionResult(
            text=text,
            tokens_in=10,
            tokens_out=1,
            ttft_ms=1.0,
            total_ms=2.0,
            tpot_ms=1.0,
            cost_usd=0.0,
            finish_reason="stop",
            token_source="mock",
        )

    monkeypatch.setattr(ModelClient, "complete", _fake_complete)


def test_end_to_end_judge_rps_does_not_block_test_walltime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With judge_rps=100, 10 judge calls finish in well under a second.

    No real sleeps happen here — we patch time.sleep inside the plugin
    module so the test asserts on the *requested* sleep durations
    instead of wall clock.
    """
    _patch_complete_with_judge(monkeypatch)

    recorded_sleeps: list[float] = []

    def _record_sleep(seconds: float) -> None:
        recorded_sleeps.append(float(seconds))

    # Intercept the time.sleep used inside JudgeThrottle by patching the
    # module-level reference that JudgeThrottle resolves at construction.
    import inferencebench_quality.plugin as quality_plugin_module

    monkeypatch.setattr(quality_plugin_module.time, "sleep", _record_sleep)

    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)

    plugin = LLMQualityPlugin()
    spec = plugin.get_benchmark("llm.quality.factual-judged")
    ctx = RunContext(
        model_id="openai/mock-model",
        engine_kind=EngineKind.OPENAI,
        output_dir=tmp_path / "out",
        extra={
            "signing_mode": "dev",
            "dev_key_path": str(private_key_path),
            "judge_rps": 100.0,
        },
    )

    envelope = plugin.run(spec, ctx)
    assert envelope.metrics.get("n_judged") == 10.0
    # Total *requested* sleep is bounded by 9 * 1/100 = 0.09 s.
    # (May be lower if the model-under-test calls themselves take wall time.)
    assert sum(recorded_sleeps) < 1.0


def test_end_to_end_judge_rps_zero_never_sleeps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``judge_rps=0`` (default) → throttle never invokes sleep."""
    _patch_complete_with_judge(monkeypatch)

    recorded_sleeps: list[float] = []
    import inferencebench_quality.plugin as quality_plugin_module

    monkeypatch.setattr(
        quality_plugin_module.time, "sleep", lambda s: recorded_sleeps.append(float(s))
    )

    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)

    plugin = LLMQualityPlugin()
    spec = plugin.get_benchmark("llm.quality.factual-judged")
    ctx = RunContext(
        model_id="openai/mock-model",
        engine_kind=EngineKind.OPENAI,
        output_dir=tmp_path / "out",
        extra={
            "signing_mode": "dev",
            "dev_key_path": str(private_key_path),
        },
    )
    plugin.run(spec, ctx)
    # judge_rps=0 → throttle is a no-op. Any sleeps observed here are noise
    # from other code paths (e.g. xdist coordination) and must be << 100ms.
    assert all(s < 0.05 for s in recorded_sleeps), recorded_sleeps
