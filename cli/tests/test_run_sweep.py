"""Tests for ``bench run --sweep`` and ``bench run --rps-sweep``.

These exercise the new closed-loop concurrency-sweep / open-loop RPS-sweep
dispatch path. Real benchmark execution is mocked by monkeypatching the
plugin's ``run`` method so the tests never need a live vLLM.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from _helpers import make_envelope  # type: ignore[import-not-found]
from typer.testing import CliRunner

from inferencebench.cli import app
from inferencebench.envelope import SigningMode, generate_dev_keypair, sign_envelope

if TYPE_CHECKING:
    import pytest

runner = CliRunner(env={"COLUMNS": "240"})


def _make_signed_envelope(point_label: str, ok_rate: float, throughput: float) -> Any:
    """Build a dev-signed envelope with deterministic-ish content per point."""
    env = make_envelope(
        model_id=f"fake-model-{point_label}",
        # Vary run_id so distinct sweep points get distinct content_hashes.
        run_id=f"01934567-89ab-7000-8000-0000000{int(float(point_label)):05d}",
        metrics={
            "throughput_tok_per_s": throughput,
            "ttft_p50_ms": 100.0,
            "ttft_p99_ms": 250.0,
            "tpot_p50_ms": 20.0,
            "total_p50_ms": 1500.0,
            "ok_rate": ok_rate,
            "compliance_rate": 0.97,
        },
    )
    return env


def _dev_key(tmp_path: Path) -> Path:
    priv, _pub = generate_dev_keypair(tmp_path / "cosign.key")
    return priv


def _install_fake_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ok_rate: float = 1.0,
    fail_points: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Stub LLMInferencePlugin.run with a fake that returns a signed envelope.

    Returns a list that the test can inspect to see what points were invoked.
    """
    calls: list[dict[str, Any]] = []

    from inferencebench_llm.plugin import LLMInferencePlugin

    def fake_run(self: Any, spec: Any, context: Any) -> Any:  # noqa: ARG001
        # Translate context.extra → point label
        if "concurrency" in context.extra:
            point = str(context.extra["concurrency"])
        elif "rps" in context.extra:
            point = f"{float(context.extra['rps']):g}"
        else:
            point = "0"
        calls.append({"point": point, "extra": dict(context.extra)})

        if point in fail_points:
            msg = f"forced failure at point {point}"
            raise RuntimeError(msg)

        env = _make_signed_envelope(point, ok_rate, 1000.0 + float(point) * 10)
        dev_key_path = context.extra.get("dev_key_path")
        if dev_key_path:
            return sign_envelope(env, mode=SigningMode.DEV, dev_key_path=Path(str(dev_key_path)))
        return env

    def fake_validate(self: Any, spec: Any, context: Any) -> list[str]:  # noqa: ARG001
        return []

    monkeypatch.setattr(LLMInferencePlugin, "run", fake_run)
    monkeypatch.setattr(LLMInferencePlugin, "validate", fake_validate)
    return calls


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_sweep_dispatches_per_point(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--sweep 1,2` runs the benchmark twice and writes two envelopes."""
    dev_key = _dev_key(tmp_path)
    calls = _install_fake_run(monkeypatch)
    out_dir = tmp_path / "results"

    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "fake/model",
            "--base-url",
            "http://localhost:8000/v1",
            "--sweep",
            "1,2",
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
            "--output",
            str(out_dir),
        ],
    )
    combined = result.stdout + (result.stderr or "")
    assert result.exit_code == 0, combined
    assert [c["point"] for c in calls] == ["1", "2"]
    # One envelope file per point, prefixed by the point label.
    files = sorted(p.name for p in out_dir.glob("*.json"))
    assert len(files) == 2, files
    assert any(f.startswith("c1-") for f in files), files
    assert any(f.startswith("c2-") for f in files), files


def test_sweep_single_point_still_uses_sweep_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--sweep 4` (one point) must still produce a c4-prefixed envelope."""
    dev_key = _dev_key(tmp_path)
    _install_fake_run(monkeypatch)
    out_dir = tmp_path / "results"

    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "fake/model",
            "--base-url",
            "http://localhost:8000/v1",
            "--sweep",
            "4",
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
            "--output",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    files = sorted(p.name for p in out_dir.glob("*.json"))
    assert len(files) == 1
    assert files[0].startswith("c4-"), files


def test_rps_sweep_dispatches_per_point(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--rps-sweep 1,2` runs twice with open_loop driver."""
    dev_key = _dev_key(tmp_path)
    calls = _install_fake_run(monkeypatch)
    out_dir = tmp_path / "results"

    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "fake/model",
            "--base-url",
            "http://localhost:8000/v1",
            "--rps-sweep",
            "1,2",
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
            "--output",
            str(out_dir),
        ],
    )
    combined = result.stdout + (result.stderr or "")
    assert result.exit_code == 0, combined
    assert [c["point"] for c in calls] == ["1", "2"]
    for c in calls:
        assert c["extra"]["driver_type"] == "open_loop"
    files = sorted(p.name for p in out_dir.glob("*.json"))
    assert len(files) == 2
    assert any(f.startswith("rps1-") for f in files), files
    assert any(f.startswith("rps2-") for f in files), files


def test_sweep_and_concurrency_are_mutually_exclusive(tmp_path: Path) -> None:
    """`--sweep 1 --concurrency 4` exits non-zero with a clear error."""
    # No need to install fake run — should fail before plugin is invoked.
    dev_key = _dev_key(tmp_path)
    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "fake/model",
            "--sweep",
            "1",
            "--concurrency",
            "4",
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
        ],
    )
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "mutually exclusive" in combined.lower()
    assert "--sweep" in combined
    assert "--concurrency" in combined


def test_sweep_and_rps_sweep_are_mutually_exclusive(tmp_path: Path) -> None:
    """`--sweep` and `--rps-sweep` cannot be combined."""
    dev_key = _dev_key(tmp_path)
    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "fake/model",
            "--sweep",
            "1",
            "--rps-sweep",
            "2",
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
        ],
    )
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "mutually exclusive" in combined.lower()


def test_rps_sweep_and_rps_are_mutually_exclusive(tmp_path: Path) -> None:
    dev_key = _dev_key(tmp_path)
    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "fake/model",
            "--rps-sweep",
            "1",
            "--rps",
            "2.0",
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
        ],
    )
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "mutually exclusive" in combined.lower()


def test_invalid_sweep_value_errors_gracefully(tmp_path: Path) -> None:
    """`--sweep abc` exits with a clear, non-traceback error."""
    dev_key = _dev_key(tmp_path)
    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "fake/model",
            "--sweep",
            "abc",
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
        ],
    )
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "invalid" in combined.lower()
    assert "--sweep" in combined


def test_invalid_rps_sweep_value_errors_gracefully(tmp_path: Path) -> None:
    dev_key = _dev_key(tmp_path)
    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "fake/model",
            "--rps-sweep",
            "1,oops",
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
        ],
    )
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "invalid" in combined.lower()


def test_sweep_writes_n_envelopes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """N points → N envelope files in the output directory."""
    dev_key = _dev_key(tmp_path)
    _install_fake_run(monkeypatch)
    out_dir = tmp_path / "results"

    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "fake/model",
            "--base-url",
            "http://localhost:8000/v1",
            "--sweep",
            "1,4,16,64",
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
            "--output",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    files = sorted(p.name for p in out_dir.glob("*.json"))
    assert len(files) == 4, files
    prefixes = {f.split("-", 1)[0] for f in files}
    assert prefixes == {"c1", "c4", "c16", "c64"}


def test_sweep_help_lists_new_flags() -> None:
    """``bench run --help`` exposes the new --sweep / --rps-sweep flags."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    for flag in ("--sweep", "--rps-sweep"):
        assert flag in result.stdout, f"missing flag: {flag}"


def test_sweep_failure_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If any sweep point throws, the command exits 1 after running the rest."""
    dev_key = _dev_key(tmp_path)
    calls = _install_fake_run(monkeypatch, fail_points=("4",))
    out_dir = tmp_path / "results"

    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "fake/model",
            "--base-url",
            "http://localhost:8000/v1",
            "--sweep",
            "1,4,16",
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
            "--output",
            str(out_dir),
        ],
    )
    assert result.exit_code != 0
    # All three points should have been attempted.
    assert [c["point"] for c in calls] == ["1", "4", "16"]
    # Only the non-failing points produce envelopes on disk.
    files = sorted(p.name for p in out_dir.glob("*.json"))
    assert len(files) == 2, files
    assert any(f.startswith("c1-") for f in files), files
    assert any(f.startswith("c16-") for f in files), files


def test_sweep_low_ok_rate_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If any point's ok_rate < 0.95 the command exits 1."""
    dev_key = _dev_key(tmp_path)
    _install_fake_run(monkeypatch, ok_rate=0.5)
    out_dir = tmp_path / "results"

    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "fake/model",
            "--base-url",
            "http://localhost:8000/v1",
            "--sweep",
            "1,2",
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
            "--output",
            str(out_dir),
        ],
    )
    assert result.exit_code != 0
    files = sorted(p.name for p in out_dir.glob("*.json"))
    # Envelopes were still written — the failure is communicated via exit code,
    # not by suppressing the corpus.
    assert len(files) == 2


# Sanity: ensure the env stays clean between tests (no stray cwd cosign.key)
def teardown_module() -> None:
    if os.path.exists("cosign.key"):
        os.unlink("cosign.key")
