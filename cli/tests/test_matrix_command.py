"""Tests for ``bench matrix`` — multi-endpoint dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from _helpers import make_envelope  # type: ignore[import-not-found]
from typer.testing import CliRunner

from inferencebench.cli import app
from inferencebench.envelope import SigningMode, generate_dev_keypair, sign_envelope

if TYPE_CHECKING:
    import pytest

runner = CliRunner(env={"COLUMNS": "240"})


# --------------------------------------------------------------------------- #
# Test plumbing                                                               #
# --------------------------------------------------------------------------- #
def _dev_key(tmp_path: Path) -> Path:
    priv, _pub = generate_dev_keypair(tmp_path / "cosign.key")
    return priv


def _install_fake_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_targets: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Stub LLMInferencePlugin.run to record per-call (target, point)."""
    calls: list[dict[str, Any]] = []

    from inferencebench_llm.plugin import LLMInferencePlugin

    def fake_run(self: Any, spec: Any, context: Any) -> Any:  # noqa: ARG001
        # Recover target name from ctx.extra (we stash it under `target_name`
        # via the matrix YAML's `extra` block in test configs).
        target_name = str(context.extra.get("target_name", "unknown"))
        point = int(context.extra.get("concurrency", 0))
        calls.append(
            {
                "target": target_name,
                "point": point,
                "model_id": context.model_id,
                "base_url": context.base_url,
                "api_key": context.api_key,
                "extra": dict(context.extra),
            }
        )
        if target_name in fail_targets:
            msg = f"forced failure for {target_name}"
            raise RuntimeError(msg)

        # Distinct run_id per (target, point) so content_hashes differ.
        salt = (abs(hash((target_name, point))) % 10**10)
        env = make_envelope(
            model_id=f"fake-{target_name}",
            run_id=f"01934567-89ab-7000-8000-0{salt:011d}"[:36],
            metrics={
                "throughput_tok_per_s": 1000.0 + point * 10,
                "ttft_p50_ms": 100.0,
                "ttft_p99_ms": 250.0,
                "tpot_p50_ms": 20.0,
                "total_p50_ms": 1500.0,
                "ok_rate": 1.0,
                "compliance_rate": 0.97,
            },
        )
        dev_key_path = context.extra.get("dev_key_path")
        if dev_key_path:
            return sign_envelope(env, mode=SigningMode.DEV, dev_key_path=Path(str(dev_key_path)))
        return env

    def fake_validate(self: Any, spec: Any, context: Any) -> list[str]:  # noqa: ARG001
        return []

    monkeypatch.setattr(LLMInferencePlugin, "run", fake_run)
    monkeypatch.setattr(LLMInferencePlugin, "validate", fake_validate)
    return calls


def _write_config(path: Path, body: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_minimal_yaml_one_target_one_point(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """1 target x 1 sweep point -> 1 envelope written."""
    dev_key = _dev_key(tmp_path)
    calls = _install_fake_run(monkeypatch)
    out_dir = tmp_path / "results"
    cfg = _write_config(
        tmp_path / "matrix.yaml",
        {
            "schema": "inferencebench.matrix.v1",
            "suite_id": "llm.inference",
            "sweep": [1],
            "targets": [
                {
                    "name": "vllm-a",
                    "model": "fake/model",
                    "engine": "vllm",
                    "base_url": "http://localhost:8000/v1",
                    "extra": {"target_name": "vllm-a"},
                },
            ],
        },
    )

    result = runner.invoke(
        app,
        [
            "matrix",
            str(cfg),
            "--output",
            str(out_dir),
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
        ],
    )
    combined = result.stdout + (result.stderr or "")
    assert result.exit_code == 0, combined
    assert len(calls) == 1
    assert calls[0]["target"] == "vllm-a"
    assert calls[0]["point"] == 1
    files = sorted(p.name for p in out_dir.glob("*.json"))
    assert len(files) == 1, files
    assert files[0].startswith("vllm-a-c1-"), files


def test_two_targets_three_sweep_points_six_envelopes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """2 targets x 3 sweep points = 6 envelopes; filenames include target+point."""
    dev_key = _dev_key(tmp_path)
    calls = _install_fake_run(monkeypatch)
    out_dir = tmp_path / "results"
    cfg = _write_config(
        tmp_path / "matrix.yaml",
        {
            "schema": "inferencebench.matrix.v1",
            "suite_id": "llm.inference",
            "sweep": [1, 4, 16],
            "targets": [
                {
                    "name": "vllm-a",
                    "model": "fake/model-a",
                    "engine": "vllm",
                    "base_url": "http://localhost:8000/v1",
                    "extra": {"target_name": "vllm-a"},
                },
                {
                    "name": "vllm-b",
                    "model": "fake/model-b",
                    "engine": "vllm",
                    "base_url": "http://localhost:8001/v1",
                    "extra": {"target_name": "vllm-b"},
                },
            ],
        },
    )

    result = runner.invoke(
        app,
        [
            "matrix",
            str(cfg),
            "--output",
            str(out_dir),
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
        ],
    )
    combined = result.stdout + (result.stderr or "")
    assert result.exit_code == 0, combined
    assert len(calls) == 6
    files = sorted(p.name for p in out_dir.glob("*.json"))
    assert len(files) == 6, files
    prefixes = {f.rsplit("-", 1)[0] for f in files}
    expected = {
        "vllm-a-c1",
        "vllm-a-c4",
        "vllm-a-c16",
        "vllm-b-c1",
        "vllm-b-c4",
        "vllm-b-c16",
    }
    assert prefixes == expected, (prefixes, files)


def test_missing_required_field_suite_id(tmp_path: Path) -> None:
    """Missing `suite_id` → exit 2 with helpful error."""
    dev_key = _dev_key(tmp_path)
    cfg = _write_config(
        tmp_path / "matrix.yaml",
        {
            "schema": "inferencebench.matrix.v1",
            "sweep": [1],
            "targets": [
                {
                    "name": "vllm-a",
                    "model": "fake/model",
                    "engine": "vllm",
                },
            ],
        },
    )

    result = runner.invoke(
        app,
        [
            "matrix",
            str(cfg),
            "--output",
            str(tmp_path / "results"),
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
        ],
    )
    combined = result.stdout + (result.stderr or "")
    assert result.exit_code == 2, combined
    assert "suite_id" in combined
    assert "invalid" in combined.lower()


def test_missing_api_key_env_skips_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An openai-like target with missing api_key_env is skipped; others run."""
    monkeypatch.delenv("NONEXISTENT_VAR_FOR_MATRIX_TEST", raising=False)
    dev_key = _dev_key(tmp_path)
    calls = _install_fake_run(monkeypatch)
    out_dir = tmp_path / "results"
    cfg = _write_config(
        tmp_path / "matrix.yaml",
        {
            "schema": "inferencebench.matrix.v1",
            "suite_id": "llm.inference",
            "sweep": [1],
            "targets": [
                {
                    "name": "openai-skip",
                    "model": "gpt-4o-mini",
                    "engine": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "NONEXISTENT_VAR_FOR_MATRIX_TEST",
                    "extra": {"target_name": "openai-skip"},
                },
                {
                    "name": "vllm-ok",
                    "model": "fake/model",
                    "engine": "vllm",
                    "base_url": "http://localhost:8000/v1",
                    "extra": {"target_name": "vllm-ok"},
                },
            ],
        },
    )

    result = runner.invoke(
        app,
        [
            "matrix",
            str(cfg),
            "--output",
            str(out_dir),
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
        ],
    )
    combined = result.stdout + (result.stderr or "")
    assert result.exit_code == 0, combined
    # warning about the skip
    assert "NONEXISTENT_VAR_FOR_MATRIX_TEST" in combined
    assert "skip" in combined.lower()
    # Only the vllm target actually called the plugin
    assert [c["target"] for c in calls] == ["vllm-ok"]
    files = sorted(p.name for p in out_dir.glob("*.json"))
    assert len(files) == 1, files
    assert files[0].startswith("vllm-ok-c1-"), files


def test_no_continue_on_error_stops_on_first_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--no-continue-on-error`: matrix stops after the failing target, exit 1."""
    dev_key = _dev_key(tmp_path)
    calls = _install_fake_run(monkeypatch, fail_targets=("vllm-b",))
    out_dir = tmp_path / "results"
    cfg = _write_config(
        tmp_path / "matrix.yaml",
        {
            "schema": "inferencebench.matrix.v1",
            "suite_id": "llm.inference",
            "sweep": [1],
            "targets": [
                {
                    "name": "vllm-a",
                    "model": "fake/model-a",
                    "engine": "vllm",
                    "extra": {"target_name": "vllm-a"},
                },
                {
                    "name": "vllm-b",
                    "model": "fake/model-b",
                    "engine": "vllm",
                    "extra": {"target_name": "vllm-b"},
                },
                {
                    "name": "vllm-c",
                    "model": "fake/model-c",
                    "engine": "vllm",
                    "extra": {"target_name": "vllm-c"},
                },
            ],
        },
    )

    result = runner.invoke(
        app,
        [
            "matrix",
            str(cfg),
            "--output",
            str(out_dir),
            "--no-continue-on-error",
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
        ],
    )
    combined = result.stdout + (result.stderr or "")
    assert result.exit_code == 1, combined
    # The third target should never run.
    seen = [c["target"] for c in calls]
    assert "vllm-a" in seen
    assert "vllm-b" in seen
    assert "vllm-c" not in seen
    # vllm-a should have written one envelope; vllm-b failed.
    files = sorted(p.name for p in out_dir.glob("*.json"))
    assert any(f.startswith("vllm-a-c1-") for f in files), files
    assert not any(f.startswith("vllm-b-c1-") for f in files), files


def test_continue_on_error_keeps_going(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--continue-on-error` (default): summary shows ✗ for the failed target, exit 0."""
    dev_key = _dev_key(tmp_path)
    calls = _install_fake_run(monkeypatch, fail_targets=("vllm-b",))
    out_dir = tmp_path / "results"
    cfg = _write_config(
        tmp_path / "matrix.yaml",
        {
            "schema": "inferencebench.matrix.v1",
            "suite_id": "llm.inference",
            "sweep": [1],
            "targets": [
                {
                    "name": "vllm-a",
                    "model": "fake/model-a",
                    "engine": "vllm",
                    "extra": {"target_name": "vllm-a"},
                },
                {
                    "name": "vllm-b",
                    "model": "fake/model-b",
                    "engine": "vllm",
                    "extra": {"target_name": "vllm-b"},
                },
                {
                    "name": "vllm-c",
                    "model": "fake/model-c",
                    "engine": "vllm",
                    "extra": {"target_name": "vllm-c"},
                },
            ],
        },
    )

    result = runner.invoke(
        app,
        [
            "matrix",
            str(cfg),
            "--output",
            str(out_dir),
            "--continue-on-error",
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
        ],
    )
    combined = result.stdout + (result.stderr or "")
    assert result.exit_code == 0, combined
    # All three targets attempted.
    seen = [c["target"] for c in calls]
    assert seen == ["vllm-a", "vllm-b", "vllm-c"]
    files = sorted(p.name for p in out_dir.glob("*.json"))
    # Two envelopes (a and c), not the failing b.
    assert len(files) == 2, files
    assert any(f.startswith("vllm-a-c1-") for f in files), files
    assert any(f.startswith("vllm-c-c1-") for f in files), files
    # Summary table should mark the failed row with ✗.
    assert "✗" in combined or "x" in combined.lower()


def test_schema_key_loaded_from_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Schema string `inferencebench.matrix.v1` is read from YAML and accepted."""
    dev_key = _dev_key(tmp_path)
    _install_fake_run(monkeypatch)
    out_dir = tmp_path / "results"
    cfg_body = {
        "schema": "inferencebench.matrix.v1",
        "suite_id": "llm.inference",
        "sweep": [1],
        "targets": [
            {
                "name": "vllm-a",
                "model": "fake/model",
                "engine": "vllm",
                "base_url": "http://localhost:8000/v1",
                "extra": {"target_name": "vllm-a"},
            }
        ],
    }
    cfg = _write_config(tmp_path / "matrix.yaml", cfg_body)

    # Sanity-check the YAML round-trips so the schema key is what we expect.
    loaded = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert loaded["schema"] == "inferencebench.matrix.v1"

    result = runner.invoke(
        app,
        [
            "matrix",
            str(cfg),
            "--output",
            str(out_dir),
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
        ],
    )
    combined = result.stdout + (result.stderr or "")
    assert result.exit_code == 0, combined

    # And a wrong schema should be rejected.
    bad = dict(cfg_body)
    bad["schema"] = "inferencebench.matrix.v2"
    bad_cfg = _write_config(tmp_path / "bad.yaml", bad)
    bad_result = runner.invoke(
        app,
        [
            "matrix",
            str(bad_cfg),
            "--output",
            str(out_dir),
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
        ],
    )
    assert bad_result.exit_code == 2
    assert "schema" in (bad_result.stdout + (bad_result.stderr or ""))
