"""Tests for ``bench run`` (ticket 0025).

These exercise the CLI wiring — plugin lookup, --list, error messages on
missing plugin. End-to-end execution against a live engine is out of scope
here (covered by integration tests in ``plugins/llm-inference/tests/``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from typer.testing import CliRunner

from inferencebench.cli import app
from inferencebench.commands import run as run_module

if TYPE_CHECKING:
    import pytest

runner = CliRunner()


def test_run_with_no_plugins_prints_helpful_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With zero plugins discovered, ``bench run`` exits non-zero with a hint."""
    monkeypatch.setattr(run_module, "_entry_points", lambda: [])

    result = runner.invoke(
        app,
        ["run", "llm.inference", "--model", "foo"],
    )
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "No plugin" in combined or "no plugin" in combined.lower()
    assert "llm.inference" in combined


def test_run_with_unknown_suite_lists_installed() -> None:
    """If at least one plugin is installed but the requested suite isn't, list what IS."""
    real_eps = run_module._entry_points()
    # The llm-inference plugin is part of this workspace so should be discoverable.
    assert any(ep.name == "llm.inference" for ep in real_eps), (
        "expected the llm.inference plugin to be installed in the workspace"
    )

    result = runner.invoke(
        app,
        ["run", "voice.realtime", "--model", "foo"],
    )
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "voice.realtime" in combined
    # Should mention the installed plugin in the "Installed:" list.
    assert "llm.inference" in combined


def test_run_list_prints_benchmark_ids() -> None:
    """``bench run llm.inference --list`` prints available benchmark_ids and exits 0."""
    result = runner.invoke(app, ["run", "llm.inference", "--list"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert "llm.inference.sharegpt-v3" in result.stdout


def test_run_help_lists_new_flags() -> None:
    """``bench run --help`` exposes the new flags added in ticket 0025."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    for flag in ("--signing-mode", "--dev-key", "--strict", "--list", "--base-url"):
        assert flag in result.stdout, f"missing flag: {flag}"


def test_run_help_lists_prices_file_flag() -> None:
    """``bench run --help`` exposes the ``--prices-file`` flag."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--prices-file" in result.stdout


def test_run_prices_file_missing_errors(tmp_path: Path) -> None:
    """A nonexistent ``--prices-file`` aborts before any plugin work."""
    missing = tmp_path / "absent.yaml"
    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "foo",
            "--prices-file",
            str(missing),
            "--signing-mode",
            "keyless",
        ],
    )
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "absent.yaml" in combined or "not found" in combined


def test_run_prices_file_forwards_into_run_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--prices-file <path>`` sets ``RunContext.extra['prices_file']`` to the resolved path."""
    from inferencebench_llm.plugin import LLMInferencePlugin

    prices = tmp_path / "custom.yaml"
    prices.write_text(
        """
schema: inferencebench.pricing.v1
entries:
  - provider: acme
    model: acme/Bigfoot-9B
    input_per_million_usd: 0.42
    output_per_million_usd: 1.00
""",
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def fake_run(
        self: LLMInferencePlugin,  # noqa: ARG001
        spec: Any,  # noqa: ARG001
        context: Any,
    ) -> Any:
        captured["extra"] = dict(context.extra)
        # Raise to short-circuit the rest of the CLI's envelope-writing path.
        msg = "stop after capturing context"
        raise RuntimeError(msg)

    def fake_validate(
        self: LLMInferencePlugin,  # noqa: ARG001
        spec: Any,  # noqa: ARG001
        context: Any,  # noqa: ARG001
    ) -> list[str]:
        return []

    monkeypatch.setattr(LLMInferencePlugin, "run", fake_run)
    monkeypatch.setattr(LLMInferencePlugin, "validate", fake_validate)

    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "acme/Bigfoot-9B",
            "--prices-file",
            str(prices),
            "--signing-mode",
            "keyless",
            "--output",
            str(tmp_path / "results"),
        ],
    )

    # The fake_run raises so we expect a non-zero exit, but the capture must
    # have happened before the raise.
    assert result.exit_code != 0
    assert "extra" in captured, (
        f"plugin.run was never reached. output: {result.stdout + (result.stderr or '')}"
    )
    assert captured["extra"].get("prices_file") == str(prices.resolve())


def test_run_help_lists_judge_flags() -> None:
    """``bench run --help`` exposes the LLM-as-judge flags."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--judge-model" in result.stdout
    assert "--judge-max-questions" in result.stdout


def _capture_extra_via_fake_run(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    """Patch the llm-inference plugin so we can inspect ``ctx.extra``.

    The plugin discovery path goes through the real entry-point registry;
    we hijack ``run`` + ``validate`` so the test never actually executes a
    benchmark. The ``run`` callback records ``ctx.extra`` and then raises
    to short-circuit the rest of the CLI.
    """
    from inferencebench_llm.plugin import LLMInferencePlugin

    def fake_run(
        self: LLMInferencePlugin,  # noqa: ARG001
        spec: Any,  # noqa: ARG001
        context: Any,
    ) -> Any:
        captured["extra"] = dict(context.extra)
        msg = "stop after capturing context"
        raise RuntimeError(msg)

    def fake_validate(
        self: LLMInferencePlugin,  # noqa: ARG001
        spec: Any,  # noqa: ARG001
        context: Any,  # noqa: ARG001
    ) -> list[str]:
        return []

    monkeypatch.setattr(LLMInferencePlugin, "run", fake_run)
    monkeypatch.setattr(LLMInferencePlugin, "validate", fake_validate)


def test_run_judge_model_flag_forwards_into_run_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--judge-model openai/gpt-4o-mini`` → ``RunContext.extra['judge_model']``."""
    captured: dict[str, Any] = {}
    _capture_extra_via_fake_run(monkeypatch, captured)

    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "openai/some-model",
            "--judge-model",
            "openai/gpt-4o-mini",
            "--signing-mode",
            "keyless",
            "--output",
            str(tmp_path / "results"),
        ],
    )

    assert result.exit_code != 0  # fake_run raises after capturing
    assert "extra" in captured, (
        f"plugin.run was never reached. output: {result.stdout + (result.stderr or '')}"
    )
    assert captured["extra"].get("judge_model") == "openai/gpt-4o-mini"


def test_run_judge_max_questions_flag_forwards_into_run_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--judge-max-questions 5`` → ``RunContext.extra['judge_max_questions']``."""
    captured: dict[str, Any] = {}
    _capture_extra_via_fake_run(monkeypatch, captured)

    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "openai/some-model",
            "--judge-max-questions",
            "5",
            "--signing-mode",
            "keyless",
            "--output",
            str(tmp_path / "results"),
        ],
    )

    assert result.exit_code != 0
    assert "extra" in captured
    assert captured["extra"].get("judge_max_questions") == 5


def test_run_help_lists_judge_rps_flag() -> None:
    """``bench run --help`` exposes the ``--judge-rps`` throttle flag."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--judge-rps" in result.stdout


def test_run_judge_rps_flag_forwards_into_run_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--judge-rps 5.0`` → ``RunContext.extra['judge_rps'] == 5.0``."""
    captured: dict[str, Any] = {}
    _capture_extra_via_fake_run(monkeypatch, captured)

    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "openai/some-model",
            "--judge-rps",
            "5.0",
            "--signing-mode",
            "keyless",
            "--output",
            str(tmp_path / "results"),
        ],
    )

    assert result.exit_code != 0
    assert "extra" in captured
    assert captured["extra"].get("judge_rps") == 5.0


def test_run_judge_rps_zero_omits_extra_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--judge-rps 0`` (the default) doesn't set the extra key at all."""
    captured: dict[str, Any] = {}
    _capture_extra_via_fake_run(monkeypatch, captured)

    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "openai/some-model",
            "--signing-mode",
            "keyless",
            "--output",
            str(tmp_path / "results"),
        ],
    )

    assert result.exit_code != 0
    assert "extra" in captured
    assert "judge_rps" not in captured["extra"]
