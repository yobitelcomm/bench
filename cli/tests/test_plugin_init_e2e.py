"""End-to-end tests for ``bench plugin init``.

These tests don't `pip install` the scaffolded plugin (that would mutate the
workspace venv mid-test) — instead they prove the scaffold is syntactically
valid Python and contains the runtime surface required by the plugin contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from inferencebench.cli import app

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()


def _scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = "demo") -> Path:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["plugin", "init", name, "--modality", "llm", "--kind", "perf"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    return tmp_path / "plugins" / name


def test_scaffolded_plugin_py_parses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every generated .py file must be syntactically valid."""
    base = _scaffold(tmp_path, monkeypatch)
    for py in [
        base / "src" / "inferencebench_demo" / "__init__.py",
        base / "src" / "inferencebench_demo" / "schemas.py",
        base / "src" / "inferencebench_demo" / "plugin.py",
        base / "tests" / "test_plugin.py",
    ]:
        src = py.read_text(encoding="utf-8")
        # compile() raises SyntaxError if the file is invalid.
        compile(src, str(py), "exec")


def test_scaffold_creates_all_expected_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scaffold ships every file required to install and run."""
    base = _scaffold(tmp_path, monkeypatch)
    for relpath in (
        "pyproject.toml",
        "README.md",
        "src/inferencebench_demo/__init__.py",
        "src/inferencebench_demo/schemas.py",
        "src/inferencebench_demo/plugin.py",
        "tests/test_plugin.py",
    ):
        assert (base / relpath).exists(), f"missing scaffolded file: {relpath}"


def test_generated_plugin_mentions_contract_methods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plugin class implements the four contract methods."""
    base = _scaffold(tmp_path, monkeypatch)
    plugin_py = (base / "src" / "inferencebench_demo" / "plugin.py").read_text(encoding="utf-8")
    for symbol in ("suite_id", "list_benchmarks", "get_benchmark", "validate", "run"):
        assert symbol in plugin_py, f"missing {symbol} in scaffolded plugin.py"


def test_hyphenated_name_yields_snake_case_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hyphenated plugin name becomes a snake_case package and CapCase class."""
    base = _scaffold(tmp_path, monkeypatch, name="voice-quality")
    pkg_dir = base / "src" / "inferencebench_voice_quality"
    assert pkg_dir.is_dir()
    plugin_py = (pkg_dir / "plugin.py").read_text(encoding="utf-8")
    assert "class VoiceQualityPlugin" in plugin_py
    assert 'suite_id = "voice-quality"' in plugin_py


def test_scaffolded_plugin_includes_signing_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generated run() signs the envelope — no further wiring required."""
    base = _scaffold(tmp_path, monkeypatch)
    plugin_py = (base / "src" / "inferencebench_demo" / "plugin.py").read_text(encoding="utf-8")
    assert "sign_envelope" in plugin_py
    assert "SigningMode.DEV" in plugin_py
    assert "dev_key_path" in plugin_py
