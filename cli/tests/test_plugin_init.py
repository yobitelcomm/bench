"""Tests for ``bench plugin init`` (ticket 0028)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from inferencebench.cli import app

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()


def test_plugin_init_scaffolds_expected_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`bench plugin init foo --modality voice` creates the full skeleton."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["plugin", "init", "foo", "--modality", "voice"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    base = tmp_path / "plugins" / "foo"
    expected = [
        base / "pyproject.toml",
        base / "README.md",
        base / "src" / "inferencebench_foo" / "__init__.py",
        base / "src" / "inferencebench_foo" / "plugin.py",
        base / "tests" / "test_plugin.py",
    ]
    for p in expected:
        assert p.exists(), f"missing scaffolded file: {p}"

    pyproj = (base / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "inferencebench-foo"' in pyproj
    assert '"foo" = "inferencebench_foo.plugin:FooPlugin"' in pyproj

    plugin_py = (base / "src" / "inferencebench_foo" / "plugin.py").read_text(encoding="utf-8")
    assert "class FooPlugin" in plugin_py
    assert 'suite_id = "foo"' in plugin_py
    assert "voice" in plugin_py  # modality embedded in docstring

    test_py = (base / "tests" / "test_plugin.py").read_text(encoding="utf-8")
    assert "FooPlugin().suite_id == \"foo\"" in test_py


def test_plugin_init_rejects_invalid_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Underscores and uppercase are rejected."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["plugin", "init", "Bad_Name"])
    assert result.exit_code != 0
    assert "Invalid plugin name" in result.stdout + (result.stderr or "")
    assert not (tmp_path / "plugins" / "Bad_Name").exists()


def test_plugin_init_refuses_existing_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuses to overwrite an existing plugin directory."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "plugins" / "bar").mkdir(parents=True)
    result = runner.invoke(app, ["plugin", "init", "bar"])
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "already exists" in combined
