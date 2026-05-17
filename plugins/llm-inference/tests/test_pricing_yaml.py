"""Tests for the YAML-backed pricing registry.

Covers the migration from a hardcoded ``_REGISTRY`` dict to a bundled
``prices.yaml`` plus the new ``load_pricing`` / ``set_pricing`` /
``validate_pricing_file`` helpers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inferencebench_llm import pricing as pricing_mod
from inferencebench_llm.pricing import (
    _BUILTIN_FALLBACK,
    _REGISTRY,
    ModelPricing,
    load_pricing,
    lookup,
    set_pricing,
    validate_pricing_file,
)


def test_bundled_yaml_loads_at_import_with_expected_count() -> None:
    """The bundled ``prices.yaml`` produces the same number of entries as the legacy dict."""
    assert len(_REGISTRY) == len(_BUILTIN_FALLBACK)
    assert len(_REGISTRY) == 15


def test_bundled_yaml_groq_llama_8b_lookup() -> None:
    """The Groq Llama-3.1-8B entry survived the migration."""
    p = lookup("groq", "meta-llama/Llama-3.1-8B-Instruct")
    assert p is not None
    assert p.provider == "groq"
    assert p.model == "meta-llama/Llama-3.1-8B-Instruct"
    assert p.input_per_million_usd == pytest.approx(0.05)
    assert p.output_per_million_usd == pytest.approx(0.08)


def test_load_pricing_with_custom_yaml(tmp_path: Path) -> None:
    """``load_pricing(<custom yaml>)`` returns the entries in that file."""
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        """
schema: inferencebench.pricing.v1
currency: USD
per_million_tokens: true
entries:
  - provider: acme
    model: acme/Bigfoot-9B
    input_per_million_usd: 0.42
    output_per_million_usd: 1.00
    notes: "Internal estimate"
""",
        encoding="utf-8",
    )

    registry = load_pricing(custom)

    assert ("acme", "acme/Bigfoot-9B") in registry
    entry = registry[("acme", "acme/Bigfoot-9B")]
    assert isinstance(entry, ModelPricing)
    assert entry.input_per_million_usd == pytest.approx(0.42)
    assert entry.output_per_million_usd == pytest.approx(1.00)
    assert entry.notes == "Internal estimate"


def test_load_pricing_none_returns_bundled_copy() -> None:
    """``load_pricing(None)`` returns a non-empty registry matching the bundled count."""
    registry = load_pricing(None)
    assert len(registry) == len(_REGISTRY)
    assert ("openai", "gpt-4o") in registry


def test_malformed_entry_is_skipped_not_fatal(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An entry missing ``model`` is skipped with a warning; loading still succeeds."""
    yaml_file = tmp_path / "partial.yaml"
    yaml_file.write_text(
        """
schema: inferencebench.pricing.v1
entries:
  - provider: good
    model: good/Model
    input_per_million_usd: 1.0
    output_per_million_usd: 2.0
  - provider: bad
    input_per_million_usd: 1.0
    output_per_million_usd: 2.0
  - provider: also-bad
    model: ""
    input_per_million_usd: 1.0
    output_per_million_usd: 2.0
""",
        encoding="utf-8",
    )

    with caplog.at_level("WARNING", logger="inferencebench_llm.pricing"):
        registry = load_pricing(yaml_file)

    assert ("good", "good/Model") in registry
    assert len(registry) == 1
    # Two entries were skipped — both should have produced warnings.
    assert sum("Skipping invalid pricing entry" in r.message for r in caplog.records) >= 2


def test_load_pricing_nonexistent_file_raises_clear_error(tmp_path: Path) -> None:
    """A missing file raises ``FileNotFoundError`` with the path in the message."""
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError, match=r"nope\.yaml"):
        load_pricing(missing)


def test_validate_pricing_file_counts_valid_and_skipped(tmp_path: Path) -> None:
    """``validate_pricing_file`` returns 1 valid + 1 skipped on a mixed file."""
    yaml_file = tmp_path / "mixed.yaml"
    yaml_file.write_text(
        """
schema: inferencebench.pricing.v1
entries:
  - provider: ok
    model: ok/Model
    input_per_million_usd: 1.0
    output_per_million_usd: 2.0
  - provider: not-ok
    model: bad/Model
    input_per_million_usd: "this should be a number"
    output_per_million_usd: 2.0
""",
        encoding="utf-8",
    )
    stats = validate_pricing_file(yaml_file)
    assert stats.valid == 1
    assert stats.skipped == 1
    assert stats.errors


def test_set_pricing_replaces_module_registry(tmp_path: Path) -> None:
    """``set_pricing`` swaps the module-level registry; ``lookup`` reflects the change."""
    original = dict(_REGISTRY)
    try:
        custom = {
            ("acme", "acme/Bigfoot-9B"): ModelPricing(
                provider="acme",
                model="acme/Bigfoot-9B",
                input_per_million_usd=0.42,
                output_per_million_usd=1.00,
            ),
        }
        set_pricing(custom)
        assert lookup("acme", "acme/Bigfoot-9B") is not None
        # The bundled entry is no longer visible.
        assert lookup("openai", "gpt-4o") is None
    finally:
        set_pricing(original)
        assert lookup("openai", "gpt-4o") is not None


def test_validate_pricing_file_missing_raises(tmp_path: Path) -> None:
    """``validate_pricing_file`` propagates ``FileNotFoundError`` for missing inputs."""
    with pytest.raises(FileNotFoundError):
        validate_pricing_file(tmp_path / "absent.yaml")


def test_bundled_fallback_is_complete() -> None:
    """``_BUILTIN_FALLBACK`` must remain in sync with the bundled YAML.

    Both should produce the same ``(provider, model)`` key set so a corrupt
    install can still be used as a last resort without surprises.
    """
    yaml_keys = set(pricing_mod._load_bundled().keys())
    fallback_keys = set(_BUILTIN_FALLBACK.keys())
    assert yaml_keys == fallback_keys
