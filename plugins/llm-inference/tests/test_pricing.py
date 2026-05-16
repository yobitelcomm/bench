"""Tests for the pricing registry."""

from __future__ import annotations

import pytest

from inferencebench_llm.pricing import (
    ModelPricing,
    all_providers,
    estimate_cost,
    lookup,
    models_for,
)


def test_lookup_known_model() -> None:
    p = lookup("openai", "gpt-4o-mini")
    assert p is not None
    assert p.provider == "openai"
    assert p.input_per_million_usd > 0
    assert p.output_per_million_usd > 0


def test_lookup_case_insensitive_provider() -> None:
    p = lookup("OpenAI", "gpt-4o-mini")
    assert p is not None


def test_lookup_unknown_returns_none() -> None:
    assert lookup("unknown-provider", "nonexistent-model") is None


def test_lookup_handles_provider_prefixed_model() -> None:
    """If the user passes ``openai/gpt-4o`` as the model, split on /."""
    p = lookup("anywhere", "openai/gpt-4o")
    assert p is not None
    assert p.model == "gpt-4o"


def test_cost_for_basic_math() -> None:
    p = ModelPricing(
        provider="x",
        model="y",
        input_per_million_usd=1.0,
        output_per_million_usd=2.0,
    )
    # 500K in @ $1/M + 1M out @ $2/M = $0.50 + $2.00 = $2.50
    assert p.cost_for(500_000, 1_000_000) == pytest.approx(2.50)


def test_cost_zero_tokens_is_zero() -> None:
    p = lookup("openai", "gpt-4o-mini")
    assert p is not None
    assert p.cost_for(0, 0) == 0.0


def test_estimate_cost_unknown_model_returns_none() -> None:
    assert estimate_cost("unknown", "model", 100, 100) is None


def test_estimate_cost_round_trip() -> None:
    """Estimating + querying through the convenience function gives the same answer."""
    direct = lookup("openai", "gpt-4o-mini").cost_for(1000, 500)  # type: ignore[union-attr]
    via_helper = estimate_cost("openai", "gpt-4o-mini", 1000, 500)
    assert direct == via_helper


def test_all_providers_returns_sorted_set() -> None:
    providers = all_providers()
    assert providers == sorted(providers)
    assert len(providers) >= 3  # openai, anthropic, google at minimum


def test_models_for_provider() -> None:
    openai_models = models_for("openai")
    assert "gpt-4o" in openai_models
    assert "gpt-4o-mini" in openai_models
