"""Provider pricing registry — converts token counts to $ when the provider
doesn't include cost in its API response.

The registry is a small in-memory dict of ``(provider, model)`` → input/output
$/M-token rates. Phase 1 ships a starter set of public list prices; Phase 2
adds a YAML file and a community PR flow to keep prices fresh.

For LiteLLM-routed providers (OpenAI, Anthropic, Together, etc.) LiteLLM
usually returns ``response_cost`` directly — use that when available.
This registry covers the fallback path: self-hosted vLLM/SGLang/local where
the user wants a $-per-million-tokens estimate based on the underlying model.

All prices are USD per million tokens, list price as of 2026-05 (verify before
quoting publicly).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Public list price for one model on one provider."""

    provider: str
    model: str
    input_per_million_usd: float
    output_per_million_usd: float
    notes: str = ""

    def cost_for(self, tokens_in: int, tokens_out: int) -> float:
        """Compute the USD cost for one request."""
        return (
            tokens_in / 1_000_000.0 * self.input_per_million_usd
            + tokens_out / 1_000_000.0 * self.output_per_million_usd
        )


# Provider/model → ModelPricing. Phase 1 starter set — keep it small + verified.
# Prices in USD per million tokens. Update via PR with a citation to the
# provider's pricing page.
_REGISTRY: dict[tuple[str, str], ModelPricing] = {
    ("openai", "gpt-4o"): ModelPricing(
        provider="openai",
        model="gpt-4o",
        input_per_million_usd=2.50,
        output_per_million_usd=10.00,
        notes="OpenAI listed 2026-05",
    ),
    ("openai", "gpt-4o-mini"): ModelPricing(
        provider="openai",
        model="gpt-4o-mini",
        input_per_million_usd=0.15,
        output_per_million_usd=0.60,
    ),
    ("anthropic", "claude-opus-4-7"): ModelPricing(
        provider="anthropic",
        model="claude-opus-4-7",
        input_per_million_usd=15.00,
        output_per_million_usd=75.00,
    ),
    ("anthropic", "claude-sonnet-4-6"): ModelPricing(
        provider="anthropic",
        model="claude-sonnet-4-6",
        input_per_million_usd=3.00,
        output_per_million_usd=15.00,
    ),
    ("anthropic", "claude-haiku-4-5"): ModelPricing(
        provider="anthropic",
        model="claude-haiku-4-5",
        input_per_million_usd=0.25,
        output_per_million_usd=1.25,
    ),
    ("google", "gemini-3.1-pro"): ModelPricing(
        provider="google",
        model="gemini-3.1-pro",
        input_per_million_usd=1.25,
        output_per_million_usd=5.00,
    ),
    ("together", "meta-llama/Llama-4-Maverick"): ModelPricing(
        provider="together",
        model="meta-llama/Llama-4-Maverick",
        input_per_million_usd=0.60,
        output_per_million_usd=0.60,
        notes="Together blended rate",
    ),
    ("fireworks", "meta-llama/Llama-4-Maverick"): ModelPricing(
        provider="fireworks",
        model="meta-llama/Llama-4-Maverick",
        input_per_million_usd=0.50,
        output_per_million_usd=0.50,
    ),
    ("groq", "meta-llama/Llama-4-Maverick"): ModelPricing(
        provider="groq",
        model="meta-llama/Llama-4-Maverick",
        input_per_million_usd=0.59,
        output_per_million_usd=0.79,
    ),
    ("together", "meta-llama/Llama-3.1-8B-Instruct"): ModelPricing(
        provider="together",
        model="meta-llama/Llama-3.1-8B-Instruct",
        input_per_million_usd=0.18,
        output_per_million_usd=0.18,
    ),
    ("fireworks", "meta-llama/Llama-3.1-8B-Instruct"): ModelPricing(
        provider="fireworks",
        model="meta-llama/Llama-3.1-8B-Instruct",
        input_per_million_usd=0.20,
        output_per_million_usd=0.20,
    ),
    ("groq", "meta-llama/Llama-3.1-8B-Instruct"): ModelPricing(
        provider="groq",
        model="meta-llama/Llama-3.1-8B-Instruct",
        input_per_million_usd=0.05,
        output_per_million_usd=0.08,
    ),
    ("together", "meta-llama/Llama-3.1-70B-Instruct"): ModelPricing(
        provider="together",
        model="meta-llama/Llama-3.1-70B-Instruct",
        input_per_million_usd=0.88,
        output_per_million_usd=0.88,
    ),
    ("fireworks", "meta-llama/Llama-3.1-70B-Instruct"): ModelPricing(
        provider="fireworks",
        model="meta-llama/Llama-3.1-70B-Instruct",
        input_per_million_usd=0.90,
        output_per_million_usd=0.90,
    ),
    ("groq", "meta-llama/Llama-3.1-70B-Instruct"): ModelPricing(
        provider="groq",
        model="meta-llama/Llama-3.1-70B-Instruct",
        input_per_million_usd=0.59,
        output_per_million_usd=0.79,
    ),
}


def lookup(provider: str, model: str) -> ModelPricing | None:
    """Return the pricing entry for ``(provider, model)``, or None if not registered."""
    key = (provider.lower().strip(), model.strip())
    if key in _REGISTRY:
        return _REGISTRY[key]
    # Tolerate ``openai/gpt-4o`` style: split on first slash
    if "/" in model:
        head, tail = model.split("/", 1)
        return _REGISTRY.get((head.lower(), tail))
    return None


def all_providers() -> list[str]:
    """List unique providers in the registry, sorted."""
    return sorted({p for p, _ in _REGISTRY})


def models_for(provider: str) -> list[str]:
    """List models registered for one provider, sorted."""
    return sorted(m for p, m in _REGISTRY if p == provider.lower())


def providers_for(model: str) -> list[ModelPricing]:
    """List ``ModelPricing`` entries registered for one canonical model id, sorted by provider.

    Used when we have a model id and want to know which providers serve it.
    The model id is the canonical HF-style id (e.g. ``meta-llama/Llama-3.1-8B-Instruct``);
    callers should strip routing prefixes like ``openai/`` before looking up.
    """
    key = model.strip()
    return sorted(
        (entry for (_, m), entry in _REGISTRY.items() if m == key),
        key=lambda e: e.provider,
    )


def estimate_cost(
    provider: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> float | None:
    """Estimate USD cost for one request. Returns None if the model isn't priced."""
    p = lookup(provider, model)
    if p is None:
        return None
    return p.cost_for(tokens_in, tokens_out)
