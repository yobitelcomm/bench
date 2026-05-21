"""Provider pricing registry — converts token counts to $ when the provider
doesn't include cost in its API response.

The registry is a small in-memory dict of ``(provider, model)`` → input/output
$/M-token rates. The data lives in :file:`prices.yaml` shipped with the
plugin; users can override it by passing ``--prices-file <path>`` to
``bench cost`` or ``bench run``.

For LiteLLM-routed providers (OpenAI, Anthropic, Together, etc.) LiteLLM
usually returns ``response_cost`` directly — use that when available.
This registry covers the fallback path: self-hosted vLLM/SGLang/local where
the user wants a $-per-million-tokens estimate based on the underlying model.

All prices are USD per million tokens, list price as of 2026-05 (verify before
quoting publicly).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


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


# Hardcoded fallback used only if the bundled :file:`prices.yaml` is missing
# or corrupt. Keeps the module importable in adversarial install states.
_BUILTIN_FALLBACK: dict[tuple[str, str], ModelPricing] = {
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


@dataclass(frozen=True, slots=True)
class _ParseStats:
    """Validation summary for a single YAML pricing file."""

    valid: int
    skipped: int
    errors: list[str]


def _parse_entry(
    raw: Any,  # noqa: ANN401 -- YAML payload is arbitrary user input
    *,
    index: int,
    source: str,
) -> ModelPricing | str:
    """Validate one YAML entry. Returns a :class:`ModelPricing` or an error message.

    Required keys: ``provider`` (str), ``model`` (str),
    ``input_per_million_usd`` (number), ``output_per_million_usd`` (number).
    Optional: ``notes`` (str).
    """
    if not isinstance(raw, dict):
        return f"entry {index} in {source}: expected mapping, got {type(raw).__name__}"
    required = ("provider", "model", "input_per_million_usd", "output_per_million_usd")
    missing = [k for k in required if k not in raw]
    if missing:
        return f"entry {index} in {source}: missing required keys {missing}"
    provider = raw["provider"]
    model = raw["model"]
    input_rate = raw["input_per_million_usd"]
    output_rate = raw["output_per_million_usd"]
    notes = raw.get("notes", "")
    if not isinstance(provider, str) or not provider.strip():
        return f"entry {index} in {source}: 'provider' must be a non-empty string"
    if not isinstance(model, str) or not model.strip():
        return f"entry {index} in {source}: 'model' must be a non-empty string"
    if not isinstance(input_rate, int | float) or isinstance(input_rate, bool):
        return f"entry {index} in {source}: 'input_per_million_usd' must be a number"
    if not isinstance(output_rate, int | float) or isinstance(output_rate, bool):
        return f"entry {index} in {source}: 'output_per_million_usd' must be a number"
    if notes is None:
        notes = ""
    if not isinstance(notes, str):
        return f"entry {index} in {source}: 'notes' must be a string"
    return ModelPricing(
        provider=provider,
        model=model,
        input_per_million_usd=float(input_rate),
        output_per_million_usd=float(output_rate),
        notes=notes,
    )


def _build_registry_from_yaml(
    payload: Any,  # noqa: ANN401 -- YAML payload is arbitrary user input
    *,
    source: str,
) -> tuple[dict[tuple[str, str], ModelPricing], _ParseStats]:
    """Build a ``(provider, model) -> ModelPricing`` dict from a parsed YAML payload.

    Invalid entries are logged as warnings and skipped (no exception). The
    returned :class:`_ParseStats` summarises what was kept vs skipped.
    """
    if not isinstance(payload, dict):
        msg = f"{source}: top-level YAML must be a mapping, got {type(payload).__name__}"
        return {}, _ParseStats(valid=0, skipped=0, errors=[msg])
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        msg = f"{source}: 'entries' must be a list, got {type(entries).__name__}"
        return {}, _ParseStats(valid=0, skipped=0, errors=[msg])

    registry: dict[tuple[str, str], ModelPricing] = {}
    errors: list[str] = []
    valid = 0
    skipped = 0
    for i, raw in enumerate(entries):
        parsed = _parse_entry(raw, index=i, source=source)
        if isinstance(parsed, str):
            logger.warning("Skipping invalid pricing entry: %s", parsed)
            errors.append(parsed)
            skipped += 1
            continue
        registry[(parsed.provider.lower().strip(), parsed.model.strip())] = parsed
        valid += 1
    return registry, _ParseStats(valid=valid, skipped=skipped, errors=errors)


def _load_bundled() -> dict[tuple[str, str], ModelPricing]:
    """Load the bundled :file:`prices.yaml` resource. Fall back to the builtin dict."""
    try:
        resource = files("inferencebench_llm") / "prices.yaml"
        with as_file(resource) as path:
            if not path.is_file():
                logger.warning("Bundled prices.yaml missing; using _BUILTIN_FALLBACK")
                return dict(_BUILTIN_FALLBACK)
            text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        logger.warning("Could not read bundled prices.yaml (%s); using _BUILTIN_FALLBACK", exc)
        return dict(_BUILTIN_FALLBACK)
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        logger.warning("Bundled prices.yaml is malformed (%s); using _BUILTIN_FALLBACK", exc)
        return dict(_BUILTIN_FALLBACK)
    registry, _stats = _build_registry_from_yaml(payload, source="<bundled prices.yaml>")
    if not registry:
        logger.warning("Bundled prices.yaml produced an empty registry; using _BUILTIN_FALLBACK")
        return dict(_BUILTIN_FALLBACK)
    return registry


def load_pricing(
    path: Path | str | None = None,
) -> dict[tuple[str, str], ModelPricing]:
    """Load a pricing registry from disk.

    With ``path=None`` returns a fresh copy of the bundled registry (the same
    one ``lookup``/``estimate_cost`` consult by default).

    With a path, parses that YAML file and returns the resulting registry
    dict. Does NOT mutate the module-level ``_REGISTRY`` — pass the result
    around explicitly or call :func:`set_pricing` to install it process-wide.

    Raises ``FileNotFoundError`` if ``path`` is set but doesn't exist, and
    ``ValueError`` if the YAML is malformed.
    """
    if path is None:
        return _load_bundled()
    p = Path(path)
    if not p.is_file():
        msg = f"prices file not found: {p}"
        raise FileNotFoundError(msg)
    try:
        payload = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"failed to parse YAML at {p}: {exc}"
        raise ValueError(msg) from exc
    registry, _stats = _build_registry_from_yaml(payload, source=str(p))
    return registry


def validate_pricing_file(path: Path | str) -> _ParseStats:
    """Parse a YAML pricing file and return a :class:`_ParseStats` summary.

    Used by ``bench cost --validate-prices <path>``. Does not raise on
    invalid entries — they're surfaced as ``errors`` on the result. Does
    raise ``FileNotFoundError`` if the file is missing and ``ValueError``
    if the YAML itself is malformed.
    """
    p = Path(path)
    if not p.is_file():
        msg = f"prices file not found: {p}"
        raise FileNotFoundError(msg)
    try:
        payload = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"failed to parse YAML at {p}: {exc}"
        raise ValueError(msg) from exc
    _registry, stats = _build_registry_from_yaml(payload, source=str(p))
    return stats


# Module-level registry consulted by ``lookup``/``estimate_cost``/etc.
# Built from the bundled YAML at import time; can be replaced via
# :func:`set_pricing` for callers that want process-wide custom pricing.
_REGISTRY: dict[tuple[str, str], ModelPricing] = _load_bundled()


def set_pricing(registry: dict[tuple[str, str], ModelPricing]) -> None:
    """Replace the module-level pricing registry with ``registry`` in place.

    This is **process-wide global state** — every subsequent call to
    :func:`lookup`, :func:`estimate_cost`, :func:`all_providers`,
    :func:`models_for`, and :func:`providers_for` consults the new dict.
    Prefer :func:`load_pricing` + passing the result explicitly when
    possible; reach for ``set_pricing`` only when retrofitting code paths
    that go through the module-level helpers.
    """
    global _REGISTRY
    _REGISTRY = registry


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
