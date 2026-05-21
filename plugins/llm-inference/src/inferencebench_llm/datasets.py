"""Dataset loaders for the llm-inference plugin.

Phase 1 supports:
- ``builtin://`` — 5-prompt fallback shipped with the plugin (no network)
- ``hf://<repo>/<config>`` — Hugging Face datasets (graceful fallback to builtin if offline)
- ``file://<path>`` — local JSONL files (one prompt per line, ``{"prompt": "..."}`` shape)
- ``fixtures://<key>`` — pre-fetched dataset from ``bench fixtures fetch <key>``
  (resolved against ``$BENCH_FIXTURES_ROOT`` or
  ``~/.cache/inferencebench/fixtures/``)

The dataset hash in :class:`DatasetConfig.hash` is *not* recomputed here —
it's a manifest fingerprint that ships in the YAML. If the HF dataset content
changes upstream, the YAML's hash must be bumped via an ADR.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from inferencebench_llm.schemas import DatasetConfig

_BUILTIN_PROMPTS: list[str] = [
    "Explain the difference between TCP and UDP in two sentences.",
    "Write a Python function that returns the nth Fibonacci number iteratively.",
    "Summarise the plot of Hamlet in three sentences.",
    "What's the time complexity of merge sort? Explain why.",
    "Describe how a CPU cache hierarchy works at a high level.",
    "Compare Adam and AdamW optimisers — when would you pick each?",
    "Outline three causes of a sudden P99 latency spike in a steady RPS workload.",
    "What is bandwidth-delay product and why does it matter for TCP throughput?",
    "When would you use a circuit breaker vs a retry-with-backoff?",
    "Briefly: how does paged attention reduce KV cache fragmentation?",
]


def load_prompts(spec: DatasetConfig, *, max_n: int | None = None) -> list[str]:
    """Resolve a :class:`DatasetConfig.uri` into a list of prompt strings.

    Args:
        spec: The DatasetConfig from a BenchmarkSpec.
        max_n: Optional override of ``spec.sampling.n``.

    Returns:
        A list of prompt strings of length ``min(spec.sampling.n, available)``.

    Raises:
        ValueError: If the URI scheme is unsupported.
        FileNotFoundError: For ``file://`` URIs pointing nowhere.
    """
    n = max_n if max_n is not None else spec.sampling.n
    uri = spec.uri.strip()

    if uri.startswith("builtin://") or not uri:
        prompts = list(_BUILTIN_PROMPTS)
    elif uri.startswith("file://"):
        prompts = _load_file_prompts(uri[len("file://") :])
    elif uri.startswith("hf://"):
        prompts = _load_hf_prompts(uri[len("hf://") :], spec.sampling.seed)
    elif uri.startswith("fixtures://"):
        prompts = _load_fixture_prompts(uri[len("fixtures://") :])
    else:
        msg = f"unsupported dataset URI scheme: {uri.split('://')[0]}://"
        raise ValueError(msg)

    if not prompts:
        # Always return something so the driver has a workload — graceful empty fallback
        prompts = list(_BUILTIN_PROMPTS)

    # Round-robin / truncate to n
    if n <= len(prompts):
        return prompts[:n]
    repeat = (n // len(prompts)) + 1
    return (prompts * repeat)[:n]


def compute_dataset_hash(prompts: list[str]) -> str:
    """SHA-256 over the canonical JSON-ordered prompt list. Used for the envelope."""
    canonical = json.dumps(prompts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Per-scheme loaders                                                          #
# --------------------------------------------------------------------------- #
def _load_file_prompts(path: str) -> list[str]:
    """JSONL with ``{"prompt": "..."}`` per line. Other shapes are skipped silently."""
    p = Path(path)
    if not p.exists():
        msg = f"dataset file not found: {p}"
        raise FileNotFoundError(msg)
    prompts: list[str] = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("prompt"), str):
                prompts.append(obj["prompt"])
            elif isinstance(obj, str):
                prompts.append(obj)
    except OSError:
        pass
    return prompts


def _fixtures_cache_root() -> Path:
    """Resolve the bench-fixtures cache root.

    Honours ``BENCH_FIXTURES_ROOT`` for test/power-user overrides and otherwise
    matches the default ``bench fixtures fetch`` writes to.
    """
    override = os.environ.get("BENCH_FIXTURES_ROOT")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "inferencebench" / "fixtures"


def _load_fixture_prompts(key: str) -> list[str]:
    """Load a ``fixtures://<key>`` dataset previously written by ``bench fixtures fetch``.

    Each line is a JSON object; we pick the first plausible string field
    (``prompt``, ``question``, ``query``, ``source``) so the perf driver can
    feed prompts to its engines regardless of the upstream dataset shape.

    Raises:
        FileNotFoundError: If the fixture has not been fetched. The message
            guides the user to ``bench fixtures fetch <key>``.
    """
    cache_path = _fixtures_cache_root() / f"{key}.jsonl"
    if not cache_path.exists():
        msg = f"fixture not cached: {cache_path}. Run `bench fixtures fetch {key}` first."
        raise FileNotFoundError(msg)

    prompts: list[str] = []
    with cache_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            for field in ("prompt", "question", "query", "source"):
                value = obj.get(field)
                if isinstance(value, str) and value.strip():
                    prompts.append(value)
                    break
    return prompts


def _load_hf_prompts(repo_path: str, seed: int) -> list[str]:
    """Load prompts from a Hugging Face dataset repo. Falls back to [] on any error.

    The caller's :func:`load_prompts` handles fallback to the builtin set, so
    a network-less CI environment is OK — it just gets the fallback prompts.

    The repo_path looks like ``<org>/<repo>`` or ``<org>/<repo>/<config>``.
    We pull the first column with type ``string`` from the test split.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    # Parse out config + split
    parts = repo_path.split("/")
    if len(parts) < 2:
        return []
    repo = "/".join(parts[:2])
    config = parts[2] if len(parts) > 2 else None
    split = "test"

    try:
        ds = load_dataset(repo, config, split=split, streaming=True)
    except Exception:
        return []

    prompts: list[str] = []
    for i, row in enumerate(ds):
        if i >= 1000:  # safety cap
            break
        for k in ("prompt", "text", "instruction", "input"):
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                prompts.append(v)
                break

    # Deterministic shuffle for reproducibility
    if prompts:
        import random

        random.Random(seed).shuffle(prompts)

    return prompts
