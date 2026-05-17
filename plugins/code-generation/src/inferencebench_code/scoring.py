"""Pure scoring helpers for the code-generation plugin.

Two helpers:

* :func:`extract_python_code` extracts the first fenced ``python`` block
  from a model response. The HumanEval prompting convention is to ask the
  model to return code in a markdown fence; we strip the fence and return
  the inner code. When no fence is present we treat the whole response
  as code (some smaller models skip the fences when given a function
  signature stub).

* :func:`compute_pass_at_k` returns the unbiased pass@k estimator from
  the HumanEval paper (Chen et al. 2021): for ``n`` samples with ``c``
  passing, ``pass@k = 1 - C(n-c, k) / C(n, k)``. For Phase 1 we only
  run ``k=1`` (so this collapses to the mean) but the helper ships so
  future revisions can compute pass@10/100 from richer sampling runs.
"""

from __future__ import annotations

import math
import re

# Match a fenced python block — accept ``` python``` or ``` py ```. We
# extract only the inner body and intentionally do not require a closing
# newline so partial streamed responses still parse.
_PY_FENCE = re.compile(
    r"```(?:python|py)\s*\n(.*?)(?:```|\Z)",
    re.IGNORECASE | re.DOTALL,
)
# Bare ``` fence (no language tag) — accepted as a fallback when the
# model omits the language hint.
_BARE_FENCE = re.compile(r"```\s*\n(.*?)(?:```|\Z)", re.DOTALL)


def extract_python_code(text: str) -> str:
    """Extract the first fenced ``python`` block from ``text``.

    Returns the inner code with leading/trailing whitespace stripped.
    Falls back to a bare triple-fence, then to the whole response (also
    stripped) when no fence is present.

    Multi-fence responses return the **first** block — HumanEval-style
    prompts ask for one solution; later fences usually contain test
    repetition or example I/O.
    """
    match = _PY_FENCE.search(text)
    if match is not None:
        return match.group(1).strip()
    match = _BARE_FENCE.search(text)
    if match is not None:
        return match.group(1).strip()
    return text.strip()


def compute_pass_at_k(results: list[bool], k: int) -> float:
    """Return the HumanEval-paper unbiased pass@k estimator.

    Arguments:
        results: list of per-sample pass/fail booleans (one model attempt each).
        k: how many top samples per task we would have picked.

    Formula: ``pass@k = 1 - C(n - c, k) / C(n, k)`` where ``n = len(results)``
    and ``c = sum(results)``. When ``n - c < k`` the binomial coefficient is
    zero and pass@k collapses to 1.0 (every k-subset must contain at least
    one passing sample). When ``k > n`` we clip ``k = n`` — the estimator is
    only defined for ``k <= n``.
    """
    n = len(results)
    if n == 0:
        return 0.0
    if k < 1:
        return 0.0
    k = min(k, n)
    c = sum(1 for r in results if r)
    if n - c < k:
        return 1.0
    # math.comb is exact for ints; cast to float for the ratio.
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
