"""Percentile estimator with bootstrap confidence intervals.

Required for every latency report. We use bootstrap CIs (1000 resamples,
95% by default) on the percentile estimator itself rather than Gaussian
approximations — latency distributions are heavy-tailed and Gaussian
intervals badly underestimate P99/P99.9 uncertainty.

Public API::

    from inferencebench.harness.metrics import Percentiles

    p = Percentiles(samples)
    print(p.p50, p.p99, p.p99_ci_low, p.p99_ci_high)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_PERCENTILES = (50.0, 75.0, 90.0, 95.0, 99.0, 99.9)
DEFAULT_N_RESAMPLES = 1000
DEFAULT_CI = 95.0  # produces (2.5, 97.5) interval


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """Bootstrap confidence interval for one percentile."""

    percentile: float
    estimate: float
    ci_low: float
    ci_high: float
    ci_pct: float
    n_resamples: int


class Percentiles:
    """Compute percentiles + bootstrap CIs from a 1-D array of samples.

    Args:
        samples: Iterable of floats (latencies, RTFs, anything).
        percentiles: Which percentiles to compute. Defaults to a sensible set.
        bootstrap: If True (default), compute bootstrap CIs for each percentile.
        n_resamples: Bootstrap resample count. 1000 is fast + tight; 10000 is
            overkill for most cases.
        ci_pct: Confidence level (e.g. 95.0). Symmetric quantile interval.
        seed: PRNG seed for reproducibility.

    After construction the percentile values are accessible as attributes
    (e.g. ``p.p50``, ``p.p99``) and the full BootstrapCI list as ``p.cis``.
    """

    def __init__(
        self,
        samples: list[float] | np.ndarray,
        *,
        percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
        bootstrap: bool = True,
        n_resamples: int = DEFAULT_N_RESAMPLES,
        ci_pct: float = DEFAULT_CI,
        seed: int = 42,
    ) -> None:
        arr = np.asarray([float(x) for x in samples if np.isfinite(x)], dtype=np.float64)
        if arr.size == 0:
            msg = "Percentiles needs at least one finite sample"
            raise ValueError(msg)
        self._arr = arr
        self.percentiles = percentiles
        self.n = int(arr.size)
        self.mean = float(arr.mean())
        self.min = float(arr.min())
        self.max = float(arr.max())

        # Point estimates
        self._values: dict[float, float] = {
            p: float(np.percentile(arr, p, method="linear")) for p in percentiles
        }

        # Bootstrap CIs
        self.cis: list[BootstrapCI] = []
        if bootstrap and arr.size >= 2:
            rng = np.random.default_rng(seed)
            for p in percentiles:
                ci = bootstrap_percentile_ci(
                    arr, p, n_resamples=n_resamples, ci_pct=ci_pct, rng=rng
                )
                self.cis.append(ci)

    # Conveniently-named attribute access: p.p50, p.p99, p.p99_ci_low, p.p99_ci_high
    def __getattr__(self, name: str) -> float:
        if name.startswith("p") and "_ci_" not in name:
            try:
                p = float(name[1:].replace("_", "."))
            except ValueError:
                pass
            else:
                if p in self._values:
                    return self._values[p]
        if "_ci_" in name and name.startswith("p"):
            # e.g. p99_ci_low, p99_9_ci_high
            try:
                head, side = name.split("_ci_")
                p = float(head[1:].replace("_", "."))
            except ValueError:
                pass
            else:
                for ci in self.cis:
                    if ci.percentile == p:
                        return ci.ci_low if side == "low" else ci.ci_high
        raise AttributeError(name)

    def as_dict(self) -> dict[str, float]:
        """Return a flat dict for embedding in envelopes / JSON.

        Keys: ``"p50_ms"``, ``"p99_ms"``, ``"p99_ci_low_ms"``, ``"p99_ci_high_ms"``,
        plus ``"mean"``, ``"min"``, ``"max"``, ``"n"``.
        """
        out: dict[str, float] = {
            "n": float(self.n),
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
        }
        for p, val in self._values.items():
            key = f"p{_format_p(p)}"
            out[key] = val
        for ci in self.cis:
            base = f"p{_format_p(ci.percentile)}"
            out[f"{base}_ci_low"] = ci.ci_low
            out[f"{base}_ci_high"] = ci.ci_high
        return out


def bootstrap_percentile_ci(
    samples: np.ndarray,
    percentile: float,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_pct: float = DEFAULT_CI,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """Compute bootstrap CI for one percentile of one sample array.

    Resamples ``n_resamples`` times with replacement (size = len(samples)),
    takes the percentile of each resample, and returns the
    (lower, upper) quantiles for the requested confidence interval.

    Args:
        samples: 1-D float array of N samples.
        percentile: Which percentile (0-100) to estimate.
        n_resamples: Number of bootstrap resamples. 1000 default.
        ci_pct: Confidence level (e.g. 95.0). Symmetric interval.
        rng: numpy Generator. Defaults to deterministic seed 42.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(samples)
    if n < 2:
        # Degenerate case: CI = point estimate ± 0
        v = float(np.percentile(samples, percentile, method="linear"))
        return BootstrapCI(
            percentile=percentile,
            estimate=v,
            ci_low=v,
            ci_high=v,
            ci_pct=ci_pct,
            n_resamples=n_resamples,
        )

    # Vectorised resampling: (n_resamples, n) index matrix
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled = samples[idx]
    boot_estimates = np.percentile(resampled, percentile, axis=1, method="linear")

    alpha = (100.0 - ci_pct) / 2.0
    lo = float(np.percentile(boot_estimates, alpha))
    hi = float(np.percentile(boot_estimates, 100.0 - alpha))
    est = float(np.percentile(samples, percentile, method="linear"))

    return BootstrapCI(
        percentile=percentile,
        estimate=est,
        ci_low=lo,
        ci_high=hi,
        ci_pct=ci_pct,
        n_resamples=n_resamples,
    )


def _format_p(p: float) -> str:
    """Format a percentile value for use in dict keys: 50.0 -> '50', 99.9 -> '99_9'."""
    if p == int(p):
        return str(int(p))
    return str(p).replace(".", "_")
