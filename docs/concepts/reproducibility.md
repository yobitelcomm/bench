# Reproducibility

The reason InferenceBench exists is to make benchmark results reproducible. Four mechanisms enforce reproducibility:

1. **Warm-up discipline.** Three runs are discarded before measurement.
2. **A convergence gate.** Measurement does not start until the system is statistically stable.
3. **Explicit seeds.** Every stochastic step uses a seed that is recorded in the envelope.
4. **The hardware fingerprint.** Every result is bound to the configuration that produced it.

Together, these give two readers of an envelope enough information to attempt the same run.

## Warm-up discipline

```text
Run 0  → discarded (cold caches, JIT warm, allocator settling)
Run 1  → discarded
Run 2  → discarded
Run 3+ → measurement begins, conditional on convergence
```

Discarding three warm-ups is not the only behavior worth measuring. Cold-start latency is reported separately in `metrics.cold_start_ms` and is sampled in a dedicated phase. The warm steady-state numbers are what `ttft_p50_ms` and friends measure.

## The convergence gate

After the discarded warm-ups, the harness waits for the coefficient of variation of TTFT to drop below 5% across a rolling window of 30 requests. If the gate does not pass within 60 seconds, the harness records a warning in the envelope and proceeds anyway, so the run is not lost.

The gate exists because cold-start variance can persist longer than three warm iterations on large models with paged-attention KV caches.

## Seeds

Every stochastic step uses a seed:

- The dataset sampler
- The request inter-arrival generator (Poisson)
- The prompt shuffle
- The decoding generator (when sampling > 0 temperature)

The seed is recorded in `envelope.seed`. Re-running with the same seed, same dataset hash, same engine config hash, and same hardware fingerprint should produce identical traces modulo non-determinism in the kernel (which is itself measured and reported in `warnings`).

## Three independent process launches

Cross-engine comparisons require three independent process launches with different seeds, then aggregating via bootstrap CI. This catches engine-internal warm-up effects and allocator nondeterminism that a single run hides.

## Bootstrap percentile CIs

Percentiles are reported with a 95% bootstrap confidence interval (1000 resamples). The harness never reports a percentile as a single point estimate.

Expected output snippet:

```
ttft_p50_ms   142.0  [139.4, 144.7]
ttft_p99_ms   280.3  [262.1, 304.5]
```

## What you need to reproduce a run

From an envelope alone, a reader needs:

- The model id and revision (`model.id`, `model.revision`)
- The engine name, version, and config hash (`engine.*`)
- The dataset id and hash (`dataset.*`)
- The seed (`seed`)
- The driver options (`driver_options`)
- A matching hardware configuration (`hardware_fingerprint`)

Phase 1 ships everything but the engine config replay tool. Recovering the engine config from `config_hash` is on the v0.2 roadmap.

## See also

- [Hardware fingerprinting](fingerprinting.md)
- [The signed envelope](envelope.md)
- [Methodology](methodology.md)
