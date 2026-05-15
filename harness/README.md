# inferencebench-harness

The core measurement engine. Drivers (open-loop Poisson, closed-loop), telemetry samplers (NVML, RAPL), percentile math with bootstrap CI, hardware fingerprinting.

## Status

Phase 1 active development.

## Public API (in development)

```python
from inferencebench.harness import BenchmarkRun, OpenLoopDriver, ClosedLoopDriver
from inferencebench.harness.telemetry import NVMLSampler, RAPLSampler
from inferencebench.harness.metrics import Percentiles, BootstrapCI, GoodputAtSLO
from inferencebench.harness.fingerprint import compute_hardware_fingerprint
```

See [docs/concepts/harness.md](../docs/concepts/harness.md) for the full conceptual guide.
