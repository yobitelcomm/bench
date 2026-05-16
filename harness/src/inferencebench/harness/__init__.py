"""Core measurement engine for InferenceBench.

Public API (Phase 1 surface):

    # Model invocation
    from inferencebench.harness import ModelClient, CompletionResult, ClientError

    # Drivers (request scheduling)
    from inferencebench.harness import OpenLoopDriver, ClosedLoopDriver, Sample

    # Telemetry (background samplers)
    from inferencebench.harness import NVMLSampler, RAPLSampler

    # Convergence / warmup gate
    from inferencebench.harness import ConvergenceGate

    # Metrics
    from inferencebench.harness.metrics import Percentiles, BootstrapCI, GoodputAtSLO

    # Top-level run
    from inferencebench.harness import BenchmarkRun, RunResult

    # Health diagnostic
    from inferencebench.harness import run_diagnostic, DiagnosticReport

    # Hardware + software provenance
    from inferencebench.harness.fingerprint import (
        collect_hardware_fingerprint, collect_software_provenance,
    )
"""

from inferencebench.harness.client import (
    ClientError,
    CompletionResult,
    ModelClient,
    detect_endpoint_health,
    env_api_key,
)
from inferencebench.harness.convergence import ConvergenceGate, ConvergenceState
from inferencebench.harness.doctor import (
    CheckResult,
    CheckStatus,
    DiagnosticReport,
    run_diagnostic,
)
from inferencebench.harness.drivers import ClosedLoopDriver, OpenLoopDriver, Sample
from inferencebench.harness.fingerprint import (
    collect_hardware_fingerprint,
    collect_software_provenance,
)
from inferencebench.harness.metrics import BootstrapCI, Percentiles
from inferencebench.harness.run import BenchmarkRun, RunResult
from inferencebench.harness.telemetry import NVMLSampler, RAPLSampler

__all__ = [
    "BenchmarkRun",
    "BootstrapCI",
    "CheckResult",
    "CheckStatus",
    "ClientError",
    "ClosedLoopDriver",
    "CompletionResult",
    "ConvergenceGate",
    "ConvergenceState",
    "DiagnosticReport",
    "ModelClient",
    "NVMLSampler",
    "OpenLoopDriver",
    "Percentiles",
    "RAPLSampler",
    "RunResult",
    "Sample",
    "collect_hardware_fingerprint",
    "collect_software_provenance",
    "detect_endpoint_health",
    "env_api_key",
    "run_diagnostic",
]
