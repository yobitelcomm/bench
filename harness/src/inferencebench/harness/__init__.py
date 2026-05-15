"""Core measurement engine for InferenceBench.

Public API (Phase 1 surface so far):

    # Model invocation
    from inferencebench.harness import ModelClient, CompletionResult, ClientError

    # Drivers (request scheduling)
    from inferencebench.harness.drivers import OpenLoopDriver, ClosedLoopDriver, Sample

    # Telemetry (background samplers)
    from inferencebench.harness.telemetry import NVMLSampler, RAPLSampler

    # Convergence / warmup gate
    from inferencebench.harness.convergence import ConvergenceGate

    # Metrics
    from inferencebench.harness.metrics import Percentiles, BootstrapCI

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
from inferencebench.harness.telemetry import NVMLSampler, RAPLSampler

__all__ = [
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
    "Sample",
    "collect_hardware_fingerprint",
    "collect_software_provenance",
    "detect_endpoint_health",
    "env_api_key",
    "run_diagnostic",
]
