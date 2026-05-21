"""LLMInferencePlugin — Plugin entry point for ``llm.inference`` benchmarks.

Implements the contract every InferenceBench plugin must satisfy:

- :meth:`list_benchmarks` — what benchmark specs are bundled with this plugin
- :meth:`get_benchmark` — look one up by id
- :meth:`run` — execute a spec and produce a signed envelope
- :meth:`validate` — fast sanity check before run
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from inferencebench.envelope import (
    DatasetSpec as EnvDatasetSpec,
)
from inferencebench.envelope import (
    EngineConfig,
    Envelope,
    EnvelopeBuilder,
    ModelConfig,
    Quantization,
    SigningMode,
    sign_envelope,
)
from inferencebench.harness import (
    BenchmarkRun,
    ClosedLoopDriver,
    CompletionResult,
    OpenLoopDriver,
    Sample,
    collect_hardware_fingerprint,
    collect_software_provenance,
)
from inferencebench.harness.convergence import ConvergenceGate
from inferencebench.harness.metrics import SLOPredicate, summarise_energy
from inferencebench_llm.datasets import compute_dataset_hash, load_prompts
from inferencebench_llm.engines import (
    Engine,
    EngineUnavailableError,
    LlamaCppEngine,
    MLXEngine,
    SGLangEngine,
    TRTLLMEngine,
    VLLMEngine,
)
from inferencebench_llm.pricing import ModelPricing, load_pricing, providers_for
from inferencebench_llm.schemas import BenchmarkSpec, EngineKind, RunContext
from inferencebench_llm.slo_profiles import (
    HardwareClass,
    classify,
    format_resolved,
    scale_slos,
)

if TYPE_CHECKING:
    from inferencebench.harness.run import RunResult


# Metrics this plugin is expected to emit on a healthy run. Used by
# ``bench coverage`` to flag silent-failure envelopes (e.g. NVML samples
# failed → no power_avg_w).
EXPECTED_METRICS: tuple[str, ...] = (
    "throughput_tok_per_s",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "tpot_p50_ms",
    "tpot_p99_ms",
    "total_p50_ms",
    "total_p99_ms",
    "ok_rate",
    "compliance_rate",
    "power_avg_w",
    "power_peak_w",
    "energy_joules_total",
    "joules_per_token",
    "req_per_s_passing",
    "req_per_s_all",
)


def _json_num(v: float) -> str:
    """JSON-safe numeric encoder: NaN/inf become null."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "null"
    return repr(v)


def _json_str(v: str | None) -> str:
    """JSON-safe string encoder."""
    return json.dumps(v if v is not None else "")


# Blended input/output weighting for the reference-cost fallback. Matches
# ``bench cost``'s default (3:1 input:output).
_BLEND_INPUT_SHARE = 0.75
_BLEND_OUTPUT_SHARE = 1.0 - _BLEND_INPUT_SHARE


def _strip_routing_prefix(model_id: str) -> str:
    """Strip LiteLLM-style routing prefixes (e.g. ``openai/``) from a model id.

    The pricing registry stores canonical HF model ids; users who plumb in
    ``openai/meta-llama/Llama-3.1-8B-Instruct`` should still hit the registry.
    Mirrors ``lookup()``'s prefix tolerance in :mod:`inferencebench_llm.pricing`.
    """
    if model_id.startswith("openai/"):
        return model_id[len("openai/") :]
    return model_id


def _registry_reference_cost(
    model_id: str,
    *,
    custom_registry: dict[tuple[str, str], ModelPricing] | None = None,
) -> tuple[float, str] | None:
    """Return ``(blended_rate_usd_per_million, cheapest_provider)`` for ``model_id``.

    Returns ``None`` if no provider in the (bundled or custom) registry offers
    this model — callers should omit the cost metric entirely in that case
    rather than emit a misleading zero.

    When ``custom_registry`` is supplied (typically from
    ``RunContext.extra["prices_file"]``) it overrides the bundled registry.
    """
    canonical = _strip_routing_prefix(model_id)
    if custom_registry is not None:
        entries = sorted(
            (e for (_, m), e in custom_registry.items() if m == canonical),
            key=lambda e: e.provider,
        )
    else:
        entries = providers_for(canonical)
    if not entries:
        return None
    best = min(
        entries,
        key=lambda e: (
            _BLEND_INPUT_SHARE * e.input_per_million_usd
            + _BLEND_OUTPUT_SHARE * e.output_per_million_usd
        ),
    )
    blended = (
        _BLEND_INPUT_SHARE * best.input_per_million_usd
        + _BLEND_OUTPUT_SHARE * best.output_per_million_usd
    )
    return blended, best.provider


# --------------------------------------------------------------------------- #
# Engine registry                                                             #
# --------------------------------------------------------------------------- #
_ENGINES: dict[EngineKind, type[Engine]] = {
    EngineKind.VLLM: VLLMEngine,
    EngineKind.SGLANG: SGLangEngine,
    EngineKind.LLAMACPP: LlamaCppEngine,
    EngineKind.TRTLLM: TRTLLMEngine,
    EngineKind.MLX: MLXEngine,
}


def _engine_for(kind: EngineKind) -> Engine:
    cls = _ENGINES.get(kind)
    if cls is None:
        supported = ", ".join(sorted(k.value for k in _ENGINES))
        msg = f"Engine '{kind.value}' is not implemented yet. Supported engines: {supported}."
        raise EngineUnavailableError(msg)
    return cls()


_FALLBACK_PROMPTS: list[str] = [
    "Explain the difference between TCP and UDP in two sentences.",
    "Write a Python function that returns the nth Fibonacci number iteratively.",
    "Summarise the plot of Hamlet in three sentences.",
    "What's the time complexity of merge sort? Explain why.",
    "Describe how a CPU cache hierarchy works at a high level.",
]


_SLO_TEMPLATES: dict[str, list[SLOPredicate]] = {
    "llm.standard": [
        SLOPredicate("ttft", "ttft_ms", "<", 200.0),
        SLOPredicate("tpot", "tpot_ms", "<", 50.0),
        SLOPredicate("total", "total_ms", "<", 3000.0),
    ],
    "llm.relaxed": [
        SLOPredicate("ttft", "ttft_ms", "<", 1000.0),
        SLOPredicate("total", "total_ms", "<", 10000.0),
    ],
}


class LLMInferencePlugin:
    """Plugin entry point. Registered via ``inferencebench.plugins`` entrypoint group."""

    suite_id = "llm.inference"
    version = "0.0.0"
    description = "LLM inference performance benchmarks (TTFT/TPOT/throughput/goodput)."

    def list_benchmarks(self) -> list[BenchmarkSpec]:
        bench_dir = self._benchmarks_dir()
        specs: list[BenchmarkSpec] = []
        if not bench_dir.exists():
            return specs
        for yml in sorted(bench_dir.glob("*.yaml")):
            specs.append(self._load_yaml(yml))
        return specs

    def get_benchmark(self, benchmark_id: str) -> BenchmarkSpec:
        for spec in self.list_benchmarks():
            if spec.benchmark_id == benchmark_id:
                return spec
        msg = f"benchmark_id not found: {benchmark_id}"
        raise KeyError(msg)

    def validate(self, spec: BenchmarkSpec, context: RunContext) -> list[str]:
        warnings: list[str] = []
        try:
            engine = _engine_for(context.engine_kind)
            engine.probe(context)
        except EngineUnavailableError as exc:
            warnings.append(f"engine: {exc}")
        if not context.model_id:
            warnings.append("model_id is empty")
        if context.engine_kind == EngineKind.VLLM and not context.base_url:
            warnings.append("vLLM needs base_url (e.g. http://localhost:8000/v1)")
        return warnings

    def run(self, spec: BenchmarkSpec, context: RunContext) -> Envelope:
        """Execute the benchmark and return a SIGNED envelope."""
        engine = _engine_for(context.engine_kind)
        engine_version = engine.probe(context)
        client = engine.build_client(context)

        driver = self._build_driver(spec, context)
        gate = ConvergenceGate(
            warmup_runs=spec.warmup.discard_runs,
            window=spec.warmup.convergence_window,
            threshold=spec.warmup.convergence_cov_threshold,
        )
        prompts = self._workload_prompts(spec)
        actual_dataset_hash = compute_dataset_hash(prompts)

        def _request_fn(idx: int, item: Any) -> Sample:
            prompt = str(item)
            t_arrival = time.perf_counter() * 1000.0
            try:
                result: CompletionResult = client.complete(prompt, max_tokens=128, stream=True)
            except Exception as exc:
                return Sample(
                    request_idx=idx,
                    arrival_ms=t_arrival,
                    start_ms=t_arrival,
                    ttft_ms=float("nan"),
                    total_ms=float("nan"),
                    tpot_ms=float("nan"),
                    tokens_in=0,
                    tokens_out=0,
                    cost_usd=0.0,
                    finish_reason="error",
                    ok=False,
                    error=str(exc),
                )
            return Sample(
                request_idx=idx,
                arrival_ms=t_arrival,
                start_ms=t_arrival,
                ttft_ms=result.ttft_ms,
                total_ms=result.total_ms,
                tpot_ms=result.tpot_ms,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                cost_usd=result.cost_usd,
                finish_reason=result.finish_reason,
                ok=True,
                extra={"token_source": result.token_source},
            )

        nvml_interval_ms = int(context.extra.get("nvml_interval_ms", 50))
        rapl_interval_ms = int(context.extra.get("rapl_interval_ms", 100))
        bench = BenchmarkRun(
            driver=driver,
            workload=prompts,
            request_fn=_request_fn,
            convergence=gate,
            nvml_interval_ms=nvml_interval_ms,
            rapl_interval_ms=rapl_interval_ms,
        )
        raw = bench.execute()
        base_slos = _SLO_TEMPLATES.get(spec.slo_template, [])
        # Hardware-aware SLO scaling: the base numbers are anchored to an H100
        # (1.0x). Detect the host's hardware class and rescale before scoring
        # goodput so that lighter hardware isn't unfairly penalised.
        hw_fp = collect_hardware_fingerprint()
        hw_class: HardwareClass = classify(hw_fp)
        slos = scale_slos(base_slos, hw_class)
        resolved_template = format_resolved(slos)
        result: RunResult = raw.compute_metrics(slos=slos)

        # Dump raw samples alongside the envelope so failures stay diagnosable
        # (envelope.metrics is aggregated; without this you lose the per-request
        # error strings when ok_rate < 1).
        self._dump_samples(context, result.samples)

        envelope = self._build_envelope(
            spec,
            context,
            engine_version,
            result,
            dataset_hash=actual_dataset_hash,
            hardware_fingerprint=hw_fp,
            hw_class=hw_class,
            resolved_template=resolved_template,
        )
        signing_mode = context.extra.get("signing_mode", "dev")
        dev_key_path = context.extra.get("dev_key_path")
        if signing_mode == "dev":
            if not dev_key_path:
                msg = "dev signing requires context.extra['dev_key_path']"
                raise ValueError(msg)
            return sign_envelope(
                envelope,
                mode=SigningMode.DEV,
                dev_key_path=Path(str(dev_key_path)),
            )
        return sign_envelope(envelope, mode=SigningMode.KEYLESS)

    def _dump_samples(self, context: RunContext, samples: list[Sample]) -> None:
        """Write per-request samples to ``<output_dir>/samples-<ts>.jsonl``.

        We use a timestamp-suffixed filename rather than the content_hash
        because samples are written before the envelope is finalised and the
        content_hash isn't available yet. Errors here never block the run —
        diagnostics are best-effort.
        """
        try:
            out_dir = Path(context.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            path = out_dir / f"samples-{ts}.jsonl"
            with path.open("w", encoding="utf-8") as fp:
                for s in samples:
                    fp.write(
                        '{"request_idx":'
                        + str(s.request_idx)
                        + ',"ok":'
                        + ("true" if s.ok else "false")
                        + ',"ttft_ms":'
                        + _json_num(s.ttft_ms)
                        + ',"total_ms":'
                        + _json_num(s.total_ms)
                        + ',"tpot_ms":'
                        + _json_num(s.tpot_ms)
                        + ',"tokens_in":'
                        + str(s.tokens_in)
                        + ',"tokens_out":'
                        + str(s.tokens_out)
                        + ',"finish_reason":"'
                        + (s.finish_reason or "")
                        + '"'
                        + (',"error":' + _json_str(s.error) if s.error else "")
                        + "}\n"
                    )
        except OSError:
            pass  # diagnostics-only — never block the run

    def _custom_pricing_registry(
        self, context: RunContext
    ) -> dict[tuple[str, str], ModelPricing] | None:
        """Return a user-supplied pricing registry if ``RunContext.extra['prices_file']`` is set.

        Returns ``None`` (i.e. fall through to the bundled registry) when the
        key is absent, empty, or the file can't be loaded. Load failures are
        surfaced as warnings via stderr rather than aborting the run — cost
        is an optional metric.
        """
        raw = context.extra.get("prices_file")
        if not raw:
            return None
        path = Path(str(raw))
        try:
            return load_pricing(path)
        except (FileNotFoundError, ValueError, OSError) as exc:
            import sys

            print(
                f"warning: failed to load --prices-file {path}: {exc}",
                file=sys.stderr,
            )
            return None

    def _benchmarks_dir(self) -> Path:
        return Path(__file__).parent / "benchmarks"

    def _load_yaml(self, path: Path) -> BenchmarkSpec:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return BenchmarkSpec.model_validate(raw)

    def _workload_prompts(self, spec: BenchmarkSpec) -> list[str]:
        """Resolve the workload prompts for this benchmark spec.

        Phase 1 supports ``builtin://``, ``file://``, and best-effort ``hf://`` URIs;
        see :mod:`inferencebench_llm.datasets`.
        """
        return load_prompts(spec.dataset)

    def _build_driver(
        self,
        spec: BenchmarkSpec,
        context: RunContext | None = None,
    ) -> OpenLoopDriver | ClosedLoopDriver:
        """Build a driver from the spec, allowing CLI overrides via ``context.extra``.

        Recognised override keys (all optional):
            - ``rps`` (float): override open-loop arrival rate
            - ``concurrency`` (int): override closed-loop concurrency
            - ``duration_s`` (int): override the measurement window
            - ``driver_type`` ("open_loop"|"closed_loop"): switch driver kind
        """
        extra: dict[str, Any] = dict(context.extra) if context is not None else {}
        driver_type = str(extra.get("driver_type") or spec.driver.type)
        duration_s = int(extra.get("duration_s") or spec.driver.duration_s)

        if driver_type == "open_loop":
            rps_override = extra.get("rps")
            rps = (
                float(rps_override)
                if rps_override is not None
                else (spec.driver.rps[0] if spec.driver.rps else 1.0)
            )
            return OpenLoopDriver(
                mean_rps=rps,
                duration_s=duration_s,
                warmup_requests=spec.warmup.discard_runs,
            )
        concurrency_override = extra.get("concurrency")
        concurrency = (
            int(concurrency_override)
            if concurrency_override is not None
            else (spec.driver.concurrency[0] if spec.driver.concurrency else 1)
        )
        return ClosedLoopDriver(
            concurrency=concurrency,
            duration_s=duration_s,
            warmup_requests=spec.warmup.discard_runs,
        )

    def _build_envelope(
        self,
        spec: BenchmarkSpec,
        context: RunContext,
        engine_version: str,
        result: RunResult,
        *,
        dataset_hash: str | None = None,
        hardware_fingerprint: Any | None = None,
        hw_class: HardwareClass | None = None,
        resolved_template: str | None = None,
    ) -> Envelope:
        hw = (
            hardware_fingerprint
            if hardware_fingerprint is not None
            else (collect_hardware_fingerprint())
        )
        sw = collect_software_provenance()

        # Prefer the freshly-computed hash over the spec's declared one,
        # since the loader may have substituted the fallback prompt set.
        if dataset_hash:
            ds_hash = dataset_hash
        else:
            ds_hash = spec.dataset.hash
            if ds_hash.startswith("sha256:"):
                ds_hash = ds_hash[len("sha256:") :]

        metrics: dict[str, float | int | str | None] = {}
        if result.ttft_percentiles is not None:
            metrics["ttft_p50_ms"] = result.ttft_percentiles.p50
            metrics["ttft_p99_ms"] = result.ttft_percentiles.p99
        if result.tpot_percentiles is not None:
            metrics["tpot_p50_ms"] = result.tpot_percentiles.p50
            metrics["tpot_p99_ms"] = result.tpot_percentiles.p99
        if result.total_percentiles is not None:
            metrics["total_p50_ms"] = result.total_percentiles.p50
            metrics["total_p99_ms"] = result.total_percentiles.p99
        if result.goodput is not None:
            metrics["req_per_s_passing"] = result.goodput.req_per_s_passing
            metrics["req_per_s_all"] = result.goodput.req_per_s_all
            metrics["compliance_rate"] = result.goodput.compliance_rate
            metrics["ok_rate"] = result.goodput.ok_rate

        tokens_out_total = sum(s.tokens_out for s in result.samples if s.ok)
        if tokens_out_total and result.duration_s > 0:
            metrics["throughput_tok_per_s"] = tokens_out_total / result.duration_s

        cost_total = sum(s.cost_usd for s in result.samples if s.ok)
        # Only emit cost if the provider actually reported one. A 0.0 here is
        # not "free" — it's "no pricing data" and would be misleading on the
        # leaderboard. Self-hosted vLLM never reports cost; in that case we
        # fall back to the bundled pricing registry's cheapest blended rate,
        # tagging the source so consumers can tell measured from reference cost.
        if tokens_out_total and cost_total > 0:
            metrics["cost_usd_per_million_tokens"] = (cost_total / tokens_out_total) * 1e6
            metrics["cost_source"] = "provider"
        else:
            custom_registry = self._custom_pricing_registry(context)
            registry_cost = _registry_reference_cost(
                context.model_id, custom_registry=custom_registry
            )
            if registry_cost is not None:
                rate, provider = registry_cost
                metrics["cost_usd_per_million_tokens"] = rate
                metrics["cost_source"] = f"registry:{provider}"

        # Energy / power summary from telemetry
        energy = summarise_energy(
            result.gpu_telemetry,
            result.rapl_telemetry,
            result.samples,
            duration_s=result.duration_s,
        )
        if energy.gpu_power_avg_w > 0:
            metrics["power_avg_w"] = energy.gpu_power_avg_w
            metrics["power_peak_w"] = energy.gpu_power_peak_w
        if energy.total_energy_joules > 0:
            metrics["energy_joules_total"] = energy.total_energy_joules
        if not (energy.joules_per_token != energy.joules_per_token):  # not NaN
            metrics["joules_per_token"] = energy.joules_per_token

        # Record which hardware class drove the SLO thresholds and the resolved
        # numbers themselves — gives leaderboards a way to filter / explain why
        # a result either passed or failed its SLO budget.
        if hw_class is not None:
            metrics["slo_hardware_class"] = hw_class.key
        if resolved_template is not None:
            metrics["slo_template_resolved"] = resolved_template

        # Envelope.metrics requires at least one entry — guarantee with sample count
        if not metrics:
            metrics["n_samples"] = float(len(result.samples))

        builder = EnvelopeBuilder(
            suite_id=spec.benchmark_id,
            suite_version=spec.suite_version,
            model=ModelConfig(
                id=context.model_id,
                revision=context.model_revision,
                provider=context.engine_kind.value,
                endpoint_hash="0" * 64,
            ),
            engine=EngineConfig(
                name=context.engine_kind.value,
                version=engine_version or "unknown",
                config_hash="0" * 64,
            ),
            hardware_fingerprint=hw,
            software_provenance=sw,
            dataset=EnvDatasetSpec(id=spec.dataset.id, hash=ds_hash),
            seed=spec.dataset.sampling.seed,
            quantization=(
                Quantization(format=context.quantization_format)
                if context.quantization_format
                else None
            ),
            metrics=metrics,
            slo_template=spec.slo_template,
        )
        return builder.build()
