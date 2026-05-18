"""CodeGenerationPlugin — entry point for ``code.generation`` benchmarks.

HumanEval-style execution-based scoring: for each fixture row we send the
function-signature prompt to the model, extract Python code from the
response, execute it against the bundled unit tests in an isolated
subprocess, and aggregate per-task pass/fail into a ``pass_at_1`` headline.

Structural twin of :class:`inferencebench_quality.plugin.LLMQualityPlugin`:
plugin contract, ModelClient wiring, signing flow, and envelope shape all
mirror that module so cross-plugin tooling (summary / compare / diff /
audit) treats code envelopes the same as quality envelopes.

**Safety:** :func:`run` prints a yellow banner on every invocation as a
reminder that model output is executed locally. See the package README for
the full safety boundary; ``runner.py`` is best-effort defence-in-depth, not
a sandbox.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

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
    CompletionResult,
    ModelClient,
    Sample,
    collect_hardware_fingerprint,
    collect_software_provenance,
)
from inferencebench.harness.metrics import Percentiles
from inferencebench_code.runner import RunResult, run_unit_tests
from inferencebench_code.schemas import BenchmarkSpec, EngineKind, RunContext
from inferencebench_code.scoring import extract_python_code

_SAFETY_BANNER = (
    "\033[33m"  # yellow
    "WARNING: code.generation executes model-generated Python locally. "
    "The runner is best-effort (python -I subprocess with timeout + forbidden-import "
    "pre-scan) — NOT a sandbox. Use only with trusted models and bundled fixtures."
    "\033[0m"
)

# Engines that require ``base_url`` (self-hosted OpenAI-compatible servers).
_SELF_HOSTED_ENGINES = frozenset({EngineKind.VLLM, EngineKind.SGLANG})


def _fixtures_cache_root() -> Path:
    """Resolve the bench-fixtures cache root for ``fixtures://`` dataset URIs."""
    override = os.environ.get("BENCH_FIXTURES_ROOT")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "inferencebench" / "fixtures"


def _json_num(v: float) -> str:
    """JSON-safe numeric encoder: NaN/inf become null."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "null"
    return repr(v)


def _compute_fixture_hash(items: list[dict[str, str]]) -> str:
    """SHA-256 over the canonical-JSON-encoded fixture rows."""
    canonical = json.dumps(items, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_client(context: RunContext, *, timeout_s: float = 60.0) -> ModelClient:
    """Build a :class:`ModelClient` from the run context.

    Same routing convention as the quality plugin: self-hosted OpenAI-
    compatible engines (vLLM, SGLang) get the ``openai/`` LiteLLM prefix
    applied exactly once; provider-hosted engines (OpenAI, Anthropic)
    leave the model id untouched.
    """
    model_id = context.model_id
    api_key: str | None
    if context.engine_kind in _SELF_HOSTED_ENGINES:
        if model_id.startswith("openai/"):
            model_id = model_id[len("openai/") :]
        model_id = f"openai/{model_id}"
        api_key = context.api_key or "EMPTY"
    else:
        api_key = context.api_key or None
    return ModelClient(
        model=model_id,
        api_key=api_key,
        base_url=context.base_url or None,
        timeout_s=timeout_s,
    )


def _ensure_signature_present(prompt: str, generated: str) -> str:
    """Prepend ``prompt`` (function signature + docstring) to ``generated`` if missing.

    Some models reply with the function body only; we glue the signature back
    on so the subprocess can call the named entry point. We detect "missing
    signature" by looking for the first non-empty line of ``prompt`` (the
    ``def ...`` line) in ``generated``; if it isn't present we prepend.
    """
    head = ""
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("def "):
            head = stripped
            break
    if head and head not in generated:
        return prompt.rstrip() + "\n" + generated
    return generated


# Metrics this plugin is expected to emit. Consumed by ``bench coverage``.
EXPECTED_METRICS: tuple[str, ...] = (
    "pass_at_1",
    "pass_at_1_p05",
    "pass_at_1_p50",
    "pass_at_1_p95",
    "timeout_rate",
    "ok_rate",
    "n_samples",
    "ttft_p50_ms",
    "total_p50_ms",
)


class CodeGenerationPlugin:
    """Plugin entry point. Registered via ``inferencebench.plugins`` entrypoint group."""

    suite_id = "code.generation"
    version = "0.0.2"
    description = (
        "Code-generation benchmarks (HumanEval-style execution-based scoring; "
        "executes model output locally — see README for the safety boundary)."
    )

    # ----------------------------------------------------------- benchmarks #
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

    # ------------------------------------------------------------- validate #
    def validate(self, spec: BenchmarkSpec, context: RunContext) -> list[str]:
        warnings: list[str] = []
        if not context.model_id:
            warnings.append("model_id is empty")
        if context.engine_kind in _SELF_HOSTED_ENGINES and not context.base_url:
            warnings.append(
                f"{context.engine_kind.value} needs base_url "
                "(e.g. http://localhost:8000/v1)"
            )
        if not self._dataset_path(spec).exists():
            warnings.append(f"fixture not found: {spec.dataset.path}")
        return warnings

    # ------------------------------------------------------------------ run #
    def run(self, spec: BenchmarkSpec, context: RunContext) -> Envelope:
        """Execute the benchmark and return a SIGNED envelope.

        Prints the safety banner to stderr on every call.
        """
        print(_SAFETY_BANNER, file=sys.stderr, flush=True)

        client = _build_client(context)
        items = self._load_fixture(spec)
        fixture_hash = _compute_fixture_hash(items)

        samples, passed_flags, timeout_flags = self._score_items(
            client, items, timeout_s=spec.timeout_s
        )

        # Best-effort diagnostic dump — never blocks the run on I/O errors.
        self._dump_samples(context, samples)

        envelope = self._build_envelope(
            spec,
            context,
            samples=samples,
            passed_flags=passed_flags,
            timeout_flags=timeout_flags,
            dataset_hash=fixture_hash,
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

    # -------------------------------------------------------- core scoring #
    def _score_items(
        self,
        client: ModelClient,
        items: list[dict[str, str]],
        *,
        timeout_s: float,
    ) -> tuple[list[Sample], list[bool], list[bool]]:
        """Iterate fixture items sequentially, scoring each model response.

        For each fixture row:

        1. Send the prompt to the model (max_tokens=512, streamed).
        2. Extract a Python block from the response.
        3. Glue the function signature back on if the model omitted it.
        4. Execute the result against the row's unit tests with a
           ``timeout_s`` wall clock.

        Records a :class:`Sample` per row (including ``passed``,
        ``duration_s``, ``timeout``, and an ``error_summary`` string) and
        returns parallel lists of per-row ``passed`` and ``timeout`` flags
        for the aggregator.
        """
        samples: list[Sample] = []
        passed_flags: list[bool] = []
        timeout_flags: list[bool] = []

        for idx, item in enumerate(items):
            prompt = item["prompt"]
            tests = item["tests"]
            t_arrival = time.perf_counter() * 1000.0
            try:
                result: CompletionResult = client.complete(
                    prompt, stream=True, max_tokens=512
                )
            except Exception as exc:
                samples.append(
                    Sample(
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
                )
                passed_flags.append(False)
                timeout_flags.append(False)
                continue

            extracted = extract_python_code(result.text)
            solution = _ensure_signature_present(prompt, extracted)
            run_result: RunResult = run_unit_tests(
                solution, tests, timeout_s=timeout_s
            )
            passed_flags.append(run_result.passed)
            timeout_flags.append(run_result.timeout)

            error_summary = self._summarize_error(run_result)
            sample_extra: dict[str, str | int | float | bool] = {
                "task_id": item.get("task_id", ""),
                "passed": run_result.passed,
                "duration_s": run_result.duration_s,
                "timeout_flag": run_result.timeout,
            }
            if error_summary:
                sample_extra["error_summary"] = error_summary

            samples.append(
                Sample(
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
                    extra=sample_extra,
                )
            )
        return samples, passed_flags, timeout_flags

    @staticmethod
    def _summarize_error(result: RunResult) -> str:
        """Distil ``RunResult`` into a short ``error_summary`` string.

        Empty when the run passed. Otherwise: ``"timeout"`` for wall-clock
        kills, ``"forbidden_import: <name>"`` for pre-scan refusals (the
        stderr is already shaped that way by the runner), and the last
        line of stderr (typically the AssertionError or Exception class)
        for normal failures.
        """
        if result.passed:
            return ""
        if result.timeout:
            return "timeout"
        if result.stderr.startswith("forbidden_import"):
            return result.stderr.split("\n", 1)[0]
        last_lines = [
            line.strip() for line in result.stderr.strip().splitlines() if line.strip()
        ]
        return last_lines[-1] if last_lines else "exit_nonzero"

    # ------------------------------------------------------------ samples #
    def _dump_samples(self, context: RunContext, samples: list[Sample]) -> None:
        """Write per-task samples (incl. pass flag + duration) to ``samples-<ts>.jsonl``.

        Mirrors the llm-quality plugin's diagnostic dump — failures here
        never block the run.
        """
        try:
            out_dir = Path(context.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            path = out_dir / f"samples-{ts}.jsonl"
            with path.open("w", encoding="utf-8") as fp:
                for s in samples:
                    extra = s.extra or {}
                    passed = bool(extra.get("passed", False))
                    duration = extra.get("duration_s")
                    timeout_flag = bool(extra.get("timeout_flag", False))
                    duration_part = (
                        ',"duration_s":' + _json_num(float(duration))
                        if isinstance(duration, (int, float))
                        else ""
                    )
                    fp.write(
                        '{"request_idx":' + str(s.request_idx)
                        + ',"ok":' + ("true" if s.ok else "false")
                        + ',"passed":' + ("true" if passed else "false")
                        + ',"timeout":' + ("true" if timeout_flag else "false")
                        + ',"ttft_ms":' + _json_num(s.ttft_ms)
                        + ',"total_ms":' + _json_num(s.total_ms)
                        + ',"tokens_out":' + str(s.tokens_out)
                        + duration_part
                        + ',"finish_reason":"' + (s.finish_reason or "") + '"'
                        + "}\n"
                    )
        except OSError:
            pass  # diagnostics-only — never block the run

    # ---------------------------------------------------------- file paths #
    def _benchmarks_dir(self) -> Path:
        return Path(__file__).parent / "benchmarks"

    def _datasets_dir(self) -> Path:
        return Path(__file__).parent / "datasets"

    def _dataset_path(self, spec: BenchmarkSpec) -> Path:
        raw = spec.dataset.path
        if raw.startswith("fixtures://"):
            return _fixtures_cache_root() / f"{raw[len('fixtures://'):]}.jsonl"
        return self._datasets_dir() / raw

    def _load_yaml(self, path: Path) -> BenchmarkSpec:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return BenchmarkSpec.model_validate(raw)

    def _load_fixture(self, spec: BenchmarkSpec) -> list[dict[str, str]]:
        path = self._dataset_path(spec)
        if not path.exists():
            if spec.dataset.path.startswith("fixtures://"):
                key = spec.dataset.path[len("fixtures://") :]
                msg = (
                    f"fixture not cached: {path}. "
                    f"Run `bench fixtures fetch {key}` first."
                )
                raise FileNotFoundError(msg)
            msg = f"fixture not found: {path}"
            raise FileNotFoundError(msg)
        items: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                if "task_id" not in obj or "prompt" not in obj or "tests" not in obj:
                    continue
                items.append(
                    {
                        "task_id": str(obj["task_id"]),
                        "prompt": str(obj["prompt"]),
                        "tests": str(obj["tests"]),
                        "canonical_solution": str(obj.get("canonical_solution", "")),
                        "entry_point": str(obj.get("entry_point", "")),
                    }
                )
        if not items:
            msg = f"fixture is empty: {path}"
            raise ValueError(msg)
        return items

    # ---------------------------------------------------------- envelope #
    def _build_envelope(
        self,
        spec: BenchmarkSpec,
        context: RunContext,
        *,
        samples: list[Sample],
        passed_flags: list[bool],
        timeout_flags: list[bool],
        dataset_hash: str,
    ) -> Envelope:
        hw = collect_hardware_fingerprint()
        sw = collect_software_provenance()

        metrics: dict[str, float | int | str | None] = {}

        ok_samples = [s for s in samples if s.ok]
        n_total = len(samples)
        metrics["n_samples"] = float(n_total)
        metrics["n_ok"] = float(len(ok_samples))
        metrics["ok_rate"] = float(len(ok_samples)) / float(n_total) if n_total else 0.0

        if passed_flags:
            scores = [1.0 if p else 0.0 for p in passed_flags]
            mean_pass = sum(scores) / len(scores)
            metrics["pass_at_1"] = mean_pass
            if len(scores) >= 2:
                pcts = Percentiles(scores, percentiles=(5.0, 50.0, 95.0))
                metrics["pass_at_1_p05"] = pcts.p5
                metrics["pass_at_1_p50"] = pcts.p50
                metrics["pass_at_1_p95"] = pcts.p95
            else:
                metrics["pass_at_1_p05"] = mean_pass
                metrics["pass_at_1_p50"] = mean_pass
                metrics["pass_at_1_p95"] = mean_pass

        if timeout_flags:
            metrics["timeout_rate"] = float(sum(timeout_flags)) / float(
                len(timeout_flags)
            )

        # Latency aggregates — useful for "quality at what cost" comparisons.
        ttft_vals = [s.ttft_ms for s in ok_samples if math.isfinite(s.ttft_ms)]
        total_vals = [s.total_ms for s in ok_samples if math.isfinite(s.total_ms)]
        if ttft_vals:
            metrics["ttft_p50_ms"] = Percentiles(ttft_vals).p50
        if total_vals:
            metrics["total_p50_ms"] = Percentiles(total_vals).p50

        tokens_out_total = sum(s.tokens_out for s in ok_samples)
        if tokens_out_total:
            metrics["tokens_out_total"] = float(tokens_out_total)

        cost_total = sum(s.cost_usd for s in ok_samples)
        if tokens_out_total and cost_total > 0:
            metrics["cost_usd_per_million_tokens"] = (
                cost_total / tokens_out_total
            ) * 1e6
            metrics["cost_source"] = "provider"

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
                version=context.engine_version or "unknown",
                config_hash="0" * 64,
            ),
            hardware_fingerprint=hw,
            software_provenance=sw,
            dataset=EnvDatasetSpec(id=spec.dataset.id, hash=dataset_hash),
            seed=0,
            quantization=(
                Quantization(format=context.quantization_format)
                if context.quantization_format
                else None
            ),
            metrics=metrics,
            slo_template=spec.slo_template,
        )
        return builder.build()
