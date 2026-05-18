"""VisionUnderstandingPlugin — entry point for ``vision.understanding`` benchmarks.

Mirrors the llm-quality plugin shape but sends image-bearing multimodal
requests via :class:`MultimodalClient`. Scoring is deterministic against
bundled fixture answers (``exact_match`` / ``substring_match``) with
LLM-as-judge available for free-form responses.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable
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
from inferencebench_vision.multimodal_client import MultimodalClient
from inferencebench_vision.schemas import BenchmarkSpec, EngineKind, RunContext
from inferencebench_vision.scoring import SCORERS, ScoreContext

_DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"


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


def _json_str(v: str | None) -> str:
    """JSON-safe string encoder."""
    return json.dumps(v if v is not None else "")


# Engines that require ``base_url`` (self-hosted OpenAI-compatible servers).
# OPENAI / ANTHROPIC are provider-hosted; base_url is optional for them.
_SELF_HOSTED_ENGINES = frozenset({EngineKind.VLLM, EngineKind.SGLANG})


def _compute_fixture_hash(items: list[dict[str, str]]) -> str:
    """SHA-256 over the canonical-JSON-encoded fixture rows."""
    canonical = json.dumps(items, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_transport(context: RunContext, *, timeout_s: float = 120.0) -> ModelClient:
    """Build the underlying text :class:`ModelClient` used as multimodal transport.

    Self-hosted OpenAI-compatible servers (vLLM, SGLang) need the
    ``openai/<model>`` LiteLLM routing prefix; provider-hosted endpoints
    (``OPENAI``, ``ANTHROPIC``) leave the model id untouched so LiteLLM's
    own provider routing kicks in.
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


# Metrics this plugin is expected to emit. Consumed by ``bench coverage``.
EXPECTED_METRICS: tuple[str, ...] = (
    "accuracy",
    "accuracy_p05",
    "accuracy_p50",
    "accuracy_p95",
    "n_samples",
    "n_ok",
    "ok_rate",
    "ttft_p50_ms",
    "total_p50_ms",
    "tokens_out_total",
)


class VisionUnderstandingPlugin:
    """Plugin entry point. Registered via ``inferencebench.plugins`` entrypoint group."""

    suite_id = "vision.understanding"
    version = "0.0.2"
    description = (
        "Vision-language understanding benchmarks (accuracy on bundled image+question "
        "fixtures via multimodal chat completions)."
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
        """Execute the benchmark and return a SIGNED envelope."""
        transport = _build_transport(context)
        client = MultimodalClient(transport)
        items = self._load_fixture(spec)
        fixture_hash = _compute_fixture_hash(items)
        scorer = SCORERS[spec.scoring]

        judge_client: ModelClient | None = None
        if spec.scoring == "judge_llm":
            judge_client = self._build_judge_client(spec, context)

        judge_errors: list[str] = []
        judge_cost_usd: list[float] = []
        samples, scores = self._score_items(
            client,
            items,
            scorer,
            judge_client=judge_client,
            judge_errors=judge_errors,
            judge_cost_usd=judge_cost_usd,
            is_judge=spec.scoring == "judge_llm",
        )

        # Best-effort diagnostic dump — never blocks the run on I/O errors.
        self._dump_samples(context, samples)

        envelope = self._build_envelope(
            spec,
            context,
            samples=samples,
            scores=scores,
            dataset_hash=fixture_hash,
            judge_errors=judge_errors if spec.scoring == "judge_llm" else None,
            judge_cost_usd=judge_cost_usd if spec.scoring == "judge_llm" else None,
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
        client: MultimodalClient,
        items: list[dict[str, str]],
        scorer: Callable[[ScoreContext], float],
        *,
        judge_client: ModelClient | None = None,
        judge_errors: list[str] | None = None,
        judge_cost_usd: list[float] | None = None,
        is_judge: bool = False,
    ) -> tuple[list[Sample], list[float]]:
        """Iterate fixture items sequentially, scoring each multimodal response.

        Rows whose ``image_path`` doesn't resolve on disk are skipped with a
        ``finish_reason='missing_image'`` sample (ok=False) so the envelope's
        ``ok_rate`` truthfully reflects the input quality.
        """
        samples: list[Sample] = []
        scores: list[float] = []
        _judge_errors = judge_errors if judge_errors is not None else []
        _judge_cost = judge_cost_usd if judge_cost_usd is not None else []
        datasets_dir = self._datasets_dir()

        for idx, item in enumerate(items):
            question = item["question"]
            reference = item["answer"]
            image_rel = item["image_path"]
            image_path = (datasets_dir / image_rel).resolve()
            t_arrival = time.perf_counter() * 1000.0

            if not image_path.exists():
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
                        finish_reason="missing_image",
                        ok=False,
                        error=f"image not found: {image_rel}",
                    )
                )
                continue

            try:
                result: CompletionResult = client.complete_multimodal(
                    image_path, question, max_tokens=128
                )
            except Exception as exc:  # network errors are per-request
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
                continue

            if is_judge:
                score_ctx = ScoreContext(
                    reference=reference,
                    hypothesis=result.text,
                    question=question,
                    judge_client=judge_client,
                    judge_errors=_judge_errors,
                    judge_cost_usd=_judge_cost,
                )
            else:
                score_ctx = ScoreContext(
                    reference=reference,
                    hypothesis=result.text,
                    question=question,
                )
            score = float(scorer(score_ctx))
            scores.append(score)

            sample_extra: dict[str, str | int | float | bool] = {
                "task": item.get("task", ""),
                "score": score,
            }
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
        return samples, scores

    # ------------------------------------------------------------ samples #
    def _dump_samples(self, context: RunContext, samples: list[Sample]) -> None:
        """Write per-request samples (incl. score) to ``<output_dir>/samples-<ts>.jsonl``.

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
                    score = s.extra.get("score") if s.extra else None
                    score_part = (
                        ',"score":' + _json_num(float(score))
                        if isinstance(score, (int, float))
                        else ""
                    )
                    fp.write(
                        '{"request_idx":' + str(s.request_idx)
                        + ',"ok":' + ("true" if s.ok else "false")
                        + ',"ttft_ms":' + _json_num(s.ttft_ms)
                        + ',"total_ms":' + _json_num(s.total_ms)
                        + ',"tokens_in":' + str(s.tokens_in)
                        + ',"tokens_out":' + str(s.tokens_out)
                        + score_part
                        + ',"finish_reason":"' + (s.finish_reason or "") + '"'
                        + (',"error":' + _json_str(s.error) if s.error else "")
                        + "}\n"
                    )
        except OSError:
            pass  # diagnostics-only — never block the run

    # ---------------------------------------------------------- judge #
    def _build_judge_client(
        self, spec: BenchmarkSpec, context: RunContext
    ) -> ModelClient:
        """Construct the (text-only) judge :class:`ModelClient`.

        Model id precedence: spec.judge_model > extra['judge_model'] >
        ``openai/gpt-4o-mini``. The judge reuses the run-context engine kind,
        base_url and api_key; most graders are OpenAI-compatible endpoints.
        """
        judge_model = spec.judge_model
        if not judge_model:
            extra_model = context.extra.get("judge_model")
            if isinstance(extra_model, str) and extra_model:
                judge_model = extra_model
        if not judge_model:
            judge_model = _DEFAULT_JUDGE_MODEL
        judge_context = context.model_copy(update={"model_id": judge_model})
        return _build_transport(judge_context)

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
                if (
                    "image_path" not in obj
                    or "question" not in obj
                    or "answer" not in obj
                ):
                    continue
                items.append(
                    {
                        "image_path": str(obj["image_path"]),
                        "question": str(obj["question"]),
                        "answer": str(obj["answer"]),
                        "task": str(obj.get("task", "")),
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
        scores: list[float],
        dataset_hash: str,
        judge_errors: list[str] | None = None,
        judge_cost_usd: list[float] | None = None,
    ) -> Envelope:
        hw = collect_hardware_fingerprint()
        sw = collect_software_provenance()

        metrics: dict[str, float | int | str | None] = {}

        ok_samples = [s for s in samples if s.ok]
        n_ok = len(ok_samples)
        metrics["n_samples"] = float(len(samples))
        metrics["n_ok"] = float(n_ok)
        metrics["ok_rate"] = (
            float(n_ok) / float(len(samples)) if samples else 0.0
        )

        if scores:
            mean_acc = sum(scores) / len(scores)
            metrics["accuracy"] = mean_acc
            if len(scores) >= 2:
                pcts = Percentiles(scores, percentiles=(5.0, 50.0, 95.0))
                metrics["accuracy_p05"] = pcts.p5
                metrics["accuracy_p50"] = pcts.p50
                metrics["accuracy_p95"] = pcts.p95
            else:
                metrics["accuracy_p05"] = mean_acc
                metrics["accuracy_p50"] = mean_acc
                metrics["accuracy_p95"] = mean_acc

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
        judge_cost_total = sum(judge_cost_usd) if judge_cost_usd else 0.0
        combined_cost = cost_total + judge_cost_total
        if tokens_out_total and combined_cost > 0:
            metrics["cost_usd_per_million_tokens"] = (
                combined_cost / tokens_out_total
            ) * 1e6
            metrics["cost_source"] = "provider"
            if judge_cost_total > 0:
                metrics["judge_cost_usd_total"] = judge_cost_total

        if judge_errors is not None:
            metrics["judge_errors"] = float(len(judge_errors))

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
