"""LLMQualityPlugin — entry point for ``llm.quality`` benchmarks.

The skeleton sibling to the ``llm.inference`` perf plugin. Implements the same
contract (``list_benchmarks`` / ``get_benchmark`` / ``validate`` / ``run``) but
emits accuracy as its headline metric instead of throughput.

Scoring is deterministic exact-match / substring-match / token-F1 against
bundled fixture answers — see :mod:`inferencebench_quality.scoring`.
LLM-as-judge is deferred to a later revision; the contract surface should be
stable enough that swapping the scorer is a one-file change.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

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
from inferencebench_quality.schemas import BenchmarkSpec, EngineKind, RunContext
from inferencebench_quality.scoring import SCORERS

if TYPE_CHECKING:
    from collections.abc import Callable


def _json_num(v: float) -> str:
    """JSON-safe numeric encoder: NaN/inf become null."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "null"
    return repr(v)


def _json_str(v: str | None) -> str:
    """JSON-safe string encoder."""
    return json.dumps(v if v is not None else "")


# Engines that require ``base_url`` (self-hosted OpenAI-compatible servers).
# OPENAI here means "provider-hosted OpenAI endpoint" — base_url is optional.
_SELF_HOSTED_ENGINES = frozenset({EngineKind.VLLM, EngineKind.SGLANG})


def _compute_fixture_hash(items: list[dict[str, str]]) -> str:
    """SHA-256 over the canonical-JSON-encoded fixture rows."""
    canonical = json.dumps(items, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_client(context: RunContext, *, timeout_s: float = 60.0) -> ModelClient:
    """Build a :class:`ModelClient` from the run context.

    The engine flavour only changes which env-var the API key defaults from and
    whether ``base_url`` is required. The wire protocol is OpenAI-compatible
    for every engine kind we support here.
    """
    return ModelClient(
        model=context.model_id,
        api_key=context.api_key or None,
        base_url=context.base_url or None,
        timeout_s=timeout_s,
    )


class LLMQualityPlugin:
    """Plugin entry point. Registered via ``inferencebench.plugins`` entrypoint group."""

    suite_id = "llm.quality"
    version = "0.0.0"
    description = "LLM quality benchmarks (accuracy on bundled fixtures; LLM-as-judge deferred)."

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
        client = _build_client(context)
        items = self._load_fixture(spec)
        fixture_hash = _compute_fixture_hash(items)
        scorer = SCORERS[spec.scoring]

        samples, scores = self._score_items(client, items, scorer)

        # Best-effort diagnostic dump — never blocks the run on I/O errors.
        self._dump_samples(context, samples)

        envelope = self._build_envelope(
            spec,
            context,
            samples=samples,
            scores=scores,
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
        scorer: Callable[[str, str], float],
    ) -> tuple[list[Sample], list[float]]:
        """Iterate fixture items sequentially, scoring each model response.

        Quality runs are per-question and order-independent — no driver
        machinery is required. We still emit ``Sample`` objects so the
        envelope-building path stays uniform with the perf plugin.
        """
        samples: list[Sample] = []
        scores: list[float] = []
        for idx, item in enumerate(items):
            question = item["question"]
            reference = item["answer"]
            t_arrival = time.perf_counter() * 1000.0
            try:
                result: CompletionResult = client.complete(
                    question, stream=True, max_tokens=128
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
                continue
            score = float(scorer(result.text, reference))
            scores.append(score)
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
                    extra={
                        "score": score,
                        "category": item.get("category", ""),
                    },
                )
            )
        return samples, scores

    # ------------------------------------------------------------ samples #
    def _dump_samples(self, context: RunContext, samples: list[Sample]) -> None:
        """Write per-request samples (incl. score) to ``<output_dir>/samples-<ts>.jsonl``.

        Mirrors the llm-inference plugin's diagnostic dump — failures here
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

    # ---------------------------------------------------------- file paths #
    def _benchmarks_dir(self) -> Path:
        return Path(__file__).parent / "benchmarks"

    def _datasets_dir(self) -> Path:
        return Path(__file__).parent / "datasets"

    def _dataset_path(self, spec: BenchmarkSpec) -> Path:
        return self._datasets_dir() / spec.dataset.path

    def _load_yaml(self, path: Path) -> BenchmarkSpec:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return BenchmarkSpec.model_validate(raw)

    def _load_fixture(self, spec: BenchmarkSpec) -> list[dict[str, str]]:
        path = self._dataset_path(spec)
        if not path.exists():
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
                if "question" not in obj or "answer" not in obj:
                    continue
                items.append(
                    {
                        "question": str(obj["question"]),
                        "answer": str(obj["answer"]),
                        "category": str(obj.get("category", "")),
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
    ) -> Envelope:
        hw = collect_hardware_fingerprint()
        sw = collect_software_provenance()

        metrics: dict[str, float | int | str | None] = {}

        ok_samples = [s for s in samples if s.ok]
        n_ok = len(ok_samples)
        metrics["n_samples"] = float(len(samples))
        metrics["n_ok"] = float(n_ok)
        metrics["ok_rate"] = float(n_ok) / float(len(samples)) if samples else 0.0

        if scores:
            mean_acc = sum(scores) / len(scores)
            metrics["accuracy"] = mean_acc
            # Bootstrap CI on per-sample scores via the existing Percentiles
            # machinery. p05/p50/p95 are the quantile-on-resampled-mean band
            # that consumers like ``bench diff`` already know how to read.
            if len(scores) >= 2:
                pcts = Percentiles(scores, percentiles=(5.0, 50.0, 95.0))
                metrics["accuracy_p05"] = pcts.p5
                metrics["accuracy_p50"] = pcts.p50
                metrics["accuracy_p95"] = pcts.p95
            else:
                metrics["accuracy_p05"] = mean_acc
                metrics["accuracy_p50"] = mean_acc
                metrics["accuracy_p95"] = mean_acc

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

        # Cost: only emit when the provider actually reported it. Self-hosted
        # vLLM / SGLang never do; the perf plugin's pricing-registry fallback
        # is intentionally NOT mirrored here — quality runs are cheap enough
        # that a missing-cost row is more honest than an estimated one.
        cost_total = sum(s.cost_usd for s in ok_samples)
        if tokens_out_total and cost_total > 0:
            metrics["cost_usd_per_million_tokens"] = (cost_total / tokens_out_total) * 1e6
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
