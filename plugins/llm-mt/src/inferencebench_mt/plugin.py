"""LLMMTPlugin — entry point for ``llm.mt`` benchmarks.

The fifth-modality sibling to the perf / quality / voice / embeddings plugins.
Implements the same contract (``list_benchmarks`` / ``get_benchmark`` /
``validate`` / ``run``) but its headline metric is translation accuracy —
chrF (character n-gram F-score) by default, with token-BLEU and exact-match
as alternates. Scoring is deterministic and dependency-free; see
:mod:`inferencebench_mt.scoring`.

The plugin drives a real :class:`ModelClient` per fixture row, constructs an
MT prompt (``"Translate from {src} to {tgt}: ..."``), and scores the model's
hypothesis against the bundled reference. Self-hosted OpenAI-compatible
endpoints (vLLM, SGLang) get the LiteLLM ``openai/`` routing prefix added
exactly once — same convention as the llm-quality plugin.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
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
from inferencebench.harness.metrics import EnergyReport, Percentiles, TelemetryWindow
from inferencebench_mt.schemas import BenchmarkSpec, EngineKind, RunContext
from inferencebench_mt.scoring import SCORERS

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
# OPENAI / COHERE here mean "provider-hosted endpoint" — base_url is optional.
_SELF_HOSTED_ENGINES = frozenset({EngineKind.VLLM, EngineKind.SGLANG})


def _fixtures_cache_root() -> Path:
    """Resolve the bench-fixtures cache root for ``fixtures://`` dataset URIs."""
    override = os.environ.get("BENCH_FIXTURES_ROOT")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "inferencebench" / "fixtures"


def _compute_fixture_hash(items: list[dict[str, str]]) -> str:
    """SHA-256 over the canonical-JSON-encoded fixture rows."""
    canonical = json.dumps(items, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_client(context: RunContext, *, timeout_s: float = 60.0) -> ModelClient:
    """Build a :class:`ModelClient` from the run context.

    OpenAI-compatible self-hosted servers (vLLM, SGLang) require the LiteLLM
    ``openai/<model>`` routing prefix; we add it here exactly once, stripping
    any user-supplied prefix first so a double ``openai/openai/...`` never
    reaches LiteLLM. Provider-hosted engines (``OPENAI``, ``COHERE``) leave
    the model id untouched.
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


def _build_prompt(source: str, source_lang: str, target_lang: str) -> str:
    """Construct the translation prompt for one fixture row."""
    return f"Translate from {source_lang} to {target_lang}:\n\n{source}\n\nTranslation:"


# Metrics this plugin is expected to emit. Consumed by ``bench coverage``.
EXPECTED_METRICS: tuple[str, ...] = (
    "chrf_mean",
    "chrf_p50",
    "chrf_p95",
    "ok_rate",
    "n_samples",
    "ttft_p50_ms",
    "total_p50_ms",
)


class LLMMTPlugin:
    """Plugin entry point. Registered via ``inferencebench.plugins`` entrypoint group."""

    suite_id = "llm.mt"
    version = "0.0.2"
    description = (
        "LLM machine-translation benchmarks (chrF / token-BLEU / exact-match on bundled fixtures)."
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
                f"{context.engine_kind.value} needs base_url (e.g. http://localhost:8000/v1)"
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

        samples, scores, telemetry = self._score_items(client, items, spec, scorer)

        # Best-effort diagnostic dump — never blocks the run on I/O errors.
        self._dump_samples(context, samples)

        envelope = self._build_envelope(
            spec,
            context,
            samples=samples,
            scores=scores,
            dataset_hash=fixture_hash,
            energy=telemetry.summarise(samples),
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
        spec: BenchmarkSpec,
        scorer: Callable[[str, str], float],
    ) -> tuple[list[Sample], list[float], TelemetryWindow]:
        """Iterate fixture items, call the model, score each hypothesis.

        MT runs are per-sentence and order-independent — no driver machinery
        is required. We still emit ``Sample`` objects so the envelope-building
        path stays uniform with the perf plugin.
        """
        samples: list[Sample] = []
        scores: list[float] = []
        telemetry = TelemetryWindow()
        with telemetry:
            for idx, item in enumerate(items):
                source = item["source"]
                reference = item["reference"]
                prompt = _build_prompt(source, spec.source_lang, spec.target_lang)
                t_arrival = time.perf_counter() * 1000.0
                try:
                    result: CompletionResult = client.complete(
                        prompt, stream=True, max_tokens=256
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

                score = float(scorer(reference, result.text))
                scores.append(score)

                sample_extra: dict[str, str | int | float | bool] = {
                    "domain": item.get("domain", ""),
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
        return samples, scores, telemetry

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
                        '{"request_idx":'
                        + str(s.request_idx)
                        + ',"ok":'
                        + ("true" if s.ok else "false")
                        + ',"ttft_ms":'
                        + _json_num(s.ttft_ms)
                        + ',"total_ms":'
                        + _json_num(s.total_ms)
                        + ',"tokens_in":'
                        + str(s.tokens_in)
                        + ',"tokens_out":'
                        + str(s.tokens_out)
                        + score_part
                        + ',"finish_reason":"'
                        + (s.finish_reason or "")
                        + '"'
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
        raw = spec.dataset.path
        if raw.startswith("fixtures://"):
            return _fixtures_cache_root() / f"{raw[len('fixtures://') :]}.jsonl"
        return self._datasets_dir() / raw

    def _load_yaml(self, path: Path) -> BenchmarkSpec:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return BenchmarkSpec.model_validate(raw)

    def _load_fixture(self, spec: BenchmarkSpec) -> list[dict[str, str]]:
        path = self._dataset_path(spec)
        if not path.exists():
            if spec.dataset.path.startswith("fixtures://"):
                key = spec.dataset.path[len("fixtures://") :]
                msg = f"fixture not cached: {path}. Run `bench fixtures fetch {key}` first."
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
                if "source" not in obj or "reference" not in obj:
                    continue
                items.append(
                    {
                        "source": str(obj["source"]),
                        "reference": str(obj["reference"]),
                        "domain": str(obj.get("domain", "")),
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
        energy: EnergyReport | None = None,
    ) -> Envelope:
        hw = collect_hardware_fingerprint()
        sw = collect_software_provenance()

        metrics: dict[str, float | int | str | None] = {}

        ok_samples = [s for s in samples if s.ok]
        n_ok = len(ok_samples)
        metrics["n_samples"] = float(len(samples))
        metrics["n_ok"] = float(n_ok)
        metrics["ok_rate"] = float(n_ok) / float(len(samples)) if samples else 0.0

        # Headline scoring metric — keyed by the spec's scoring strategy so
        # downstream ``bench diff`` knows the direction (all three MT scorers
        # are higher-is-better).
        if scores:
            mean_score = sum(scores) / len(scores)
            if spec.scoring == "exact_match":
                metrics["exact_match_rate"] = mean_score
            else:
                prefix = spec.scoring  # "chrf" | "bleu_token"
                # Keep the metric key short for BLEU (``bleu_mean``, not
                # ``bleu_token_mean``) so diff's _HIGHER_IS_BETTER policy and
                # the leaderboard column headers stay readable.
                key_prefix = "bleu" if prefix == "bleu_token" else prefix
                metrics[f"{key_prefix}_mean"] = mean_score
                if len(scores) >= 2:
                    pcts = Percentiles(scores, percentiles=(50.0, 95.0))
                    metrics[f"{key_prefix}_p50"] = pcts.p50
                    metrics[f"{key_prefix}_p95"] = pcts.p95
                else:
                    metrics[f"{key_prefix}_p50"] = mean_score
                    metrics[f"{key_prefix}_p95"] = mean_score

        # Latency aggregates — "quality at what cost" comparisons.
        ttft_vals = [s.ttft_ms for s in ok_samples if math.isfinite(s.ttft_ms)]
        total_vals = [s.total_ms for s in ok_samples if math.isfinite(s.total_ms)]
        if ttft_vals:
            metrics["ttft_p50_ms"] = Percentiles(ttft_vals).p50
        if total_vals:
            metrics["total_p50_ms"] = Percentiles(total_vals).p50

        tokens_out_total = sum(s.tokens_out for s in ok_samples)
        if tokens_out_total:
            metrics["tokens_out_total"] = float(tokens_out_total)

        # Energy / power summary from telemetry (None on plugins that haven't
        # threaded a TelemetryWindow through yet). Mirrors llm-inference.
        if energy is not None:
            if energy.gpu_power_avg_w > 0:
                metrics["power_avg_w"] = energy.gpu_power_avg_w
                metrics["power_peak_w"] = energy.gpu_power_peak_w
            if energy.total_energy_joules > 0:
                metrics["energy_joules_total"] = energy.total_energy_joules
                if energy.joules_per_token == energy.joules_per_token:  # not NaN
                    metrics["joules_per_token"] = energy.joules_per_token

        # Cost: only emit when the provider actually reported it. Self-hosted
        # vLLM / SGLang never do; the perf plugin's pricing-registry fallback
        # is intentionally NOT mirrored here — MT runs are cheap enough
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
