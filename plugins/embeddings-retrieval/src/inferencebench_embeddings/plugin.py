"""EmbeddingsRetrievalPlugin — entry point for ``embeddings.retrieval`` benchmarks.

Phase-2-quality skeleton: produces a real signed envelope by deterministically
ranking the corpus per query via ``sha256(query + doc_id)`` sort, then scoring
the top-k against the fixture's relevant set with recall@5 / mrr@10 / nDCG@10.

Future revisions wire a real embedding model call into :meth:`_rank_corpus`;
the rest of the pipeline (signing, aggregation, sample dump) is production-shaped.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
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
    Sample,
    collect_hardware_fingerprint,
    collect_software_provenance,
)
from inferencebench.harness.metrics import EnergyReport, Percentiles, TelemetryWindow
from inferencebench_embeddings.schemas import BenchmarkSpec, EngineKind, RunContext
from inferencebench_embeddings.scoring import METRICS


def _json_num(v: float) -> str:
    """JSON-safe numeric encoder: NaN/inf become null."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "null"
    return repr(v)


# Engines that require ``base_url`` (self-hosted TEI servers).
_SELF_HOSTED_ENGINES = frozenset({EngineKind.TEI})


def _fixtures_cache_root() -> Path:
    """Resolve the bench-fixtures cache root for ``fixtures://`` dataset URIs."""
    override = os.environ.get("BENCH_FIXTURES_ROOT")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "inferencebench" / "fixtures"


def _compute_fixture_hash(queries: list[dict[str, object]], corpus: list[dict[str, str]]) -> str:
    """SHA-256 over the canonical-JSON-encoded queries + corpus."""
    canonical = json.dumps(
        {"queries": queries, "corpus": corpus},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _rank_corpus(query: str, corpus_ids: list[str]) -> list[str]:
    """Deterministically rank corpus doc-ids for a query.

    Sorts by ``sha256(query + doc_id).hexdigest()`` so the ranking is
    reproducible across machines and Python versions but uncorrelated with
    actual relevance — exactly what we want for a contract-validation
    skeleton that should produce a non-degenerate metric in [0, 1].
    """

    def _key(doc_id: str) -> str:
        h = hashlib.sha256()
        h.update(query.encode("utf-8"))
        h.update(b"\x00")
        h.update(doc_id.encode("utf-8"))
        return h.hexdigest()

    return sorted(corpus_ids, key=_key)


# Metrics this plugin is expected to emit. Consumed by ``bench coverage``.
EXPECTED_METRICS: tuple[str, ...] = (
    "recall_at_5_mean",
    "recall_at_5_p50",
    "recall_at_5_p95",
    "mrr_at_10_mean",
    "ndcg_at_10_mean",
    "ok_rate",
    "n_queries",
    "corpus_size",
)


class EmbeddingsRetrievalPlugin:
    """Plugin entry point. Registered via ``inferencebench.plugins`` entrypoint group."""

    suite_id = "embeddings.retrieval"
    version = "0.0.0"
    description = (
        "Embeddings retrieval benchmarks (deterministic hash ranking on bundled "
        "corpora; real embedding-model invocation deferred)."
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
                f"{context.engine_kind.value} needs base_url (e.g. http://localhost:8080)"
            )
        if not self._queries_path(spec).exists():
            warnings.append(f"queries fixture not found: {spec.dataset.path}")
        if not self._corpus_path(spec).exists():
            warnings.append(f"corpus fixture not found: {spec.dataset.corpus_path}")
        return warnings

    # ------------------------------------------------------------------ run #
    def run(self, spec: BenchmarkSpec, context: RunContext) -> Envelope:
        """Execute the benchmark and return a SIGNED envelope."""
        queries = self._load_queries(spec)
        corpus = self._load_corpus(spec)
        fixture_hash = _compute_fixture_hash(queries, corpus)
        k, scorer = METRICS[spec.metric]

        corpus_ids = [doc["doc_id"] for doc in corpus]

        samples: list[Sample] = []
        scores: list[float] = []
        telemetry = TelemetryWindow()
        with telemetry:
            for idx, q in enumerate(queries):
                query_text = str(q["query"])
                raw_relevant = q.get("relevant_doc_ids") or []
                # ``_load_queries`` guarantees this is a list[str], but mypy sees
                # the dict as ``dict[str, object]`` — narrow explicitly.
                assert isinstance(raw_relevant, list)
                relevant = [str(x) for x in raw_relevant]
                t_arrival = time.perf_counter() * 1000.0
                t_start = time.perf_counter()
                ranking = _rank_corpus(query_text, corpus_ids)
                score = float(scorer(ranking, relevant, k))
                total_ms = (time.perf_counter() - t_start) * 1000.0
                scores.append(score)
                samples.append(
                    Sample(
                        request_idx=idx,
                        arrival_ms=t_arrival,
                        start_ms=t_arrival,
                        ttft_ms=float("nan"),
                        total_ms=total_ms,
                        tpot_ms=float("nan"),
                        tokens_in=len(query_text.split()),
                        tokens_out=k,
                        cost_usd=0.0,
                        finish_reason="stop",
                        ok=True,
                        extra={
                            "score": score,
                            "k": k,
                            "n_relevant": len(relevant),
                            "topk": ranking[:k],
                        },
                    )
                )

        # Best-effort diagnostic dump — never blocks the run on I/O errors.
        self._dump_samples(context, samples)

        envelope = self._build_envelope(
            spec,
            context,
            samples=samples,
            scores=scores,
            corpus_size=len(corpus),
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

    # ------------------------------------------------------------ samples #
    def _dump_samples(self, context: RunContext, samples: list[Sample]) -> None:
        """Write per-query samples (incl. score) to ``<output_dir>/samples-<ts>.jsonl``.

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
                        + "}\n"
                    )
        except OSError:
            pass  # diagnostics-only — never block the run

    # ---------------------------------------------------------- file paths #
    def _benchmarks_dir(self) -> Path:
        return Path(__file__).parent / "benchmarks"

    def _datasets_dir(self) -> Path:
        return Path(__file__).parent / "datasets"

    def _queries_path(self, spec: BenchmarkSpec) -> Path:
        raw = spec.dataset.path
        if raw.startswith("fixtures://"):
            return _fixtures_cache_root() / f"{raw[len('fixtures://') :]}.jsonl"
        return self._datasets_dir() / raw

    def _corpus_path(self, spec: BenchmarkSpec) -> Path:
        raw = spec.dataset.corpus_path
        if raw.startswith("fixtures://"):
            return _fixtures_cache_root() / f"{raw[len('fixtures://') :]}.jsonl"
        return self._datasets_dir() / raw

    def _load_yaml(self, path: Path) -> BenchmarkSpec:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return BenchmarkSpec.model_validate(raw)

    def _load_queries(self, spec: BenchmarkSpec) -> list[dict[str, object]]:
        path = self._queries_path(spec)
        if not path.exists():
            msg = f"queries fixture not found: {path}"
            raise FileNotFoundError(msg)
        items: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                if "query" not in obj or "relevant_doc_ids" not in obj:
                    continue
                items.append(
                    {
                        "query": str(obj["query"]),
                        "relevant_doc_ids": [str(x) for x in obj["relevant_doc_ids"]],
                    }
                )
        if not items:
            msg = f"queries fixture is empty: {path}"
            raise ValueError(msg)
        return items

    def _load_corpus(self, spec: BenchmarkSpec) -> list[dict[str, str]]:
        path = self._corpus_path(spec)
        if not path.exists():
            msg = f"corpus fixture not found: {path}"
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
                if "doc_id" not in obj or "text" not in obj:
                    continue
                items.append({"doc_id": str(obj["doc_id"]), "text": str(obj["text"])})
        if not items:
            msg = f"corpus fixture is empty: {path}"
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
        corpus_size: int,
        dataset_hash: str,
        energy: EnergyReport | None = None,
    ) -> Envelope:
        hw = collect_hardware_fingerprint()
        sw = collect_software_provenance()

        metrics: dict[str, float | int | str | None] = {}

        ok_samples = [s for s in samples if s.ok]
        n_ok = len(ok_samples)
        metrics["n_queries"] = float(len(samples))
        metrics["n_ok"] = float(n_ok)
        metrics["ok_rate"] = float(n_ok) / float(len(samples)) if samples else 0.0
        metrics["corpus_size"] = float(corpus_size)

        # Headline retrieval metric — keyed by the spec's metric so downstream
        # `bench diff` knows higher-is-better (see _HIGHER_IS_BETTER in
        # cli/commands/diff.py).
        if scores:
            mean_score = sum(scores) / len(scores)
            metric_prefix = spec.metric  # "recall_at_5" | "mrr_at_10" | "ndcg_at_10"
            metrics[f"{metric_prefix}_mean"] = mean_score
            if len(scores) >= 2:
                pcts = Percentiles(scores, percentiles=(50.0, 95.0))
                metrics[f"{metric_prefix}_p50"] = pcts.p50
                metrics[f"{metric_prefix}_p95"] = pcts.p95
            else:
                metrics[f"{metric_prefix}_p50"] = mean_score
                metrics[f"{metric_prefix}_p95"] = mean_score

        total_vals = [s.total_ms for s in ok_samples if math.isfinite(s.total_ms)]
        if total_vals:
            metrics["total_p50_ms"] = Percentiles(total_vals).p50

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
