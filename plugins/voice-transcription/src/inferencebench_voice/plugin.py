"""VoiceTranscriptionPlugin — entry point for ``voice.transcription`` benchmarks.

Phase-2: drives a REAL Whisper-compatible HTTP endpoint (OpenAI's audio API,
faster-whisper-server, vLLM-audio, …) for every fixture row. The bundled
fixtures ship with tiny synthetic WAVs (pure sine tones, 16 kHz mono PCM) so
the plugin remains runnable without external dependencies; a Whisper server
will return garbage for sine waves, but the request shape is exercised.

Users with real audio swap in real WAV files. The plugin reads each WAV,
ships it to ``<base_url>/audio/transcriptions``, parses the JSON response,
and scores the returned text against the fixture reference (WER / CER /
exact-match — selectable per benchmark).

Failures (missing WAV, transport error, non-2xx response) degrade gracefully:
the offending sample is marked ``ok=False`` and the envelope still signs.
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
    Sample,
    collect_hardware_fingerprint,
    collect_software_provenance,
)
from inferencebench.harness.metrics import Percentiles
from inferencebench_voice.audio_client import TranscriptionResult, transcribe
from inferencebench_voice.schemas import BenchmarkSpec, EngineKind, RunContext
from inferencebench_voice.scoring import SCORERS

if TYPE_CHECKING:
    from collections.abc import Callable


def _json_num(v: float) -> str:
    """JSON-safe numeric encoder: NaN/inf become null."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "null"
    return repr(v)


# Engines that require ``base_url`` (self-hosted OpenAI-compatible audio servers).
# OPENAI / COHERE here mean "provider-hosted endpoint" — base_url is optional.
_SELF_HOSTED_ENGINES = frozenset({EngineKind.WHISPER_HTTP})

# Default base URLs for provider-hosted audio APIs. Self-hosted engines must
# supply ``context.base_url`` explicitly (validate() warns otherwise).
_DEFAULT_BASE_URLS: dict[EngineKind, str] = {
    EngineKind.OPENAI: "https://api.openai.com/v1",
    EngineKind.COHERE: "https://api.cohere.ai/v1",
}


def _fixtures_cache_root() -> Path:
    """Resolve the bench-fixtures cache root for ``fixtures://`` dataset URIs."""
    override = os.environ.get("BENCH_FIXTURES_ROOT")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "inferencebench" / "fixtures"


def _compute_fixture_hash(items: list[dict[str, str | float]]) -> str:
    """SHA-256 over the canonical-JSON-encoded fixture rows."""
    canonical = json.dumps(items, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_base_url(context: RunContext) -> str:
    """Pick the base URL: explicit context.base_url overrides provider defaults."""
    if context.base_url:
        return context.base_url
    return _DEFAULT_BASE_URLS.get(context.engine_kind, "")


# Metrics this plugin is expected to emit. Consumed by ``bench coverage``.
EXPECTED_METRICS: tuple[str, ...] = (
    "wer_mean",
    "wer_p50",
    "wer_p95",
    "ok_rate",
    "n_samples",
    "total_audio_duration_s",
    "total_p50_ms",
    "audio_path_resolved_count",
)


class VoiceTranscriptionPlugin:
    """Plugin entry point. Registered via ``inferencebench.plugins`` entrypoint group."""

    suite_id = "voice.transcription"
    version = "0.1.0"
    description = (
        "Voice transcription benchmarks against any Whisper-compatible /v1/audio/"
        "transcriptions endpoint (OpenAI, faster-whisper-server, vLLM-audio)."
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
        items = self._load_fixture(spec)
        fixture_hash = _compute_fixture_hash(items)
        scorer = SCORERS[spec.scoring]

        samples, scores, durations, n_resolved = self._score_items(items, scorer, context)

        # Best-effort diagnostic dump — never blocks the run on I/O errors.
        self._dump_samples(context, samples)

        envelope = self._build_envelope(
            spec,
            context,
            samples=samples,
            scores=scores,
            durations=durations,
            dataset_hash=fixture_hash,
            audio_path_resolved_count=n_resolved,
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
        items: list[dict[str, str | float]],
        scorer: Callable[[str, str], float],
        context: RunContext,
    ) -> tuple[list[Sample], list[float], list[float], int]:
        """Iterate fixture items, call the audio transcription endpoint, score each.

        Returns ``(samples, scores, durations, audio_path_resolved_count)``.
        ``samples`` is the harness-compatible per-utterance list, ``scores``
        is the raw scoring output (WER/CER/EM error rates), ``durations`` is
        the per-row audio seconds, ``audio_path_resolved_count`` is the
        number of fixture rows whose ``audio_path`` resolved to a file on
        disk (rows with a missing WAV are recorded as failed samples but
        excluded from the resolved count + from the scoring set).
        """
        base_url = _resolve_base_url(context)
        api_key = context.api_key or "EMPTY"
        datasets_root = self._datasets_dir()

        samples: list[Sample] = []
        scores: list[float] = []
        durations: list[float] = []
        n_resolved = 0

        for idx, item in enumerate(items):
            reference = str(item["reference"])
            duration_s = float(item.get("duration_s") or 0.0)
            rel_audio = str(item["audio_path"])
            audio_path = (datasets_root / rel_audio).resolve()
            t_arrival = time.perf_counter() * 1000.0

            if not audio_path.exists():
                # Record as a failed sample; do not count toward scoring or
                # toward audio_path_resolved_count.
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
                        error=f"audio file not found: {rel_audio}",
                        extra={
                            "reference": reference,
                            "hypothesis": "",
                            "duration_s": duration_s,
                            "audio_path": rel_audio,
                        },
                    )
                )
                continue

            n_resolved += 1
            result = self._invoke_transcribe(
                audio_path,
                base_url=base_url,
                model=context.model_id,
                api_key=api_key,
            )
            samples.append(
                self._to_sample(
                    idx=idx,
                    arrival_ms=t_arrival,
                    result=result,
                    reference=reference,
                    duration_s=duration_s,
                    rel_audio=rel_audio,
                    scorer=scorer,
                )
            )
            if result.ok:
                scores.append(float(scorer(reference, result.text)))
                durations.append(duration_s)

        return samples, scores, durations, n_resolved

    # Injection seam — tests patch ``_invoke_transcribe`` to avoid real HTTP.
    def _invoke_transcribe(
        self,
        audio_path: Path,
        *,
        base_url: str,
        model: str,
        api_key: str,
    ) -> TranscriptionResult:
        """Call the audio client. Exists as a method so tests can monkeypatch it."""
        return transcribe(
            audio_path,
            base_url=base_url,
            model=model,
            api_key=api_key,
        )

    def _to_sample(
        self,
        *,
        idx: int,
        arrival_ms: float,
        result: TranscriptionResult,
        reference: str,
        duration_s: float,
        rel_audio: str,
        scorer: Callable[[str, str], float],
    ) -> Sample:
        """Fold a :class:`TranscriptionResult` into a harness :class:`Sample`."""
        score = float(scorer(reference, result.text)) if result.ok else float("nan")
        return Sample(
            request_idx=idx,
            arrival_ms=arrival_ms,
            start_ms=arrival_ms,
            ttft_ms=result.ttft_ms,
            total_ms=result.total_ms,
            tpot_ms=float("nan"),
            tokens_in=0,
            tokens_out=result.tokens_out,
            cost_usd=result.cost_usd,
            finish_reason=result.finish_reason,
            ok=result.ok,
            error=result.error or "",
            extra={
                "score": score,
                "reference": reference,
                "hypothesis": result.text,
                "duration_s": duration_s,
                "audio_path": rel_audio,
            },
        )

    # ------------------------------------------------------------ samples #
    def _dump_samples(self, context: RunContext, samples: list[Sample]) -> None:
        """Write per-utterance samples (incl. score) to ``<output_dir>/samples-<ts>.jsonl``.

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

    def _dataset_path(self, spec: BenchmarkSpec) -> Path:
        raw = spec.dataset.path
        if raw.startswith("fixtures://"):
            return _fixtures_cache_root() / f"{raw[len('fixtures://') :]}.jsonl"
        return self._datasets_dir() / raw

    def _load_yaml(self, path: Path) -> BenchmarkSpec:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return BenchmarkSpec.model_validate(raw)

    def _load_fixture(self, spec: BenchmarkSpec) -> list[dict[str, str | float]]:
        path = self._dataset_path(spec)
        if not path.exists():
            if spec.dataset.path.startswith("fixtures://"):
                key = spec.dataset.path[len("fixtures://") :]
                msg = f"fixture not cached: {path}. Run `bench fixtures fetch {key}` first."
                raise FileNotFoundError(msg)
            msg = f"fixture not found: {path}"
            raise FileNotFoundError(msg)
        items: list[dict[str, str | float]] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                if "audio_path" not in obj or "reference" not in obj:
                    continue
                items.append(
                    {
                        "audio_path": str(obj["audio_path"]),
                        "reference": str(obj["reference"]),
                        "duration_s": float(obj.get("duration_s", 0.0)),
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
        durations: list[float],
        dataset_hash: str,
        audio_path_resolved_count: int,
    ) -> Envelope:
        hw = collect_hardware_fingerprint()
        sw = collect_software_provenance()

        metrics: dict[str, float | int | str | None] = {}

        ok_samples = [s for s in samples if s.ok]
        n_ok = len(ok_samples)
        metrics["n_samples"] = float(len(samples))
        metrics["n_ok"] = float(n_ok)
        metrics["ok_rate"] = float(n_ok) / float(len(samples)) if samples else 0.0
        metrics["audio_path_resolved_count"] = float(audio_path_resolved_count)

        # Headline scoring metric — keyed by the spec's scoring strategy
        # so downstream `bench diff` knows whether lower or higher is better
        # (see _LOWER_IS_BETTER / _HIGHER_IS_BETTER policy in cli/commands/diff.py).
        if scores:
            mean_score = sum(scores) / len(scores)
            metric_prefix = spec.scoring  # "wer" | "cer" | "exact_match"
            metrics[f"{metric_prefix}_mean"] = mean_score
            if len(scores) >= 2:
                pcts = Percentiles(scores, percentiles=(50.0, 95.0))
                metrics[f"{metric_prefix}_p50"] = pcts.p50
                metrics[f"{metric_prefix}_p95"] = pcts.p95
            else:
                metrics[f"{metric_prefix}_p50"] = mean_score
                metrics[f"{metric_prefix}_p95"] = mean_score

        # Audio aggregates — useful for "quality at what throughput" comparisons.
        total_audio_s = sum(durations)
        if total_audio_s > 0:
            metrics["total_audio_duration_s"] = float(total_audio_s)

        total_vals = [s.total_ms for s in ok_samples if math.isfinite(s.total_ms)]
        if total_vals:
            metrics["total_p50_ms"] = Percentiles(total_vals).p50

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
