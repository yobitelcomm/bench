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
from inferencebench.harness.metrics import EnergyReport, Percentiles, TelemetryWindow
from inferencebench_quality.schemas import BenchmarkSpec, EngineKind, RunContext
from inferencebench_quality.scoring import (
    SCORERS,
    PersonaConsistencyResult,
    ScoreContext,
    judge_llm_persona,
    persona_consistency,
)

_DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"


def _fixtures_cache_root() -> Path:
    """Resolve the bench-fixtures cache root for ``fixtures://`` dataset URIs.

    Honours ``BENCH_FIXTURES_ROOT`` for tests/power-users; otherwise matches
    the default location ``bench fixtures fetch`` writes to.
    """
    override = os.environ.get("BENCH_FIXTURES_ROOT")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "inferencebench" / "fixtures"


class JudgeThrottle:
    """Single-threaded rate limiter for judge API calls.

    Spaces successive :meth:`acquire` calls at least ``1/rps`` seconds apart
    by sleeping the residual delta. ``rps <= 0`` disables throttling — the
    helper becomes a no-op so the unlimited default path costs nothing.

    The ``clock`` and ``sleep`` callables are injected for testability: real
    code uses :func:`time.monotonic` + :func:`time.sleep`; tests pass mocks
    so no wall time elapses.
    """

    def __init__(
        self,
        rps: float,
        *,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._rps = float(rps)
        self._interval = 1.0 / self._rps if self._rps > 0 else 0.0
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    @property
    def rps(self) -> float:
        """Configured requests-per-second cap (0 = unlimited)."""
        return self._rps

    def acquire(self) -> None:
        """Block until at least ``1/rps`` seconds have passed since the last call."""
        if self._interval <= 0:
            return
        clock = self._clock if self._clock is not None else time.monotonic
        sleep = self._sleep if self._sleep is not None else time.sleep
        now = clock()
        if self._last is not None:
            elapsed = now - self._last
            wait = self._interval - elapsed
            if wait > 0:
                sleep(wait)
                now = clock()
        self._last = now


def _coerce_judge_rps(raw: str | int | float | bool | None) -> float:
    """Coerce ``RunContext.extra['judge_rps']`` to a non-negative float.

    Accepts numeric or string-numeric values; anything else (or a negative
    number) silently falls back to 0 (= unlimited) so a bogus override
    never blocks a run.
    """
    if raw is None or isinstance(raw, bool):
        return 0.0
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return 0.0
        try:
            value = float(stripped)
        except ValueError:
            return 0.0
    else:
        value = float(raw)
    return value if value > 0 else 0.0


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


def _compute_fixture_hash_multi_turn(cases: list[dict[str, object]]) -> str:
    """SHA-256 over canonical-JSON-encoded multi-turn cases.

    The nested ``turns`` / ``markers`` lists are encoded as-is; sort_keys
    plus tight separators makes the hash stable across Python versions.
    """
    canonical = json.dumps(cases, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_client(context: RunContext, *, timeout_s: float = 60.0) -> ModelClient:
    """Build a :class:`ModelClient` from the run context.

    OpenAI-compatible self-hosted servers (vLLM, SGLang) require the LiteLLM
    ``openai/<model>`` routing prefix; we add it here exactly once, stripping
    any user-supplied prefix first so a double ``openai/openai/...`` never
    reaches LiteLLM. Provider-hosted engines (``OPENAI`` kind) leave the
    model id untouched.
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
)


# Scoring strategies that operate on a multi-turn conversation rather than
# a single ``(question, answer)`` pair.
_MULTI_TURN_SCORERS = frozenset({"persona_consistency", "judge_llm_persona"})


def _render_multi_turn_prompt(
    prior_turns: list[tuple[str, str]],
    next_question: str,
) -> str:
    """Render the conversation history into a single prompt string.

    The underlying ``ModelClient.complete`` takes one prompt + an optional
    system message. To preserve conversational context across turns without
    re-engineering the client API, we fold the prior turns into the prompt
    body as plain text. The caller passes the persona directive separately
    via ``ModelClient.complete``'s ``system`` kwarg.
    """
    if not prior_turns:
        return next_question
    lines: list[str] = []
    for q, r in prior_turns:
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {r}")
    lines.append(f"User: {next_question}")
    lines.append("Assistant:")
    return "\n".join(lines)


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
                f"{context.engine_kind.value} needs base_url (e.g. http://localhost:8000/v1)"
            )
        if not self._dataset_path(spec).exists():
            warnings.append(f"fixture not found: {spec.dataset.path}")
        return warnings

    # ------------------------------------------------------------------ run #
    def run(self, spec: BenchmarkSpec, context: RunContext) -> Envelope:
        """Execute the benchmark and return a SIGNED envelope."""
        if spec.multi_turn or spec.scoring in _MULTI_TURN_SCORERS:
            return self._run_multi_turn(spec, context)
        client = _build_client(context)
        items = self._load_fixture(spec)
        fixture_hash = _compute_fixture_hash(items)
        scorer = SCORERS[spec.scoring]

        judge_client: ModelClient | None = None
        judge_max_questions: int | None = None
        judge_throttle: JudgeThrottle | None = None
        if spec.scoring == "judge_llm":
            judge_client = self._build_judge_client(spec, context)
            raw_cap = context.extra.get("judge_max_questions")
            if isinstance(raw_cap, (int, float)) and not isinstance(raw_cap, bool):
                judge_max_questions = int(raw_cap)
            elif isinstance(raw_cap, str) and raw_cap.strip():
                try:
                    judge_max_questions = int(raw_cap)
                except ValueError:
                    judge_max_questions = None
            judge_throttle = JudgeThrottle(_coerce_judge_rps(context.extra.get("judge_rps")))

        judge_errors: list[str] = []
        judge_cost_usd: list[float] = []
        samples, scores, n_judged, telemetry = self._score_items(
            client,
            items,
            scorer,
            judge_client=judge_client,
            judge_max_questions=judge_max_questions,
            judge_errors=judge_errors,
            judge_cost_usd=judge_cost_usd,
            is_judge=spec.scoring == "judge_llm",
            judge_throttle=judge_throttle,
        )

        # Best-effort diagnostic dump — never blocks the run on I/O errors.
        self._dump_samples(context, samples)

        envelope = self._build_envelope(
            spec,
            context,
            samples=samples,
            scores=scores,
            dataset_hash=fixture_hash,
            n_judged=n_judged if spec.scoring == "judge_llm" else None,
            judge_errors=judge_errors if spec.scoring == "judge_llm" else None,
            judge_cost_usd=judge_cost_usd if spec.scoring == "judge_llm" else None,
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
        scorer: Callable[[ScoreContext], float],
        *,
        judge_client: ModelClient | None = None,
        judge_max_questions: int | None = None,
        judge_errors: list[str] | None = None,
        judge_cost_usd: list[float] | None = None,
        is_judge: bool = False,
        judge_throttle: JudgeThrottle | None = None,
    ) -> tuple[list[Sample], list[float], int, TelemetryWindow]:
        """Iterate fixture items sequentially, scoring each model response.

        Quality runs are per-question and order-independent — no driver
        machinery is required. We still emit ``Sample`` objects so the
        envelope-building path stays uniform with the perf plugin.

        When ``is_judge`` is True, each successful model response is graded
        by the supplied ``judge_client`` (capped at ``judge_max_questions``
        if non-None). Judged questions contribute to ``scores``; un-judged
        ones do not. The returned ``n_judged`` is the count of questions
        actually passed through the judge.
        """
        samples: list[Sample] = []
        scores: list[float] = []
        n_judged = 0
        # Output channels for judge metrics; the caller owns the lists so
        # they survive across this call boundary.
        _judge_errors = judge_errors if judge_errors is not None else []
        _judge_cost = judge_cost_usd if judge_cost_usd is not None else []
        telemetry = TelemetryWindow()
        with telemetry:
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

                score: float | None
                if is_judge:
                    if judge_max_questions is not None and n_judged >= judge_max_questions:
                        score = None  # over the cap — skip the judge entirely
                    else:
                        if judge_throttle is not None:
                            judge_throttle.acquire()
                        score_ctx = ScoreContext(
                            reference=reference,
                            hypothesis=result.text,
                            question=question,
                            judge_client=judge_client,
                            judge_errors=_judge_errors,
                            judge_cost_usd=_judge_cost,
                        )
                        score = float(scorer(score_ctx))
                        n_judged += 1
                else:
                    score_ctx = ScoreContext(
                        reference=reference,
                        hypothesis=result.text,
                        question=question,
                    )
                    score = float(scorer(score_ctx))

                if score is not None:
                    scores.append(score)

                sample_extra: dict[str, str | int | float | bool] = {
                    "category": item.get("category", ""),
                }
                if score is not None:
                    sample_extra["score"] = score
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
        return samples, scores, n_judged, telemetry

    # ------------------------------------------------------- multi-turn #
    def _run_multi_turn(self, spec: BenchmarkSpec, context: RunContext) -> Envelope:
        """Execute a multi-turn persona-consistency benchmark.

        Each fixture row is one conversation case. We send the system prompt
        once per case and replay every prior turn in the prompt body so the
        model has full context for the next reply. Each case contributes
        exactly one ``Sample`` whose score is the persona consistency over
        that case's turns.
        """
        client = _build_client(context)
        cases = self._load_multi_turn_fixture(spec)
        fixture_hash = _compute_fixture_hash_multi_turn(cases)

        judge_client: ModelClient | None = None
        judge_errors: list[str] = []
        judge_cost_usd: list[float] = []
        judge_throttle: JudgeThrottle | None = None
        if spec.scoring == "judge_llm_persona":
            judge_client = self._build_judge_client(spec, context)
            judge_throttle = JudgeThrottle(_coerce_judge_rps(context.extra.get("judge_rps")))

        samples: list[Sample] = []
        scores: list[float] = []
        drift_misses: list[int] = []  # 0-indexed first-miss per drifting case
        n_drifted = 0
        n_judged = 0

        for idx, case in enumerate(cases):
            # _load_multi_turn_fixture validates these types; the dict value
            # type is ``object`` to avoid a TypedDict declaration just for
            # this helper. Cast locally so the rest of the loop is typed.
            system_prompt = str(case["system_prompt"])
            raw_markers = case["markers"]
            markers: list[str] = (
                [str(m) for m in raw_markers] if isinstance(raw_markers, list) else []
            )
            case_id = str(case.get("case_id") or f"case-{idx}")
            raw_turns = case["turns"]
            turn_questions: list[str] = (
                [str(t) for t in raw_turns] if isinstance(raw_turns, list) else []
            )

            t_arrival = time.perf_counter() * 1000.0
            collected_turns: list[tuple[str, str]] = []
            ttft_first: float = float("nan")
            total_sum: float = 0.0
            tokens_in_sum = 0
            tokens_out_sum = 0
            cost_sum = 0.0
            ok = True
            error_msg: str | None = None
            for t_idx, question in enumerate(turn_questions):
                prompt_text = _render_multi_turn_prompt(collected_turns, question)
                try:
                    result: CompletionResult = client.complete(
                        prompt_text,
                        stream=True,
                        max_tokens=128,
                        system=system_prompt,
                    )
                except Exception as exc:  # pragma: no cover - mocked tests
                    ok = False
                    error_msg = str(exc)
                    break
                collected_turns.append((question, result.text))
                if t_idx == 0 and math.isfinite(result.ttft_ms):
                    ttft_first = result.ttft_ms
                if math.isfinite(result.total_ms):
                    total_sum += result.total_ms
                tokens_in_sum += result.tokens_in
                tokens_out_sum += result.tokens_out
                cost_sum += result.cost_usd

            sample_extra: dict[str, str | int | float | bool] = {
                "case_id": case_id,
                "n_turns": float(len(collected_turns)),
            }

            score: float
            persona_result: PersonaConsistencyResult | None = None
            if not ok or not collected_turns:
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
                        error=error_msg or "no turns collected",
                        extra=sample_extra,
                    )
                )
                continue

            if spec.scoring == "judge_llm_persona":
                if judge_throttle is not None:
                    judge_throttle.acquire()
                score = float(
                    judge_llm_persona(
                        collected_turns,
                        system_prompt=system_prompt,
                        judge_client=judge_client,
                        judge_errors=judge_errors,
                        judge_cost_usd=judge_cost_usd,
                    )
                )
                n_judged += 1
            else:
                persona_result = persona_consistency(collected_turns, markers=markers)
                score = float(persona_result.score)
                if persona_result.drift_first_miss_turn is not None:
                    n_drifted += 1
                    drift_misses.append(persona_result.drift_first_miss_turn)
                    sample_extra["drift_first_miss_turn"] = float(
                        persona_result.drift_first_miss_turn
                    )

            scores.append(score)
            sample_extra["score"] = score

            tpot = (
                (total_sum - ttft_first) / max(tokens_out_sum - 1, 1)
                if math.isfinite(ttft_first) and tokens_out_sum > 1
                else float("nan")
            )
            samples.append(
                Sample(
                    request_idx=idx,
                    arrival_ms=t_arrival,
                    start_ms=t_arrival,
                    ttft_ms=ttft_first,
                    total_ms=total_sum if total_sum > 0 else float("nan"),
                    tpot_ms=tpot,
                    tokens_in=tokens_in_sum,
                    tokens_out=tokens_out_sum,
                    cost_usd=cost_sum,
                    finish_reason="stop",
                    ok=True,
                    extra=sample_extra,
                )
            )

        self._dump_samples(context, samples)

        envelope = self._build_multi_turn_envelope(
            spec,
            context,
            samples=samples,
            scores=scores,
            drift_misses=drift_misses,
            n_drifted=n_drifted,
            dataset_hash=fixture_hash,
            n_judged=n_judged if spec.scoring == "judge_llm_persona" else None,
            judge_errors=(judge_errors if spec.scoring == "judge_llm_persona" else None),
            judge_cost_usd=(judge_cost_usd if spec.scoring == "judge_llm_persona" else None),
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

    # ---------------------------------------------------------- judge #
    def _build_judge_client(self, spec: BenchmarkSpec, context: RunContext) -> ModelClient:
        """Construct the judge :class:`ModelClient`.

        Model id precedence: spec.judge_model > extra['judge_model'] >
        ``openai/gpt-4o-mini``. The judge reuses ``context``'s engine kind,
        base_url and api_key — most judges are OpenAI-compatible endpoints,
        so the same self-hosted vs. provider routing applies.
        """
        judge_model = spec.judge_model
        if not judge_model:
            extra_model = context.extra.get("judge_model")
            if isinstance(extra_model, str) and extra_model:
                judge_model = extra_model
        if not judge_model:
            judge_model = _DEFAULT_JUDGE_MODEL

        # Use a shallow copy of the context with model_id swapped out, so the
        # existing _build_client routing applies unchanged.
        judge_context = context.model_copy(update={"model_id": judge_model})
        return _build_client(judge_context)

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

    def _load_multi_turn_fixture(self, spec: BenchmarkSpec) -> list[dict[str, object]]:
        """Load and validate a multi-turn persona fixture.

        Expected per-row shape::

            {
              "case_id": str,
              "system_prompt": str,
              "markers": [str, ...],
              "turns": [{"question": str, ...}, ...]
            }

        Rows missing required fields are silently skipped — matches the
        single-turn loader's behaviour. Raises ``FileNotFoundError`` /
        ``ValueError`` on a missing or empty fixture so the run halts loudly
        rather than producing an empty envelope.
        """
        path = self._dataset_path(spec)
        if not path.exists():
            if spec.dataset.path.startswith("fixtures://"):
                key = spec.dataset.path[len("fixtures://") :]
                msg = f"fixture not cached: {path}. Run `bench fixtures fetch {key}` first."
                raise FileNotFoundError(msg)
            msg = f"fixture not found: {path}"
            raise FileNotFoundError(msg)
        cases: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                if "system_prompt" not in obj or "turns" not in obj or "markers" not in obj:
                    continue
                turns_raw = obj["turns"]
                if not isinstance(turns_raw, list):
                    continue
                turn_questions: list[str] = []
                for t in turns_raw:
                    if isinstance(t, dict) and "question" in t:
                        turn_questions.append(str(t["question"]))
                if not turn_questions:
                    continue
                markers_raw = obj["markers"]
                if not isinstance(markers_raw, list):
                    continue
                markers = [str(m) for m in markers_raw]
                cases.append(
                    {
                        "case_id": str(obj.get("case_id", "")),
                        "system_prompt": str(obj["system_prompt"]),
                        "markers": markers,
                        "turns": turn_questions,
                    }
                )
        if not cases:
            msg = f"multi-turn fixture is empty: {path}"
            raise ValueError(msg)
        return cases

    # ---------------------------------------------------------- envelope #
    def _build_envelope(
        self,
        spec: BenchmarkSpec,
        context: RunContext,
        *,
        samples: list[Sample],
        scores: list[float],
        dataset_hash: str,
        n_judged: int | None = None,
        judge_errors: list[str] | None = None,
        judge_cost_usd: list[float] | None = None,
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
        # is intentionally NOT mirrored here — quality runs are cheap enough
        # that a missing-cost row is more honest than an estimated one.
        cost_total = sum(s.cost_usd for s in ok_samples)
        judge_cost_total = sum(judge_cost_usd) if judge_cost_usd else 0.0
        combined_cost = cost_total + judge_cost_total
        if tokens_out_total and combined_cost > 0:
            metrics["cost_usd_per_million_tokens"] = (combined_cost / tokens_out_total) * 1e6
            metrics["cost_source"] = "provider"
            if judge_cost_total > 0:
                metrics["judge_cost_usd_total"] = judge_cost_total

        # LLM-as-judge bookkeeping. ``n_judged`` is the count of questions
        # whose score came from the judge (NOT counting cap-skipped ones);
        # ``judge_errors`` is the count of judge calls that raised.
        if n_judged is not None:
            metrics["n_judged"] = float(n_judged)
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

    def _build_multi_turn_envelope(
        self,
        spec: BenchmarkSpec,
        context: RunContext,
        *,
        samples: list[Sample],
        scores: list[float],
        drift_misses: list[int],
        n_drifted: int,
        dataset_hash: str,
        n_judged: int | None = None,
        judge_errors: list[str] | None = None,
        judge_cost_usd: list[float] | None = None,
    ) -> Envelope:
        """Build a signed envelope for a multi-turn persona-consistency run.

        Mirrors :meth:`_build_envelope` but emits persona-specific aggregates
        (``persona_consistency_mean / _p50 / _p95``, ``drift_rate``,
        ``mean_drift_turn``) instead of the single-shot ``accuracy`` row.
        ``accuracy``-keyed metrics are also written so existing tooling that
        ranks on accuracy keeps working.
        """
        hw = collect_hardware_fingerprint()
        sw = collect_software_provenance()

        metrics: dict[str, float | int | str | None] = {}

        ok_samples = [s for s in samples if s.ok]
        n_ok = len(ok_samples)
        metrics["n_samples"] = float(len(samples))
        metrics["n_ok"] = float(n_ok)
        metrics["ok_rate"] = float(n_ok) / float(len(samples)) if samples else 0.0

        if scores:
            mean_score = sum(scores) / len(scores)
            metrics["persona_consistency_mean"] = mean_score
            # Alias under accuracy so downstream tooling (bench diff,
            # bench coverage) that ranks on `accuracy` keeps working.
            metrics["accuracy"] = mean_score
            if len(scores) >= 2:
                pcts = Percentiles(scores, percentiles=(5.0, 50.0, 95.0))
                metrics["persona_consistency_p50"] = pcts.p50
                metrics["persona_consistency_p95"] = pcts.p95
                metrics["accuracy_p05"] = pcts.p5
                metrics["accuracy_p50"] = pcts.p50
                metrics["accuracy_p95"] = pcts.p95
            else:
                metrics["persona_consistency_p50"] = mean_score
                metrics["persona_consistency_p95"] = mean_score
                metrics["accuracy_p05"] = mean_score
                metrics["accuracy_p50"] = mean_score
                metrics["accuracy_p95"] = mean_score

        # Drift bookkeeping. ``drift_rate`` is the share of scored cases that
        # ever lost the persona; ``mean_drift_turn`` averages the 0-indexed
        # first-miss turn across drifting cases only (NaN-safe via absence).
        if scores:
            metrics["drift_rate"] = float(n_drifted) / float(len(scores))
        if drift_misses:
            metrics["mean_drift_turn"] = sum(drift_misses) / len(drift_misses)

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
            metrics["cost_usd_per_million_tokens"] = (combined_cost / tokens_out_total) * 1e6
            metrics["cost_source"] = "provider"
            if judge_cost_total > 0:
                metrics["judge_cost_usd_total"] = judge_cost_total

        if n_judged is not None:
            metrics["n_judged"] = float(n_judged)
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
