"""Tests for the embeddings-retrieval plugin scaffold + ranking pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from inferencebench_embeddings import (
    BenchmarkSpec,
    EmbeddingsRetrievalPlugin,
    EngineKind,
    RunContext,
)


# --------------------------------------------------------------------------- #
# Plugin contract                                                             #
# --------------------------------------------------------------------------- #
def test_plugin_metadata() -> None:
    plugin = EmbeddingsRetrievalPlugin()
    assert plugin.suite_id == "embeddings.retrieval"
    assert plugin.version
    assert plugin.description


def test_plugin_lists_bundled_benchmarks() -> None:
    plugin = EmbeddingsRetrievalPlugin()
    specs = plugin.list_benchmarks()
    assert len(specs) >= 4
    ids = {s.benchmark_id for s in specs}
    assert {
        "embeddings.retrieval.beir-mini",
        "embeddings.retrieval.long-doc",
        "embeddings.retrieval.msmarco-style",
        "embeddings.retrieval.query-expansion",
    }.issubset(ids)


def test_get_benchmark_msmarco_style_resolves() -> None:
    plugin = EmbeddingsRetrievalPlugin()
    spec = plugin.get_benchmark("embeddings.retrieval.msmarco-style")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.metric == "mrr_at_10"
    assert spec.dataset.path == "msmarco-style-queries.jsonl"
    assert spec.dataset.corpus_path == "msmarco-style-corpus.jsonl"


def test_get_benchmark_query_expansion_resolves() -> None:
    plugin = EmbeddingsRetrievalPlugin()
    spec = plugin.get_benchmark("embeddings.retrieval.query-expansion")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.metric == "recall_at_5"
    assert spec.dataset.path == "query-expansion-queries.jsonl"
    assert spec.dataset.corpus_path == "query-expansion-corpus.jsonl"


def test_plugin_get_benchmark_beir_mini() -> None:
    plugin = EmbeddingsRetrievalPlugin()
    spec = plugin.get_benchmark("embeddings.retrieval.beir-mini")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.modality == "embeddings"
    assert spec.kind == "retrieval"
    assert spec.metric == "recall_at_5"
    assert spec.dataset.path == "beir-mini-queries.jsonl"
    assert spec.dataset.corpus_path == "beir-mini-corpus.jsonl"


def test_plugin_get_benchmark_long_doc_uses_ndcg() -> None:
    plugin = EmbeddingsRetrievalPlugin()
    spec = plugin.get_benchmark("embeddings.retrieval.long-doc")
    assert spec.metric == "ndcg_at_10"


def test_plugin_get_benchmark_missing_id_raises_keyerror() -> None:
    plugin = EmbeddingsRetrievalPlugin()
    with pytest.raises(KeyError):
        plugin.get_benchmark("nonexistent.benchmark")


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def test_validate_warns_when_self_hosted_base_url_missing() -> None:
    plugin = EmbeddingsRetrievalPlugin()
    spec = plugin.get_benchmark("embeddings.retrieval.beir-mini")
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.TEI,
        base_url="",
        output_dir=Path("/tmp/bench"),
    )
    warnings = plugin.validate(spec, ctx)
    assert any("base_url" in w.lower() for w in warnings)


def test_validate_provider_hosted_engine_does_not_require_base_url() -> None:
    plugin = EmbeddingsRetrievalPlugin()
    spec = plugin.get_benchmark("embeddings.retrieval.beir-mini")
    ctx = RunContext(
        model_id="openai/text-embedding-3-small",
        engine_kind=EngineKind.OPENAI,
        base_url="",
        output_dir=Path("/tmp/bench"),
    )
    warnings = plugin.validate(spec, ctx)
    assert not any("base_url" in w.lower() for w in warnings)


# --------------------------------------------------------------------------- #
# End-to-end run (no embedding-model invocation)                              #
# --------------------------------------------------------------------------- #
def test_run_beir_mini_produces_signed_envelope_with_known_recall(
    make_run_context,
) -> None:
    """Hash ranking on the bundled fixture yields recall@5 mean = 0.2."""
    plugin = EmbeddingsRetrievalPlugin()
    spec = plugin.get_benchmark("embeddings.retrieval.beir-mini")
    ctx = make_run_context()

    envelope = plugin.run(spec, ctx)

    assert envelope.signature is not None
    assert envelope.signature.method == "dev-key"
    assert envelope.signature.bundle

    recall_mean = envelope.metrics.get("recall_at_5_mean")
    assert recall_mean is not None
    assert isinstance(recall_mean, (int, float))
    # Per-query scores: [0, 0.5, 0, 0, 0.5] → mean = 0.2.
    assert float(recall_mean) == pytest.approx(0.2)

    assert envelope.metrics.get("recall_at_5_p50") is not None
    assert envelope.metrics.get("n_queries") == 5.0
    assert envelope.metrics.get("corpus_size") == 20.0
    assert envelope.metrics.get("ok_rate") == 1.0


def test_run_long_doc_produces_signed_envelope_with_known_ndcg(
    make_run_context,
) -> None:
    """Hash ranking on the long-doc fixture yields nDCG@10 mean ≈ 0.4187."""
    plugin = EmbeddingsRetrievalPlugin()
    spec = plugin.get_benchmark("embeddings.retrieval.long-doc")
    ctx = make_run_context()

    envelope = plugin.run(spec, ctx)

    ndcg_mean = envelope.metrics.get("ndcg_at_10_mean")
    assert ndcg_mean is not None
    # Computed exactly from the bundled fixture (see test_scoring values).
    assert float(ndcg_mean) == pytest.approx(0.4187285611, abs=1e-6)
    assert envelope.metrics.get("n_queries") == 3.0
    assert envelope.metrics.get("corpus_size") == 10.0


def test_run_writes_samples_jsonl_alongside_envelope(make_run_context) -> None:
    """The diagnostic samples-<ts>.jsonl is written to output_dir."""
    plugin = EmbeddingsRetrievalPlugin()
    spec = plugin.get_benchmark("embeddings.retrieval.beir-mini")
    ctx = make_run_context()

    plugin.run(spec, ctx)
    samples_files = list(ctx.output_dir.glob("samples-*.jsonl"))
    assert len(samples_files) == 1
    lines = samples_files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5  # one per query
