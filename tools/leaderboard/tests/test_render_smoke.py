"""Smoke test: render a 3-envelope corpus to tmp_path and inspect the output."""

from __future__ import annotations

import json
from pathlib import Path

from inferencebench_leaderboard import render_site


def test_render_site_creates_index_and_category_pages(
    envelope_corpus: Path, tmp_path: Path
) -> None:
    out = tmp_path / "site"
    result = render_site(envelope_corpus, out)

    assert result.envelopes_loaded == 3
    assert result.envelopes_skipped == 0
    assert set(result.categories) == {"llm.inference", "embeddings.retrieval"}
    assert result.categories["llm.inference"] == 2
    assert result.categories["embeddings.retrieval"] == 1

    # Top-level index exists and references both categories.
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "llm.inference" in index
    assert "embeddings.retrieval" in index

    # Per-category page exists and contains a model id from that category.
    llm = (out / "llm.inference" / "index.html").read_text(encoding="utf-8")
    assert "meta-llama/Llama-4-Maverick" in llm
    assert "mistralai/Mistral-Large" in llm

    # Entry detail page exists and contains the verify snippet.
    entry_path = out / "llm.inference" / "01934567-89ab-7000-8000-000000000001.html"
    assert entry_path.exists()
    entry_html = entry_path.read_text(encoding="utf-8")
    assert "bench verify" in entry_html
    assert "meta-llama/Llama-4-Maverick" in entry_html

    # Static assets copied.
    assert (out / "static" / "site.css").exists()
    assert (out / "static" / "sort.js").exists()

    # Raw envelope JSONs copied through.
    assert (out / "envelopes" / "01-llama.json").exists()

    # Machine-readable dump is valid JSON.
    data = json.loads((out / "data" / "leaderboard.json").read_text(encoding="utf-8"))
    assert data["schema"] == "inferencebench-leaderboard.v1"
    suites = {c["suite_id"] for c in data["categories"]}
    assert suites == {"llm.inference", "embeddings.retrieval"}


def test_render_site_with_custom_base_url(envelope_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "site"
    result = render_site(envelope_corpus, out, base_url="/bench/")

    assert result.envelopes_loaded == 3
    index = (out / "index.html").read_text(encoding="utf-8")
    # Links should be prefixed with the supplied base.
    assert 'href="/bench/llm.inference/"' in index
    assert 'href="/bench/static/site.css"' in index


def test_render_site_empty_dir(tmp_path: Path) -> None:
    src = tmp_path / "empty"
    src.mkdir()
    out = tmp_path / "site"
    result = render_site(src, out)
    assert result.envelopes_loaded == 0
    assert result.categories == {}
    assert (out / "index.html").exists()


def test_category_page_contains_filter_input(envelope_corpus: Path, tmp_path: Path) -> None:
    """Rendered category page exposes the client-side filter input + script."""
    out = tmp_path / "site"
    render_site(envelope_corpus, out)
    llm = (out / "llm.inference" / "index.html").read_text(encoding="utf-8")
    # The filter input is rendered server-side so the page works even before
    # JS executes; filter.js attaches the input event handler.
    assert 'class="ib-filter"' in llm
    assert 'type="search"' in llm
    assert "static/filter.js" in llm


def test_filter_js_written_to_disk(envelope_corpus: Path, tmp_path: Path) -> None:
    """The filter.js asset is copied alongside sort.js and site.css."""
    out = tmp_path / "site"
    render_site(envelope_corpus, out)
    assert (out / "static" / "filter.js").exists()


def test_filter_js_has_no_external_urls(envelope_corpus: Path, tmp_path: Path) -> None:
    """filter.js must not reference any external CDN — purely browser-native."""
    out = tmp_path / "site"
    render_site(envelope_corpus, out)
    js = (out / "static" / "filter.js").read_text(encoding="utf-8")
    assert "http://" not in js
    assert "https://" not in js
