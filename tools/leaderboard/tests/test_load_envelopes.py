"""Tests for envelope loading + malformed-file skipping."""

from __future__ import annotations

from pathlib import Path

from inferencebench_leaderboard import load_envelopes, render_site


def test_load_envelopes_skips_garbage(corpus_with_garbage: Path) -> None:
    loaded = load_envelopes(corpus_with_garbage)
    # 3 valid + 2 broken = 5 files; only 3 should load.
    assert len(loaded) == 3
    filenames = {item.source_filename for item in loaded}
    assert "broken-syntax.json" not in filenames
    assert "broken-schema.json" not in filenames


def test_load_envelopes_nonexistent_dir(tmp_path: Path) -> None:
    assert load_envelopes(tmp_path / "does-not-exist") == []


def test_render_site_records_skipped_count(corpus_with_garbage: Path, tmp_path: Path) -> None:
    out = tmp_path / "site"
    result = render_site(corpus_with_garbage, out)
    assert result.envelopes_loaded == 3
    assert result.envelopes_skipped == 2
    # Site still renders despite bad files.
    assert (out / "index.html").exists()
