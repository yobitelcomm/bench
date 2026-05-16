"""Static-site leaderboard renderer for InferenceBench signed envelopes.

Reads a directory of canonical envelope JSONs, groups them by ``suite_id``,
and emits a static HTML + CSS + JSON site suitable for hosting on GitHub Pages
(target: https://yobitelcomm.github.io/bench).

Public surface:

    render_site(envelopes_dir, out_dir, *, base_url="/") -> SiteRenderResult
    SiteRenderResult       — summary of a render pass (counts, skipped, paths)
    LoadedEnvelope         — parsed envelope plus its source filename
    load_envelopes(dir)    — collect parseable envelopes from a directory
    compute_pareto(...)    — Pareto-frontier classifier for arbitrary axes

The renderer is deliberately framework-free: plain HTML + a small static CSS
file, no client-side bundler, no JS frameworks. A tiny vanilla sort script
ships with the site to make the tables sortable.
"""

from __future__ import annotations

from inferencebench_leaderboard.data import (
    PARETO_DIRECTIONS,
    LoadedEnvelope,
    compute_pareto,
    load_envelopes,
)
from inferencebench_leaderboard.render import SiteRenderResult, render_site

__all__ = [
    "PARETO_DIRECTIONS",
    "LoadedEnvelope",
    "SiteRenderResult",
    "compute_pareto",
    "load_envelopes",
    "render_site",
]

__version__ = "0.0.0"
