"""Typer CLI entry point for the static leaderboard renderer.

Invoke with::

    python -m inferencebench_leaderboard build envelopes/ site/
    inferencebench-leaderboard build envelopes/ site/

The ``build`` subcommand is the only operation today; future subcommands
(e.g. ``validate``, ``diff``) plug in here.
"""

from __future__ import annotations

from pathlib import Path

import typer

from inferencebench_leaderboard.render import render_site

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Render the static InferenceBench leaderboard site.",
)


@app.command("build")
def build(
    envelopes_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory containing signed envelope JSON files.",
    ),
    out_dir: Path = typer.Argument(
        ...,
        file_okay=False,
        dir_okay=True,
        help="Destination directory for the generated static site.",
    ),
    base_url: str = typer.Option(
        "/",
        "--base-url",
        help="URL prefix the site will be served from (e.g. '/bench/').",
    ),
) -> None:
    """Render the static leaderboard site to ``out_dir``."""
    result = render_site(envelopes_dir, out_dir, base_url=base_url)
    typer.echo(
        f"Rendered {result.envelopes_loaded} envelope(s) "
        f"across {len(result.categories)} category(ies) "
        f"to {result.out_dir} "
        f"({result.envelopes_skipped} skipped, {result.pages_written} files written)."
    )
    for suite_id, count in sorted(result.categories.items()):
        typer.echo(f"  - {suite_id}: {count}")


if __name__ == "__main__":  # pragma: no cover
    app()
