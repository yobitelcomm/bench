"""``bench schema`` — emit JSON Schema for envelopes (and friends).

Non-Python consumers (Go services, Rust verifiers, web frontends) need a
stable JSON Schema for the envelope format. Pydantic v2 generates it via
``Envelope.model_json_schema()`` — this command wraps that, plus the
benchmark-spec schema from the LLM plugin and a hand-written schema for
the ``inferencebench.mirror.v1`` index emitted by ``bench publish --to local``.

The filename is ``schema_cmd.py`` rather than ``schema.py`` to avoid
shadowing the stdlib's notion of ``schema`` in IDEs and to mirror the
``list_cmd.py`` convention used elsewhere in this package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from inferencebench.envelope import SCHEMA_VERSION, Envelope

console = Console()
err_console = Console(stderr=True)


_TARGETS = {"envelope", "benchmark-spec", "mirror-index"}


def schema(
    target: Annotated[
        str,
        typer.Option(
            "--target",
            help=(
                "Which schema to emit: envelope (default), benchmark-spec "
                "(requires inferencebench-llm), or mirror-index."
            ),
        ),
    ] = "envelope",
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Write JSON to this path instead of stdout.",
        ),
    ] = None,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Print just the envelope schema version (e.g. 'v1') and exit.",
        ),
    ] = False,
) -> None:
    """Emit a JSON Schema for an envelope, benchmark spec, or mirror index."""
    if version:
        console.print(SCHEMA_VERSION)
        return

    if target not in _TARGETS:
        err_console.print(
            f"[red]Unknown --target value:[/red] {target} "
            f"(expected one of: {', '.join(sorted(_TARGETS))})"
        )
        raise typer.Exit(code=2)

    schema_dict = _build_schema(target)
    rendered = json.dumps(schema_dict, indent=2, sort_keys=True)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
        return

    # Plain print so the JSON is machine-parseable (no Rich highlighting/wrapping).
    print(rendered)


# --------------------------------------------------------------------------- #
# Builders                                                                    #
# --------------------------------------------------------------------------- #
def _build_schema(target: str) -> dict[str, Any]:
    if target == "envelope":
        return Envelope.model_json_schema()
    if target == "benchmark-spec":
        return _benchmark_spec_schema()
    if target == "mirror-index":
        return _mirror_index_schema()
    # Defensive: _TARGETS guard above means we never get here.
    raise AssertionError(f"unreachable target: {target}")


def _benchmark_spec_schema() -> dict[str, Any]:
    """Lazy-import the LLM plugin so this command works without it installed."""
    try:
        from inferencebench_llm.schemas import BenchmarkSpec
    except ImportError as exc:
        err_console.print(
            "[red]inferencebench-llm plugin is not installed.[/red] "
            "Install it: [bold]pip install inferencebench-llm[/bold]"
        )
        raise typer.Exit(code=2) from exc
    return BenchmarkSpec.model_json_schema()


def _mirror_index_schema() -> dict[str, Any]:
    """Hand-written JSON Schema for the local mirror's ``index.json``.

    Matches the payload written by ``bench publish --to local`` —
    see ``commands/publish.py::_update_mirror_index``.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://inferencebench.dev/schemas/mirror-index.v1.json",
        "title": "InferenceBench Mirror Index",
        "description": (
            "Self-describing index of a local envelope mirror, written by "
            "`bench publish --to local`. One entry per published envelope."
        ),
        "type": "object",
        "required": ["schema", "n_entries", "entries"],
        "additionalProperties": False,
        "properties": {
            "schema": {
                "type": "string",
                "const": "inferencebench.mirror.v1",
                "description": "Schema identifier for this index.",
            },
            "n_entries": {
                "type": "integer",
                "minimum": 0,
                "description": "Number of entries (must equal len(entries)).",
            },
            "entries": {
                "type": "array",
                "items": {"$ref": "#/$defs/MirrorEntry"},
            },
        },
        "$defs": {
            "MirrorEntry": {
                "type": "object",
                "required": [
                    "suite_id",
                    "suite_slug",
                    "model_id",
                    "engine",
                    "content_hash",
                    "path",
                    "signed",
                    "tag",
                    "timestamp",
                ],
                "additionalProperties": False,
                "properties": {
                    "suite_id": {
                        "type": "string",
                        "description": "Suite identifier (e.g. 'llm.inference').",
                    },
                    "suite_slug": {
                        "type": "string",
                        "description": "Filesystem-safe slug (dots → dashes).",
                    },
                    "model_id": {
                        "type": "string",
                        "description": "Model identifier from the envelope.",
                    },
                    "engine": {
                        "type": "string",
                        "description": "Engine label, e.g. 'vllm v0.7.2'.",
                    },
                    "content_hash": {
                        "type": "string",
                        "pattern": "^[0-9a-f]{64}$",
                        "description": "SHA-256 of the canonical envelope payload.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to the envelope JSON, relative to the mirror root.",
                    },
                    "signed": {
                        "type": "boolean",
                        "description": "Whether the envelope carries a signature.",
                    },
                    "tag": {
                        "type": "string",
                        "description": "Optional tag attached at publish time ('' if none).",
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "ISO 8601 timestamp from the envelope.",
                    },
                },
            },
        },
    }
