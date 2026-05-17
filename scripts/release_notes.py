#!/usr/bin/env python3
"""Extract a single version's section from CHANGELOG.md as GitHub-Release markdown.

Reads ``CHANGELOG.md`` at the repo root, finds the ``## [X.Y.Z]`` header
matching ``--version`` (or the latest entry below ``## [Unreleased]`` if
``--version`` is omitted), and prints the body to stdout — ready to paste
into a GitHub Release.

This is infrastructure tooling. argparse + stdlib only — no third-party deps.

Usage::

    python scripts/release_notes.py --version 0.0.2
    python scripts/release_notes.py --version 0.0.2 --out RELEASE_NOTES_v0.0.2.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"

# A version section header looks like ``## [0.0.2] — 2026-05-17`` (em-dash
# or ASCII hyphen, optional date). We capture the version and the rest of
# the line to render a clean title.
SECTION_RE = re.compile(
    r"^##\s+\[(?P<version>[^\]]+)\](?P<rest>.*)$",
    re.MULTILINE,
)


def extract_section(
    changelog_text: str, version: str | None
) -> tuple[str, str]:
    """Return ``(resolved_version, section_markdown)`` for the requested version.

    If ``version`` is None, return the latest non-Unreleased entry.

    Raises ``ValueError`` if the requested version is not present, or if the
    changelog has no concrete version entries (only ``[Unreleased]``).
    """
    matches = list(SECTION_RE.finditer(changelog_text))
    if not matches:
        msg = "CHANGELOG.md has no '## [<version>]' section headers."
        raise ValueError(msg)

    if version is None:
        # Walk past `[Unreleased]` (case-insensitive) to find the latest real
        # released version.
        target = next(
            (m for m in matches if m.group("version").strip().lower() != "unreleased"),
            None,
        )
        if target is None:
            msg = "CHANGELOG.md contains only [Unreleased] — no released version yet."
            raise ValueError(msg)
    else:
        target = next(
            (m for m in matches if m.group("version").strip() == version),
            None,
        )
        if target is None:
            available = ", ".join(
                m.group("version").strip()
                for m in matches
                if m.group("version").strip().lower() != "unreleased"
            )
            msg = (
                f"version {version!r} not found in CHANGELOG.md. "
                f"Available: {available or '(none)'}"
            )
            raise ValueError(msg)

    resolved = target.group("version").strip()
    rest = target.group("rest").rstrip()

    # Body runs from the end of this header line up to the next ``## [...]``.
    body_start = target.end()
    next_section = next(
        (m for m in matches if m.start() > target.start()), None
    )
    body_end = next_section.start() if next_section is not None else len(changelog_text)
    body = changelog_text[body_start:body_end].strip("\n")

    # Strip the link-reference footer (``[Unreleased]: <url>``) that
    # `Keep a Changelog` puts at the very bottom — those refs belong to the
    # changelog as a whole, not a single section.
    body = re.sub(
        r"^\[[^\]]+\]:\s+http[^\n]+$",
        "",
        body,
        flags=re.MULTILINE,
    ).rstrip()

    title_suffix = rest.lstrip(" —-").strip()
    header = f"# v{resolved}" + (f" — {title_suffix}" if title_suffix else "")
    rendered = f"{header}\n\n{body}\n" if body else f"{header}\n"
    return resolved, rendered


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="release_notes.py",
        description=(
            "Extract a single version's section from CHANGELOG.md as "
            "GitHub-Release-ready markdown."
        ),
    )
    parser.add_argument(
        "--version",
        default=None,
        help=(
            "Version string to extract (e.g. '0.0.2'). If omitted, uses the "
            "latest released entry (skipping [Unreleased])."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        type=Path,
        help="Write to this path instead of stdout.",
    )
    parser.add_argument(
        "--changelog",
        default=None,
        type=Path,
        help="Override CHANGELOG path (default: <repo-root>/CHANGELOG.md).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    changelog_path = args.changelog if args.changelog is not None else CHANGELOG_PATH
    if not changelog_path.is_file():
        sys.stderr.write(f"error: changelog not found: {changelog_path}\n")
        return 1

    text = changelog_path.read_text(encoding="utf-8")

    try:
        _resolved, rendered = extract_section(text, args.version)
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
