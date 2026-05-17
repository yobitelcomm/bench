#!/usr/bin/env python3
"""Release dry-run: validate every workspace package can produce a wheel.

Runs ``uv build`` against every workspace package, inspects the produced wheel
+ sdist, and prints a Rich summary table. Exit 0 if all packages succeed.

Usage::

    python scripts/release_dry_run.py [--check-only] [--clean] [--verbose]
                                      [--bump VERSION]

This is infrastructure tooling (not user-facing), so it deliberately uses
``argparse`` and only stdlib + ``rich`` (already a workspace dep). The script
never imports workspace packages; it shells out to ``uv build``.
"""

from __future__ import annotations

import argparse
import email
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from email.message import Message
from pathlib import Path

from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent

# Globs (relative to REPO_ROOT) for workspace package pyprojects. The root
# pyproject is intentionally excluded — it's the workspace marker.
PACKAGE_GLOBS: tuple[str, ...] = (
    "*/pyproject.toml",
    "plugins/*/pyproject.toml",
    "integrations/*/pyproject.toml",
    "tools/*/pyproject.toml",
)

# ``version = "..."`` line — single-line, double-quoted form, with optional
# whitespace. We restrict the rewrite to this exact PEP 621 shape to avoid
# nuking ``version`` keys inside [tool.*] tables.
VERSION_RE = re.compile(r'^(\s*version\s*=\s*)"[^"]*"(\s*)$', re.MULTILINE)

# Required PEP 621 metadata fields we check on every wheel.
REQUIRED_METADATA_FIELDS: tuple[str, ...] = (
    "Name",
    "Version",
    "Summary",
    "License",
    "Requires-Python",
)


@dataclass
class PackageResult:
    """Outcome of a single package's dry-run build."""

    path: Path
    name: str = ""
    version: str = ""
    wheel_size: int = 0
    sdist_size: int = 0
    status: str = ""  # "ok" | "fail"
    error: str = ""
    metadata_issues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_packages(repo_root: Path | None = None) -> list[Path]:
    """Return absolute Paths to every workspace package directory.

    The root ``pyproject.toml`` is excluded — it's the uv workspace marker,
    not a buildable package.
    """
    root = repo_root if repo_root is not None else REPO_ROOT
    seen: set[Path] = set()
    out: list[Path] = []
    for glob in PACKAGE_GLOBS:
        for pyproject in sorted(root.glob(glob)):
            pkg_dir = pyproject.parent.resolve()
            if pkg_dir == root.resolve():
                continue
            if pkg_dir in seen:
                continue
            seen.add(pkg_dir)
            out.append(pkg_dir)
    return out


# ---------------------------------------------------------------------------
# Version bump
# ---------------------------------------------------------------------------


def update_versions(
    target: str,
    *,
    dry_run: bool = False,
    repo_root: Path | None = None,
) -> list[Path]:
    """Rewrite ``version = "..."`` in every workspace pyproject.

    Args:
        target: New version string (e.g. ``"0.0.2"``).
        dry_run: If True, return the list of files that WOULD change without
            writing anything. If False, perform the rewrite in place.
        repo_root: Optional repo root override (used by tests with tmp_path).

    Returns:
        List of pyproject paths that were (or would be) modified.
    """
    root = repo_root if repo_root is not None else REPO_ROOT
    replacement = rf'\g<1>"{target}"\g<2>'
    changed: list[Path] = []
    for pkg_dir in discover_packages(root):
        pyproject = pkg_dir / "pyproject.toml"
        original = pyproject.read_text(encoding="utf-8")
        new_text, n_subs = VERSION_RE.subn(replacement, original, count=1)
        if n_subs == 0 or new_text == original:
            continue
        changed.append(pyproject)
        if not dry_run:
            pyproject.write_text(new_text, encoding="utf-8")
    return changed


# ---------------------------------------------------------------------------
# Build + wheel inspection
# ---------------------------------------------------------------------------


def _find_wheel_and_sdist(out_dir: Path) -> tuple[Path | None, Path | None]:
    wheels = sorted(out_dir.glob("*.whl"))
    sdists = sorted(out_dir.glob("*.tar.gz"))
    wheel = wheels[0] if len(wheels) == 1 else None
    sdist = sdists[0] if len(sdists) == 1 else None
    return wheel, sdist


def _parse_wheel_metadata(wheel: Path) -> tuple[Message | None, list[str]]:
    """Open a wheel and return its METADATA message plus the file list."""
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        metadata_name = next(
            (n for n in names if n.endswith(".dist-info/METADATA")), None
        )
        if metadata_name is None:
            return None, names
        raw = zf.read(metadata_name)
    return email.message_from_bytes(raw), names


def validate_wheel(wheel: Path) -> list[str]:
    """Inspect a wheel and return a list of human-readable issues.

    Empty list means the wheel is structurally and metadata-wise sound. We
    check:
      - zip is valid (``python -m zipfile -t``)
      - METADATA + RECORD present
      - PEP 621 required metadata fields are present
      - at least one ``.py`` file in the actual package directory
    """
    issues: list[str] = []

    # 1) valid zip
    test_proc = subprocess.run(
        [sys.executable, "-m", "zipfile", "-t", str(wheel)],
        capture_output=True,
        text=True,
        check=False,
    )
    if test_proc.returncode != 0:
        issues.append(f"zip integrity check failed: {test_proc.stderr.strip()}")
        return issues  # nothing else we can usefully say

    metadata, names = _parse_wheel_metadata(wheel)
    if metadata is None:
        issues.append("missing dist-info/METADATA")
        return issues

    # 2) RECORD present, non-empty
    record_name = next((n for n in names if n.endswith(".dist-info/RECORD")), None)
    if record_name is None:
        issues.append("missing dist-info/RECORD")
    else:
        with zipfile.ZipFile(wheel) as zf:
            if not zf.read(record_name).strip():
                issues.append("dist-info/RECORD is empty")

    # 3) required metadata fields
    for field_name in REQUIRED_METADATA_FIELDS:
        value = metadata.get(field_name)
        if value is None or not str(value).strip():
            issues.append(f"missing metadata field: {field_name}")

    # 4) at least one .py file outside dist-info
    py_files = [
        n
        for n in names
        if n.endswith(".py") and ".dist-info/" not in n and ".data/" not in n
    ]
    if not py_files:
        issues.append("wheel contains no .py files in package directory")

    return issues


def build_package(
    pkg_dir: Path,
    *,
    check_only: bool,
    verbose: bool,
    console: Console,
) -> PackageResult:
    """Build a single package and inspect its wheel."""
    result = PackageResult(path=pkg_dir)
    with tempfile.TemporaryDirectory(prefix="bench-dryrun-") as tmp:
        out_dir = Path(tmp)
        cmd = ["uv", "build", "--out-dir", str(out_dir)]
        if verbose:
            console.print(f"[dim]$ (cd {pkg_dir.relative_to(REPO_ROOT)} && {' '.join(cmd)})[/]")
        proc = subprocess.run(
            cmd,
            cwd=pkg_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            result.status = "fail"
            result.error = (proc.stderr or proc.stdout).strip()
            return result

        wheel, sdist = _find_wheel_and_sdist(out_dir)
        if wheel is None:
            result.status = "fail"
            result.error = (
                f"expected exactly one .whl in {out_dir}, found "
                f"{sorted(p.name for p in out_dir.glob('*.whl'))}"
            )
            return result
        if sdist is None:
            result.status = "fail"
            result.error = (
                f"expected exactly one .tar.gz in {out_dir}, found "
                f"{sorted(p.name for p in out_dir.glob('*.tar.gz'))}"
            )
            return result

        result.wheel_size = wheel.stat().st_size
        result.sdist_size = sdist.stat().st_size

        # Parse name/version straight from METADATA so we report what the
        # built artifact actually contains (not what the pyproject says).
        metadata, _ = _parse_wheel_metadata(wheel)
        if metadata is not None:
            result.name = str(metadata.get("Name", "") or "")
            result.version = str(metadata.get("Version", "") or "")

        if not check_only:
            issues = validate_wheel(wheel)
            result.metadata_issues = issues
            if issues:
                result.status = "fail"
                result.error = "; ".join(issues)
                return result

    result.status = "ok"
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _human_size(n: int) -> str:
    if n == 0:
        return "-"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n = int(n / 1024)
    return f"{n} B"


def _clean_dist_dirs(packages: list[Path], console: Console) -> None:
    for pkg_dir in packages:
        dist = pkg_dir / "dist"
        if dist.exists():
            shutil.rmtree(dist)
            console.print(f"[dim]cleaned {dist.relative_to(REPO_ROOT)}[/]")


def _render_table(results: list[PackageResult]) -> Table:
    table = Table(title="Release dry-run", show_lines=False)
    table.add_column("package")
    table.add_column("version")
    table.add_column("wheel size", justify="right")
    table.add_column("sdist size", justify="right")
    table.add_column("status", justify="center")
    for r in results:
        status = "[green]ok[/]" if r.status == "ok" else "[red]fail[/]"
        rel = r.path.relative_to(REPO_ROOT)
        name = r.name or str(rel)
        table.add_row(
            f"{name} [dim]({rel})[/]",
            r.version or "-",
            _human_size(r.wheel_size),
            _human_size(r.sdist_size),
            status,
        )
    return table


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="release_dry_run.py",
        description="Validate every workspace package can produce a wheel.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Skip wheel-content inspection; only verify uv build exit code.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete each package's dist/ directory before building.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each uv build invocation.",
    )
    parser.add_argument(
        "--bump",
        metavar="VERSION",
        default=None,
        help=(
            "Rewrite every workspace pyproject's version = \"...\" to VERSION "
            "before building. Mutually exclusive with --check-only."
        ),
    )
    args = parser.parse_args(argv)
    if args.bump is not None and args.check_only:
        parser.error("--bump and --check-only are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()

    packages = discover_packages()
    if not packages:
        console.print("[red]no workspace packages found[/]")
        return 1

    console.print(f"Discovered [bold]{len(packages)}[/] workspace packages.")

    if args.clean:
        _clean_dist_dirs(packages, console)

    if args.bump is not None:
        changed = update_versions(args.bump, dry_run=False)
        console.print(
            f"Rewrote version to [bold]{args.bump}[/] in {len(changed)} pyproject.toml files."
        )

    results: list[PackageResult] = []
    for pkg_dir in packages:
        rel = pkg_dir.relative_to(REPO_ROOT)
        console.print(f"[bold]Building[/] {rel} ...")
        r = build_package(
            pkg_dir,
            check_only=args.check_only,
            verbose=args.verbose,
            console=console,
        )
        results.append(r)
        marker = "[green]ok[/]" if r.status == "ok" else "[red]fail[/]"
        console.print(f"  -> {marker}")

    console.print(_render_table(results))

    failures = [r for r in results if r.status != "ok"]
    if failures:
        first = failures[0]
        console.print(
            f"\n[red]First failure: {first.path.relative_to(REPO_ROOT)}[/]"
        )
        console.print(first.error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
