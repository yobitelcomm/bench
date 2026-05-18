"""``bench plugin`` — manage benchmark plugins.

Subcommands: list, init, install, info, discover.
Plugin discovery via Python entry points (installed) and a curated JSON
registry (available to install).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from importlib import metadata, resources
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()
err_console = Console(stderr=True)

_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*$")

_REGISTRY_RESOURCE = "plugin-registry.json"
_VALID_MODALITIES = {"llm", "voice", "code", "embeddings", "mt", "other"}
_VALID_STATUSES = {"core", "community", "experimental", "archived"}


def _discover_plugins() -> list[metadata.EntryPoint]:
    """Return all registered plugin entry points (Phase 1: typically empty)."""
    try:
        return list(metadata.entry_points(group="inferencebench.plugins"))
    except TypeError:
        # Pre-3.10 compat path — shouldn't trigger on 3.12
        return list(metadata.entry_points().get("inferencebench.plugins", []))  # type: ignore[attr-defined]


@app.command("list")
def list_plugins() -> None:
    """List installed plugins."""
    eps = _discover_plugins()
    if not eps:
        console.print("[yellow]No plugins installed.[/yellow]")
        console.print("Install one: [bold]pip install inferencebench-llm[/bold]")
        return

    table = Table(title="Installed plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Module", style="dim")
    table.add_column("Distribution", style="green")
    for ep in eps:
        dist = ep.dist.name if ep.dist else "?"
        table.add_row(ep.name, ep.value, dist)
    console.print(table)


@app.command("init")
def init_plugin(
    name: Annotated[str, typer.Argument(help="New plugin name (e.g. 'voice').")],
    kind: Annotated[str, typer.Option("--kind", help="perf, quality, or both.")] = "both",
    modality: Annotated[str, typer.Option("--modality", help="llm, voice, video, 3d, ...")] = "llm",
) -> None:
    """Scaffold a working plugin package under ``./plugins/<name>/``.

    The scaffolded plugin is end-to-end runnable: it ships a builtin "echo"
    engine that returns canned completions, so users can install it and
    immediately ``bench run <name>.smoke --signing-mode dev`` to produce a
    signed envelope. Replace the echo engine with the real workload to wire
    a new modality.
    """
    if not _NAME_RE.match(name):
        err_console.print(
            f"[red]Invalid plugin name:[/red] {name!r} — must match [a-z][a-z0-9-]* "
            "(lowercase, no underscores)."
        )
        raise typer.Exit(code=1)

    target = Path.cwd() / "plugins" / name
    if target.exists():
        err_console.print(f"[red]Target directory already exists:[/red] {target}")
        raise typer.Exit(code=1)

    snake = name.replace("-", "_")
    cap_name = "".join(part.capitalize() for part in name.split("-"))
    class_name = f"{cap_name}Plugin"
    eff_modality = modality or "llm"
    eff_kind = kind or "both"

    subs = {
        "name": name,
        "Name": cap_name,
        "snake": snake,
        "pkg": snake,  # backwards-compat alias for old templates
        "cls": class_name,
        "modality": eff_modality,
        "kind": eff_kind,
    }

    src_pkg_dir = target / "src" / f"inferencebench_{snake}"
    tests_dir = target / "tests"
    src_pkg_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    (target / "pyproject.toml").write_text(
        _PYPROJECT_TEMPLATE.format(**subs), encoding="utf-8"
    )
    (target / "README.md").write_text(
        _README_TEMPLATE.format(**subs), encoding="utf-8"
    )
    (src_pkg_dir / "__init__.py").write_text(
        _INIT_TEMPLATE.format(**subs), encoding="utf-8"
    )
    (src_pkg_dir / "schemas.py").write_text(
        _SCHEMAS_TEMPLATE.format(**subs), encoding="utf-8"
    )
    (src_pkg_dir / "plugin.py").write_text(
        _PLUGIN_TEMPLATE.format(**subs), encoding="utf-8"
    )
    (tests_dir / "test_plugin.py").write_text(
        _TEST_TEMPLATE.format(**subs), encoding="utf-8"
    )

    console.print(f"[green]Scaffolded plugin[/green] [bold]{name}[/bold] at {target}")
    console.print("Next steps:")
    console.print(f"  [bold]pip install -e ./plugins/{name}[/bold]")
    console.print(
        f"  [bold]bench run {name}.smoke --signing-mode dev --dev-key cosign.key[/bold]"
    )


# --------------------------------------------------------------------------- #
# Templates                                                                   #
# --------------------------------------------------------------------------- #
_PYPROJECT_TEMPLATE = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "inferencebench-{name}"
version = "0.0.0"
description = "InferenceBench plugin: {name}"
readme = "README.md"
requires-python = ">=3.12"
license = {{ text = "Apache-2.0" }}
dependencies = [
    "inferencebench-envelope",
    "inferencebench-harness",
    "pydantic~=2.9",
]

[project.entry-points."inferencebench.plugins"]
"{name}" = "inferencebench_{snake}.plugin:{cls}"

[tool.hatch.build.targets.wheel]
packages = ["src/inferencebench_{snake}"]
"""

_README_TEMPLATE = """\
# inferencebench-{name}

InferenceBench plugin for the `{name}` suite (modality: {modality}, kind: {kind}).
Generated by `bench plugin init`.

## Install

```bash
pip install -e .
```

## Quickstart

The scaffolded plugin ships with a builtin **echo** engine that returns canned
completions, so the suite runs end-to-end out of the box:

```bash
bench plugin init {name}            # already done if you're reading this
cd plugins/{name} && pip install -e .
bench envelope keygen --out cosign.key
bench run {name}.smoke --signing-mode dev --dev-key cosign.key
```

The first invocation produces a signed envelope under
``./runs/<run-id>/envelope.json``.

## Implementing a real workload

Replace the ``_echo_complete`` function in
``src/inferencebench_{snake}/plugin.py`` with your real model invocation, and
add real benchmark specs to ``list_benchmarks()``. See
``plugins/llm-inference`` for the canonical multi-engine example.
"""

_INIT_TEMPLATE = '''\
"""InferenceBench {name} plugin package."""

from inferencebench_{snake}.plugin import {cls}
from inferencebench_{snake}.schemas import BenchmarkSpec, EngineKind, RunContext

__all__ = ["BenchmarkSpec", "EngineKind", "RunContext", "{cls}"]
'''

_SCHEMAS_TEMPLATE = '''\
"""Pydantic schemas for the {name} plugin.

These mirror the shape used by the canonical ``inferencebench-llm`` plugin so
that the CLI's ``_resolve_plugin_schemas`` discovery (which looks for
``RunContext`` and ``EngineKind`` at the package top level) works unchanged.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class EngineKind(StrEnum):
    """Engines this plugin can drive.

    The scaffold ships with a single in-process ``ECHO`` engine that returns
    canned text — replace with real engines as you implement them.
    """

    ECHO = "echo"


class BenchmarkSpec(BaseModel):
    """One benchmark — minimal scaffold shape.

    Real plugins extend this with dataset / driver / metrics fields. We keep
    the surface tight here so the generated plugin runs without any external
    configuration.
    """

    model_config = ConfigDict(extra="forbid")
    benchmark_id: Annotated[str, Field(min_length=1)]
    suite_version: Annotated[str, Field(pattern=r"^\\d+\\.\\d+\\.\\d+(-[\\w.]+)?$")]
    description: str = ""
    modality: Literal["{modality}"] = "{modality}"
    kind: Literal["perf", "quality", "both"] = "{kind}"
    slo_template: str = "{name}.standard"


class RunContext(BaseModel):
    """Per-invocation context (where to send requests, where to write results).

    Mirrors the llm-inference plugin's shape so the CLI's run command can
    construct one without special-casing.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    model_id: Annotated[str, Field(min_length=1)] = "echo-model"
    model_revision: Annotated[str, Field(min_length=7, max_length=40)] = "unknown00"
    engine_kind: EngineKind = EngineKind.ECHO
    engine_version: str = ""
    base_url: str = ""
    api_key: str = ""
    quantization_format: str = ""
    hardware_class: str = ""
    output_dir: Path = Path("./runs")
    extra: dict[str, str | int | float | bool] = Field(default_factory=dict)
'''

_PLUGIN_TEMPLATE = '''\
"""{cls} — InferenceBench plugin for the ``{name}`` suite.

Modality: {modality}
Kind:     {kind}

Implements the plugin contract every InferenceBench plugin must satisfy:

- :meth:`list_benchmarks` — what benchmark specs are bundled with this plugin
- :meth:`get_benchmark` — look one up by id
- :meth:`validate` — fast sanity check before run
- :meth:`run` — execute a spec and produce a signed envelope

The scaffold ships with an in-process *echo* engine: ``_echo_complete`` just
returns a canned reply. That's deliberate — it means ``bench run
{name}.smoke`` produces a real signed envelope on a fresh machine with no
external services, GPU, or dataset access. Replace ``_echo_complete`` with
your real workload to make this a real benchmark.
"""

from __future__ import annotations

from pathlib import Path

from inferencebench.envelope import (
    DatasetSpec as EnvDatasetSpec,
)
from inferencebench.envelope import (
    EngineConfig,
    Envelope,
    EnvelopeBuilder,
    ModelConfig,
    SigningMode,
    sign_envelope,
)
from inferencebench.harness import (
    Percentiles,
    Sample,
    collect_hardware_fingerprint,
    collect_software_provenance,
)
from inferencebench_{snake}.schemas import BenchmarkSpec, EngineKind, RunContext

_SMOKE_BENCHMARK = BenchmarkSpec(
    benchmark_id="{name}.smoke",
    suite_version="1.0.0",
    description=(
        "Smoke benchmark for the {name} suite. Drives the builtin echo engine "
        "and produces a signed envelope with deterministic latency metrics."
    ),
)

_ECHO_REPLIES: tuple[str, ...] = (
    "echo: hello",
    "echo: this is the {name} smoke benchmark",
    "echo: replace me with a real engine",
    "echo: still running",
    "echo: done",
)


def _echo_complete(prompt: str, *, idx: int) -> str:
    """Builtin echo engine — returns a canned reply per request index.

    Replace with real model invocation (HTTP call, local model, ...) to turn
    this scaffold into a real benchmark.
    """
    _ = prompt  # unused in the echo engine
    return _ECHO_REPLIES[idx % len(_ECHO_REPLIES)]


class {cls}:
    """Entry point for the ``{name}`` benchmark suite."""

    suite_id = "{name}"
    version = "0.0.0"
    description = (
        "InferenceBench {name} plugin (modality: {modality}, kind: {kind}). "
        "Scaffolded by `bench plugin init`; ships with a builtin echo engine."
    )

    def list_benchmarks(self) -> list[BenchmarkSpec]:
        """Return the BenchmarkSpec list shipped with this plugin."""
        return [_SMOKE_BENCHMARK]

    def get_benchmark(self, benchmark_id: str) -> BenchmarkSpec:
        """Look up a benchmark spec by id."""
        for spec in self.list_benchmarks():
            if spec.benchmark_id == benchmark_id:
                return spec
        msg = f"benchmark_id not found: {{benchmark_id}}"
        raise KeyError(msg)

    def validate(self, spec: BenchmarkSpec, context: RunContext) -> list[str]:
        """Return a list of human-readable warnings (empty = OK)."""
        warnings: list[str] = []
        if context.engine_kind != EngineKind.ECHO:
            warnings.append(
                f"engine_kind '{{context.engine_kind.value}}' is not implemented; "
                "the scaffold only supports the builtin echo engine."
            )
        if not spec.benchmark_id.startswith("{name}."):
            warnings.append(
                f"benchmark_id '{{spec.benchmark_id}}' does not match suite prefix '{name}.'"
            )
        return warnings

    def run(self, spec: BenchmarkSpec, context: RunContext) -> Envelope:
        """Execute the benchmark and return a SIGNED envelope.

        Drives the builtin echo engine for ``n_samples`` deterministic
        requests, builds an Envelope from the resulting Percentiles, and
        signs it via ``sign_envelope``. Signing config is read from
        ``context.extra`` (``signing_mode``, ``dev_key_path``).
        """
        n_samples = int(context.extra.get("n_samples", 5))
        samples: list[Sample] = []
        for idx in range(n_samples):
            prompt = f"echo prompt #{{idx}}"
            reply = _echo_complete(prompt, idx=idx)
            samples.append(
                Sample(
                    request_idx=idx,
                    arrival_ms=float(idx * 10),
                    start_ms=float(idx * 10),
                    ttft_ms=10.0,
                    total_ms=100.0,
                    tpot_ms=1.0,
                    tokens_in=len(prompt.split()),
                    tokens_out=len(reply.split()),
                    cost_usd=0.0,
                    finish_reason="stop",
                    ok=True,
                )
            )

        envelope = self._build_envelope(spec, context, samples)
        signing_mode = str(context.extra.get("signing_mode", "dev"))
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

    def _build_envelope(
        self,
        spec: BenchmarkSpec,
        context: RunContext,
        samples: list[Sample],
    ) -> Envelope:
        hw = collect_hardware_fingerprint()
        sw = collect_software_provenance()

        ttft = Percentiles([s.ttft_ms for s in samples], bootstrap=False)
        tpot = Percentiles([s.tpot_ms for s in samples], bootstrap=False)
        total = Percentiles([s.total_ms for s in samples], bootstrap=False)

        metrics: dict[str, float | int | str | None] = {{
            "ttft_p50_ms": ttft.p50,
            "ttft_p99_ms": ttft.p99,
            "tpot_p50_ms": tpot.p50,
            "tpot_p99_ms": tpot.p99,
            "total_p50_ms": total.p50,
            "total_p99_ms": total.p99,
            "n_samples": float(len(samples)),
            "tokens_out_total": float(sum(s.tokens_out for s in samples)),
        }}

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
                version=context.engine_version or "0.0.0",
                config_hash="0" * 64,
            ),
            hardware_fingerprint=hw,
            software_provenance=sw,
            dataset=EnvDatasetSpec(id="{name}.echo", hash="0" * 64),
            seed=42,
            metrics=metrics,
            slo_template=spec.slo_template,
        )
        return builder.build()
'''

_TEST_TEMPLATE = '''\
"""Smoke tests for the generated {name} plugin."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from inferencebench.envelope import generate_dev_keypair
from inferencebench_{snake} import {cls}, EngineKind, RunContext

if TYPE_CHECKING:
    from inferencebench.envelope import Envelope


def test_plugin_suite_id() -> None:
    assert {cls}().suite_id == "{name}"


def test_plugin_lists_smoke_benchmark() -> None:
    specs = {cls}().list_benchmarks()
    assert len(specs) >= 1
    assert all(s.benchmark_id.startswith("{name}.") for s in specs)


def test_plugin_run_produces_signed_envelope(tmp_path: Path) -> None:
    """End-to-end: the scaffolded plugin runs and produces a signed envelope."""
    key_path = tmp_path / "cosign.key"
    generate_dev_keypair(key_path)

    plugin = {cls}()
    spec = plugin.list_benchmarks()[0]
    ctx = RunContext(
        model_id="echo-test",
        engine_kind=EngineKind.ECHO,
        output_dir=tmp_path / "runs",
        extra={{
            "signing_mode": "dev",
            "dev_key_path": str(key_path),
        }},
    )

    envelope: Envelope = plugin.run(spec, ctx)

    assert envelope.signature is not None
    assert envelope.signature.bundle != ""
    assert envelope.signature.method == "dev-key"
    numeric_metrics = [v for v in envelope.metrics.values() if isinstance(v, (int, float))]
    assert numeric_metrics, "expected at least one numeric metric in the envelope"
'''


# --------------------------------------------------------------------------- #
# Registry discovery                                                          #
# --------------------------------------------------------------------------- #
def _cache_registry_path() -> Path:
    """Return the user-cache path for a registry refreshed via --refresh."""
    override = os.environ.get("BENCH_CACHE_ROOT")
    root = Path(override) if override else Path.home() / ".cache" / "inferencebench"
    return root / "plugin-registry.json"


def _bundled_registry_text() -> str:
    """Return the registry JSON shipped inside the CLI wheel."""
    return resources.files("inferencebench").joinpath(
        f"data/{_REGISTRY_RESOURCE}"
    ).read_text(encoding="utf-8")


def _parse_registry(raw: str, *, source: str) -> dict[str, Any]:
    """Parse and minimally validate a registry document.

    ``source`` is purely for error messages so users can tell which file
    failed to parse.
    """
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"registry at {source} is not valid JSON: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(data, dict):
        msg = f"registry at {source}: top-level must be a JSON object"
        raise ValueError(msg)
    plugins = data.get("plugins")
    if not isinstance(plugins, list):
        msg = f"registry at {source}: 'plugins' must be a list"
        raise ValueError(msg)
    return data


def _read_url(url: str, *, timeout: float = 10.0) -> str:
    """Fetch a remote registry document. Only ``http(s)`` schemes accepted."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        msg = f"unsupported URL scheme: {parsed.scheme!r}"
        raise ValueError(msg)
    req = urllib.request.Request(  # noqa: S310 - scheme validated above
        url, headers={"User-Agent": "bench-plugin-discover"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - scheme validated
        data: bytes = resp.read()
    return data.decode("utf-8")


def _load_registry(
    override: str | None,
    *,
    prefer_cache: bool = True,
) -> tuple[dict[str, Any], str]:
    """Load the registry. Returns ``(parsed, source_label)``.

    Resolution order:

    1. ``override`` — explicit path or URL. Required to load if set.
    2. Local refresh cache at ``~/.cache/inferencebench/plugin-registry.json``
       (only if ``prefer_cache`` and the file exists).
    3. The bundled copy shipped inside the CLI wheel.
    """
    if override is not None:
        parsed = urllib.parse.urlparse(override)
        if parsed.scheme in {"http", "https"}:
            raw = _read_url(override)
            return _parse_registry(raw, source=override), f"url:{override}"
        path = Path(override).expanduser()
        if not path.exists() or not path.is_file():
            msg = f"registry path not found: {path}"
            raise FileNotFoundError(msg)
        raw = path.read_text(encoding="utf-8")
        return _parse_registry(raw, source=str(path)), f"path:{path}"

    if prefer_cache:
        cache = _cache_registry_path()
        if cache.exists() and cache.is_file():
            raw = cache.read_text(encoding="utf-8")
            return _parse_registry(raw, source=str(cache)), f"cache:{cache}"

    raw = _bundled_registry_text()
    return _parse_registry(raw, source="<bundled>"), "bundled"


def _installed_plugin_names() -> set[str]:
    """Return entry-point names of plugins installed in the current env."""
    return {ep.name for ep in _discover_plugins()}


def _filter_plugins(
    plugins: list[dict[str, Any]],
    *,
    installed_filter: bool,
    available_filter: bool,
    modality: str | None,
    status: str | None,
    installed_names: set[str],
) -> list[dict[str, Any]]:
    """Apply the discover-CLI filter flags."""
    out: list[dict[str, Any]] = []
    for entry in plugins:
        name = str(entry.get("name", ""))
        if installed_filter and name not in installed_names:
            continue
        if available_filter and name in installed_names:
            continue
        if modality is not None and str(entry.get("modality", "")) != modality:
            continue
        if status is not None and str(entry.get("status", "")) != status:
            continue
        out.append(entry)
    return out


@app.command("discover")
def discover_plugins(
    installed: Annotated[
        bool,
        typer.Option(
            "--installed/--no-installed",
            help="Show only plugins currently installed locally.",
        ),
    ] = False,
    available: Annotated[
        bool,
        typer.Option(
            "--available/--no-available",
            help="Show only plugins not yet installed locally.",
        ),
    ] = False,
    modality: Annotated[
        str | None,
        typer.Option(
            "--modality",
            help=(
                "Filter by modality: llm, voice, code, embeddings, mt, other."
            ),
        ),
    ] = None,
    status: Annotated[
        str | None,
        typer.Option(
            "--status",
            help="Filter by status: core, community, experimental, archived.",
        ),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option(
            "--json/--no-json",
            help="Emit the (filtered) registry as JSON instead of a table.",
        ),
    ] = False,
    registry: Annotated[
        str | None,
        typer.Option(
            "--registry",
            help=(
                "Override the registry source. Accepts a local file path or "
                "an http(s) URL. Default: the bundled registry shipped in "
                "the CLI wheel (or the refresh cache if present)."
            ),
        ),
    ] = None,
    refresh: Annotated[
        str | None,
        typer.Option(
            "--refresh",
            help=(
                "Fetch a fresh registry from URL, write it to the local "
                "cache at ~/.cache/inferencebench/plugin-registry.json, "
                "then use it. Subsequent invocations prefer the cached "
                "version over the bundled one."
            ),
        ),
    ] = None,
) -> None:
    """Discover known InferenceBench plugins.

    Reads a curated JSON registry of plugins (the bundled copy by default)
    and renders it as a Rich table. Use ``--installed`` / ``--available``
    to cross-reference against locally installed entry points,
    ``--modality`` / ``--status`` to filter by metadata, and ``--json``
    for machine-readable output.
    """
    if modality is not None and modality not in _VALID_MODALITIES:
        err_console.print(
            f"[red]invalid --modality:[/red] {modality!r} "
            f"(valid: {', '.join(sorted(_VALID_MODALITIES))})"
        )
        raise typer.Exit(code=2)
    if status is not None and status not in _VALID_STATUSES:
        err_console.print(
            f"[red]invalid --status:[/red] {status!r} "
            f"(valid: {', '.join(sorted(_VALID_STATUSES))})"
        )
        raise typer.Exit(code=2)
    if installed and available:
        err_console.print(
            "[red]--installed and --available are mutually exclusive.[/red]"
        )
        raise typer.Exit(code=2)

    refreshed_source: str | None = None
    if refresh is not None:
        try:
            raw = _read_url(refresh)
            # Validate before persisting so we don't cache garbage.
            _parse_registry(raw, source=refresh)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            err_console.print(f"[red]failed to refresh registry from {refresh}:[/red] {exc}")
            raise typer.Exit(code=2) from exc
        cache_path = _cache_registry_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(raw, encoding="utf-8")
        refreshed_source = f"refresh:{refresh}"

    try:
        data, source = _load_registry(registry)
    except (FileNotFoundError, ValueError, OSError, urllib.error.URLError) as exc:
        err_console.print(f"[red]failed to load registry:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    if refreshed_source is not None:
        source = refreshed_source

    plugins_raw = data.get("plugins", [])
    plugins: list[dict[str, Any]] = [p for p in plugins_raw if isinstance(p, dict)]

    installed_names = _installed_plugin_names()
    filtered = _filter_plugins(
        plugins,
        installed_filter=installed,
        available_filter=available,
        modality=modality,
        status=status,
        installed_names=installed_names,
    )

    if json_out:
        out_doc = {
            "schema": data.get("schema"),
            "updated_iso": data.get("updated_iso"),
            "source": source,
            "plugins": filtered,
        }
        console.print_json(json.dumps(out_doc))
        return

    if not filtered:
        console.print(
            f"[yellow]no plugins match the given filters[/yellow]  (source: {source})"
        )
        return

    title = (
        f"InferenceBench plugin registry "
        f"(updated {data.get('updated_iso', '?')}, source: {source})"
    )
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("name", style="cyan")
    table.add_column("modality", style="green")
    table.add_column("status", style="magenta")
    table.add_column("install", style="bold")
    table.add_column("repo", style="dim")
    for entry in filtered:
        name = str(entry.get("name", ""))
        marker = " [green](installed)[/green]" if name in installed_names else ""
        table.add_row(
            f"{name}{marker}",
            str(entry.get("modality", "")),
            str(entry.get("status", "")),
            str(entry.get("install", "")),
            str(entry.get("repo", "")),
        )
    console.print(table)
    console.print(f"[dim]registry source: {source}[/dim]")


@app.command("install")
def install_plugin(
    package: Annotated[str, typer.Argument(help="Plugin package name (e.g. inferencebench-llm).")],
) -> None:
    """Install a plugin from PyPI.

    Phase 1 stub — ticket 0028. (User can already do this with pip directly.)
    """
    err_console.print(
        f"[yellow][stub][/yellow] bench plugin install [bold]{package}[/bold] — "
        "not yet implemented in v0.0.0 (ticket 0028). Use [bold]pip install[/bold] instead."
    )


@app.command("info")
def info_plugin(
    name: Annotated[str, typer.Argument(help="Plugin name to introspect.")],
) -> None:
    """Show information about an installed plugin."""
    eps = _discover_plugins()
    matched = [ep for ep in eps if ep.name == name]
    if not matched:
        err_console.print(f"[red]Plugin not found:[/red] {name}")
        err_console.print(f"Installed: {[ep.name for ep in eps] or 'none'}")
        raise typer.Exit(code=1)

    ep = matched[0]
    console.print(f"[bold]{ep.name}[/bold]")
    console.print(f"  module:        {ep.value}")
    if ep.dist:
        console.print(f"  distribution:  {ep.dist.name} {ep.dist.version}")
        meta = ep.dist.metadata
        if meta:
            console.print(f"  summary:       {meta.get('Summary', '<none>')}")
            console.print(f"  homepage:      {meta.get('Home-page', '<none>')}")
