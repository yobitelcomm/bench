# InferenceBench plugin registry

A curated JSON catalogue of known third-party plugins for the `bench` CLI.
The CLI's `bench plugin discover` command reads this file and shows what
plugins exist, what they do, and how to install them.

This is **not** a package index — packages are still distributed on PyPI.
The registry is a discovery channel: it lets users (and the CLI) answer
"what plugins are out there?" without scraping PyPI or knowing each
plugin's name in advance.

## Schema

The file is a single JSON object with three top-level keys:

| Key | Type | Description |
|---|---|---|
| `schema` | string | Schema identifier. Currently `inferencebench.plugin-registry.v1`. |
| `updated_iso` | string | ISO-8601 date the registry was last updated (`YYYY-MM-DD`). |
| `plugins` | array | Plugin entries (see below). |

Each plugin entry is an object with the following required fields:

| Field | Type | Description |
|---|---|---|
| `name` | string | Entry-point name as installed plugins expose it (e.g. `llm.inference`). Must be unique within the registry. |
| `package` | string | PyPI distribution name (e.g. `inferencebench-llm`). |
| `version` | string | Latest known version (semver). |
| `install` | string | Single-line install command users can copy/paste. |
| `modality` | enum | One of `llm`, `voice`, `code`, `embeddings`, `mt`, `other`. |
| `kind` | enum | One of `perf`, `quality`, `both`. |
| `repo` | string | Canonical source-repository URL. |
| `license` | string | SPDX identifier (e.g. `Apache-2.0`, `MIT`). |
| `status` | enum | One of `core`, `community`, `experimental`, `archived`. |
| `description` | string | One-line human-readable summary. |
| `engines_supported` | array of strings | Inference engines the plugin can drive (informational; may be empty). |
| `maintainer` | string | GitHub org/user that owns the package. |

### Status values

- **`core`** — maintained by `yobitelcomm`. Shipped in the `bench` monorepo.
- **`community`** — third-party plugin, PR-merged into the registry,
  passed curation review. Stable and recommended.
- **`experimental`** — third-party, accepted but beta-quality. Use at
  your own risk; APIs may change.
- **`archived`** — listed for historical context only; no longer
  maintained. Will not appear in default `bench plugin discover` output
  in a future release.

## Proposing additions (community plugins)

To add a plugin to the registry:

1. Fork the [yobitelcomm/bench](https://github.com/yobitelcomm/bench)
   repository.
2. Open `tools/plugin-registry/registry.json` and add a new entry to the
   `plugins` array. Keep entries sorted by `name`.
3. Bump the top-level `updated_iso` field to the current date.
4. Open a pull request with title `registry: add <name>`.
5. The PR description must include:
   - Link to the plugin's source repository.
   - Link to the published PyPI package.
   - A signed commit (or DCO sign-off) demonstrating you have rights to
     submit the entry.

A maintainer will review against the curation policy below and either
merge (entry becomes `community` or `experimental`) or request changes.

## Curation policy

Every accepted entry passes the following review gates:

1. **License check.** The plugin must declare an SPDX-recognised
   open-source licence (Apache-2.0, MIT, BSD-3-Clause, MPL-2.0, or
   compatible). Copyleft licences (AGPL, GPL) are not accepted for the
   `community` tier because they alter downstream re-distribution
   semantics; they may be listed as `experimental` with a footnote.
2. **Package authenticity.** Reviewers verify:
   - The PyPI package and the linked GitHub repository are owned by the
     same identity (matching maintainer email or org).
   - The PyPI package metadata's `Home-page` or `Project-URL` points back
     to the claimed repository.
   - The package has at least one signed release (sigstore/cosign,
     PEP 740 attestations, or GPG signature on the git tag).
3. **Plugin contract.** The package must declare an entry point under
   `inferencebench.plugins` and expose the standard plugin shape
   (`list_benchmarks`, `get_benchmark`, `validate`, `run`).
4. **Security review.** Reviewers spot-check the plugin source for
   obvious red flags (network calls to unexpected hosts, file-system
   writes outside the run directory, code that disables signing).
   Plugins doing anything unusual must declare it in the PR description.
5. **Naming.** The `name` field must not collide with any existing
   `core` plugin and should follow the `<modality>.<task>` convention.

First acceptance lands the entry as `experimental`. Two consecutive
quarterly registry refreshes without unresolved security issues
promote it to `community`.

## How the CLI consumes the registry

The CLI loads the registry in this priority order:

1. Path passed via `bench plugin discover --registry <path-or-url>`.
2. The user's local refresh cache at
   `~/.cache/inferencebench/plugin-registry.json` (written by
   `bench plugin discover --refresh <url>`).
3. The bundled copy shipped inside the CLI wheel at
   `inferencebench/data/plugin-registry.json` (a build-time copy of
   this file).

The bundled copy and `tools/plugin-registry/registry.json` are kept in
sync by `tests/test_plugin_registry_sync.py`, which fails the build if
they diverge.

## Hosted registry (Phase 2+)

A live mirror of this file will eventually be published at
`https://yobitelcomm.github.io/bench/plugin-registry.json` so plugin
authors can ship updates without users having to upgrade the `bench`
CLI. The `--refresh` flag already supports this URL today; the hosting
endpoint is reserved but not yet active.
