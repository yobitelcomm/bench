# Changelog

This page is intended to mirror the repository's `CHANGELOG.md` once it exists. Until then, releases are tracked here.

## Format

The project follows [Keep a Changelog](https://keepachangelog.com/) and uses [Semantic Versioning](https://semver.org/). The CLI, the envelope schema, and each plugin are versioned independently.

## [Unreleased]

Phase 1 development is in progress. Highlights so far:

- Typer-based `bench` CLI with `run`, `compare`, `publish`, `verify`, `leaderboard`, `doctor`, `cost`, and `plugin` subcommands
- Pydantic v2 envelope models with content hashing
- Hardware diagnostic (`bench doctor`) backed by NVML
- Plugin discovery via Python entry points
- This documentation site

The first published release will be `v0.1.0` on PyPI, targeting late 2026.

## See also

- [Repository CHANGELOG.md](https://github.com/yobitelcomm/bench/blob/main/CHANGELOG.md)
- [Releases on GitHub](https://github.com/yobitelcomm/bench/releases)
