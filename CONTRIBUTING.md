# Contributing to InferenceBench

Thank you for your interest in contributing. This project is in early Phase 1 (solo dev + Claude Code). External PRs are welcome but please open an issue first to discuss scope.

## Getting started

```bash
git clone https://github.com/yobitelcomm/bench
cd bench
uv sync --all-extras --dev
pre-commit install
make all   # lint + typecheck + test
```

## Workflow

1. **Find or open an issue.** All work tracks to a ticket in `TICKETS/phase-N/` or an issue.
2. **Branch off `main`.** Use `<author>/<scope>/<short-description>` naming.
3. **Read [CONVENTIONS.md](CONVENTIONS.md)** before writing code. It's enforced via CI.
4. **Write tests first** when the spec is clear. See [skills/writing-tests/SKILL.md](skills/writing-tests/SKILL.md).
5. **Open a PR** following the [pull request template](.github/PULL_REQUEST_TEMPLATE.md).
6. **CI must be green.** Lint + typecheck + tests + security + license check.

## What kind of contributions we want

- **Bug fixes** for documented issues
- **Plugin additions** following [skills/new-plugin/SKILL.md](skills/new-plugin/SKILL.md) (Phase 2+ mainly)
- **Methodology improvements** to existing benchmarks — open as a `methodology-issue.md` first
- **Hardware support** if you have access to a vendor we don't (MI300X, RTX 5090, M5 Max, etc.)
- **Documentation fixes and improvements**

## What we won't accept

- Changes that compromise vendor neutrality
- Benchmarks without signed envelopes
- New benchmarks without a methodology review (the `benchmark-validator` agent)
- Changes that bypass the convergence gate or warmup discipline
- Code without tests

## Conventional Commits

We enforce [Conventional Commits](https://www.conventionalcommits.org/) via CI. Examples:

```
feat(plugin-llm): add SGLang engine support
fix(envelope): correct content_hash canonical ordering
docs(quickstart): clarify HF Hub publish flow
```

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating, you agree to abide by its terms.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Questions?

Open a [Discussion](https://github.com/yobitelcomm/bench/discussions) or reach out at `bench@yobitel.com`.
