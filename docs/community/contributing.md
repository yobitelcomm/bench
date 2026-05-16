# Contributing

External contributions are welcome. The project is in early Phase 1, so please open an issue before starting non-trivial work.

The canonical contributing guide lives in the repository at [CONTRIBUTING.md](https://github.com/yobitelcomm/bench/blob/main/CONTRIBUTING.md). Highlights below.

## Getting started

```bash
git clone https://github.com/yobitelcomm/bench
cd bench
uv sync --all-extras --dev
pre-commit install
make all
```

The `make all` target runs lint, type check, and the full test suite.

## What we welcome

- Bug fixes for documented issues
- New plugins, following the methodology review process
- Methodology improvements for existing benchmarks
- Hardware support for vendors we do not yet cover (MI300X, RTX 5090, M5 Max)
- Documentation fixes and improvements

## What we do not accept

- Changes that compromise vendor neutrality
- Benchmarks without signed envelopes
- New benchmarks without a methodology review
- Code without tests
- Changes that bypass the convergence gate or the warm-up discipline

## Workflow

1. Find or open an issue.
2. Branch off `main` with the project naming scheme: `<type>/<scope>/<ticket-id>-<short-description>`.
3. Write tests first when the spec is clear.
4. Open a PR using the template. CI must be green.
5. A maintainer reviews and merges.

## Conventional Commits

We enforce [Conventional Commits](https://www.conventionalcommits.org/). Examples:

```
feat(plugin-llm): add SGLang engine support
fix(envelope): correct content_hash canonical ordering
docs(quickstart): clarify HF Hub publish flow
```

## See also

- [Code of conduct](code-of-conduct.md)
- [CONTRIBUTING.md on GitHub](https://github.com/yobitelcomm/bench/blob/main/CONTRIBUTING.md)
- [Methodology](../concepts/methodology.md)
