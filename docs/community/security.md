# Security policy

The canonical policy lives in the repository at [SECURITY.md](https://github.com/yobitelcomm/bench/blob/main/SECURITY.md). This page mirrors it for the docs nav.

## Reporting a vulnerability

If you discover a security vulnerability in InferenceBench, please report it privately:

- **Email**: `security@yobitel.com`
- **Subject**: `[security] <one-line summary>`
- Include: reproduction steps, affected versions, suggested fix if any, and your name for credit (or "anonymous").

Do **not** open a public issue for security vulnerabilities. We will respond within 5 business days and coordinate disclosure.

## Supported versions

| Version | Supported |
|---|---|
| 0.x | yes |

Phase 1 is pre-1.0. Once we ship v1.0, we will commit to security patches for the latest minor version.

## Scope

In scope:

- The `bench` CLI and all sub-packages (`inferencebench-envelope`, `inferencebench-harness`, plugins).
- Our GitHub Actions workflows and CI configuration.
- The signing infrastructure (`bench-attest` when it exists).

Out of scope:

- Third-party inference engines we benchmark (vLLM, SGLang, etc.) — report to their teams.
- Third-party model providers (Anthropic, OpenAI, etc.) — report to their security teams.
- Issues in the docs site that are not a real software vulnerability.

## Supply chain

We sign every release with Sigstore. Verify a release tarball with:

```bash
cosign verify-blob \
    --certificate-identity-regexp "https://github.com/yobitelcomm/bench/.*" \
    --certificate-oidc-issuer https://token.actions.githubusercontent.com \
    --bundle <release>.bundle \
    <release>.tar.gz
```

We also publish SBOMs (Software Bill of Materials) via Syft alongside every release.

## Bug bounty

Not currently active. We will launch a HackerOne program in Phase 5+.

## See also

- [SECURITY.md on GitHub](https://github.com/yobitelcomm/bench/blob/main/SECURITY.md)
- [Code of conduct](code-of-conduct.md)
- [Contributing](contributing.md)
