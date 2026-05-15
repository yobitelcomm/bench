# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability in InferenceBench, please report it privately:

- **Email**: `security@yobitel.com`
- **Subject**: `[security] <one-line summary>`
- Include: reproduction steps, affected versions, suggested fix if any, your name for credit (or "anonymous")

Do **not** open a public issue for security vulnerabilities. We will respond within 5 business days and coordinate disclosure.

## Supported versions

| Version | Supported          |
|---------|--------------------|
| 0.x     | :white_check_mark: |

Phase 1 is pre-1.0. Once we ship v1.0, we'll commit to security patches for the latest minor version.

## Scope

In scope:
- The `bench` CLI and all sub-packages (`inferencebench-envelope`, `inferencebench-harness`, plugins)
- Our GitHub Actions workflows and CI configuration
- The signing infrastructure (`bench-attest` when it exists)

Out of scope:
- Third-party inference engines we benchmark (vLLM, SGLang, etc.) — report to their teams
- Third-party model providers (Anthropic, OpenAI, etc.) — report to their security teams
- Issues in our docs site that aren't a real software vulnerability

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

Not currently active. Phase 5+ we'll launch a HackerOne program.
