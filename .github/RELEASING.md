# Releasing InferenceBench packages

> **Audience**: abishekvr (the only human who can cut a release in Phase 1).
> **Source of truth**: this doc + `.github/workflows/release.yml`.
>
> **Phase 1 rule**: every PyPI publish is manual, dry-run-first, single-package.
> See `HUMAN_REVIEW_GATES.md` item #3 (absolute prohibition on autopublish).

## What ships from this repo

| Input key | PyPI name                  | Source dir              | Notes                                                  |
|-----------|----------------------------|-------------------------|--------------------------------------------------------|
| `cli`     | `inferencebench`           | `cli/`                  | Installs `bench` + `inferencebench` binary aliases.    |
| `envelope`| `inferencebench-envelope`  | `envelope/`             | Canonical signed-envelope spec + Sigstore signer.      |
| `harness` | `inferencebench-harness`   | `harness/`              | Measurement engine: drivers, telemetry, percentiles.   |
| `llm`     | `inferencebench-llm`       | `plugins/llm-inference/`| First-party LLM inference plugin (vLLM in Phase 1).    |

Versions are independent. SemVer applies per package.

---

## 1. Prepare the release locally

Releases require a CHANGELOG entry and a version bump merged to `main` **before**
the workflow is invoked. Do this from a feature branch and open a PR:

```bash
# Example: bump the CLI to 0.1.0
make release-prep PACKAGE=cli
```

`make release-prep` (to be implemented in ticket 0035) should:

1. Prompt for the new version (or read it from `--bump=minor|patch|major`).
2. Update `cli/pyproject.toml`'s `version` field.
3. Move `## [Unreleased]` items in `cli/CHANGELOG.md` to a `## [X.Y.Z] — YYYY-MM-DD` section.
4. Commit on a `release/cli-vX.Y.Z` branch.
5. Open a PR titled `release(cli): vX.Y.Z` for human review.

Merge that PR to `main` before continuing.

---

## 2. Dry-run the publish workflow

From the GitHub UI:

1. Go to **Actions** → **Release** → **Run workflow**.
2. Select branch `main` (or the release tag commit).
3. Set:
   - `package`: e.g. `cli`
   - `dry_run`: **`true`** (default)
4. Click **Run workflow**.

The workflow will:

- Run lint + typecheck + the PR-CI pytest set.
- Build the named package with `uv build --package <name>`.
- Sigstore-sign the wheel + sdist via the GitHub OIDC token (keyless).
- Upload the signed artifacts as a workflow artifact named `signed-dist-<package>`.
- Print a summary listing what *would* be uploaded — but skip the PyPI step.

**Download `signed-dist-<package>` and inspect**:

- File names and version match what you expected.
- SHA-256 sums in `SHA256SUMS` look sane.
- `.sigstore`/`.sigstore.json` bundles are present.
- The wheel installs into a clean venv (`pip install dist/<wheel>`).

If anything is wrong, fix it and re-run the dry-run. Never skip step 2.

---

## 3. Promote to a real publish

Once the dry-run artifacts look correct:

1. Go to **Actions** → **Release** → **Run workflow** again.
2. Same `package` value.
3. Set `dry_run`: **`false`**.
4. The workflow will only proceed past the build/sign stage to the `publish`
   job, which is also gated on the `pypi` GitHub Environment — configure that
   environment with a required reviewer (= abishekvr) so a second click is
   needed before upload.

The `publish` job uses **PyPI Trusted Publishers (OIDC)** — there is no API
token in repo secrets. See the next section for one-time setup.

---

## 4. One-time PyPI setup (human action, required before first publish)

For each of the four PyPI projects, configure a Trusted Publisher in the
project settings on `pypi.org`:

- **Owner**: `yobitelcomm`
- **Repository name**: `bench`
- **Workflow filename**: `release.yml`
- **Environment name**: `pypi`

Plus the matching GitHub Environment in `Settings → Environments → New environment`:

- Name: `pypi`
- Required reviewers: `abishekvr`
- Deployment branches: `main` only

> **OPEN ITEM for human**: this Trusted-Publisher binding cannot be done from
> Claude Code — it has to be clicked through on pypi.org while logged in as the
> owner of each project. Track this in `TICKETS/phase-1/0033-release-workflow.md`.

---

## 5. Verify after publish

```bash
# In a fresh venv
python -m venv /tmp/verify-bench && . /tmp/verify-bench/bin/activate
pip install --no-cache-dir inferencebench=={VERSION}
bench --version
bench doctor
```

For the non-CLI packages:

```bash
pip install --no-cache-dir inferencebench-envelope=={VERSION}
python -c "import inferencebench.envelope; print(inferencebench.envelope.__version__)"
```

### Verify the Sigstore signature

Download the wheel and the matching `.sigstore` bundle from the workflow
artifact. Then:

```bash
pip install sigstore
sigstore verify identity \
  --bundle inferencebench-{VERSION}-py3-none-any.whl.sigstore \
  --cert-identity "https://github.com/yobitelcomm/bench/.github/workflows/release.yml@refs/heads/main" \
  --cert-oidc-issuer "https://token.actions.githubusercontent.com" \
  inferencebench-{VERSION}-py3-none-any.whl
```

A successful verify proves the wheel was built by *this* workflow on a commit
to `main`, not by anyone with a stolen API token.

---

## 6. Post-release

- Tag the release commit: `git tag cli/v{VERSION} && git push origin cli/v{VERSION}`.
- Draft a GitHub Release pointing at that tag, attach the signed artifacts.
- Announce per the Phase-1 plan (HN/r/LocalLLaMA — only after CLI v0.1.0+).

---

## Emergency: stop a release mid-flight

The workflow's `concurrency` group is `release-global` with
`cancel-in-progress: false`, so a started release will run to completion. If
you need to stop a publish:

1. **Before the `publish` job** — cancel the run from the Actions UI; only the
   `dist-*` and `signed-dist-*` artifacts will exist, nothing on PyPI.
2. **After publish has started** — you cannot unpublish from PyPI. Yank with
   `twine yank inferencebench {VERSION} --reason "..."` and immediately cut a
   `{VERSION}+1` with the fix.

---

## Why this is so manual

InferenceBench's whole moat is reproducibility and trust. A bad release (wrong
artifact, leaked key, malicious dep) corrodes the moat permanently. Two-click
manual is cheap insurance.
