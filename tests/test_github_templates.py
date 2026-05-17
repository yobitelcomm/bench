"""GitHub templates and community docs sanity checks.

These guard the public-release polish: issue templates must be valid YAML,
the PR template must include the test-plan section we tell contributors to
fill in, and the docs site must ship the 10-minute tour page that
quickstart.md links to.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ISSUE_TEMPLATE_DIR = ROOT / ".github" / "ISSUE_TEMPLATE"
PR_TEMPLATE = ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"
TOUR_DOC = ROOT / "docs" / "tour.md"


def test_every_issue_template_yml_parses() -> None:
    """Every `.yml` under `.github/ISSUE_TEMPLATE/` must be valid YAML."""
    yml_files = sorted(ISSUE_TEMPLATE_DIR.glob("*.yml"))
    assert yml_files, f"No YAML issue templates found in {ISSUE_TEMPLATE_DIR}"
    for path in yml_files:
        with path.open(encoding="utf-8") as fh:
            try:
                yaml.safe_load(fh)
            except yaml.YAMLError as exc:  # pragma: no cover - failure path
                raise AssertionError(f"{path.name} is not valid YAML: {exc}") from exc


def test_pull_request_template_contains_test_plan() -> None:
    """The PR template must include a `## Test plan` section."""
    assert PR_TEMPLATE.exists(), f"Missing {PR_TEMPLATE}"
    body = PR_TEMPLATE.read_text(encoding="utf-8")
    assert "## Test plan" in body, "PR template is missing a `## Test plan` section"


def test_tour_doc_exists_and_mentions_bench_doctor() -> None:
    """The 10-minute tour page must exist and reference `bench doctor`."""
    assert TOUR_DOC.exists(), f"Missing {TOUR_DOC}"
    body = TOUR_DOC.read_text(encoding="utf-8")
    assert "bench doctor" in body, "docs/tour.md is missing the `bench doctor` walk-through"
