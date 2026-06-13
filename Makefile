# Top-level convenience targets. Wraps `uv` for the monorepo.
# Run `make help` to see all targets.

.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-fast test-gpu test-paid all clean release-dry-run release-bump-0.0.2 release-notes

help: ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make <target>\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: ## uv sync the whole workspace.
	uv sync --all-extras --dev

lint: ## ruff check + ruff format --check (no auto-fix; matches CI).
	uv run ruff check .
	uv run ruff format --check .

format: ## ruff format + ruff check --fix.
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## mypy strict mode.
	uv run mypy

test-fast: ## Unit tests only (no integration, no GPU, no paid).
	uv run pytest -m "not integration and not gpu and not paid and not slow"

test: ## All PR-CI tests (unit + integration, CPU only).
	uv run pytest -m "not gpu and not paid"

test-gpu: ## GPU tests (run only on TestBM).
	uv run pytest -m "gpu"

test-paid: ## Paid-API tests (requires --paid token + cost budget).
	uv run pytest --paid -m "paid"

all: lint typecheck test ## Everything except GPU/paid.

clean: ## Remove caches.
	rm -rf .ruff_cache .mypy_cache .pytest_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +

release-dry-run: ## Build every workspace package and inspect each wheel.
	python scripts/release_dry_run.py

release-bump-0.0.2: ## Rewrite every workspace pyproject to 0.0.2, then dry-run build.
	python scripts/release_dry_run.py --bump 0.0.2

release-notes: ## Print GitHub-Release-ready markdown for v0.0.2.
	python scripts/release_notes.py --version 0.0.2
