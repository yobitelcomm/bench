# Install

InferenceBench is a Python package. The `bench` binary and the `inferencebench` alias both land in your `PATH`.

## Quickest — PyPI (v0.0.2, recommended)

```bash
pip install inferencebench inferencebench-llm
bench --version
bench plugins
```

The 4 core packages on PyPI today: `inferencebench` (the CLI), `inferencebench-envelope` (schema + signing), `inferencebench-harness` (drivers + telemetry + metrics), `inferencebench-llm` (vLLM perf plugin). This is enough to run any LLM perf benchmark against a vLLM/SGLang endpoint and to verify any signed envelope.

## Recommended for daily use — uv tool

If you have [`uv`](https://docs.astral.sh/uv/) installed, this puts `bench` on your PATH globally without the venv dance:

```bash
uv tool install inferencebench --with inferencebench-llm
```

Now `bench --help` works from any shell, any cwd. Upgrade later with `uv tool upgrade inferencebench`.

## With all plugins (in-progress; 4 of 12 on PyPI today)

The full plugin matrix has 12 packages: 4 core + 8 plugins/integrations. As of v0.0.2 the 4 core packages are on PyPI; the other 8 (`inferencebench-quality`, `-mt`, `-code`, `-voice`, `-embeddings`, `-vision`, `-hf-publisher`, `-leaderboard`) are pending publication on a PyPI rate-limit window and will land in v0.0.3.

Until then, the cleanest way to get the full set is to install from a clone:

```bash
git clone https://github.com/yobitelcomm/bench
cd bench
uv sync --all-packages --dev --prerelease=allow
uv run bench --help
```

Or inject the in-tree wheels into a `uv tool` install:

```bash
git clone https://github.com/yobitelcomm/bench
cd bench
for pkg in envelope harness plugins/llm-inference plugins/llm-quality plugins/llm-mt \
           plugins/code-generation plugins/voice-transcription plugins/embeddings-retrieval \
           plugins/vision-understanding integrations/hf-publisher tools/leaderboard; do
  (cd "$pkg" && uv build)
done

uv tool install inferencebench \
  --with inferencebench-llm \
  --with ./dist/inferencebench_quality-*.whl \
  --with ./dist/inferencebench_mt-*.whl \
  --with ./dist/inferencebench_code-*.whl \
  --with ./dist/inferencebench_voice-*.whl \
  --with ./dist/inferencebench_embeddings-*.whl \
  --with ./dist/inferencebench_vision-*.whl \
  --with ./dist/inferencebench_hf_publisher-*.whl \
  --with ./dist/inferencebench_leaderboard-*.whl \
  --prerelease=allow
```

## Verify the install

```bash
bench --version
bench plugins
bench list
```

You should see at least `llm.inference` listed under plugins, and the 5 bundled benchmark specs it ships.

## Optional extras

| Extra | Adds | When you need it |
|---|---|---|
| `[publish]` | `inferencebench-hf-publisher` (after v0.0.3) | If you want `bench publish --to hf` |
| `[leaderboard]` | `inferencebench-leaderboard` (after v0.0.3) | If you want `bench leaderboard --build` or `bench dashboard` |
| `[fixtures]` | `datasets~=4.0` | If you want `bench fixtures fetch <hf-dataset>` |
| `[all]` | All of the above | One-shot full-fat install |

```bash
pip install 'inferencebench[fixtures]'      # works today
pip install 'inferencebench[all]'           # works after v0.0.3
```

## Supported platforms

| Platform | Status |
|---|---|
| Linux x86_64 with NVIDIA H100 + driver 580.x | **Validated** (50-envelope marathon, 2026-05-18) |
| Linux x86_64, other NVIDIA GPUs (RTX 4090, A100) | Best-effort; SLO multipliers in `bench doctor --show-slo` adjust thresholds per GPU class |
| Linux ARM (Grace, Ampere Altra) | Phase 2 |
| macOS (Apple Silicon) | Phase 2 — MLX engine adapter is in tree but unvalidated against real `mlx_lm.server` |
| Windows | Not planned |

## Python versions

Python 3.12 or newer. Older versions are not supported.

## Uninstall

```bash
pip uninstall inferencebench inferencebench-envelope inferencebench-harness inferencebench-llm
# or, if installed via uv tool:
uv tool uninstall inferencebench
```
