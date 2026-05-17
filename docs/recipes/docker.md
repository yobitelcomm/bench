# Recipe: Run bench in a container

The `Dockerfile` at the repo root builds a self-contained image with the `bench` CLI and all bundled plugins installed. Use it when you want a clean, reproducible environment that doesn't touch the host Python install — typical in CI runners, on multi-tenant build hosts, or when sharing a quick benchmark with a teammate.

## 1. Build the image

From the repo root:

```bash
docker build -t yobitelcomm/bench:0.0.2 .
```

The build is multi-stage. Stage 1 uses `uv build --all-packages` to produce wheels for every workspace package (`cli`, `harness`, `envelope`, the plugins, the integrations). Stage 2 is a slim `python:3.12-slim` runtime that `pip install`s those wheels under a non-root `bench` user. The final image is ~120 MB compressed.

## 2. Sanity-check it

```bash
docker run --rm yobitelcomm/bench:0.0.2 --help
docker run --rm yobitelcomm/bench:0.0.2 list
docker run --rm yobitelcomm/bench:0.0.2 doctor
```

`--help`, `list`, and `doctor` need no external state, so they confirm the image is wired up correctly.

## 3. Use the helper script

`scripts/run_in_container.sh` wraps a standard `docker run` invocation:

```bash
./scripts/run_in_container.sh run llm.inference \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --engine vllm \
    --endpoint http://localhost:8000/v1
```

The helper:
- Mounts the current working directory at `/work` and `cd`'s into it (so envelopes the CLI writes are visible on the host).
- Uses `--network host` so the container can reach a vLLM/SGLang/TRT-LLM/llama.cpp/MLX endpoint running on `localhost` of the host.
- Picks the image from `$BENCH_IMAGE`, defaulting to `yobitelcomm/bench:latest`. Pin a specific tag when reproducibility matters:

```bash
BENCH_IMAGE=yobitelcomm/bench:0.0.2 ./scripts/run_in_container.sh list
```

## Caveats

- **Networking.** `--network host` is the simplest way to reach a local inference server. On Docker Desktop (macOS / Windows) `--network host` doesn't behave the same way — use `--add-host=host.docker.internal:host-gateway` and point `--endpoint` at `http://host.docker.internal:8000/v1` instead.
- **GPU access.** This image is CPU-only by design; the CLI itself doesn't need a GPU. The inference engine (vLLM, SGLang, TRT-LLM, llama.cpp, mlx_lm.server) runs *outside* the container and the CLI just talks to it over HTTP. If you want to run the engine in a container too, see the engine's own image.
- **Signing keys.** `bench publish` and signed-envelope flows need access to a Sigstore dev key or OIDC credentials. Mount the key explicitly: `-v ~/.config/bench/cosign.key:/home/bench/.config/bench/cosign.key:ro`. The `.dockerignore` deliberately excludes `cosign.key` / `cosign.pub` so they never end up baked into the image.
- **Output directories.** The CLI writes envelopes and per-request samples to `--output` (default `validation-runs/`). Mount that path in if you want results to persist on the host — the helper script handles this by mounting `$(pwd)` at `/work`.
