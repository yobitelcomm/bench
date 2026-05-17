#!/usr/bin/env bash
# Convenience: run `bench` inside the container, mounting cwd at /work.
set -euo pipefail
IMAGE="${BENCH_IMAGE:-yobitelcomm/bench:latest}"
exec docker run --rm -it \
    --network host \
    -v "$(pwd)":/work \
    -w /work \
    "$IMAGE" "$@"
