# Multi-stage build for the InferenceBench CLI.
# Stage 1: build a wheel for every workspace package using uv.
# Stage 2: slim runtime with only the installed wheels.

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH
COPY . .
RUN uv build --all-packages --out-dir /wheels

FROM python:${PYTHON_VERSION}-slim AS runtime
LABEL org.opencontainers.image.title="InferenceBench"
LABEL org.opencontainers.image.source="https://github.com/yobitelcomm/bench"
LABEL org.opencontainers.image.licenses="Apache-2.0"
RUN useradd -m -u 1000 bench
USER bench
WORKDIR /home/bench
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --user /wheels/*.whl
ENV PATH=/home/bench/.local/bin:$PATH
ENTRYPOINT ["bench"]
CMD ["--help"]
