# inferencebench-llm

LLM inference plugin for InferenceBench Suite. Drives requests through any
inference engine and produces a signed envelope with TTFT/TPOT/throughput/
goodput-at-SLO metrics.

## Status

Phase 1 ships **vLLM-only** support. SGLang, TensorRT-LLM, llama.cpp, MLX in Phase 2.

## Install

```bash
pip install inferencebench inferencebench-llm
```

## Quickstart

```bash
# Start a vLLM server (separately) on :8000, then:
bench run llm.inference \
    --model meta-llama/Llama-4-Maverick \
    --engine vllm \
    --endpoint http://localhost:8000/v1 \
    --concurrency 1,4,16,64 \
    --duration 300
```

See `docs/plugins/llm-inference.md` for the full reference.
