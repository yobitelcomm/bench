# inferencebench (CLI)

The `bench` command-line tool. `pip install inferencebench` gives you both `bench` and `inferencebench` aliases.

## Status

Phase 1, in development. Skeleton implemented; most commands are stubs.

## Quickstart (when complete)

```bash
pip install inferencebench
bench run llm.inference --model meta-llama/Llama-4-Maverick --engine vllm --hardware h100
bench verify ~/.cache/inferencebench/runs/latest/envelope.json
bench publish ~/.cache/inferencebench/runs/latest --to hf
```

See the [docs site](https://yobitelcomm.github.io/bench) for the full command reference (ships at v0.1).
