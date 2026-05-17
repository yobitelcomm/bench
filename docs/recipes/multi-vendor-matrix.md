# Recipe: multi-vendor matrix

Drive one benchmark across multiple endpoints from a single config and a single command. The classic use case: Llama vs Qwen on two vLLM endpoints, captured side-by-side at conc=1 and conc=16 so you can compare throughput, TTFT, and energy at the same load.

## 1. Define the targets

`matrix.yaml`:

```yaml
schema: inferencebench.matrix.v1
suite_id: llm.inference.chatbot-short
duration_s: 60
sweep: [1, 16]
targets:
  - name: llama-vllm
    model: meta-llama/Llama-3.1-8B-Instruct
    engine: vllm
    base_url: http://localhost:8000/v1
    quant: fp16
  - name: qwen-vllm
    model: Qwen/Qwen2.5-7B-Instruct
    engine: vllm
    base_url: http://localhost:8001/v1
    quant: fp16

  # Uncomment to add a hosted-OpenAI comparison.
  # Set OPENAI_API_KEY in your shell before invoking bench matrix; the target
  # is skipped (yellow warning) when the env var is missing rather than failing
  # the whole matrix.
  # - name: openai-gpt4o
  #   model: gpt-4o-mini
  #   engine: openai
  #   base_url: https://api.openai.com/v1
  #   api_key_env: OPENAI_API_KEY
```

`name` is the filename prefix on the resulting envelopes. `engine` is the engine kind enum exposed by the plugin — Phase 1 supports `vllm` (shipping), `sglang` (skeleton), `llama-cpp`, and `openai` (provider-hosted, OpenAI-compatible).

## 2. Run the matrix

```bash
bench matrix matrix.yaml --output ./matrix-results
```

Each `(target, point)` produces a signed envelope under `./matrix-results/<target>-c<point>-<hash>.json`. Real conc=16 numbers from `validation-runs/2026-05-16-cross-model-corpus/`:

```
                                 Matrix results
 target        point  throughput_tok_per_s  ttft_p50_ms  ok_rate  envelope                            status
 llama-vllm    1      122.2                 13.98        1.000    llama-vllm-c1-814953250c16.json       ✓
 llama-vllm    16     1384                  41.69        1.000    llama-vllm-c16-60be8efd6d21.json      ✓
 qwen-vllm     1      120.0                 13.40        1.000    qwen-vllm-c1-07b69e640395.json        ✓
 qwen-vllm     16     1362                  40.98        1.000    qwen-vllm-c16-8d7ef1b17fb7.json       ✓
```

A missing API-key env var (e.g. for the commented OpenAI target) yields a `skip` row rather than aborting the matrix; an engine error yields an `✗` row and, with `--continue-on-error` on (default), the rest of the matrix proceeds.

## 3. Read the corpus

The eight envelopes — really four when only `vllm` is enabled — slot directly into the rest of the toolchain:

```bash
bench summary ./matrix-results
bench compare ./matrix-results/*.json --report pareto
bench history ./matrix-results --metric throughput_tok_per_s
```

At conc=16 the two vLLM endpoints land within 1.5 % on throughput and within 1 % on TTFT — both about 1380 tok/s, both around 0.70 J/tok — so the matrix surfaces the model gap as roughly architectural noise at saturation. At conc=1 Qwen's `joules_per_token` opens a real lead.

## 4. Wire it into a comparison report

```bash
bench compare ./matrix-results/*.json --report json | \
  jq '[.runs[] | {model: .model_id, conc: .metrics.concurrency, tput: .metrics.throughput_tok_per_s, j_per_tok: .metrics.joules_per_token}]'
```

The same envelopes feed [`bench leaderboard`](../cli/bench-leaderboard.md) and a regression baseline check — everything downstream treats matrix outputs the same as `bench run` outputs.

## Where to go next

- [bench matrix reference](../cli/bench-matrix.md) — full schema + flag table
- [Recipes: cross-model](cross-model.md) — the single-command precursor (manual two-run loop)
- [Vendor neutrality](../concepts/vendor-neutrality.md)
