# Multi-vendor marathon

A single `bench` install can drive **9 production-class LLMs across 5 vendors and 10 benchmarks** in one afternoon on an 8×H100 box. This page is the writeup of one such run.

## Setup

- 8× NVIDIA H100 80GB HBM3 (TestBM)
- vLLM 0.21.0 serving every model except where noted
- bf16 weights, `--max-model-len 4096`
- 1 H100 for ≤9B models, 2× for Qwen2-VL, 4× for Llama-3.1-70B
- Each benchmark run was 20s of measurement, dev-key signed

## Results

Every row below is a real signed envelope. The full 50-envelope corpus is published as one dataset repo per run under [huggingface.co/Yobitel](https://huggingface.co/Yobitel). Verify any of them with:

```bash
pip install inferencebench inferencebench-hf-publisher
bench fetch hf://datasets/Yobitel/qwen-qwen2-5-7b-instruct__llm-inference-chatbot-short__019e3b88f8d6
bench verify ~/.cache/inferencebench/fetched/*.json \
  --dev-public-key trust/cosign-2026-05-18-marathon.pub
```

The public key is committed at [trust/cosign-2026-05-18-marathon.pub](https://github.com/yobitelcomm/bench/blob/main/trust/cosign-2026-05-18-marathon.pub) and mirrored at [huggingface.co/datasets/Yobitel/bench-trust-anchors](https://huggingface.co/datasets/Yobitel/bench-trust-anchors). All 50 envelopes pass audit against this key.

### Keyless-signed mirror

The same 50 envelopes are also published with **Sigstore keyless signatures** (no key file needed) at [huggingface.co/datasets/Yobitel/marathon-keyless-v0.0.2](https://huggingface.co/datasets/Yobitel/marathon-keyless-v0.0.2). They were re-signed end-to-end through GitHub Actions' OIDC token — see [`.github/workflows/keyless-sign-marathon.yml`](https://github.com/yobitelcomm/bench/blob/main/.github/workflows/keyless-sign-marathon.yml) — so each signature ties back to a specific workflow run that anyone can audit on the Rekor transparency log:

```bash
bench fetch hf://datasets/Yobitel/marathon-keyless-v0.0.2/<one-of-the-files>
bench verify ~/.cache/inferencebench/fetched/*.json \
  --require-issuer https://token.actions.githubusercontent.com \
  --require-identity-pattern 'github\.com/yobitelcomm/bench/'
```

See the [Sigstore keyless verify recipe](sigstore-verification.md) for the full story on what the policy flags do and why this is stronger than dev-key trust.

| Model | Vendor | tok/s | factual% | arith% | persona | chrF en→fr | HumanEval | MBPP | OCR | chartQA | reason% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | Meta | 228 | 100% | 75% | 0.96 | 0.77 | 100% | — | — | — | — |
| Llama-3.1-70B-Instruct | Meta | 195 | 100% | 100% | **1.00** | 0.88 | 80% | — | — | — | — |
| Qwen2.5-7B-Instruct | Alibaba | 541 | 100% | 88% | 0.92 | **0.89** | 100% | — | — | — | — |
| Qwen2.5-Coder-7B-Instruct | Alibaba | 529 | 100% | — | 0.76 | — | 100% | 100% | — | — | — |
| Qwen2-VL-7B-Instruct | Alibaba | 216 | 100% | — | — | — | — | — | 80% | 60% | — |
| Mistral-7B-Instruct-v0.3 | Mistral | 472 | 100% | 50% | 0.96 | 0.77 | 80% | — | — | — | — |
| Phi-3.5-mini-instruct | Microsoft | **716** | 100% | 75% | 0.96 | — | 100% | — | — | — | 20% |
| DeepSeek-Coder-V2-Lite | DeepSeek | 134 | 100% | — | 0.88 | — | 100% | 100% | — | — | — |
| gemma-2-9b-it | Google | 385 | 100% | 100% | — | 0.75 | 100% | — | — | — | — |

Bold = highest in column. `—` = benchmark not run for that model (vision models skip text-only suites; specialist coders skip MT; etc.).

## Things we found

- **Llama-3.1-70B is the most consistent generalist**: top of factual, arithmetic, persona-consistency, second on translation. Slowest tok/s in the cohort, naturally (4-way TP overhead).
- **Phi-3.5-mini is the throughput king**: 716 tok/s on a single H100 from a 3.8B-param model, but its `reasoning-mini` score (20%) is a scoring-strategy mismatch — Phi answers in prose ("The answer is fifteen") and the spec uses `exact_match` against `"15"`. This is real data, not a Phi failure: the right fix is a `numeric_match` scorer.
- **Coder specialists pay a generality tax**: Qwen2.5-Coder-7B is 100% on both HumanEval and MBPP but only 0.76 on persona-consistency vs 0.92 for the general Qwen2.5-7B.
- **Gemma's persona-consistency run failed** (ok_rate=0) because Gemma's chat template rejects the `system` role; the plugin currently sends the persona prompt as a system message. Known limitation — to be fixed by detecting Gemma family and merging system into the first user turn.
- **Real multimodal works**: Qwen2-VL-7B scored 80% on synthetic OCR and 60% on chart-QA from images generated programmatically in the plugin — the `vision.understanding` plugin's request/response shape was validated end-to-end against a real vision model.

## Reproduce it yourself

1. Get an 8×H100 (or smaller cluster — most of the cohort runs on 1 GPU).
2. Install bench + the LLM plugin:

   ```bash
   git clone https://github.com/yobitelcomm/bench
   cd bench
   uv sync --all-packages --dev --prerelease=allow
   ```

3. Start a model via vLLM (any of the 9 above; non-gated examples: Qwen2.5-7B-Instruct, Phi-3.5-mini-instruct, DeepSeek-Coder-V2-Lite-Instruct):

   ```bash
   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --gpu-memory-utilization 0.85
   ```

4. Run the full mini-suite:

   ```bash
   uv run python -c "from inferencebench.envelope import generate_dev_keypair; generate_dev_keypair('cosign.key')"

   for SPEC in llm.inference.chatbot-short llm.quality.factual-mini \
               llm.quality.arithmetic-mini llm.quality.persona-consistency-mini \
               llm.mt.flores-200-mini-en-fr code.generation.humaneval-mini; do
     bench run "$SPEC" \
       --model Qwen/Qwen2.5-7B-Instruct --engine vllm \
       --base-url http://localhost:8000/v1 \
       --signing-mode dev --dev-key cosign.key \
       --output ./envelopes
   done
   ```

5. Render the leaderboard:

   ```bash
   bench leaderboard --build --envelopes ./envelopes --out ./site
   bench audit ./envelopes
   bench summary ./envelopes
   ```

6. (Optional) Repeat with another model + `bench compare` to overlay:

   ```bash
   bench compare ./envelopes/a-*.json ./envelopes/b-*.json --report pareto
   ```

## What this doesn't yet cover

- True audio benchmarks against a real Whisper server — the `voice.transcription` plugin's wire format is validated against a stub HTTP server but the marathon didn't include a faster-whisper-server run.
- Llama-3.2-Vision is gated on a separate access list distinct from the Llama-3.1 family; we substituted Qwen2-VL-7B for the multimodal slot.
- Reasoning-mini's exact-match scorer punishes verbose models. Use `judge_llm` scoring or wait for the planned `numeric_match` strategy.
- Cost numbers reflect registered list prices, not measured spot rates. For self-hosted models cost is synthesized from the cheapest provider in the bundled pricing registry.
