**Title:** I built an open-source benchmark that signs every result and works offline

---

I got tired of comparing local-LLM numbers across blog posts where nobody publishes the engine config, the prompt distribution, the GPU temperature, or even the driver version. So I started building `bench` — a vendor-neutral benchmark CLI where every result is a Sigstore-signed envelope you can verify locally.

It's Apache 2.0, `pip install inferencebench`, runs offline (signing falls back to a local dev key if you don't want OIDC), and two people on two 4090s can run the same suite and produce envelopes that are byte-comparable.

## What's useful for a home/lab GPU operator

- **Local vLLM benchmarking.** Open-loop (Poisson, fixed RPS) and closed-loop (bounded concurrency) drivers. Three discarded warmups, convergence gate, bootstrap CIs on percentiles.
- **Joules per token, not just tokens/sec.** RAPL + NVML sampled at 50–100 ms. If you care which quant is more efficient on your power bill, this is the number.
- **Hardware fingerprint.** DMI UUID, GPU PCI IDs and serials, VBIOS, BIOS settings, driver, CUDA, NCCL — all SHA-256'd into the envelope. If your neighbor's "same 4090" is actually a different VBIOS, the fingerprints don't match.
- **Consumer GPUs are first-class.** 3090, 4090, 5090 are valid `--hardware` targets. Reference Phase 1 runs happen on H100 because that's what the project has access to, but the abstraction is per-device.

## The flow

```bash
pip install inferencebench

bench doctor
# [screenshot: bench doctor output showing GPU model, VBIOS,
#  current temp, ECC state, driver/CUDA/NCCL versions, and a
#  red "REFUSED: GPU clock throttling detected" if applicable]

bench run llm.inference.chatbot-short \
  --model Qwen/Qwen2.5-7B-Instruct \
  --engine vllm --quant bf16 \
  --hardware rtx-4090 \
  --sweep 1,4,16 --duration 90

bench verify ~/.cache/inferencebench/runs/latest/envelope.json
# Verified: signature OK, Rekor log index 12345, fingerprint matches.
```

`bench doctor` refuses to start a measurement if the GPU is thermal-throttling or showing ECC errors.

For scale reference, here's the same `chatbot-short` benchmark on the v0.1.0 reference H100 corpus (Qwen2.5-72B, TP=4, BF16):

| concurrency | throughput | TTFT p50 | joules/token |
|---:|---:|---:|---:|
| 1 | 56 tok/s | 24 ms | 37 |
| 4 | 234 tok/s | 46 ms | 9.0 |
| 16 | **891 tok/s** | 47 ms | **2.5** |

That's the same envelope schema your 4090 run will produce. The Pareto frontier (throughput vs joules) is what the leaderboard renders; vendor-neutral by construction.

## Phase 2 is where r/LocalLLaMA gets the rest

Today `bench` ships one engine (vLLM) and six modality plugins with reference envelopes (LLM inference, LLM quality, MT, code, vision, embeddings, voice). Phase 2 plan, in order: **llama.cpp** (GGUF, Metal, ROCm, Vulkan), **MLX** for Apple silicon, **AMD** (MI300X and consumer ROCm), then 3D / world-models / agents / robotics / chip kernels.

If you live mostly on llama.cpp or MLX, the right time to engage is now — the engine driver interface is small and getting locked down before v0.1.

## How to contribute

- **Hardware coverage**: if you have a GPU we don't, run the suite and PR the envelope to the reference set. Issue template is in the repo.
- **Methodology critique**: open an issue tagged `methodology`. We expect to be wrong sometimes; airing it is the point.
- **A new engine driver**: see `skills/new-plugin/SKILL.md` in the repo.

Links:

- Repo: https://github.com/yobitelcomm/bench
- Docs: https://yobitelcomm.github.io/bench

Happy to answer questions, especially about the envelope spec and where the methodology has obvious holes.
