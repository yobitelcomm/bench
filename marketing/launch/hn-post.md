**Title:** Show HN: bench — vendor-neutral AI benchmarks with signed result envelopes

---

Every AI benchmark you read today has one of three problems.

It is single-vendor: NVIDIA AIPerf measures NVIDIA hardware on NVIDIA software stacks. The numbers are real, but the framing is not neutral.

It is single-modality: MLPerf Inference covers a fixed list of models and tasks, and its tail-latency methodology is a moving target between rounds.

Or it is closed and unreproducible: aggregators like Artificial Analysis publish leaderboards built from numbers nobody outside the company can rerun. You see a chart. You cannot see the request distribution, the hardware fingerprint, the engine config, or the seed.

None of these projects ship a result you can independently verify, byte-for-byte, against a public transparency log.

`bench` is an attempt at that missing piece: a vendor-neutral, open-source benchmark CLI where every result is a Sigstore-signed envelope you can verify on someone else's machine.

## What it does

`bench` is a Python 3.12 CLI (`pip install inferencebench`) that drives a benchmark, captures full hardware and software provenance, packages the measurements into a canonical envelope, signs it with Sigstore (keyless OIDC, logged in Rekor), and optionally publishes it as a permanent dataset on Hugging Face Hub.

The envelope captures, at minimum:

- DMI UUID, GPU PCI IDs and serials, VBIOS, BIOS (resizable BAR, above-4G), CPU microcode
- Driver, CUDA, NCCL versions
- Container image digest, `pip freeze` hash, git commit, `nvidia-smi -q` hash
- Dataset id and hash, seed, engine config hash
- Full TTFT and TPOT distributions (not just a P50), throughput, goodput-at-SLO, joules-per-token, $/M tokens
- A Sigstore signature bundle with the Rekor log index

Everything is hashed into one `fingerprint_sha256` so two runs from "the same machine" that are not actually the same machine fail to match. The full schema is in `envelope/` in the repo.

## A concrete example

```bash
pip install inferencebench

bench doctor             # refuses to run if the GPU is thermal-throttling
                         # or if ECC errors are present

bench run llm.inference \
  --model meta-llama/Llama-4-Maverick \
  --engine vllm --quant fp8 \
  --hardware h100 \
  --driver open-loop --rps 10 --duration 300 \
  --slo-template llm.standard

bench verify ~/.cache/inferencebench/runs/latest/envelope.json
bench publish ~/.cache/inferencebench/runs/latest --to hf
```

The envelope that lands on disk looks roughly like this (abridged):

```json
{
  "envelope_version": "v1",
  "suite_id": "llm.inference",
  "run_id": "01HXY...",
  "model": {"id": "meta-llama/Llama-4-Maverick", "revision": "..."},
  "engine": {"name": "vllm", "version": "0.7.2", "config_hash": "..."},
  "hardware_fingerprint": {
    "fingerprint_sha256": "9c3a...",
    "gpus": [{"model": "H100-SXM5-80GB", "pci_id": "...", "vbios": "..."}],
    "driver": "560.35.03", "cuda": "12.6", "nccl": "2.22.3"
  },
  "metrics": {
    "ttft_p50_ms": 142.0, "ttft_p99_ms": 280.3,
    "tpot_p50_ms": 18.5,
    "throughput_tok_per_s": 1842.1,
    "goodput_at_slo": 142.3,
    "joules_per_token": 0.32
  },
  "signature": {
    "method": "sigstore-cosign",
    "rekor_log_index": 12345,
    "bundle": "..."
  }
}
```

Anyone can take that file, run `bench verify <url>`, and get a pass/fail rooted in Sigstore's public transparency log. That is the wedge.

## What it is not, yet

Phase 1 is deliberately narrow.

- One engine: vLLM. SGLang, TensorRT-LLM, llama.cpp and MLX are Phase 2.
- One hardware tier shipped end-to-end: H100. AMD MI300X, RTX 5090 and Apple silicon are Phase 2, gated on hardware partnerships.
- One modality: LLM inference. Voice, vision, 3D, world models, agents and chip kernels are all sketched in the architecture but explicitly not in v0.1.

The plugin interface and envelope schema were designed for the full surface, not retrofitted. Adding a modality is a new package; adding an engine is a driver subclass. There is no benchmark hosted in `bench` today that secretly favors one stack — and if one ever ships, it should be filed as a bug.

The percentile math uses bootstrap CIs (1000 resamples, 95%), three independent process launches per cross-engine comparison, a convergence gate before measurement starts, and three discarded warmups. Open-loop and closed-loop drivers. This isn't novel; it's just the discipline most public benchmark numbers skip.

## How to try it

- Repo: https://github.com/yobitelcomm/bench
- Docs: https://yobitelcomm.github.io/bench
- Install: `pip install inferencebench`

What I would most like feedback on:

- The envelope schema (`envelope/envelope.v1.json`). Is there a field you would refuse to trust a result without?
- The convergence gate and warmup discipline in `harness/`. Too strict, too loose?
- Methodology gaps you have seen reward-hacked in other suites that we should preempt.

Open an issue or hop into the methodology-disputes channel on Discord. Critiques in public are the point.
