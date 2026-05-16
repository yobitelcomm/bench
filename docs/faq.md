# FAQ

## What does InferenceBench measure?

Inference performance and efficiency across throughput, latency, cost, energy, and (where applicable) quality. Phase 1 covers LLM serving. Phase 2 expands to voice, video, image, embeddings, time-series, robotics, and agents.

## How is this different from MLPerf?

MLPerf focuses on tightly-controlled training and inference scenarios run by vendor engineers. InferenceBench focuses on day-to-day serving comparisons that any practitioner can run on their own hardware, with the full hardware fingerprint and Sigstore-signed envelope so the result is independently verifiable.

## How is this different from LMSYS Arena or LMArena?

LMSYS Arena measures user preference between model outputs. InferenceBench measures the inference system: throughput, latency, cost, and energy. The two are complementary.

## Why a signed envelope?

Without a signature, a result is just a number in a blog post. With a Sigstore-signed envelope and a Rekor log entry, anyone can verify exactly what produced the result. That is the bar for credible third-party comparisons.

## Why not use a self-signed key?

Self-signed signatures only prove that whoever has the key signed the thing. They do not prove who that party is. Sigstore keyless OIDC binds the signature to a verifiable identity (a GitHub Actions workflow, a GitHub user). Self-signed is fine for dev; published results use Sigstore.

## What hardware can I run on?

Phase 1 ships fully tested against Linux x86_64 with NVIDIA H100. The driver loads on other NVIDIA GPUs but the engine configs are not tuned for them. AMD (MI300X), consumer GPUs (RTX 5090), and Apple Silicon (M5 Max) are Phase 2.

## Can I use a hosted endpoint?

Yes. The `llm.inference` plugin supports hosted endpoints through LiteLLM. The hardware fingerprint then identifies the local driver machine; the engine and model identity in the envelope identify the remote endpoint.

## Why discard three warm-up runs?

Cold-start variance on large models is severe — JIT, paged-attention KV cache, allocator behavior. Three warm-up runs plus a convergence gate (CoV < 5% across the last 30 requests) gets us into the steady-state regime before measurement begins.

## Why open-loop Poisson?

Production traffic arrives independently of the system's current load. Closed-loop drivers (where a new request only fires after the previous one completes) systematically under-report tail latency. Open-loop Poisson at a target arrival rate is the standard correct way to characterize serving systems.

## What is `goodput_at_slo`?

The throughput rate at which the SLO template is still satisfied. Peak throughput is irrelevant if your tail latency blows past your service level — `goodput_at_slo` is what you actually plan capacity against.

## Where does the cost number come from?

The plugin uses a pricing snapshot taken at run time. Provider-side promotional pricing is not reflected. The snapshot URL and timestamp are recorded in the envelope so the source is auditable.

## Where do published results live?

On Hugging Face Hub. Every `bench publish --to hf` creates a dataset repo under `yobitel-bench-results/`. The static leaderboard at `yobitelcomm.github.io/bench` renders from this corpus.

## Is the methodology fixed?

No. Methodologies are versioned via `suite_version`. A breaking methodology change bumps the major version and ships in a new plugin release. Old envelopes remain valid; they record the version they were produced under.

## How do I dispute a result?

Open an issue with the envelope URL and the specific claim you dispute. The reproducibility path is to re-run with the same envelope inputs and compare. Methodological disputes are handled as versioned methodology changes.

## When will Studio / Enterprise tiers ship?

Not until Phase 2 at earliest. Phase 1 is CLI + envelope + one plugin only. Nothing about Studio or Enterprise exists in the codebase yet.

## See also

- [Methodology](concepts/methodology.md)
- [Reproducibility](concepts/reproducibility.md)
- [Pareto frontiers](concepts/pareto.md)
