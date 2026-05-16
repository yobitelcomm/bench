# Recipe: verify and replay

Reproducibility is the product's moat. Anyone with a signed envelope and access to compatible hardware should be able to (a) verify the envelope's signature and content hash and (b) re-run the benchmark to produce a new envelope they can diff against the original.

## 1. Fetch

Grab the envelope you want to scrutinise. Hugging Face Hub, an HTTPS mirror, or a local file all work:

```bash
bench fetch hf://datasets/yobitel-bench-results/llama-3.1-8b__chatbot-short__abcdef123456
```

The fetched payload is validated against the `Envelope` schema before the command declares success. The local cache lives at `~/.cache/inferencebench/fetched/`.

## 2. Verify

```bash
bench verify ~/.cache/inferencebench/fetched/3f9c1a2b8e7d.json
```

`bench verify` recomputes the content hash from the envelope minus the signature block, then validates the signature. Sigstore keyless and dev ed25519 keys are both supported. Any mismatch is a hard failure:

```
FAIL  ~/.cache/inferencebench/fetched/3f9c1a2b8e7d.json
  method:  cosign-dev
  reason:  content hash mismatch (stored=60be8efd6d21..., recomputed=9c2f0a14...)
```

There are no warnings — verification either passes or it doesn't.

## 3. Replay

The envelope records every input needed to re-run the benchmark (suite id, model, engine, dataset, seed, quantization, SLO template). What it deliberately omits is the live engine endpoint, because that's host-specific. Point `bench replay` at your own engine:

```bash
bench replay ~/.cache/inferencebench/fetched/3f9c1a2b8e7d.json \
  --base-url http://localhost:8000/v1 \
  --output ./replay-results
```

The command verifies the source envelope first (refuses to replay an unverified envelope unless you pass `--no-verify`), spins up the same plugin configuration, and produces a new signed envelope. The replay summary table shows source vs. replay side by side for the identity fields and headline metrics.

## 4. Diff the replay

```bash
bench diff \
  ~/.cache/inferencebench/fetched/3f9c1a2b8e7d.json \
  ./replay-results/<hash>.json
```

If the replay landed on substantially different numbers, the diff table will surface it. Acceptable cross-host variation is a few-percent band; anything larger usually points at a hardware difference, an engine version mismatch, or a dataset hash drift — all of which the envelope captures and the diff context-match block will warn about.

## Why this matters

Most "benchmarks" in the wild are screenshots. A signed envelope is a verifiable contract:

- The hardware fingerprint says exactly what silicon ran the test.
- The software provenance pins the engine version, CUDA toolkit, driver, kernel.
- The dataset hash makes "the harness used a different ShareGPT subset" detectable, not invisible.
- The Sigstore signature makes tampering with any of the above detectable.

Anyone who can run `bench verify` + `bench replay` + `bench diff` can independently check whether a published number holds up on their hardware.

## Where to go next

- [bench verify reference](../cli/bench-verify.md)
- [bench replay reference](../cli/bench-replay.md)
- [The signed envelope](../concepts/envelope.md)
- [Reproducibility concept](../concepts/reproducibility.md)
