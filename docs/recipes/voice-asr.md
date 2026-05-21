# Benchmark a Whisper ASR server

The `voice.transcription` plugin produces signed WER envelopes for any
[OpenAI-compatible audio-transcription endpoint](https://platform.openai.com/docs/api-reference/audio/createTranscription).
That covers the SaaS providers (OpenAI, Cohere) and the self-hosted servers
that follow the same wire format — primarily
[`faster-whisper-server`](https://github.com/fedirz/faster-whisper-server) and
forks. The recipe below uses `faster-whisper-server` running on a single GPU
and the bundled `voice.transcription.librispeech-clean-mini` benchmark (5 real
LibriSpeech test-clean utterances, ~18s of audio).

## Why a real ASR benchmark matters

A WER number means nothing without the audio it was measured against. The
envelope binds:

- the model id (`Systran/faster-whisper-large-v3` etc.)
- the engine kind and version
- the audio fixture set (hashed in `dataset.path` + per-file `sha256_16`)
- the hardware fingerprint
- a Sigstore or dev-key signature over the canonical hash of all of the above

So a downstream consumer can compare WER between two engines on the *same*
audio under the *same* conditions, instead of comparing your 2.1% on
LibriSpeech-test-clean to someone else's 2.8% on a private dataset.

## Setup

1. **Start the server** on a GPU box (one H100 / L40 / RTX 4090 is fine for
   whisper-large-v3, ~3GB VRAM at int8). The server exposes
   `POST /v1/audio/transcriptions` on port 8000:

   ```bash
   # Easiest: the upstream Docker image.
   docker run --rm --gpus all -p 8000:8000 \
     -e WHISPER__MODEL=Systran/faster-whisper-large-v3 \
     -e WHISPER__COMPUTE_TYPE=float16 \
     fedirz/faster-whisper-server:latest-cuda
   ```

   Wait for `INFO: Application startup complete.` in the logs (~10-30s while
   the model downloads on first run).

2. **Install bench + the voice plugin** on the box that will *call* the
   server (can be the same machine):

   ```bash
   git clone https://github.com/yobitelcomm/bench
   cd bench
   uv sync --all-packages --dev --prerelease=allow
   ```

3. **Mint a one-shot dev key** (no Sigstore network for the demo):

   ```bash
   uv run python -c "from inferencebench.envelope import generate_dev_keypair; generate_dev_keypair('cosign.key')"
   ```

## Run the benchmark

```bash
uv run bench run voice.transcription.librispeech-clean-mini \
  --model Systran/faster-whisper-large-v3 \
  --engine whisper-http \
  --base-url http://localhost:8000/v1 \
  --signing-mode dev --dev-key cosign.key \
  --output ./envelopes
```

Expected runtime: ~10s on H100, ~30s on an L40. The plugin sends each
fixture WAV in order, waits for the transcription, scores it against the
reference, then emits one signed envelope summarizing the run.

## Inspect the result

```bash
uv run bench summary ./envelopes/voice.transcription.librispeech-clean-mini-*.json
```

You should see something like:

```
suite:           voice.transcription.librispeech-clean-mini v1.0.0
model:           Systran/faster-whisper-large-v3
engine:          whisper-http
n_samples:       5
ok_rate:         1.00
wer_mean:        0.02    # 2 % — whisper-large-v3 is very strong on LS-clean
wer_p50:         0.00
wer_p95:         0.07
total_p50_ms:    420
```

Whisper-large-v3 on LibriSpeech test-clean is published at ~2 % WER; a single
H100 box should reproduce that band on this 5-utt slice (within ±2 % given
the small N). Higher WER is usually one of:

- **Endpoint default language is wrong** — pass `--engine-arg
  language=en` (see your server's docs).
- **Server still warming up** — first call after startup can include model
  compile time; re-run.
- **Reference has a quirk the normalizer doesn't strip** — the bundled
  scorer lowercases + strips ASCII punctuation; if the engine emits numerals
  ("M.A." → "MA" vs the reference "M A") that will count as substitution.
  Open an issue if you hit a normalization case that's not Whisper-style.

## Verify the envelope

```bash
uv run bench verify ./envelopes/voice.transcription.librispeech-clean-mini-*.json \
  --dev-public-key cosign.key.pub
```

```
OK
  method:           dev-key
  content_hash:     <sha256>
  suite:            voice.transcription.librispeech-clean-mini v1.0.0
  model:            Systran/faster-whisper-large-v3
  engine:           whisper-http
```

For keyless verification of a community-published envelope, see the
[Sigstore keyless verify recipe](sigstore-verification.md).

## What this benchmark isn't

- **Not a multi-engine bake-off.** It runs against whatever endpoint you
  point it at. To compare engines, run the same spec against two engines and
  combine with `bench compare`.
- **Not a license certificate.** LibriSpeech is CC BY 4.0; the bundled 5
  utterances are sourced from
  [`hf-internal-testing/librispeech_asr_dummy`](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy).
  If you publish numbers, attribute appropriately.
- **Not enough audio to make a paper claim.** 18s of speech is for smoke
  validation. For headline numbers, use a longer subset of LibriSpeech +
  CommonVoice + AMI + earnings22 — write a custom JSONL pointing at the same
  schema and `bench run` will pick it up.
