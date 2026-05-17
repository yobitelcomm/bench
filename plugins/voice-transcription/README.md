# inferencebench-voice

Voice transcription (ASR) plugin for the InferenceBench Suite.

Phase-2-quality skeleton: produces signed envelopes via deterministic fixture
processing, with placeholders for real ASR engine invocation that future
revisions wire to faster-whisper-server / OpenAI audio / Cohere transcribe.

Suite ID: `voice.transcription`

Bundled benchmarks:

- `voice.transcription.fleurs-mini` — 5 short utterances, WER scoring.
- `voice.transcription.long-form` — 3 longer references (40-100 words), WER scoring.

The skeleton does NOT actually decode audio. Each fixture's reference is
deterministically corrupted (drop last word, replace with `"end"`) to produce
a known-WER hypothesis. The plugin contract surface is the production one;
future revisions replace the stub `_synthesise_hypothesis` call with a real
audio invocation against a configured `EngineKind`.
