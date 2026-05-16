# inferencebench-hf-publisher

Publishes signed InferenceBench envelopes to the [Hugging Face Hub](https://huggingface.co/) as dataset repos under the `yobitel-bench-results` organisation. This is the public, citable, verifiable home of every result emitted by `bench publish --to hf`.

## Status

Phase 1 active development. Tickets 0029-0030.

## What it does

1. Slugifies the envelope's model id, suite id, and run hash into a deterministic dataset repo id.
2. Uploads `envelope.json`, optional `traces.parquet`, and a generated `README.md` with YAML frontmatter (the HF dataset card).
3. Optionally appends a backlink entry to the source model card's metadata (never modifies the visible body).

See [skills/hf-publishing/SKILL.md](../../skills/hf-publishing/SKILL.md) for the full design.

## Quick start

```python
from inferencebench_hf_publisher import publish_envelope_to_hf

result = publish_envelope_to_hf(envelope, hf_token=os.environ["HF_TOKEN"])
print(result.url)
```

Pass `dry_run=True` to compute the planned URL without hitting HF Hub — useful in tests and CI smoke runs.
