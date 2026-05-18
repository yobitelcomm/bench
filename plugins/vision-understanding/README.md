# inferencebench-vision

Vision-language understanding plugin for the InferenceBench Suite.

Scores vision-language model answers against bundled image+question fixtures
using deterministic exact-match, substring-match, or LLM-as-judge strategies.
Mirrors the `llm.quality` plugin contract but exercises the multimodal
chat-completions request shape that every modern VLM endpoint (vLLM, SGLang,
OpenAI, Anthropic) accepts.

Suite ID: `vision.understanding`

## Multimodal request shape

Each fixture row pairs an image with a natural-language question. The plugin
constructs an OpenAI-compatible chat-completions request with image content
inline as a base64 data URL:

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "How many bars are in this chart?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }]
}
```

vLLM, SGLang, the OpenAI Chat Completions API and Anthropic's messages API
all accept this exact shape, so a single plugin works against any of them.

## Bundled benchmarks

- `vision.understanding.ocr-mini` — 5 short OCR-style read-text-from-image
  tasks against synthetic PNGs, substring-match scoring.
- `vision.understanding.chart-qa-mini` — 5 ChartQA-style numeric-extraction
  tasks against synthetic bar charts, exact-match scoring.
