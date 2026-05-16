# Vendor neutrality

InferenceBench compares hardware vendors, cloud vendors, inference engines, and model providers. To remain credible, the project itself must not bias toward any of them. Vendor neutrality is the first non-negotiable rule in the codebase.

## What neutrality looks like in code

- **No vendor-specific shortcuts in core code.** Hardware fingerprinting, the harness, and the envelope are vendor-agnostic. Vendor-specific code lives in plugins behind a uniform interface.
- **No vendor-specific defaults that favor one stack.** Default flags and SLO templates are chosen for their representativeness, not for any one vendor's strengths.
- **No vendor-funded special treatment.** When a vendor sponsors hardware, the methodology page records the sponsorship. The result is published like any other.

## What neutrality looks like in publishing

- Every plugin must run on at least two vendors before its results land on a public leaderboard. (Phase 1 waives this rule for `llm.inference` while only one hardware class is available; the waiver is documented.)
- Methodology pages are versioned and reviewed.
- Disputes are handled transparently in public issues.

## Honest limits

InferenceBench is run by a small team. Phase 1 ships against one hardware class (H100 SXM5) on one inference engine (vLLM). That is a real limitation. We will not pretend otherwise to make the project look more multi-vendor than it is.

Phase 2 expands to additional hardware (MI300X, RTX 5090, M5 Max) and additional engines (SGLang, TensorRT-LLM, llama.cpp, MLX) as partnerships and engineering hours allow.

## How users keep us honest

- File issues when you see a vendor-biased default
- Bring up methodology concerns on the issue tracker
- Submit results from hardware we do not have access to
- Open a PR with a new plugin

## See also

- [Methodology](methodology.md)
- [Pareto frontiers](pareto.md)
- [Contributing](../community/contributing.md)
