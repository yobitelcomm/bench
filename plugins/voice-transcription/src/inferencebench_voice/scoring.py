"""Deterministic scoring strategies for the voice-transcription plugin.

Three pure functions, each ``(reference, hypothesis) -> float`` in ``[0.0, 1.0]``.
WER and CER are **error rates** — 0.0 means a perfect transcript, 1.0 means
nothing matched. Exact-match is the same forgiving compare used by the
llm-quality plugin (strip + lowercase).

No real ASR is invoked; these are pure-Python edit-distance functions on
already-decoded text.
"""

from __future__ import annotations

import re

# Standard ASCII punctuation Whisper, OpenAI audio, and Cohere transcribe all
# emit in their outputs. Stripped before tokenizing so "doesn't." in the hyp
# doesn't substitute against "doesn't" in the ref. Apostrophes are stripped too
# because LibriSpeech / FLEURS references encode "doesn't" with a real
# apostrophe but Whisper occasionally returns a Unicode curly one (U+2019).
_PUNCT_RE = re.compile(r"[.,!?;:\"'`‘’“”()\[\]\-]+")  # noqa: RUF001 — curly quotes are the point


def _normalize_text(s: str) -> str:
    """Casefold + strip ASR-irrelevant punctuation + collapse whitespace.

    The same normalizer must run on both reference and hypothesis for WER/CER
    to be meaningful. This is the minimal viable normalization — for
    Whisper-canonical strict scoring use the openai-whisper EnglishTextNormalizer
    (which also expands contractions, normalizes numerals, etc.); we deliberately
    keep this small to avoid the optional dep.
    """
    s = _PUNCT_RE.sub(" ", s.lower())
    return " ".join(s.split())


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Classic O(len(a) * len(b)) edit distance over token sequences.

    Substitutions, insertions, and deletions all cost 1. Used by both
    :func:`wer` (tokens = words) and :func:`cer` (tokens = single characters).
    Returns the integer edit distance.
    """
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, ai in enumerate(a, start=1):
        curr[0] = i
        for j, bj in enumerate(b, start=1):
            cost = 0 if ai == bj else 1
            curr[j] = min(
                prev[j] + 1,  # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev
    return prev[len(b)]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: word-level Levenshtein distance / reference word count.

    Standard ASR metric. Lower is better. Returns 0.0 when reference and
    hypothesis are token-identical (case-insensitive), and is clamped to
    [0, 1] — insertions can push WER above 1.0 in the literature but the
    envelope contract assumes [0, 1] for aggregation, so we cap. An empty
    reference is a degenerate input: returns 0.0 when hypothesis is also
    empty, 1.0 otherwise (everything is an insertion).
    """
    ref_tokens = _normalize_text(reference).split()
    hyp_tokens = _normalize_text(hypothesis).split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    distance = _levenshtein(ref_tokens, hyp_tokens)
    return min(1.0, distance / len(ref_tokens))


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate: char-level Levenshtein / reference char count.

    Companion to :func:`wer` for languages / contexts where word boundaries
    are noisy (e.g. compound words, Mandarin). Lower is better. Whitespace
    is collapsed before character-level diffing so trailing newlines don't
    inflate the score. Capped at 1.0 for the same reason as WER.
    """
    ref_chars = list(_normalize_text(reference))
    hyp_chars = list(_normalize_text(hypothesis))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    distance = _levenshtein(ref_chars, hyp_chars)
    return min(1.0, distance / len(ref_chars))


def exact_match(reference: str, hypothesis: str) -> float:
    """Return 0.0 iff hypothesis == reference (after strip + lowercase), else 1.0.

    Reported as an **error rate** to keep the direction consistent with WER
    / CER inside this plugin — 0.0 means matched, 1.0 means didn't.
    """
    return 0.0 if reference.strip().lower() == hypothesis.strip().lower() else 1.0


SCORERS = {
    "wer": wer,
    "cer": cer,
    "exact_match": exact_match,
}
