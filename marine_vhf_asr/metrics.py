"""
Evaluation metrics: word error rate and slot (critical entity) accuracy.

Plain WER treats "CHANNEL ONE SIX" -> "CHANNEL ONE SEVEN" as a single
substitution, but on the radio that one word sends a vessel to the wrong
channel. Slot accuracy checks that each safety-critical value (vessel name,
channel, position, MMSI, ...) appears verbatim in the hypothesis.
"""

from typing import Dict, Iterable, List, Tuple

from .normalize import normalize_transcript


def edit_distance(ref: List[str], hyp: List[str]) -> int:
    """Levenshtein distance between two token sequences."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        curr = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            curr[j] = min(
                prev[j] + 1,            # deletion
                curr[j - 1] + 1,        # insertion
                prev[j - 1] + (r != h)  # substitution / match
            )
        prev = curr
    return prev[-1]


def corpus_wer(pairs: Iterable[Tuple[str, str]], normalize: bool = True) -> float:
    """
    Corpus-level WER over (reference, hypothesis) pairs.

    Returns:
        Total word edits divided by total reference words
    """
    errors = 0
    words = 0
    for ref, hyp in pairs:
        if normalize:
            ref, hyp = normalize_transcript(ref), normalize_transcript(hyp)
        r, h = ref.split(), hyp.split()
        errors += edit_distance(r, h)
        words += len(r)
    return errors / words if words else 0.0


def slot_accuracy(
    examples: Iterable[Tuple[Dict[str, str], str]],
) -> Dict[str, float]:
    """
    Fraction of slot values recovered verbatim, per slot type.

    Args:
        examples: (slots, hypothesis) pairs where slots maps slot type to its
            normalized spoken value, e.g. {"channel": "ONE SIX"}

    Returns:
        Dict of slot type -> accuracy, plus "all" for the micro-average
    """
    hits: Dict[str, int] = {}
    totals: Dict[str, int] = {}
    for slots, hyp in examples:
        padded = f" {normalize_transcript(hyp)} "
        for name, value in slots.items():
            totals[name] = totals.get(name, 0) + 1
            if f" {normalize_transcript(value)} " in padded:
                hits[name] = hits.get(name, 0) + 1

    result = {name: hits.get(name, 0) / n for name, n in totals.items()}
    total = sum(totals.values())
    result["all"] = sum(hits.values()) / total if total else 0.0
    return result
