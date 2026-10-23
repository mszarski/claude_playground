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


def decoy_copy_rate(examples: Iterable[Tuple[List[str], str, str]]) -> float:
    """
    Fraction of clips where an unspoken candidate name from the prompt shows up
    in the hypothesis (and not in the reference).

    Args:
        examples: (decoy names, reference, hypothesis) triples; clips without
            decoys are ignored
    """
    copied = total = 0
    for decoys, ref, hyp in examples:
        if not decoys:
            continue
        total += 1
        ref_p = f" {normalize_transcript(ref)} "
        hyp_p = f" {normalize_transcript(hyp)} "
        if any(f" {normalize_transcript(d)} " in hyp_p and f" {normalize_transcript(d)} " not in ref_p
               for d in decoys):
            copied += 1
    return copied / total if total else 0.0


def name_cer(examples: Iterable[Tuple[Dict[str, str], str]]) -> float:
    """
    Mean character error rate of slot values (e.g. vessel names) against the
    closest span of the hypothesis, ignoring spaces ("H T M S" == "HTMS").

    Unlike slot_accuracy this rewards near misses, which matters for names the
    model has never heard: "TREKLIER JUNIOR" for "CHARIKLIA JUNIOR" scores ~0.4.

    Args:
        examples: (slots, hypothesis) pairs
    """
    total, n = 0.0, 0
    for slots, hyp in examples:
        words = normalize_transcript(hyp, expand_letters=False).split()
        for value in slots.values():
            target = normalize_transcript(value, expand_letters=False).replace(" ", "")
            if not target:
                continue
            k = len(value.split())
            best = len(target)
            for w in range(1, k + 3):
                for i in range(max(1, len(words) - w + 1)):
                    best = min(best, edit_distance(list(target), list("".join(words[i:i + w]))))
            total += min(1.0, best / len(target))
            n += 1
    return total / n if n else 0.0
