"""
Voxtral prompt construction shared by training and inference.

Voxtral's transcription prompt is
    <s>[INST][BEGIN_AUDIO][AUDIO]...[AUDIO][/INST]lang:en[TRANSCRIBE]
To bias transcription towards vessels known to be nearby (e.g. from AIS), an
optional context string is spliced in after the audio, where Voxtral's chat
format places a text instruction:
    ...[AUDIO]Candidate vessel names: A; B. Use a candidate only if it is spoken.[/INST]lang:en[TRANSCRIBE]
(see sources.prompt_text). Rows without context get exactly the stock
transcription prompt.
"""

import math
from typing import List, Optional

import torch

from .sources import SR

# Generous upper bounds on speaking rate for radio traffic, used to stop a
# runaway generation from flooding a transcript (seen on the untrained model).
MAX_CHARS_PER_SECOND = 40   # ~7 words/s; counted in characters so "370000000..." is caught too
MAX_TOKENS_PER_SECOND = 15


def build_inputs(processor, model_id: str, audios: List, contexts: Optional[List[str]] = None):
    """
    Build left-padded model inputs for a batch, with optional per-row context.

    Returns:
        BatchFeature-like dict with input_ids, attention_mask and audio features
    """
    inputs = processor.apply_transcription_request(
        language="en", audio=audios, model_id=model_id, sampling_rate=SR,
        format=["WAV"] * len(audios), return_tensors="pt",
    )
    if not contexts or not any(contexts):
        return inputs

    tok = processor.tokenizer
    end_inst = tok.convert_tokens_to_ids("[/INST]")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = []
    for ids, mask, ctx in zip(inputs["input_ids"].tolist(), inputs["attention_mask"].tolist(), contexts):
        ids = [i for i, m in zip(ids, mask) if m]  # drop left padding
        if ctx:
            at = len(ids) - 1 - ids[::-1].index(end_inst)
            ids = ids[:at] + tok(ctx, add_special_tokens=False)["input_ids"] + ids[at:]
        rows.append(ids)

    L = max(len(r) for r in rows)
    inputs["input_ids"] = torch.tensor([[pad] * (L - len(r)) + r for r in rows])
    inputs["attention_mask"] = torch.tensor([[0] * (L - len(r)) + [1] * len(r) for r in rows])
    return inputs


def max_new_tokens_for(audios: List, cap: int = 1024) -> int:
    """Token budget for a batch, from its longest clip."""
    longest = max(len(a) for a in audios) / SR
    return min(cap, math.ceil(longest * MAX_TOKENS_PER_SECOND) + 24)


def trim_runaway(text: str, duration_s: float) -> str:
    """Cut a hypothesis that is longer than anyone could say in `duration_s`."""
    limit = math.ceil(duration_s * MAX_CHARS_PER_SECOND) + 40
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[:cut if cut > 0 else limit]


def collapse_loops(text: str, max_repeats: int = 8, keep: int = 3, max_ngram: int = 4) -> str:
    """
    Cut a generation that got stuck repeating a short phrase ("VTS VTS VTS ...").

    A 1-4 word phrase repeated `max_repeats` or more times in a row is reduced to
    `keep` copies. Real radio repeats are short (MAYDAY x3, PAN PAN x3, call signs
    twice), so the thresholds leave them alone.
    """
    words = text.split()
    out, i = [], 0
    while i < len(words):
        for n in range(1, max_ngram + 1):
            gram = words[i:i + n]
            reps = 1
            while words[i + reps * n:i + (reps + 1) * n] == gram:
                reps += 1
            if len(gram) == n and reps >= max_repeats:
                out += gram * keep
                i += reps * n
                break
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def candidate_names(context: str) -> List[str]:
    """Names listed in a context built by sources.prompt_text."""
    if not context or ": " not in context:
        return []
    return context.split(": ", 1)[1].split(". Use a candidate")[0].split("; ")


def _best_span(name: str, words: List[str]):
    """(similarity, span) of the span of `words` that best matches `name`, by characters."""
    from difflib import SequenceMatcher

    target = name.replace(" ", "")
    k = len(name.split())
    best = (0.0, "")
    # Spans can be shorter than the name: "H T M S TAKSIN" is often heard as "HTMS TAXIDIAN"
    for w in range(1, k + 2):
        for i in range(max(1, len(words) - w + 1)):
            span = words[i:i + w]
            score = SequenceMatcher(None, target, "".join(span)).ratio()
            if score > best[0]:
                best = (score, " ".join(span))
    return best


def _aligned_span(words: List[str], start: int, end: int, other: List[str]) -> str:
    """Words of `other` aligned with words[start:end] (word-level diff alignment)."""
    from difflib import SequenceMatcher

    def map_index(i, is_end):
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, words, other, autojunk=False).get_opcodes():
            if i1 <= i < i2 or (is_end and i == i2 and i2 > i1):
                if tag == "equal":
                    return j1 + (i - i1)
                return j2 if is_end else j1
        return len(other)

    return " ".join(other[map_index(start, False):map_index(end, True)])


def verify_names(context_hyp: str, plain_hyp: str, candidates: List[str], threshold: float = 0.5) -> str:
    """
    Second-opinion check on candidate names copied from the prompt.

    `context_hyp` was transcribed with the candidate list, `plain_hyp` without it.
    A candidate used in `context_hyp` is kept only if something similar was heard
    without the list (character similarity >= `threshold` against the best
    matching span of `plain_hyp`). Otherwise it is replaced by what the no-list
    pass heard at the same position. This stops the model from substituting an
    unrelated name from a long list when the speaking vessel is not on it.
    Returns normalized text.
    """
    from .normalize import normalize_transcript

    # Keep spelled letters as letters so "H T M S" can match "HTMS"
    norm = lambda t: normalize_transcript(t, expand_letters=False).split()  # noqa: E731
    words, plain = norm(context_hyp), norm(plain_hyp)
    for cand in sorted(candidates, key=len, reverse=True):
        name = norm(cand)
        if not name:
            continue
        i = 0
        while i <= len(words) - len(name):
            if words[i:i + len(name)] != name:
                i += 1
                continue
            score, _ = _best_span(" ".join(name), plain)
            if score >= threshold:
                i += len(name)
                continue
            heard = _aligned_span(words, i, i + len(name), plain).split()
            words = words[:i] + heard + words[i + len(name):]
            i += max(1, len(heard))
    return " ".join(words)
