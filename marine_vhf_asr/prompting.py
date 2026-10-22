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
