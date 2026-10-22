"""Tests for prompt construction and runaway-output guards (no network)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("datasets")

from marine_vhf_asr.prompting import build_inputs, collapse_loops, max_new_tokens_for, trim_runaway  # noqa: E402
from marine_vhf_asr.sources import _vhf8k_context, prompt_text  # noqa: E402

END_INST, PAD, AUDIO = 4, 11, 24


class FakeTokenizer:
    pad_token_id = PAD
    eos_token_id = 2

    def convert_tokens_to_ids(self, token):
        return {"[/INST]": END_INST}[token]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [100 + i for i, _ in enumerate(text.split())]}


class FakeProcessor:
    """Mimics Voxtral: left-padded <s>[INST][AUDIO]*n[/INST]lang:en[TRANSCRIBE]."""
    tokenizer = FakeTokenizer()

    def apply_transcription_request(self, audio, **kwargs):
        rows = [[1, 3] + [AUDIO] * (1 + len(a) // 16000) + [END_INST, 9, 9, 34] for a in audio]
        L = max(map(len, rows))
        return {
            "input_ids": torch.tensor([[PAD] * (L - len(r)) + r for r in rows]),
            "attention_mask": torch.tensor([[0] * (L - len(r)) + [1] * len(r) for r in rows]),
            "input_features": torch.zeros(len(rows), 1),
        }


def test_no_context_is_untouched():
    p = FakeProcessor()
    audios = [np.zeros(16000), np.zeros(48000)]
    plain = p.apply_transcription_request(audios)
    out = build_inputs(p, "m", audios, ["", ""])
    assert torch.equal(out["input_ids"], plain["input_ids"])


def test_context_spliced_before_end_inst_and_left_padded():
    p = FakeProcessor()
    out = build_inputs(p, "m", [np.zeros(16000), np.zeros(48000)], ["A B C", ""])
    ids, mask = out["input_ids"].tolist(), out["attention_mask"].tolist()
    row0 = [t for t, m in zip(ids[0], mask[0]) if m]
    assert row0 == [1, 3, AUDIO, AUDIO, 100, 101, 102, END_INST, 9, 9, 34]
    row1 = [t for t, m in zip(ids[1], mask[1]) if m]
    assert row1 == [1, 3] + [AUDIO] * 4 + [END_INST, 9, 9, 34]
    # Padding goes on the left of the shorter row only
    assert mask[0] == [1] * len(row0)
    assert mask[1] == [0] + [1] * len(row1)
    assert out["input_features"].shape[0] == 2


def test_runaway_guards():
    assert trim_runaway("CHANNEL ONE SIX OVER", 2.0) == "CHANNEL ONE SIX OVER"
    runaway = "3" + "0" * 500
    assert len(trim_runaway(runaway, 1.0)) <= 80
    long_words = " ".join(["WORD"] * 400)
    trimmed = trim_runaway(long_words, 2.0)
    assert len(trimmed) <= 120 and not trimmed.endswith(" ")
    assert max_new_tokens_for([np.zeros(16000 * 2)]) == 54
    assert max_new_tokens_for([np.zeros(16000 * 600)]) == 1024


ROW = {
    "utterance_id": "u1",
    "reference_entities": ["SHAN BANG"],
    "konger_entities": ["STOLT INTEGRITY", "PAN AMBER 7"],
    "ship_names": ["ORFEOS", "SHAN BANG", "---E( A%-W-", "MONT FORT", "PAN AMBER 7"],
}


def test_vhf8k_context_modes():
    ctx, decoys = _vhf8k_context(ROW, "none")
    assert ctx == "" and decoys == []

    ctx, decoys = _vhf8k_context(ROW, "supported")
    assert "SHAN BANG" in ctx and "PAN AMBER SEVEN" in ctx
    assert decoys == ["STOLT INTEGRITY", "PAN AMBER SEVEN"]

    ctx, decoys = _vhf8k_context(ROW, "decoys")
    assert "SHAN BANG" not in ctx and "STOLT INTEGRITY" in ctx


def test_vhf8k_context_deterministic_and_clean_extras():
    a = _vhf8k_context(dict(ROW, _extra=True), "mix")
    assert a == _vhf8k_context(dict(ROW, _extra=True), "mix")
    ctx, decoys = a
    assert "%" not in ctx and "SHAN BANG" not in decoys


def test_spoken_name_position_is_shuffled():
    positions = set()
    for i in range(30):
        row = dict(ROW, utterance_id=f"u{i}")
        names = _vhf8k_context(row, "supported")[0].split(": ", 1)[1].split(". Use")[0].split("; ")
        positions.add(names.index("SHAN BANG"))
    assert len(positions) > 1


def test_prompt_text():
    assert prompt_text([]) == ""
    assert prompt_text(["A", "B"]) == "Candidate vessel names: A; B. Use a candidate only if it is spoken."


def test_collapse_loops():
    assert collapse_loops("VTS " * 50 + "OVER") == "VTS VTS VTS OVER"
    assert collapse_loops("ROGER " + "STAND BY " * 20) == "ROGER STAND BY STAND BY STAND BY"
    for legit in ["MAYDAY MAYDAY MAYDAY THIS IS RED KITE RED KITE RED KITE",
                  "PAN PAN PAN PAN PAN PAN ALL STATIONS ALL STATIONS ALL STATIONS",
                  "SECURITE SECURITE SECURITE", ""]:
        assert collapse_loops(legit) == legit


def test_vhf8k_context_padded_to_size():
    row = dict(ROW, reference_text="VTS, SHAN BANG, roger. Keeping clear of ORFEOS.")
    pool = sorted({"ORFEOS", "MONT FORT", "SHAN BANG", "GOLDEN TIDE", "SEA LION", "ALPHA STAR",
                   "BLUE WHALE", "RED ROCK", "NORTH WIND", "EAST CAPE", "WEST POINT", "SOUTH BAY"})
    ctx, decoys = _vhf8k_context(row, "supported", size=8, pool=pool)
    names = ctx.split(": ", 1)[1].split(". Use")[0].split("; ")
    assert len(names) == 8 and len(set(names)) == 8 and "SHAN BANG" in names
    assert "ORFEOS" not in names  # spoken in the clip, so never a filler decoy
    assert "SHAN BANG" not in decoys and len(decoys) == 7

    ctx, decoys = _vhf8k_context(row, "decoys", size=8, pool=pool)
    assert "SHAN BANG" not in ctx and len(decoys) == 8
    assert _vhf8k_context(row, "decoys", size=8, pool=pool) == (ctx, decoys)
