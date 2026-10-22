"""
Training / evaluation data sources.

Every loader returns a datasets.Dataset with the same columns:
    audio  - Audio(decode=False): {"bytes": ..., "path": ...}; decode with load_audio()
    text   - transcript in our canonical form (see normalize.py)
    slots  - JSON string of critical values to check verbatim ("{}" if none)
    source - source name

Sources (spec "name:split[@max_examples]", e.g. "vhf8k:train@2000"):
    vhf8k    PassbyGrocer/VHF-8K: synthetic VTS traffic, one transmission per clip.
             Splits: train, discovery, confirmation, test. Vessel names become slots.
             Transcripts with ambiguous numbers (3+ digit runs, coordinates) are
             dropped because we cannot know how the TTS voice read them.
    martts   bonalor/synthetic_maritime_radio_communication (CC-BY 4.0): synthetic
             distress / SAR dialogues, multi-speaker, one 30-125 s clip per dialogue.
             Splits: train (scenarios 1-8), test (scenarios 9-10); variant "clean"
             adds the undegraded audio (train only).
    atc      jacktol/ATC-ASR-Dataset: real accented ATC over VHF. train/validation/test
    atcosim  jlvdoorn/atcosim: clean simulated ATC. train/validation
    <path>   a local JSONL manifest written by synthesize.py
"""

import io
import json
import re
from pathlib import Path

import numpy as np

from .normalize import normalize_transcript

SR = 16000

_AMBIGUOUS_NUMBER = re.compile(r"\d{3,}|\d\s*[-.:°']\s*\d|\d[NSEW]\b")


def load_audio(audio: dict, sr: int = SR) -> np.ndarray:
    """Decode an Audio(decode=False) value to mono float32 at `sr`."""
    import soundfile as sf
    from scipy.signal import resample_poly

    src = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    x, file_sr = sf.read(src, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if file_sr != sr:
        g = np.gcd(file_sr, sr)
        x = resample_poly(x, sr // g, file_sr // g).astype(np.float32)
    return x


def _finish(ds, name):
    from datasets import Audio
    return ds.cast_column("audio", Audio(decode=False)).map(lambda _: {"source": name})


def _vhf8k(split):
    from datasets import Audio, load_dataset

    ds = load_dataset("PassbyGrocer/VHF-8K", split=split)
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.filter(lambda r: not _AMBIGUOUS_NUMBER.search(r["reference_text"]))

    def convert(r):
        norm = lambda s: normalize_transcript(s, numbers="cardinal")  # noqa: E731
        slots = {f"vessel_{i}": norm(e) for i, e in enumerate(r["reference_entities"] or [])}
        return {"text": norm(r["reference_text"]), "slots": json.dumps(slots)}

    ds = ds.map(convert).select_columns(["audio", "text", "slots"])
    return _finish(ds, "vhf8k")


def _martts(split):
    from datasets import Dataset
    from huggingface_hub import snapshot_download

    variants = ["noise"] + (["clean"] if split == "train+clean" else [])
    split = split.split("+")[0]
    if split not in ("train", "test"):
        raise ValueError("martts splits: train, train+clean, test")

    root = Path(snapshot_download(
        "bonalor/synthetic_maritime_radio_communication", repo_type="dataset",
        allow_patterns=["scenarios_preprocessed.jsonl"] + [f"dataset_{v}/*" for v in variants],
    ))
    rows = []
    for line in open(root / "scenarios_preprocessed.jsonl"):
        d = json.loads(line)
        topic, scen = d["dialog_id"].split("/")          # "Collision", "scenario_1_a"
        _, number, variant = scen.split("_")
        if (int(number) >= 9) != (split == "test"):
            continue
        text = " ".join(u["text"] for u in d["utterances"])
        text = normalize_transcript(text, numbers="cardinal", acronyms=True)
        for v in variants:
            wav = root / f"dataset_{v}" / f"{topic}_{number}_{variant}.wav"
            if wav.exists():
                rows.append({"audio": {"bytes": None, "path": str(wav)}, "text": text, "slots": "{}"})
    if not rows:
        raise FileNotFoundError(f"No MARTTS audio found under {root}")
    return _finish(Dataset.from_list(rows), "martts")


def _hf_text_audio(repo, name):
    def load(split):
        from datasets import Audio, load_dataset

        ds = load_dataset(repo, split=split).cast_column("audio", Audio(decode=False))
        ds = ds.map(lambda r: {"text": normalize_transcript(r["text"]), "slots": "{}"})
        return _finish(ds.select_columns(["audio", "text", "slots"]), name)
    return load


def _manifest(path):
    from datasets import load_dataset

    path = Path(path)
    ds = load_dataset("json", data_files=str(path), split="train")
    root = path.parent.resolve()
    ds = ds.map(lambda r: {
        "audio": {"bytes": None, "path": str(root / r["audio"])},
        "slots": json.dumps(r.get("slots") or {}),
    })
    return _finish(ds.select_columns(["audio", "text", "slots"]), path.stem)


LOADERS = {
    "vhf8k": _vhf8k,
    "martts": _martts,
    "atc": _hf_text_audio("jacktol/ATC-ASR-Dataset", "atc"),
    "atcosim": _hf_text_audio("jlvdoorn/atcosim", "atcosim"),
}


def load_source(spec: str, seed: int = 0):
    """Load one source from a spec like 'vhf8k:train@2000' or 'data/synth/manifest.jsonl'."""
    limit = None
    if "@" in spec:
        spec, n = spec.rsplit("@", 1)
        limit = int(n)
    if spec.endswith(".jsonl"):
        ds = _manifest(spec)
    else:
        name, _, split = spec.partition(":")
        if name not in LOADERS:
            raise ValueError(f"Unknown source {name!r}; choose from {sorted(LOADERS)} or a .jsonl path")
        ds = LOADERS[name](split or "train")
    if limit is not None and limit < len(ds):
        ds = ds.shuffle(seed=seed).select(range(limit))
    return ds


def load_sources(specs, seed: int = 0):
    """Load and concatenate several sources."""
    from datasets import concatenate_datasets
    return concatenate_datasets([load_source(s, seed) for s in specs])
