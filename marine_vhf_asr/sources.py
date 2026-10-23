"""
Training / evaluation data sources.

Every loader returns a datasets.Dataset with the same columns:
    audio  - Audio(decode=False): {"bytes": ..., "path": ...}; decode with load_audio()
    text   - transcript in our canonical form (see normalize.py)
    slots  - JSON string of critical values to check verbatim ("{}" if none)
    context - optional prompt text, e.g. candidate vessel names ("" if none)
    decoys  - JSON list of candidate names in the context that are NOT spoken
    source - source name

Sources (spec "name:split[#context][@max_examples]", e.g. "vhf8k:train#mix@2000"):
    vhf8k    PassbyGrocer/VHF-8K: synthetic VTS traffic, one transmission per clip.
             Splits: train, discovery, confirmation, test. Vessel names become slots.
             Transcripts with ambiguous numbers (3+ digit runs, coordinates) are
             dropped because we cannot know how the TTS voice read them.
             Context modes (candidate vessel names in the prompt):
               #none       no context (default)
               #supported  spoken name(s) + VHF-8K's phonetic decoys, shuffled
               #decoys     decoys only: tests that unspoken candidates are not copied
               #mix        training mix: 35% none, 50% supported, 15% decoys, with
                           0-5 extra names from the dialogue's vessel inventory
               #mixlong    like #mix but closer to real AIS lists: 30% none, 40% supported,
                           30% spoken name absent; list length 5-100 (log-uniform)
             #supported:N / #decoys:N pad the list to N names with random other
             vessel names from the split, like a real AIS list of ships in range.
    martts   bonalor/synthetic_maritime_radio_communication (CC-BY 4.0): synthetic
             distress / SAR dialogues, multi-speaker, one 30-125 s clip per dialogue.
             Splits: train (scenarios 1-8), test (scenarios 9-10); variant "clean"
             adds the undegraded audio (train only).
    atc      jacktol/ATC-ASR-Dataset: real accented ATC over VHF. train/validation/test
    atcosim  jlvdoorn/atcosim: clean simulated ATC. train/validation
    <path>   a local JSONL manifest written by synthesize.py; "<path>#<key>" uses each
             record's contexts[<key>] as the candidate list (see harbor.py);
             "<path>#mix" is a training mix over those lists
"""

import hashlib
import io
import json
import math
import random
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


COLUMNS = ["audio", "text", "slots", "context", "decoys"]


def _finish(ds, name):
    from datasets import Audio
    if "context" not in ds.column_names:
        ds = ds.map(lambda _: {"context": "", "decoys": "[]"})
    ds = ds.select_columns(COLUMNS).cast_column("audio", Audio(decode=False))
    return ds.map(lambda _: {"source": name})


def _vhf8k_name(name: str) -> str:
    """VHF-8K vessel name in transcript form. Letters stay letters ("M", not "MIKE"):
    that is how the TTS voice read them."""
    return normalize_transcript(name, numbers="cardinal", expand_letters=False)


_JUNK_NAME = re.compile(r"[^A-Za-z0-9 .'&/-]")


def _vhf8k_context(r, mode: str, size: int = 0, pool=()):
    """
    Pick (context, decoys) for one VHF-8K row; deterministic per utterance.

    With `size`, the list is padded to `size` names with random names from
    `pool` (never a name that is spoken in the clip).
    """
    rng = random.Random(int(hashlib.md5(f"{r['utterance_id']}/{size}".encode()).hexdigest()[:8], 16))
    if mode == "mix":
        mode = rng.choices(["none", "supported", "decoys"], weights=[35, 50, 15])[0]
    elif mode == "mixlong":
        mode = rng.choices(["none", "supported", "decoys"], weights=[30, 40, 30])[0]
        size = round(math.exp(rng.uniform(math.log(5), math.log(100))))
    if mode == "none":
        return "", []

    spoken = [_vhf8k_name(e) for e in r["reference_entities"] or []]
    decoys = [_vhf8k_name(e) for e in r["konger_entities"] or []]
    if r.get("_extra"):
        pool = [_vhf8k_name(e) for e in r["ship_names"] or [] if e and not _JUNK_NAME.search(e)]
        pool = [n for n in pool if n and n not in spoken and n not in decoys]
        decoys += rng.sample(pool, min(len(pool), rng.randint(0, 5)))
    if size:
        text = f" {_vhf8k_name(r['reference_text'])} "
        taken = set(spoken) | set(decoys)
        need = size - len(decoys) - (len(spoken) if mode == "supported" else 0)
        while need > 0:
            name = pool[rng.randrange(len(pool))]
            if name not in taken and f" {name} " not in text:
                decoys.append(name)
                taken.add(name)
                need -= 1
    names = (spoken if mode == "supported" else []) + decoys
    rng.shuffle(names)
    return prompt_text(names), decoys


def prompt_text(names) -> str:
    """Candidate-vessel context, in the wording the model is trained on."""
    if not names:
        return ""
    return f"Candidate vessel names: {'; '.join(names)}. Use a candidate only if it is spoken."


def _vhf8k(split):
    from datasets import Audio, load_dataset

    split, _, mode = split.partition("#")
    mode, _, size = (mode or "none").partition(":")
    size = int(size or 0)
    if mode not in ("none", "supported", "decoys", "mix", "mixlong") or (size and mode not in ("supported", "decoys")):
        raise ValueError(f"vhf8k context must be none/supported[:N]/decoys[:N]/mix/mixlong, got {mode!r}")

    ds = load_dataset("PassbyGrocer/VHF-8K", split=split)
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.filter(lambda r: not _AMBIGUOUS_NUMBER.search(r["reference_text"]))

    pool = []
    if size or mode == "mixlong":
        names = set()
        for cols in zip(ds["ship_names"], ds["konger_entities"], ds["reference_entities"]):
            names.update(_vhf8k_name(e) for col in cols for e in col or [] if e and not _JUNK_NAME.search(e))
        pool = sorted(n for n in names if n)

    def convert(r):
        slots = {f"vessel_{i}": _vhf8k_name(e) for i, e in enumerate(r["reference_entities"] or [])}
        context, decoys = _vhf8k_context(dict(r, _extra=mode == "mix"), mode, size, pool)
        text = normalize_transcript(r["reference_text"], numbers="cardinal", expand_letters=False)
        return {"text": text, "slots": json.dumps(slots), "context": context, "decoys": json.dumps(decoys)}

    return _finish(ds.map(convert), "vhf8k")


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
                rows.append({"audio": {"bytes": None, "path": str(wav)}, "text": text, "slots": "{}",
                             "context": "", "decoys": "[]"})
    if not rows:
        raise FileNotFoundError(f"No MARTTS audio found under {root}")
    return _finish(Dataset.from_list(rows), "martts")


def _hf_text_audio(repo, name):
    def load(split):
        from datasets import Audio, load_dataset

        ds = load_dataset(repo, split=split).cast_column("audio", Audio(decode=False))
        ds = ds.map(lambda r: {"text": normalize_transcript(r["text"]), "slots": "{}"})
        return _finish(ds, name)
    return load


def _manifest(path, context_key=""):
    """
    A local JSONL manifest. With `context_key`, each record's
    `contexts[context_key]` list of names becomes the candidate-vessel prompt
    (harbor.py writes `moving_a` and `all`); listed names that are not in the
    transcript are its decoys.
    """
    from datasets import load_dataset

    path = Path(path)
    ds = load_dataset("json", data_files=str(path), split="train")
    root = path.parent.resolve()

    def convert(r):
        padded = f" {r['text']} "
        if context_key == "mix":
            # Training mix over harbor.py lists: 30% no list, else moving_a or all;
            # the spoken names are removed from 30% of lists ("speaker not listed")
            rng = random.Random(int(hashlib.md5(r["audio"].encode()).hexdigest()[:8], 16))
            contexts = r.get("contexts") or {}
            names = [] if rng.random() < 0.3 or not contexts else list(contexts[rng.choice(sorted(contexts))] or [])
            if names and rng.random() < 0.3:
                names = [n for n in names if f" {n} " not in padded]
            rng.shuffle(names)
        else:
            names = ((r.get("contexts") or {}).get(context_key) or []) if context_key else []
        return {
            "audio_ref": {"bytes": None, "path": str(root / r["audio"])},
            # JSON slots have per-row keys, which datasets would otherwise keep
            # as a sparse struct column
            "slots_json": json.dumps({k: v for k, v in (r.get("slots") or {}).items() if v is not None}),
            "context": prompt_text(names),
            "decoys": json.dumps([n for n in names if f" {n} " not in padded]),
        }

    ds = ds.map(convert, remove_columns=[c for c in ds.column_names if c != "text"])
    ds = ds.rename_columns({"audio_ref": "audio", "slots_json": "slots"})
    return _finish(ds, path.stem)


LOADERS = {
    "vhf8k": _vhf8k,
    "martts": _martts,
    "atc": _hf_text_audio("jacktol/ATC-ASR-Dataset", "atc"),
    "atcosim": _hf_text_audio("jlvdoorn/atcosim", "atcosim"),
}


def load_source(spec: str, seed: int = 0):
    """Load one source from a spec like 'vhf8k:train#mix@2000' or 'data/synth/manifest.jsonl'."""
    limit = None
    if "@" in spec:
        spec, n = spec.rsplit("@", 1)
        limit = int(n)
    if spec.split("#")[0].endswith(".jsonl"):
        path, _, key = spec.partition("#")
        ds = _manifest(path, "" if key == "none" else key)
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
