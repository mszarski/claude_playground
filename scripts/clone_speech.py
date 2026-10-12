"""Voice the written training lines with real people's emotional delivery: zero-shot voice cloning from CREMA-D.

  # in a venv with chatterbox-tts (it pins its own torch / transformers):
  python scripts/clone_speech.py voice --heard heard_v5.jsonl --out /work/out/voiced --per-feeling 30
  # in the main env: does the cloned speech still sound like its feeling? (Whisper + the tone models)
  python scripts/clone_speech.py check --heard heard_v5.jsonl --voiced /work/out/voiced --bank tone_bank.jsonl

The audio-native responder (scripts/audio_llm.py) has to hear training lines, but the lines are written
(scripts/synth_lines.py), and plain TTS voices read as sad whatever they were asked (scripts/synth_speech.py). Each
line was already paired with a real CREMA-D clip of its feeling (scripts/synth_heard.py, field ``tone``; training
actors only), so that same clip is the cloning reference: the TTS copies the actor's voice and delivery onto the new
words. Chatterbox (ResembleAI, MIT); ``exaggeration`` scales how strongly the delivery is carried over.

``check`` runs the hearing stage (rmr.voice.Listener) on the voiced lines: the tone models' agreement with the
intended feeling, next to their agreement on the real reference clips (the ceiling: they are often wrong on real
voices too), and the word error rate of Whisper against the line, to drop garbled audio.
"""
import argparse
import io
import json
import os
import random
import re

EXAGGERATION = {"neutral": 0.35, "happy": 0.7, "sad": 0.6, "angry": 0.75, "anxious": 0.65, "surprised": 0.7}
CANON = {"angry": "angry", "disgust": "angry", "contempt": "angry", "sad": "sad", "happy": "happy",
         "fear": "anxious", "surprise": "surprised", "neutral": "neutral"}


def crema_bytes(names):
    """CREMA-D file name -> raw WAV bytes (mteb/crema-d, ODbL)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    want, out = set(names), {}
    for i in range(2):
        pf = pq.ParquetFile(hf_hub_download("mteb/crema-d", f"data/train-0000{i}-of-00002.parquet", repo_type="dataset"))
        for g in range(pf.num_row_groups):
            for r in pf.read_row_group(g, columns=["audio"]).to_pylist():
                if r["audio"]["path"] in want:
                    out[r["audio"]["path"]] = r["audio"]["bytes"]
    return out


def pick(rows, per_feeling, seed, only=None):
    rows = [r for r in rows if r.get("tone") and r.get("text") and not r["clip"].startswith("t_")]   # t_: real CREMA-D
    if only:
        rows = [r for r in rows if r["clip"] in only]
    if not per_feeling:
        return rows
    by = {}
    for r in random.Random(seed).sample(rows, len(rows)):
        by.setdefault(r["feeling"], []).append(r)
    return [r for v in by.values() for r in v[:per_feeling]]


def voice(a):
    import soundfile as sf
    import torch
    import torchaudio
    from chatterbox.tts import ChatterboxTTS

    rows = [json.loads(x) for x in open(a.heard)]
    only = set(json.load(open(a.clips))) if a.clips else None
    rows = pick(rows, a.per_feeling, a.seed, only)
    rows = rows[a.shard::a.shards]
    os.makedirs(a.out, exist_ok=True)
    rows = [r for r in rows if not os.path.exists(os.path.join(a.out, r["clip"] + ".wav"))]
    refs = crema_bytes({r["tone"] for r in rows})
    tts = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
    ref_dir = os.path.join(a.out, "_refs")
    os.makedirs(ref_dir, exist_ok=True)
    for i, r in enumerate(rows):
        ref = os.path.join(ref_dir, r["tone"])
        if not os.path.exists(ref):
            x, sr = sf.read(io.BytesIO(refs[r["tone"]]), dtype="float32")
            sf.write(ref, x, sr)
        torch.manual_seed(a.seed + i)
        wav = tts.generate(r["text"], audio_prompt_path=ref, exaggeration=EXAGGERATION.get(r["feeling"], 0.5),
                           cfg_weight=a.cfg)
        torchaudio.save(os.path.join(a.out, r["clip"] + ".wav"), wav.cpu(), tts.sr)
        if i % 50 == 0:
            print(f"voiced {i}/{len(rows)}: {r['feeling']} {r['text'][:60]!r}", flush=True)
    print(f"{len(rows)} lines voiced -> {a.out}")


def wer(ref, hyp):
    r, h = re.findall(r"[a-z']+", ref.lower()), re.findall(r"[a-z']+", hyp.lower())
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def check(a):
    from rmr.voice import Listener

    rows = {json.loads(x)["clip"]: json.loads(x) for x in open(a.heard)}
    bank = {b["file"]: b for b in map(json.loads, open(a.bank))} if a.bank else {}
    lis = Listener(device="cuda")
    out, stats = [], {}
    for f in sorted(os.listdir(a.voiced)):
        if not f.endswith(".wav"):
            continue
        r = rows[f[:-4]]
        h = lis.hear(os.path.join(a.voiced, f))
        w = wer(r["text"], h["text"])
        got = CANON.get(h["emotion"], h["emotion"])
        ref = bank.get(r["tone"])
        s = stats.setdefault(r["feeling"], [0, 0, 0, 0])
        s[0] += 1
        s[1] += got == r["feeling"]
        s[2] += bool(ref) and CANON.get(ref["emotion"], ref["emotion"]) == r["feeling"]
        s[3] += w
        out.append({"clip": r["clip"], "feeling": r["feeling"], "text": r["text"], "heard": h["text"], "wer": w,
                    "emotion": h["emotion"], "confidence": h["confidence"], "arousal": h.get("arousal"),
                    "valence": h.get("valence"), "ref_emotion": ref and ref["emotion"]})
    with open(os.path.join(a.voiced, "check.jsonl"), "w") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in out)
    print("feeling     n   tone models agree (cloned / real reference)   mean WER")
    for k, (n, ok, ref_ok, w) in sorted(stats.items()):
        print(f"{k:10s} {n:3d}   {ok / n:5.0%} / {ref_ok / n:5.0%}   {w / n:6.1%}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("voice")
    v.add_argument("--heard", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--per-feeling", type=int, default=0, help="0 = every line")
    v.add_argument("--clips", help="JSON list of clip ids to voice (e.g. the responder's training set)")
    v.add_argument("--cfg", type=float, default=0.5)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--shard", type=int, default=0)
    v.add_argument("--shards", type=int, default=1)
    c = sub.add_parser("check")
    c.add_argument("--heard", required=True)
    c.add_argument("--voiced", required=True)
    c.add_argument("--bank", default="")
    a = ap.parse_args()
    {"voice": voice, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
