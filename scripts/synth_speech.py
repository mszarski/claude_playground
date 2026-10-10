"""Voice the synthetic lines with Zonos (Apache-2.0) and check what the voice model hears in them.

  # on a GPU job (needs espeak-ng and the zonos package):
  python scripts/synth_speech.py --lines synth/lines.jsonl --refs synth/refs --out /work/out [--control]

Voices: never a real person's. Each synthetic voice is a blend of the speaker embeddings of two openly licensed
reference recordings (AMI meeting corpus and LibriSpeech, CC BY 4.0), so no output sounds like anyone in them.

Each line is spoken with the emotion it was written for (Zonos' emotion vector, plus pitch variation and speaking
rate to match), and with ``--control`` also neutrally in the same voice. The listener (rmr.voice: Whisper and the
voice-emotion model) then hears both. Writes ``wav/<id>[_ctrl].wav``, ``heard.jsonl`` (rmr.voice.Listener.hear output
+ id, feeling, condition, voice) and prints whether the emotion setting moves what the voice model hears toward
the intended feeling, and how intelligible the words stay.
"""
import argparse
import json
import os
import random
import re

# Zonos' emotion vector: happiness, sadness, disgust, fear, surprise, anger, other, neutral
EMOTION = {
    "neutral": ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9], 25.0, 15.0),
    "happy": ([0.8, 0.0, 0.0, 0.0, 0.1, 0.0, 0.05, 0.05], 70.0, 16.0),
    "sad": ([0.0, 0.8, 0.0, 0.05, 0.0, 0.0, 0.05, 0.1], 30.0, 11.0),
    "angry": ([0.0, 0.0, 0.1, 0.0, 0.0, 0.8, 0.05, 0.05], 70.0, 17.0),
}


def words(t):
    return re.findall(r"[a-z']+", t.lower())


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--refs", required=True, help="folder of reference recordings (.wav) to blend voices from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voices", type=int, default=8)
    ap.add_argument("--control", action="store_true", help="also speak each line neutrally in the same voice")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import soundfile as sf
    import torch
    import torchaudio
    from zonos.conditioning import make_cond_dict
    from zonos.model import Zonos

    from rmr.voice import TO_CANON, Listener

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cuda")
    refs = []
    for f in sorted(os.listdir(a.refs)):
        if f.endswith(".wav"):
            wav, sr = torchaudio.load(os.path.join(a.refs, f))
            refs.append(model.make_speaker_embedding(wav.mean(0, keepdim=True), sr))
    voices = []
    for v in range(a.voices):                       # blends of two references: nobody's real voice
        i, j = rng.sample(range(len(refs)), 2)
        w = rng.uniform(0.35, 0.65)
        voices.append((w * refs[i].float() + (1 - w) * refs[j].float()).bfloat16())
    lines = [json.loads(x) for x in open(a.lines)]
    os.makedirs(os.path.join(a.out, "wav"), exist_ok=True)
    jobs = []
    for k, ln in enumerate(lines):
        v = k % len(voices)
        jobs.append((ln, ln["feeling"], "emotion", v))
        if a.control:
            jobs.append((ln, "neutral", "control", v))
    lis = Listener()
    out = []
    for n, (ln, emo, cond, v) in enumerate(jobs):
        vec, pitch, rate = EMOTION[emo]
        cd = make_cond_dict(text=ln["line"], speaker=voices[v], language="en-us", emotion=vec, pitch_std=pitch,
                            speaking_rate=rate)
        codes = model.generate(model.prepare_conditioning(cd), progress_bar=False, disable_torch_compile=True)
        wav = model.autoencoder.decode(codes).cpu()[0].float()
        path = os.path.join(a.out, "wav", ln["id"] + ("_ctrl" if cond == "control" else "") + ".wav")
        sf.write(path, wav.mean(0).numpy(), model.autoencoder.sampling_rate)
        h = lis.hear(path)
        out.append({"id": ln["id"], "feeling": ln["feeling"], "condition": cond, "voice": v, "line": ln["line"],
                    "before": ln.get("before", ""), "situation": ln.get("situation", ""),
                    "wer": round(wer(ln["line"], h["text"]), 3), **h})
        if n % 20 == 0:
            print(f"{n + 1}/{len(jobs)}", flush=True)
    with open(os.path.join(a.out, "heard.jsonl"), "w") as f:
        f.writelines(json.dumps(o) + "\n" for o in out)
    # summary: does the emotion setting move what the voice model hears toward the intended feeling?
    print("\nfeeling   condition  voice model hears the intended feeling   mean P(intended)   WER")
    for feel in EMOTION:
        for cond in ("emotion", "control"):
            rows = [o for o in out if o["feeling"] == feel and o["condition"] == cond]
            if not rows:
                continue
            hit = sum(TO_CANON.get(o["emotion"], o["emotion"]) == feel for o in rows) / len(rows)
            p = sum(sum(v for k, v in o["probs"].items() if TO_CANON.get(k) == feel) for o in rows) / len(rows)
            w = sum(o["wer"] for o in rows) / len(rows)
            print(f"{feel:9s} {cond:9s} {hit:28.0%} {p:18.2f} {w:8.2f}")


if __name__ == "__main__":
    main()
