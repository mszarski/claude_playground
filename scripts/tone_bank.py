"""What the voice model hears in real voices, per feeling: run it on every CREMA-D clip (a "tone bank").

  # on a GPU job:
  python scripts/tone_bank.py --out /work/out/tone_bank.jsonl

Part of the licence-clean voice data. The student never hears audio: it sees the transcript plus the voice model's
reading (rmr.voice: categorical emotion, confidence, arousal / valence / dominance). Synthetic speech reads as the
wrong feeling (scripts/synth_speech.py: Zonos voices sound sad whatever the emotion setting), so the readings come
from real people instead. CREMA-D (mteb/crema-d, Open Database License; 7,442 acted clips, 91 actors, 6 emotions at
up to 4 intensities) gives the voice model's actual reading of each feeling; scripts/synth_heard.py pairs each written
line with a reading drawn from a clip of the same feeling. No CREMA-D audio or words reach the training data, only
the readings.

Writes one row per clip: file, actor, sentence, crema (label), intensity, feeling, and rmr.voice.Listener.hear's
emotion fields (emotion, confidence, probs, arousal, dominance, valence).
"""
import argparse
import json
import os

CREMA = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]          # mteb/crema-d label order
FEELING = {"Anger": "angry", "Disgust": "angry", "Fear": "anxious", "Happy": "happy", "Neutral": "neutral", "Sad": "sad"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="at most this many clips (0 = all), for a smoke test")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from rmr.voice import EmotionModel, load_audio

    emo, dims = EmotionModel("msp", device=a.device), EmotionModel("msp-dims", device=a.device)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    n = 0
    with open(a.out, "w") as f:
        for i in range(2):
            pf = pq.ParquetFile(hf_hub_download("mteb/crema-d", f"data/train-0000{i}-of-00002.parquet",
                                                repo_type="dataset"))
            for g in range(pf.num_row_groups):
                t = pf.read_row_group(g).to_pylist()
                for r in t:
                    name = r["audio"]["path"]                 # e.g. 1001_IEO_ANG_HI.wav
                    actor, sentence, _, level = name[:-4].split("_")
                    wav = load_audio(r["audio"]["bytes"])
                    probs = emo.predict(wav)
                    top = max(probs, key=probs.get)
                    crema = CREMA[r["label"]]
                    f.write(json.dumps({"file": name, "actor": actor, "sentence": sentence, "crema": crema,
                                        "intensity": level, "feeling": FEELING[crema], "emotion": top,
                                        "confidence": probs[top], "probs": probs, **dims.predict(wav)}) + "\n")
                    n += 1
                    if n % 500 == 0:
                        print(n, flush=True)
                    if a.limit and n >= a.limit:
                        print(f"{n} clips -> {a.out}")
                        return
    print(f"{n} clips -> {a.out}")


if __name__ == "__main__":
    main()
