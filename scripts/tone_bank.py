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
SENTENCES = {"DFA": "Don't forget a jacket.", "IEO": "It's eleven o'clock.", "IOM": "I'm on my way to the meeting.",
             "ITH": "I think I have a doctor's appointment.", "ITS": "I think I've seen this before.",
             "IWL": "I would like a new alarm clock.", "IWW": "I wonder what this is about.",
             "MTI": "Maybe tomorrow it will be cold.", "TAI": "The airplane is almost full.",
             "TIE": "That is exactly what happened.", "TSI": "The surface is slick.",
             "WSI": "We'll stop in a couple of minutes."}
FEELING = {"Anger": "angry", "Disgust": "angry", "Fear": "anxious", "Happy": "happy", "Neutral": "neutral", "Sad": "sad"}


def held_out(actor):
    """Every fifth actor (18 of 91) is kept for evaluation (scripts/eval_clean.py), never for training data."""
    return int(actor) % 5 == 0


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
