"""Run rmr.voice.Listener over MELD utterances (GPU job or CPU) -> heard.jsonl, one row per clip, with the label,
the gold transcript and the two previous utterances of the dialogue (for context).

  python scripts/listen_meld.py --split train --max-per-class 1000 --out runs/listen_train
"""
import argparse
import json
import os
import tarfile

import pandas as pd
from huggingface_hub import hf_hub_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-per-class", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from rmr.voice import Listener

    df = pd.read_csv(hf_hub_download("ajyy/MELD_audio", f"{a.split}.csv", repo_type="dataset"))
    df = df[df.Utterance.str.split().str.len().ge(3)]
    prev = {(r.Dialogue_ID, r.Utterance_ID): r.Utterance for r in df.itertuples()}
    pick = pd.concat([g.sample(min(a.max_per_class, len(g)), random_state=0) for _, g in df.groupby("Emotion")])
    want = {f"{a.split}/dia{r.Dialogue_ID}_utt{r.Utterance_ID}.flac": r for r in pick.itertuples()}
    L = Listener(device="cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    n = 0
    with tarfile.open(hf_hub_download("ajyy/MELD_audio", f"archive/{a.split}.tar.gz", repo_type="dataset")) as tf, \
            open(os.path.join(a.out, "heard.jsonl"), "w") as f:
        for m in tf:
            r = want.get(m.name)
            if r is None:
                continue
            try:
                h = L.hear(tf.extractfile(m).read())
            except Exception as e:
                print("skip", m.name, type(e).__name__, e, flush=True)
                continue
            ctx = [prev.get((r.Dialogue_ID, r.Utterance_ID - k)) for k in (2, 1)]
            f.write(json.dumps({"clip": m.name, "meld_emotion": r.Emotion, "meld_text": r.Utterance,
                                "context": [c for c in ctx if c], **h}) + "\n")
            n += 1
            if n % 200 == 0:
                print(f"heard {n}/{len(want)}", flush=True)
    print(f"done: {n} clips", flush=True)


if __name__ == "__main__":
    main()
