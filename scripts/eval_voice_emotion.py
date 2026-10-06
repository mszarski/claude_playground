"""How often do the open speech-emotion models get it right? (CPU, a few minutes per model)

  python scripts/eval_voice_emotion.py --models msp speechbrain acted --per-class 100

Four emotions every candidate can name (neutral, happy, sad, angry), balanced: ``--per-class`` clips each from
MELD's test split (conversation from Friends, ajyy/MELD_audio, GPL-3.0; evaluation only) and from CREMA-D (acted,
mteb/crema-d). Reports unweighted accuracy (chance 25%) and the confusion matrix.
"""
import argparse
import io
import json
import tarfile

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from rmr.voice import CANON, TO_CANON, EmotionModel, load_audio


def meld(n, rng):
    df = pd.read_csv(hf_hub_download("ajyy/MELD_audio", "test.csv", repo_type="dataset"))
    df["canon"] = df["Emotion"].map(TO_CANON)
    pick = pd.concat([g.sample(min(n, len(g)), random_state=0) for c, g in df[df.canon.notna()].groupby("canon")])
    want = {f"test/dia{r.Dialogue_ID}_utt{r.Utterance_ID}.flac": r.canon for r in pick.itertuples()}
    out = []
    with tarfile.open(hf_hub_download("ajyy/MELD_audio", "archive/test.tar.gz", repo_type="dataset")) as tf:
        for m in tf.getmembers():
            if m.name in want:
                out.append((load_audio(tf.extractfile(m).read()), want[m.name]))
    return out


def cremad(n, rng):
    names = {0: "angry", 3: "happy", 4: "neutral", 5: "sad"}
    df = pd.concat([pd.read_parquet(hf_hub_download("mteb/crema-d", f"data/train-0000{i}-of-00002.parquet",
                                                    repo_type="dataset")) for i in range(2)])
    df = df[df.label.isin(names)]
    pick = pd.concat([g.sample(n, random_state=0) for _, g in df.groupby("label")])
    return [(load_audio(r.audio["bytes"]), names[r.label]) for r in pick.itertuples()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["msp", "speechbrain", "acted"])
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--out", default="runs/voice_eval.json")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    sets = {"MELD (conversation)": meld(a.per_class, rng), "CREMA-D (acted)": cremad(a.per_class, rng)}
    report = {}
    for name in a.models:
        em = EmotionModel(name)
        for sname, clips in sets.items():
            C = np.zeros((4, 4), int)
            for wav, y in clips:
                q = em.canon(em.predict(wav))
                C[CANON.index(y), CANON.index(max(q, key=q.get))] += 1
            ua = float(np.mean(np.diag(C) / C.sum(1).clip(1)))
            report[f"{name} | {sname}"] = {"unweighted_accuracy": ua, "n": int(C.sum()), "confusion": C.tolist()}
            print(f"{name:12s} {sname:22s} UA {ua:.2f} (n={C.sum()})  rows=true {CANON}: {C.tolist()}", flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
