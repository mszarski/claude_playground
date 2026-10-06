"""End to end: does Reachy read people right, and does it respond appropriately? (MELD test dialogue)

  python scripts/eval_respond.py --per-class 50 --models Qwen/Qwen3-Next-80B-A3B-Instruct Qwen/Qwen3-4B-Instruct-2507

Stages (cached in --out, so reruns only do what is missing):
  1. heard.jsonl     Whisper transcript + voice emotion + arousal / valence for each clip (rmr.voice.Listener, CPU)
  2. reading         the responder's ``feeling`` vs MELD's label: tone only (the voice model), words only (LLM without
                     the voice reading), words + tone (the full system); 4-class unweighted accuracy, chance 25%
  3. response        the full system's response -> zero-shot recipe (same LLM) -> physical checks on the expanded
                     plan against what the person actually felt (MELD's label):
                       sad -> gentle    head never faster than 80 deg/s, energy <= 3
                       angry -> calm    the same: no sudden moves, low energy
                       happy -> joyful  ears up (<= 20 deg) at some point, and lively (energy >= 2.5 or rise >= 6 mm)
                       neutral -> attentive  ears never drooped (<= 100 deg), energy <= 5
"""
import argparse
import json
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

CLASSES = ["neutral", "happy", "sad", "angry"]
MELD2 = {"neutral": "neutral", "joy": "happy", "sadness": "sad", "anger": "angry"}


def clips(per_class):
    df = pd.read_csv(hf_hub_download("ajyy/MELD_audio", "test.csv", repo_type="dataset"))
    df = df[df.Emotion.isin(MELD2) & df.Utterance.str.split().str.len().ge(3)]
    pick = pd.concat([g.sample(min(per_class, len(g)), random_state=0) for _, g in df.groupby("Emotion")])
    want = {f"test/dia{r.Dialogue_ID}_utt{r.Utterance_ID}.flac": (MELD2[r.Emotion], r.Utterance) for r in pick.itertuples()}
    with tarfile.open(hf_hub_download("ajyy/MELD_audio", "archive/test.tar.gz", repo_type="dataset")) as tf:
        for m in tf.getmembers():
            if m.name in want:
                yield m.name, want[m.name], tf.extractfile(m).read()


def physical_ok(label, recipe):
    from rmr.recipe import expand
    F = expand(recipe, np.random.default_rng(0))
    ears, head, z, E = F[:, :2].mean(1), F[:, 2:5], F[:, 5], F[:, 7]
    head_speed = (np.abs(np.diff(head, axis=0)) * 25).max() if len(F) > 1 else 0.0
    if label in ("sad", "angry"):
        return bool(head_speed <= 80 and E.max() <= 3)
    if label == "happy":
        return bool(ears.min() <= 20 and (E.max() >= 2.5 or z.max() - z[0] >= 6))
    return bool(ears.max() <= 100 and E.max() <= 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3-Next-80B-A3B-Instruct", "Qwen/Qwen3-4B-Instruct-2507"])
    ap.add_argument("--out", default="runs/respond_eval")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--context", action="store_true", help="give the responder the two previous lines of the dialogue")
    ap.add_argument("--score", nargs="*", default=[], help="only score existing answer files (e.g. a student's)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.score:
        for path in a.score:
            res = [json.loads(l) for l in open(path)]
            print_row(os.path.basename(path).replace(".jsonl", ""), summarize(res))
        return

    heard_path = os.path.join(a.out, "heard.jsonl")
    if not os.path.exists(heard_path):
        from rmr.voice import Listener
        L = Listener()
        with open(heard_path, "w") as f:
            for i, (name, (label, text), audio) in enumerate(clips(a.per_class)):
                h = L.hear(audio)
                f.write(json.dumps({"clip": name, "label": label, "meld_text": text, **{k: v for k, v in h.items() if k != "probs"},
                                    "probs": h["probs"]}) + "\n")
                if i % 20 == 0:
                    print(f"  heard {i}", flush=True)
    rows = [json.loads(l) for l in open(heard_path)]
    if a.context:
        df = pd.read_csv(hf_hub_download("ajyy/MELD_audio", "test.csv", repo_type="dataset"))
        lines = {(r.Dialogue_ID, r.Utterance_ID): r.Utterance for r in df.itertuples()}
        for r in rows:
            d, u = (int(x) for x in r["clip"].split("dia")[1].split(".")[0].split("_utt"))
            r["context"] = [c for c in (lines.get((d, u - 2)), lines.get((d, u - 1))) if c]
    from rmr.voice import TO_CANON
    tone = [TO_CANON.get(r["emotion"], "other") for r in rows]
    report = {"n": len(rows), "tone_only": ua([r["label"] for r in rows], tone)}
    print(f"{len(rows)} clips | tone only (voice model): UA {report['tone_only']['ua']:.2f}", flush=True)

    from rmr.planner.write import _batch
    from rmr.respond import respond
    for model in a.models:
        tag = model.split("/")[-1] + ("+context" if a.context else "")
        path = os.path.join(a.out, f"{tag}.jsonl")
        done = {json.loads(l)["clip"]: json.loads(l) for l in open(path)} if os.path.exists(path) else {}

        def run(r):
            if r["clip"] in done:
                return done[r["clip"]]
            out = {"clip": r["clip"], "label": r["label"]}
            for cond, use_tone in (("words", False), ("both", True)):
                try:
                    out[cond] = respond(r, model=model, tone=use_tone, temperature=0.0)
                except Exception as e:
                    out[cond] = {"error": f"{type(e).__name__}: {e}"}
            try:
                prompt = out["both"]["response"]
                got, err = _batch([prompt], model)
                out["recipe"] = got[prompt]["recipe"] if prompt in got else None
            except Exception as e:
                out["recipe"] = None
            return out

        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(run, rows))
        with open(path, "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in res)
        report[tag] = summarize(res)
        print_row(tag, report[tag])
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1)


def summarize(res, conds=("words", "both")):
    """Reading accuracy per condition, physical appropriateness of the responses, share of valid recipes."""
    labels = [x["label"] for x in res]
    rep = {c: ua(labels, [x[c].get("feeling", "error") for x in res]) for c in conds if all(c in x for x in res)}
    ok = {c: [physical_ok(x["label"], x["recipe"]) for x in res if x["label"] == c and x["recipe"]] for c in CLASSES}
    rep["response_ok"] = {c: float(np.mean(v)) if v else None for c, v in ok.items()}
    rep["response_ok_mean"] = float(np.mean([v for v in rep["response_ok"].values() if v is not None]))
    # appropriate AND produced: a missing recipe counts as a failure to respond
    rep["response_ok_all"] = float(np.mean([bool(x["recipe"]) and physical_ok(x["label"], x["recipe"]) for x in res]))
    rep["recipes_valid"] = float(np.mean([x["recipe"] is not None for x in res]))
    return rep


def print_row(tag, rep):
    cond = " | ".join(f"{c} UA {rep[c]['ua']:.2f}" for c in ("words", "both") if c in rep)
    print(f"{tag:34s} {cond} | response appropriate {rep['response_ok_mean']:.2f} (counting missing as failures "
          f"{rep['response_ok_all']:.2f}) | valid recipes {rep['recipes_valid']:.2f}", flush=True)


def ua(labels, preds):
    C = np.zeros((4, 4), int)
    for y, p in zip(labels, preds):
        if p in CLASSES:
            C[CLASSES.index(y), CLASSES.index(p)] += 1
    n = np.array([sum(1 for y in labels if y == c) for c in CLASSES])
    return {"ua": float(np.mean(np.diag(C) / np.maximum(n, 1))), "confusion": C.tolist(),
            "other": int(sum(1 for p in preds if p not in CLASSES))}


if __name__ == "__main__":
    main()
