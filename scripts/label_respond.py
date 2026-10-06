"""Teacher-label MELD training clips for the distilled responder, keeping only answers the human labels agree with.

  python scripts/label_respond.py --heard runs/listen_train/heard.jsonl --teacher Qwen/Qwen3-Next-80B-A3B-Instruct \
      --out runs/respond_sft

For each clip (from scripts/listen_meld.py) the teacher writes {feeling, reading, response} (rmr.respond) and a recipe
for the response (zero-shot planner, same model). Kept when the teacher's feeling matches MELD's label and, for
neutral / happy / sad / angry, the recipe passes the physical check of scripts/eval_respond.py. Writes
labels.jsonl (everything, resumable), train.jsonl / val.jsonl (chat messages for rmr.planner.finetune train).
"""
import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from eval_respond import physical_ok  # noqa: E402

MELD = {"neutral": "neutral", "joy": "happy", "sadness": "sad", "anger": "angry", "disgust": "angry",
        "fear": "anxious", "surprise": "surprised"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heard", required=True)
    ap.add_argument("--teacher", default="Qwen/Qwen3-Next-80B-A3B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--hinted", action="store_true",
                    help="also re-ask the teacher, with the human label as a hint, on clips it misread (labels_hinted.jsonl)")
    a = ap.parse_args()
    from rmr.planner.write import _batch
    from rmr.respond import respond, student_messages

    os.makedirs(a.out, exist_ok=True)
    rows = [json.loads(l) for l in open(a.heard)]
    lab_path = os.path.join(a.out, "labels.jsonl")
    done = {json.loads(l)["clip"] for l in open(lab_path)} if os.path.exists(lab_path) else set()

    def label(r):
        try:
            t = respond(r, model=a.teacher, temperature=0.4)
            got, _ = _batch([t["response"]], a.teacher)
            t["recipe"] = got[t["response"]]["recipe"] if t["response"] in got else None
            return {"clip": r["clip"], "ok": True, **t}
        except Exception as e:
            return {"clip": r["clip"], "ok": False, "error": f"{type(e).__name__}: {e}"}

    todo = [r for r in rows if r["clip"] not in done]
    with ThreadPoolExecutor(a.workers) as ex, open(lab_path, "a") as f:
        for i, out in enumerate(ex.map(label, todo)):
            f.write(json.dumps(out) + "\n")
            f.flush()
            if i % 200 == 0:
                print(f"labelled {i}/{len(todo)}", flush=True)
    labels = {json.loads(l)["clip"]: json.loads(l) for l in open(lab_path)}
    hinted = {}
    if a.hinted:
        hint_path = os.path.join(a.out, "labels_hinted.jsonl")
        hdone = {json.loads(l)["clip"] for l in open(hint_path)} if os.path.exists(hint_path) else set()
        wrong = [r for r in rows if r["clip"] not in hdone and not (labels.get(r["clip"], {}).get("ok")
                 and labels[r["clip"]].get("feeling") == MELD[r["meld_emotion"]])]

        def label_hinted(r):
            want = MELD[r["meld_emotion"]]
            try:
                t = respond(r, model=a.teacher, temperature=0.4, hint=want)
                got, _ = _batch([t["response"]], a.teacher)
                t["recipe"] = got[t["response"]]["recipe"] if t["response"] in got else None
                return {"clip": r["clip"], "ok": True, **t}
            except Exception as e:
                return {"clip": r["clip"], "ok": False, "error": f"{type(e).__name__}: {e}"}

        with ThreadPoolExecutor(a.workers) as ex, open(hint_path, "a") as f:
            for i, out in enumerate(ex.map(label_hinted, wrong)):
                f.write(json.dumps(out) + "\n")
                f.flush()
                if i % 200 == 0:
                    print(f"hinted {i}/{len(wrong)}", flush=True)
        hinted = {json.loads(l)["clip"]: json.loads(l) for l in open(hint_path)}
    keep, stats = [], {}
    for r in rows:
        t, want = labels.get(r["clip"]), MELD[r["meld_emotion"]]
        s = stats.setdefault(want, [0, 0, 0])
        s[0] += 1
        if not (t and t.get("ok") and t.get("recipe")) or t["feeling"] != want:
            t = hinted.get(r["clip"])                 # the teacher misread it: use its hinted answer, if any
            if not (t and t.get("ok") and t.get("recipe")) or t["feeling"] != want:
                continue
        s[1] += 1
        if want in ("neutral", "happy", "sad", "angry") and not physical_ok(want, t["recipe"]):
            continue
        s[2] += 1
        keep.append({"messages": student_messages(r, t), "label": want, "clip": r["clip"]})
    random.Random(0).shuffle(keep)
    val, train = keep[:a.val], keep[a.val:]
    for name, rs in (("train", train), ("val", val)):
        with open(os.path.join(a.out, f"{name}.jsonl"), "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in rs)
    print("per feeling: clips / usable answer (agrees with the label" + (", or hinted" if a.hinted else "") +
          ") / also passes the physical check")
    for k, (n, agree, ok) in sorted(stats.items()):
        print(f"  {k:10s} {n:5d} {agree:5d} ({agree / max(n, 1):.0%}) {ok:5d}")
    print(f"train {len(train)} | val {len(val)} -> {a.out}")


if __name__ == "__main__":
    main()
