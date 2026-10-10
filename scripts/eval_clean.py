"""Licence-clean evaluation of the responder: no MELD, and a test that only tone can pass.

  python scripts/eval_clean.py build --bank runs/synth_v4/tone_bank.jsonl --lines runs/eval_clean/lines_raw.jsonl \
      --out runs/eval_clean
  python scripts/eval_clean.py teacher --heard runs/eval_clean/tone.jsonl --out runs/eval_clean/teacher_tone.jsonl [--words]
  python scripts/student_respond.py --model <student> --heard runs/eval_clean/tone.jsonl --out runs/eval_clean/v4_tone.jsonl
  python scripts/eval_clean.py score runs/eval_clean/*_tone.jsonl runs/eval_clean/*_lines.jsonl

Two sets, both from the CREMA-D actors held out of all training data (scripts/tone_bank.py held_out, 18 of 91):
  tone.jsonl   real clips of those actors saying CREMA-D's flat sentences ("It's eleven o'clock.") as neutral,
               happy, sad, angry or anxious (60 each). The words say nothing, so reading above chance means the
               responder uses the voice reading; words only (``teacher --words``) shows where chance is.
  lines.jsonl  freshly written lines (scripts/synth_lines.py with another seed; none seen in training) with
               readings from the held-out actors, six feelings: words and tone together, as in conversation.
``score`` gives unweighted accuracy of the responder's feeling over each set's classes (chance 1/5 or 1/6), the
confusions, and how often the response's recipe passes scripts/eval_respond.py's physical check for neutral / happy /
sad / angry (a missing or broken recipe fails).
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from synth_heard import pools, reading  # noqa: E402
from tone_bank import SENTENCES, held_out  # noqa: E402

TONE = ["neutral", "happy", "sad", "angry", "anxious"]


def build(a):
    rng = random.Random(0)
    bank = [json.loads(x) for x in open(a.bank)]
    os.makedirs(a.out, exist_ok=True)
    tone = []
    for f in TONE:
        clips = [b for b in bank if held_out(b["actor"]) and b["feeling"] == f and b["crema"] != "Disgust"]
        for b in rng.sample(clips, a.per_class):
            tone.append({"clip": b["file"], "label": f, "feeling": f, "text": SENTENCES[b["sentence"]], "context": [],
                         **reading(b)})
    seen = set()
    for p in a.train_lines:
        seen |= {json.loads(x)["line"].strip().lower() for x in open(p)}
    by = pools(bank, "eval")
    lines = []
    for ln in map(json.loads, open(a.lines)):
        if ln["line"].strip().lower() in seen:
            continue
        f = ln["feeling"]
        pool = "neutral" if f != "neutral" and rng.random() < a.flat else f
        lines.append({"clip": "e" + ln["id"], "label": f, "feeling": f, "text": ln["line"],
                      "context": [ln["before"]] if ln.get("before") else [], **reading(rng.choice(by[pool]))})
    for name, rows in (("tone", tone), ("lines", lines)):
        with open(os.path.join(a.out, f"{name}.jsonl"), "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
        print(f"{name}: {len(rows)} {dict(Counter(r['label'] for r in rows))}")


def teacher(a):
    from concurrent.futures import ThreadPoolExecutor

    from rmr.planner.write import _batch
    from rmr.respond import respond

    rows = [json.loads(x) for x in open(a.heard)]

    def one(r):
        try:
            t = respond(r, model=a.model, tone=not a.words, temperature=0.0)
            got, _ = _batch([t["response"]], a.model) if not a.no_recipe else ({}, None)
            return {"clip": r["clip"], "label": r["label"], "both": t,
                    "recipe": got.get(t["response"], {}).get("recipe")}
        except Exception as e:
            return {"clip": r["clip"], "label": r["label"], "both": {"feeling": "invalid"}, "recipe": None,
                    "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(a.workers) as ex:
        outs = list(ex.map(one, rows))
    with open(a.out, "w") as f:
        f.writelines(json.dumps(o) + "\n" for o in outs)
    print(f"{len(outs)} answers -> {a.out}")


def score(a):
    from eval_respond import physical_ok

    for p in a.answers:
        rows = [json.loads(x) for x in open(p)]
        classes = sorted({r["label"] for r in rows}, key=lambda c: (TONE + ["surprised"]).index(c))
        acc = {c: [r["both"]["feeling"] == c for r in rows if r["label"] == c] for c in classes}
        ua = sum(sum(v) / len(v) for v in acc.values()) / len(acc)
        phys = [bool(r.get("recipe")) and physical_ok(r["label"], r["recipe"]) for r in rows
                if r["label"] in ("neutral", "happy", "sad", "angry")]
        print(f"{os.path.basename(p):32s} UA {ua:5.1%} (chance {1 / len(classes):.0%})  physical {sum(phys) / max(len(phys), 1):5.1%}  "
              + "  ".join(f"{c[:3]} {sum(v) / len(v):.0%}" for c, v in acc.items()))
        if a.confusions:
            for c in classes:
                print(f"    {c:9s} ->", dict(Counter(r["both"]["feeling"] for r in rows if r["label"] == c).most_common()))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--bank", required=True)
    b.add_argument("--lines", required=True, help="fresh lines from scripts/synth_lines.py (another seed)")
    b.add_argument("--train-lines", nargs="*", default=[], help="lines used in training: any repeats are dropped")
    b.add_argument("--out", default="runs/eval_clean")
    b.add_argument("--per-class", type=int, default=60)
    b.add_argument("--flat", type=float, default=0.2)
    t = sub.add_parser("teacher")
    t.add_argument("--heard", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--model", default="Qwen/Qwen3-Next-80B-A3B-Instruct")
    t.add_argument("--words", action="store_true", help="words only: no voice reading")
    t.add_argument("--no-recipe", action="store_true")
    t.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("score")
    s.add_argument("answers", nargs="+")
    s.add_argument("--confusions", action="store_true")
    a = ap.parse_args()
    {"build": build, "teacher": teacher, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
