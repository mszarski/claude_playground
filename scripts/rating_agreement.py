"""Do the automatic checks agree with the human ratings? (Answer after rating, before trusting the metrics again.)

  python scripts/rating_agreement.py --items runs/rating/v2/items.json --ratings runs/rating/ratings_export

For every best-worst judgement (or a/b pick) on the response rating page, compares the two ends with the metrics
scripts/eval_respond.py reports:
* the physical check (``physical_ok`` against MELD's label: sad / angry -> calm and slow, happy -> lively,
  neutral -> attentive): among judgements where exactly one of the two passes, how often is it the one the human
  picked as better? 50% = the check says nothing about what people prefer.
* the reading: among judgements where exactly one of the two read the person's feeling as MELD labels it, how often
  is that the human's pick?

With 40 or so judgements the interval is wide (shown as a 90% Wilson interval), but a check that people disagree
with is worth knowing about before the next round of training is steered by it.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from eval_respond import physical_ok  # noqa: E402
from label_respond import MELD  # noqa: E402

LABELS = ("neutral", "happy", "sad", "angry")


def wilson(k, n, z=1.645):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load_votes(path):
    if os.path.isdir(path):
        return [json.load(open(os.path.join(dp, f))) for dp, _, fs in os.walk(path) for f in fs if f.endswith(".json")]
    raw = json.load(open(path))
    return raw if isinstance(raw, list) else raw.get("documents", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--ratings", required=True, help="the page's exported votes (folder of JSON files or a list)")
    a = ap.parse_args()
    items = {it["id"]: it for it in json.load(open(a.items))}
    tallies = {"physical check": [0, 0], "reading": [0, 0]}
    judged = 0
    for d in load_votes(a.ratings):
        for item_id, v in ((d.get("data", d)).get("votes") or {}).items():
            it = items.get(item_id)
            if not it or v.get("same"):
                continue
            answers = it.get("answers") or [it["a"], it["b"]]
            if "best" in v:
                good, bad = answers[v["best"]], answers[v["worst"]]
            elif v.get("choice") in ("a", "b"):
                good, bad = (answers[0], answers[1]) if v["choice"] == "a" else (answers[1], answers[0])
            else:
                continue
            label = MELD.get(it["label"], it["label"])
            if label not in LABELS:
                continue
            judged += 1
            for name, ok in (("physical check", lambda x: physical_ok(label, x["recipe"])),
                             ("reading", lambda x: x.get("feeling") == label)):
                g, b = ok(good), ok(bad)
                if g != b:
                    tallies[name][1] += 1
                    tallies[name][0] += int(g)
    print(f"{judged} judgements with a clear label\n")
    for name, (k, n) in tallies.items():
        lo, hi = wilson(k, n)
        verdict = ("agrees with people" if lo > 0.5 else "disagrees with people" if hi < 0.5 else "not clear yet")
        rate = f"{k / n:.0%}" if n else "n/a"
        print(f"{name:15s} decides {n:3d} judgements; the human's pick is the one it favours {rate} "
              f"(90% interval {lo:.0%} - {hi:.0%}): {verdict}")


if __name__ == "__main__":
    main()
