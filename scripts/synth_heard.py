"""Give each written line a voice reading from a real recording of the same feeling -> heard.jsonl for labelling.

  python scripts/synth_heard.py --lines runs/synth_v4/lines.jsonl --bank runs/synth_v4/tone_bank.jsonl \
      --out runs/synth_v4/heard.jsonl

The licence-clean replacement for scripts/listen_meld.py's output: the same fields rmr.voice.Listener.hear gives
(text, emotion, confidence, probs, arousal, dominance, valence) plus context and the feeling the line was written
with, which scripts/label_respond.py uses as the label in place of MELD's. Words come from scripts/synth_lines.py
(the teacher), readings from scripts/tone_bank.py (the voice model on CREMA-D clips, ODbL).

Pools: each feeling draws from the CREMA-D clips of that emotion (disgust counts as angry, fear as anxious). CREMA-D
has no surprise, so surprised lines draw from the clips the voice model itself hears as surprise, whatever the
actor was asked to play. ``--flat`` of the emotional lines get a reading from a neutral clip instead: people often say
emotional things in a flat voice, and the student should keep reading the words when the voice gives nothing.
"""
import argparse
import json
import random
from collections import Counter


def pools(bank):
    by = {}
    for b in bank:
        by.setdefault(b["feeling"], []).append(b)
    by["surprised"] = [b for b in bank if b["emotion"] == "surprise"]
    return by


def reading(b):
    return {k: b[k] for k in ("emotion", "confidence", "probs", "arousal", "dominance", "valence")} | {"tone": b["file"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flat", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    by = pools([json.loads(x) for x in open(a.bank)])
    print("pool sizes:", {k: len(v) for k, v in sorted(by.items())})
    out, src = [], Counter()
    for ln in map(json.loads, open(a.lines)):
        feel = ln["feeling"]
        pool = "neutral" if feel != "neutral" and rng.random() < a.flat else feel
        src[feel, pool] += 1
        out.append({"clip": ln["id"], "feeling": feel, "text": ln["line"],
                    "context": [ln["before"]] if ln.get("before") else [], "situation": ln.get("situation", ""),
                    "seconds": round(0.4 * len(ln["line"].split()) + 0.5, 2), **reading(rng.choice(by[pool]))})
    with open(a.out, "w") as f:
        f.writelines(json.dumps(o) + "\n" for o in out)
    print(f"{len(out)} lines -> {a.out}; reading pool per feeling: " +
          ", ".join(f"{f}<-{p} {n}" for (f, p), n in sorted(src.items())))


if __name__ == "__main__":
    main()
