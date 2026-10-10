"""Write lines for synthetic emotional speech: what a person might say to a small home robot, per feeling.

  python scripts/synth_lines.py --out runs/synth/lines.jsonl --per-feeling 15

Part of the licence-clean voice data (scripts/synth_speech.py voices them with Zonos, Apache-2.0): instead of audio
cut from a TV show (MELD), the student can learn from lines written by an open LLM (the teacher, Apache-2.0) and
voiced by an open TTS model in synthetic voices. Each line comes with the feeling it should be spoken with, a short
situation and, for some, the line before it, so the student keeps seeing context.
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

FEELINGS = ["neutral", "happy", "sad", "angry"]
STYLES = ["openly", "subtly, the words alone barely show it", "with sarcasm or irony", "tiredly", "in a rush",
          "to a friend", "talking to the robot directly", "on the phone, half to themselves"]

SCHEMA = {"type": "object", "properties": {"lines": {"type": "array", "items": {
    "type": "object", "properties": {"situation": {"type": "string"}, "before": {"type": "string"},
                                     "line": {"type": "string"}},
    "required": ["situation", "before", "line"], "additionalProperties": False}}},
    "required": ["lines"], "additionalProperties": False}

PROMPT = """Write {n} different things a person might say out loud at home, near a small expressive robot, while
feeling {feeling}. Make them sound like real speech: short (4 to 20 words), everyday topics (work, family, food,
weather, news, plans, the robot itself), varied people and situations. Express the feeling {style}.
For each give a one-sentence situation, optionally the line said just before by someone else ("" if none), and
the line itself. No stage directions, no emojis, no quotation marks inside the line."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-feeling", type=int, default=15)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--model", default="Qwen/Qwen3-Next-80B-A3B-Instruct")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from rmr.planner.llm import chat_json

    rng = random.Random(a.seed)
    jobs = [(f, rng.choice(STYLES)) for f in FEELINGS for _ in range(max(1, a.per_feeling // a.batch))]

    def one(job):
        feeling, style = job
        try:
            out = chat_json([{"role": "user", "content": PROMPT.format(n=a.batch, feeling=feeling, style=style)}],
                            SCHEMA, "lines", model=a.model, temperature=0.9)
            return [{"feeling": feeling, "style": style, **x} for x in out.get("lines", [])][:a.batch]
        except Exception as e:
            print(f"failed ({feeling}, {style}): {type(e).__name__}: {e}", flush=True)
            return []

    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        for got in ex.map(one, jobs):
            rows += got
    seen, uniq = set(), []
    for r in rows:
        key = r["line"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            uniq.append({**r, "id": f"s{len(uniq):04d}"})
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in uniq)
    print(f"{len(uniq)} lines -> {a.out}")


if __name__ == "__main__":
    main()
