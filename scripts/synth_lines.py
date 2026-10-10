"""Write lines for synthetic emotional speech: what a person might say to a small home robot, per feeling.

  python scripts/synth_lines.py --out runs/synth/lines.jsonl --per-feeling 15

Part of the licence-clean voice data: instead of lines cut from a TV show (MELD), the student learns from lines
written by an open LLM (the teacher, Apache-2.0). Each line comes with the feeling it is said with, a short situation
and, for some, the line before it, so the student keeps seeing context. scripts/synth_heard.py gives each line a
voice reading from a real recording of the same feeling (scripts/tone_bank.py); scripts/synth_speech.py was the
pilot that voiced them with TTS instead. Each batch gets a topic and a speaker drawn at random, so thousands of lines
stay varied.
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

FEELINGS = ["neutral", "happy", "sad", "angry", "anxious", "surprised"]      # rmr.respond FEELINGS
STYLES = ["openly", "subtly, the words alone barely show it", "in a rush", "to a friend", "talking to the robot directly",
          "on the phone, half to themselves"]
# styles that would change the feeling itself (sarcasm reads as annoyance, tiredness as sadness) only where they fit
EXTRA_STYLES = {"angry": ["with sarcasm or irony", "tiredly"], "sad": ["tiredly"], "anxious": ["tiredly"]}
TOPICS = ["work", "school", "family", "a partner", "kids", "a pet", "food and cooking", "the weather", "the news",
          "money", "health", "a hobby", "sport", "travel", "the house", "neighbours", "a TV show", "a game",
          "shopping", "a friend", "plans for the weekend", "the robot itself", "technology", "the commute", "sleep",
          "a phone call they just had", "a letter or message", "an appointment", "a gift", "a memory"]
SPEAKERS = ["a teenager", "a retired person", "a parent of young kids", "a student", "a nurse after a shift",
            "someone working from home", "a grandparent", "a young professional", "someone living alone",
            "a shop owner", "a musician", "a person in their thirties", "a child of about ten", "a couple's partner"]

SCHEMA = {"type": "object", "properties": {"lines": {"type": "array", "items": {
    "type": "object", "properties": {"situation": {"type": "string"}, "before": {"type": "string"},
                                     "line": {"type": "string"}},
    "required": ["situation", "before", "line"], "additionalProperties": False}}},
    "required": ["lines"], "additionalProperties": False}

PROMPT = """Write {n} different things a person might say out loud at home, near a small expressive robot, while
feeling {feeling}. Make them sound like real speech: short (4 to 20 words), everyday topics (work, family, food,
weather, news, plans, the robot itself), varied situations; this time the speaker is {speaker} and the topic
is {topic}. Express the feeling {style}.
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
    ap.add_argument("--feelings", nargs="+", default=FEELINGS)
    ap.add_argument("--prefix", default="s", help="id prefix (to merge several runs)")
    a = ap.parse_args()
    from rmr.planner.llm import chat_json

    rng = random.Random(a.seed)
    jobs = [(f, rng.choice(STYLES + EXTRA_STYLES.get(f, [])), rng.choice(TOPICS), rng.choice(SPEAKERS)) for f in a.feelings
            for _ in range(max(1, a.per_feeling // a.batch))]

    def one(job):
        feeling, style, topic, speaker = job
        try:
            out = chat_json([{"role": "user", "content": PROMPT.format(n=a.batch, feeling=feeling, style=style,
                                                                       topic=topic, speaker=speaker)}],
                            SCHEMA, "lines", model=a.model, temperature=0.9)
            return [{"feeling": feeling, "style": style, "topic": topic, "speaker": speaker, **x}
                    for x in out.get("lines", [])][:a.batch]
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
            uniq.append({**r, "id": f"{a.prefix}{len(uniq):04d}"})
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in uniq)
    print(f"{len(uniq)} lines -> {a.out}")


if __name__ == "__main__":
    main()
