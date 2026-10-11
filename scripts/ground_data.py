"""Planner training data in Pollen's house style, written with a big LLM (scripts/ground_planner.py has the library).

  python scripts/ground_data.py clips --library runs/ground/library.json --out runs/ground/clips.jsonl
  python scripts/ground_data.py restyle --library runs/ground/library.json --n 3000 --out runs/ground/restyled.jsonl

Rows in data/teacher/dataset.jsonl's format (id, prompt, idea, recipe, source, family, weight), so
``rmr.planner.sft`` builds the fine-tuning set from them like from the teacher data, with the same leak filter: any
prompt close to a held-out emotion or a probe is dropped, so those stay a real test.

* ``clips`` (source ``pollen_clip``): only 68 real clips exist (the held-out emotions and every clip of the same word
  left out). For each, the LLM sees the clip's caption, its transcribed recipe and a plain description of the motion,
  and writes ``--per-clip`` different prompts ("word. one sentence.") this exact motion would express: other
  situations, characters, wordings. Each is paired with the clip's own recipe, so the answers carry the animator's
  real timing and sizes, not an LLM's idea of them.
* ``restyle`` (source ``grounded``): ``--n`` prompts from the teacher data (the served sources) get a new recipe
  from the LLM with the real-clip library in its prompt (ground_planner.grounded_system): new concepts, house style.
"""
import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ground_planner import grounded_system, word  # noqa: E402
from judge_ratings import describe  # noqa: E402

CLIP_SCHEMA = {"type": "object", "properties": {"prompts": {"type": "array", "items": {
    "type": "object", "properties": {"prompt": {"type": "string"}, "idea": {"type": "string"}},
    "required": ["prompt", "idea"], "additionalProperties": False}}},
    "required": ["prompts"], "additionalProperties": False}

CLIP_PROMPT = """Reachy Mini is a small desktop robot: a head on a 6-axis platform, two antenna "ears", a rotating
body, no face. Below is one of its real, hand-animated motions.

Name and description: {caption}
Motion, one segment per second (e/eR/eL ears: 15 neutral, 0 up, 150 fully drooped; p pitch, + = head down; r tilt;
y turn; z height in mm; b body turn; E liveliness of fast detail 0-12):
  {recipe}
In plain words: {plain}

Write {n} different prompts that THIS EXACT motion would express well, as someone might ask the robot for a
motion. Each is "word. One sentence." like "downcast. You just heard your team lost the final." Vary them: other
situations, characters, animals, everyday moments, and other words for the feeling (synonyms, nuances), but stay
true to what the motion actually shows (its energy, direction, timing). For each, add the idea: one sentence on the
body language. Do not use the clip's own name as the word in more than one prompt."""


def clips(a):
    from rmr.planner.llm import chat_json

    lib = [x for x in json.load(open(a.library)) if not x["same_word_as_held_out"]]

    def one(x):
        msg = CLIP_PROMPT.format(caption=x["caption"], recipe=x["recipe"], plain=describe(x["recipe"]), n=a.per_clip)
        for _ in range(3):
            try:
                out = chat_json([{"role": "user", "content": msg}], CLIP_SCHEMA, "prompts", model=a.model,
                                temperature=0.9)
                return x, [p for p in out.get("prompts", []) if "." in p.get("prompt", "")][:a.per_clip]
            except Exception as e:
                print(x["name"], type(e).__name__, e, flush=True)
        return x, []

    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        for x, ps in ex.map(one, lib):
            for i, p in enumerate(ps):
                rows.append({"id": f"clip_{x['name']}_{i}", "prompt": p["prompt"].strip(), "idea": p["idea"].strip(),
                             "recipe": x["recipe"], "source": "pollen_clip", "family": word(x["name"]), "weight": 1})
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{len(rows)} prompts for {len(lib)} clips -> {a.out}")


def restyle(a):
    from rmr.planner import llm
    from rmr.planner.prompt import SCHEMA, user_message
    from rmr.recipe import check

    system = grounded_system(json.load(open(a.library)))
    data = os.path.join(os.path.dirname(__file__), "..", "data", "teacher", "dataset.jsonl")
    pool = [json.loads(x) for x in open(data)]
    pool = [r for r in pool if r["source"] in a.sources.split(",")]
    seen, uniq = set(), []
    for r in pool:
        if r["prompt"] not in seen:
            seen.add(r["prompt"])
            uniq.append(r)
    done = {json.loads(x)["prompt"] for x in open(a.out)} if os.path.exists(a.out) else set()
    todo = [r for r in random.Random(a.seed).sample(uniq, min(a.n, len(uniq))) if r["prompt"] not in done]
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]

    def one(batch):
        prompts, errors, got = [r["prompt"] for r in batch], None, {}
        for _ in range(3):
            try:
                out = llm.chat_json([{"role": "system", "content": system},
                                     {"role": "user", "content": user_message(prompts, errors)}],
                                    SCHEMA, "motion_recipes", model=a.model, temperature=0.7)
            except Exception as e:
                print(type(e).__name__, e, flush=True)
                continue
            by = {m.get("prompt", "").strip(): m for m in out.get("motions", []) if isinstance(m, dict)}
            errors = {}
            for p in prompts:
                m = by.get(p.strip())
                err = "missing from your answer" if not m or not isinstance(m.get("recipe"), str) else check(m["recipe"])
                if err:
                    errors[p] = err
                else:
                    got[p] = m
            prompts = list(errors)
            if not prompts:
                break
        return [(r, got[r["prompt"]]) for r in batch if r["prompt"] in got]

    n = 0
    with ThreadPoolExecutor(a.workers) as ex, open(a.out, "a") as f:
        for res in ex.map(one, batches):
            for r, m in res:
                f.write(json.dumps({"id": "g_" + r["id"], "prompt": r["prompt"], "idea": m.get("idea", ""),
                                    "recipe": m["recipe"], "source": "grounded", "family": r.get("family", ""),
                                    "weight": r.get("weight", 1)}) + "\n")
                n += 1
            f.flush()
    print(f"{n} restyled recipes -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clips")
    c.add_argument("--library", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--per-clip", type=int, default=15)
    c.add_argument("--model", default="moonshotai/Kimi-K3")
    c.add_argument("--workers", type=int, default=6)
    r = sub.add_parser("restyle")
    r.add_argument("--library", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--n", type=int, default=3000)
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--sources", default="claude,claude_events,seed,astra_lively")
    r.add_argument("--model", default="moonshotai/Kimi-K3")
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"clips": clips, "restyle": restyle}[a.cmd](a)


if __name__ == "__main__":
    main()
