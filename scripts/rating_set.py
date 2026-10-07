"""Build the items for the human rating page: one heard clip, two of Reachy's responses, rendered as robot videos.

  python scripts/rating_set.py --out runs/rating \
      --eval runs/respond_eval/student_v3+context.jsonl runs/respond_eval/Qwen3-Next-80B-A3B-Instruct+context.jsonl \
      --heard runs/respond_eval/heard_ctx.jsonl --n-eval 40 \
      --train runs/rating/train_samples.jsonl --train-heard runs/listen_train/heard.jsonl

Two pools, mixed on the page:
* ``eval``: MELD *test* clips, two models' answers (``--eval A B``). These give a human score for the models and are
  never trained on.
* ``train``: MELD *train* clips, two sampled answers of the student (``--train``, from ``scripts/student_respond.py
  --samples 2``). Preferences on these are the data for preference tuning (DPO).

Writes ``items.json`` (``[{id, pool, clip, label, context, text, voice, a: {model, feeling, response, reading,
recipe, video}, b: {...}}]``) and ``videos/<id>_{a,b}.mp4``. The page randomises which side each answer shows on.
Audio is not included: MELD is cut from a TV show, so raters read the transcript and the voice model's reading.
"""
import argparse
import json
import os
import random


def load(path):
    return [json.loads(line) for line in open(path)]


def answer(model, d):
    b = d["both"]
    return {"model": model, "feeling": b.get("feeling"), "response": b.get("response", ""),
            "reading": b.get("reading", ""), "recipe": d["recipe"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval", nargs=2, metavar=("A", "B"))
    ap.add_argument("--heard", default="runs/respond_eval/heard_ctx.jsonl")
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--train", help="student samples on train clips (two answers per clip, same clip id)")
    ap.add_argument("--train-heard", default="runs/listen_train/heard.jsonl")
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--ckpt", default="hf://mszarski/reachy-motion-generator/generator_v2.pt")
    ap.add_argument("--size", type=int, nargs=2, default=(360, 300))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    rng = random.Random(0)
    items = []
    if a.eval:
        heard = {r["clip"]: r for r in load(a.heard)}
        A = {d["clip"]: d for d in load(a.eval[0])}
        B = {d["clip"]: d for d in load(a.eval[1])}
        names = [os.path.basename(p).split("+")[0] for p in a.eval]
        clips = [c for c in A if c in B and A[c]["recipe"] and B[c]["recipe"] and A[c]["recipe"] != B[c]["recipe"]]
        by = {}
        for c in clips:
            by.setdefault(heard[c]["label"], []).append(c)
        for lab, cs in sorted(by.items()):
            for c in rng.sample(cs, min(len(cs), a.n_eval // len(by))):
                items.append({"pool": "eval", "clip": c, "heard": heard[c],
                              "a": answer(names[0], A[c]), "b": answer(names[1], B[c])})
    if a.train:
        heard = {r["clip"]: r for r in load(a.train_heard)}
        two = {}
        for d in load(a.train):
            if d["recipe"]:
                two.setdefault(d["clip"], []).append(d)
        cs = [c for c, ds in two.items() if len(ds) >= 2 and ds[0]["recipe"] != ds[1]["recipe"]]
        for c in rng.sample(cs, min(len(cs), a.n_train)):
            items.append({"pool": "train", "clip": c, "heard": heard[c],
                          "a": answer("student", two[c][0]), "b": answer("student", two[c][1])})
    rng.shuffle(items)
    rows = []
    for i, it in enumerate(items):
        h = it["heard"]
        rows.append({"id": f"r{i:03d}", "pool": it["pool"], "clip": it["clip"],
                     "label": h.get("label") or h.get("meld_emotion"), "context": h.get("context") or [],
                     "text": h["text"], "voice": h["emotion"],
                     **{side: {**it[side], "video": f"r{i:03d}_{side}.mp4"} for side in ("a", "b")}})
    os.makedirs(os.path.join(a.out, "videos"), exist_ok=True)
    with open(os.path.join(a.out, "items.json"), "w") as f:
        json.dump(rows, f, indent=1)
    jobs = [(i, row[side]["recipe"], row[side]["response"], os.path.join(a.out, "videos", row[side]["video"]))
            for i, row in enumerate(rows) for side in ("a", "b")]
    from multiprocessing import Pool

    with Pool(a.workers, initializer=_init, initargs=(a.ckpt, a.size)) as pool:
        for k, path in enumerate(pool.imap_unordered(_render, jobs)):
            print(f"{k + 1}/{len(jobs)} {os.path.basename(path)}", flush=True)


_W = {}


def _init(ckpt, size):
    import torch

    from rmr.generator.sample import load as load_gen
    from rmr.reach import Reach
    from rmr.renderer.sim import Sim

    torch.set_num_threads(1)
    _W["gen"], _W["reach"], _W["sim"] = load_gen(ckpt), Reach(), Sim(*size)


def _render(job):
    from rmr.generator.sample import generate_batch
    from rmr.motion import traj_to_move
    from rmr.recipe import variants
    from rmr.renderer.outputs import _write

    i, recipe, response, path = job
    if not os.path.exists(path):
        net, stats, dev = _W["gen"]
        A = generate_batch(net, stats, variants(recipe, 1, seed=i, fc=2.0, kdt=0.25), dev, seeds=[i])[0]
        move = _W["reach"].project(traj_to_move(A, response))[0]
        _write(path, _W["sim"].play(move))
    return path


if __name__ == "__main__":
    main()
