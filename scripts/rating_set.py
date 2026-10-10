"""Build the items for the human rating page: one heard clip and k of Reachy's responses, rendered as robot videos.

  python scripts/rating_set.py --out runs/rating/v2 --k 4 \
      --eval runs/respond_eval/{student_v3,student_v3-1.7b,Qwen3-Next-80B-A3B-Instruct,Qwen3-4B-Instruct-2507}+context.jsonl \
      --heard runs/respond_eval/heard_ctx.jsonl --n-eval 40 \
      --train runs/rating/train_samples.jsonl runs/rating/train_samples_more.jsonl --train-heard runs/listen_train/heard.jsonl

Two pools, mixed on the page:
* ``eval``: MELD *test* clips, one answer from each model in ``--eval`` (k models). They give a human score for the
  models and are never trained on.
* ``train``: MELD *train* clips, k sampled answers of the student (``--train``, from ``scripts/student_respond.py
  --samples N --temperature 0.9``). Preferences on these are the data for preference tuning (DPO).

With k = 2 a rater picks the better one; with k > 2 the best and the worst (best-worst scaling): one judgement then
orders 2k - 3 of the k(k - 1)/2 pairs, five of six for k = 4, so each minute of rating yields several preference
pairs instead of one.

Writes ``items.json`` (``[{id, pool, clip, label, context, text, voice, answers: [{model, feeling, response,
reading, recipe, video}, ...]}]``) and ``videos/<hash>.mp4``, named by recipe so later rounds reuse renders.
Audio is not included: MELD is cut from a TV show, so raters read the transcript and the voice model's reading.
"""
import argparse
import hashlib
import json
import os
import random


def load(path):
    return [json.loads(line) for line in open(path)]


def answer(model, d):
    b = d["both"]
    return {"model": model, "feeling": b.get("feeling"), "response": b.get("response", ""),
            "reading": b.get("reading", ""), "recipe": d["recipe"]}


def video_name(recipe):
    return hashlib.sha1(recipe.encode()).hexdigest()[:12] + ".mp4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=2, help="answers per item")
    ap.add_argument("--eval", nargs="+", help="k answer files on the test clips, one per model")
    ap.add_argument("--heard", default="runs/respond_eval/heard_ctx.jsonl")
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--train", nargs="+", help="student samples on train clips (several answers per clip id)")
    ap.add_argument("--train-heard", default="runs/listen_train/heard.jsonl")
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--ckpt", default="hf://mszarski/reachy-motion-generator/generator_v2.pt")
    ap.add_argument("--size", type=int, nargs=2, default=(360, 300))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--shard", default="0/1", help="i/n: render only every n-th video, from the i-th (parallel jobs)")
    a = ap.parse_args()
    rng = random.Random(0)
    items = []
    if a.eval:
        assert len(a.eval) == a.k, "--eval takes one answer file per answer shown (k)"
        heard = {r["clip"]: r for r in load(a.heard)}
        files = [{d["clip"]: d for d in load(p)} for p in a.eval]
        names = [os.path.basename(p).split("+")[0] for p in a.eval]
        clips = [c for c in files[0] if all(c in f and f[c]["recipe"] for f in files)
                 and len({f[c]["recipe"] for f in files}) == a.k]
        by = {}
        for c in clips:
            by.setdefault(heard[c]["label"], []).append(c)
        for lab, cs in sorted(by.items()):
            for c in rng.sample(cs, min(len(cs), a.n_eval // len(by))):
                items.append({"pool": "eval", "clip": c, "heard": heard[c],
                              "answers": [answer(n, f[c]) for n, f in zip(names, files)]})
    if a.train:
        heard = {r["clip"]: r for r in load(a.train_heard)}
        many = {}
        for p in a.train:
            for d in load(p):
                if d["recipe"] and d["recipe"] not in {x["recipe"] for x in many.get(d["clip"], [])}:
                    many.setdefault(d["clip"], []).append(d)
        cs = sorted(c for c, ds in many.items() if len(ds) >= a.k)
        for c in rng.sample(cs, min(len(cs), a.n_train)):
            items.append({"pool": "train", "clip": c, "heard": heard[c],
                          "answers": [answer("student", d) for d in many[c][:a.k]]})
    rng.shuffle(items)
    rows = []
    for i, it in enumerate(items):
        h = it["heard"]
        rows.append({"id": f"r{i:03d}", "pool": it["pool"], "clip": it["clip"],
                     "label": h.get("label") or h.get("meld_emotion"), "context": h.get("context") or [],
                     "text": h["text"], "voice": h["emotion"],
                     "answers": [{**x, "video": video_name(x["recipe"])} for x in it["answers"]]})
    os.makedirs(os.path.join(a.out, "videos"), exist_ok=True)
    with open(os.path.join(a.out, "items.json"), "w") as f:
        json.dump(rows, f, indent=1)
    jobs = {x["video"]: (x["recipe"], x["response"], os.path.join(a.out, "videos", x["video"]))
            for row in rows for x in row["answers"]}
    i, n = map(int, a.shard.split("/"))
    todo = [j for k, j in enumerate(sorted(jobs.values())) if k % n == i]
    print(f"{len(rows)} items ({sum(r['pool'] == 'eval' for r in rows)} eval), {len(jobs)} videos, "
          f"{len(todo)} in shard {a.shard}", flush=True)
    from multiprocessing import Pool

    with Pool(a.workers, initializer=_init, initargs=(a.ckpt, a.size)) as pool:
        for k, path in enumerate(pool.imap_unordered(_render, todo)):
            print(f"{k + 1}/{len(todo)} {os.path.basename(path)}", flush=True)


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

    recipe, response, path = job
    if not os.path.exists(path):
        net, stats, dev = _W["gen"]
        seed = int(os.path.basename(path)[:8], 16) % 100000          # from the recipe hash: the same video every run
        A = generate_batch(net, stats, variants(recipe, 1, seed=seed, fc=2.0, kdt=0.25), dev, seeds=[seed])[0]
        move = _W["reach"].project(traj_to_move(A, response))[0]
        _write(path, _W["sim"].play(move))
    return path


if __name__ == "__main__":
    main()
