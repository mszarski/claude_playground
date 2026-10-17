"""Rejection-sampling data for the CadQuery writer: for DeepCAD parts, decode one greedy program and --n_samples sampled
programs with vLLM, score each against the GT solid (scripts/cq_iou.score, GT normalisation) and save everything.
scripts/build_rft.py then keeps the best program per part as a fine-tuning target (train_vlm_cq.py --rft).

Each part gets one input, drawn like train_vlm_rl.py: view1 (one random fixed-camera view), view4 (2x2 grid) or
view1+expert (one view + the expert prompt). Prompts are formatted exactly as mm.vlm_cad.encode_batch does, and the
image processor settings come from the model dir (448 px -> 196 visual tokens), so vLLM sees what training saw.
Parts are processed in chunks written to <out>/chunk_<k>.pkl (existing chunks are skipped: the run resumes); scoring
of chunk k runs in a CPU pool while chunk k+1 is generated.

  python scripts/sample_rft.py --model results/vlm_cq_v1/model --data data --out results/rft_v1 --n_parts 30000
  python scripts/sample_rft.py ... --split validation --n_parts 256 --n_samples 0 --modes view4=1   # greedy check
"""
import argparse
import json
import os
import pickle
import random
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def gt_vec(data, u):
    with h5py.File(os.path.join(data, "cad_vec", u + ".h5"), "r") as f:
        return f["vec"][:].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--deepcad_cq", default="data/deepcad_cq.parquet")
    ap.add_argument("--split", default="train", choices=["train", "validation"])
    ap.add_argument("--n_parts", type=int, default=30000)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--modes", default="view1=0.4,view4=0.4,view1+expert=0.2")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--max_len", type=int, default=4096, help="vLLM context: prompt + image + program")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 3))
    ap.add_argument("--score_timeout", type=float, default=60.0)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    from pebble import ProcessPool
    from scripts.cq_iou import score
    pool = ProcessPool(max_workers=a.workers)            # before vLLM: workers fork without CUDA state

    rng = random.Random(a.seed)
    render_dir = os.path.join(a.data, "renders")
    with open(os.path.join(a.data, "text_index.json")) as f:
        text_index = json.load(f)
    dc = pd.read_parquet(a.deepcad_cq)
    with open(os.path.join(render_dir, "status.json")) as f:
        rendered = {u for u, st in json.load(f).items() if st in ("ok", "skip")}
    dc = dc[dc.ok & dc.uid.isin(rendered) & (dc.split == a.split)]
    if a.split == "train":
        dc = dc.drop_duplicates("code")
    else:                                             # the 256 val parts train_vlm_cq.py validates on
        dc = dc[(dc.iou >= 0.95) & (dc.iou <= 1.001) & (dc.code.str.len() <= 2500)]
        dc = dc[dc.uid.map(lambda u: bool(text_index.get(u, {}).get("expert")))]
    uids = [u for u in dc.uid if os.path.exists(os.path.join(a.data, "cad_vec", u + ".h5"))
            and os.path.exists(os.path.join(render_dir, f"{u}_3.png"))]
    if a.split == "train":
        rng.shuffle(uids)
    uids = uids[: a.n_parts]
    modes = {k: float(v) for k, v in (kv.split("=") for kv in a.modes.split(","))}
    names, p = list(modes), np.array(list(modes.values())) / sum(modes.values())
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(a.model)
    max_prompt = a.max_len - a.max_new_tokens - 400   # room for the template and 196 image tokens
    plan, n_long = [], 0                              # (uid, mode, views, prompt)
    for u in uids:
        mode = names[int(np.searchsorted(np.cumsum(p), rng.random()))]
        prompt = text_index.get(u, {}).get("expert") if "expert" in mode else None
        if prompt and len(processor.tokenizer(prompt)["input_ids"]) > max_prompt:
            prompt, n_long = None, n_long + 1         # rare very long prompt: image only
        if "expert" in mode and not prompt:
            mode = "view1"
        views = [0, 1, 2, 3] if mode == "view4" else ([0] if a.split == "validation" else [rng.randrange(4)])
        plan.append((u, mode, views, prompt))
    del text_index
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)
    print(f"[rft] {len(plan)} {a.split} parts, modes {modes}, {a.n_samples} samples at T={a.temp} + greedy; "
          f"{n_long} over-long prompts dropped", flush=True)

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")   # FlashInfer JIT-compiles its sampler: needs nvcc
    from vllm import LLM, SamplingParams
    from mm.vlm_cad import _messages, compose_views, configure
    print(f"[vlm] view grid {configure(a.model)} px", flush=True)   # the layout this model was trained on
    from train_vlm_cad import load_views
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_len, limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, generation_config="vllm", seed=a.seed)
    stop = [processor.tokenizer.convert_tokens_to_ids("<|im_end|>"), processor.tokenizer.eos_token_id]
    sp_greedy = SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens, stop_token_ids=stop)
    sp_sample = SamplingParams(n=max(a.n_samples, 1), temperature=a.temp, top_p=1.0, max_tokens=a.max_new_tokens,
                               stop_token_ids=stop)          # engine-level seed only: a per-request seed can repeat the n samples

    def requests(rows):
        out = []
        for u, mode, views, prompt in rows:
            image = compose_views(load_views("", os.path.join(render_dir, u), views))
            text = processor.apply_chat_template(_messages(image, prompt), tokenize=False, add_generation_prompt=True)
            out.append({"prompt": text, "multi_modal_data": {"image": image}})
        return out

    def finish(k, rows, codes, futs, t_gen):
        recs = []
        for (u, mode, views, prompt), cs, fs in zip(rows, codes, futs):
            rs = []
            for fu in fs:
                try:
                    r = fu.result()
                except Exception:  # noqa: BLE001 - timeout / crash: invalid, flagged
                    r = {"valid": False, "gt": 0.0, "own": 0.0, "failed": True}
                rs.append(r)
            recs.append({"uid": u, "mode": mode, "views": views, "prompt": prompt, "greedy": cs[0],
                         "samples": cs[1:], "scores": rs})
        ok = [r for r in recs if r["scores"][0] is not None]
        g = np.array([r["scores"][0]["gt"] for r in ok])
        best = np.array([max(s["gt"] for s in r["scores"]) for r in ok])
        failed = sum(bool(s and s.get("failed")) for r in recs for s in r["scores"])
        with open(os.path.join(a.out, f"chunk_{k}.pkl"), "wb") as f:
            pickle.dump(recs, f)
        print(json.dumps({"chunk": k, "parts": len(recs), "greedy_iou": round(float(g.mean()), 4),
                          "best_iou": round(float(best.mean()), 4),
                          "improved_0.02": round(float(np.mean(best > g + 0.02)), 4),
                          "greedy_invalid": round(float(np.mean([not r["scores"][0]["valid"] for r in ok])), 4),
                          "timeout_or_crash": failed, "t_gen": round(t_gen), "t_total": round(time.time() - t0)}),
              flush=True)

    pending = None
    t0 = time.time()
    for k, s in enumerate(range(0, len(plan), a.chunk)):
        if os.path.exists(os.path.join(a.out, f"chunk_{k}.pkl")):
            print(f"[rft] chunk {k} exists, skipping", flush=True)
            continue
        rows = plan[s: s + a.chunk]
        t1 = time.time()
        reqs = requests(rows)
        greedy = [o.outputs[0].text for o in llm.generate(reqs, sp_greedy, use_tqdm=False)]
        sampled = ([[c.text for c in o.outputs] for o in llm.generate(reqs, sp_sample, use_tqdm=False)]
                   if a.n_samples else [[] for _ in rows])
        t_gen = time.time() - t1
        codes = [[(c.strip() or None) for c in [g] + ss] for g, ss in zip(greedy, sampled)]
        futs = [[pool.schedule(score, args=((gt_vec(a.data, u), c, "cadquery"),), timeout=a.score_timeout)
                 for c in cs] for (u, _, _, _), cs in zip(rows, codes)]
        if pending:
            finish(*pending)                          # scores of the previous chunk, computed during this generate
        pending = (k, rows, codes, futs, t_gen)
    if pending:
        finish(*pending)
    pool.close()
    print(f"[done] {a.out}", flush=True)


if __name__ == "__main__":
    main()
