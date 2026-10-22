"""Revision-turn training data from rich parts (scripts/gen_rich_data.py shards): the model's own attempts, with the
generator's program as the target. Same output format as scripts/build_revision.py (DeepCAD), plus an `input_stem`
column (the input renders), so train_vlm_cq.py --revision can mix both.

Per part: one input (view4 or one random view), one greedy program and --n_samples sampled ones (vLLM). Each attempt
is built, scored against the GT program (own IoU, scripts/rich_iou.py) and rendered with the input cameras, or its
build error is kept. Rows: the greedy attempt, plus one flawed sample (own IoU < --bad or failed) when there is one.
Target: the attempt itself when it is already right (own IoU >= --keep), else the GT program.

  python scripts/sample_rich_rev.py --model results/vlm_grpo_rich/model --rich data/rich/train --n_parts 15000 \
      --out data/rich_revision
Writes <out>/revision.parquet and <out>/attempts/<rid>_{0..3}.png.
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.sysutil import usable_cpus  # noqa: E402


def work(args):
    """Score + render one attempt -> (own IoU or -1, error or None, attempt stem or None)."""
    gt_code, code, rid, out = args
    from scripts.build_revision import work as render_attempt
    from scripts.rich_iou import score
    if not code:
        return -1.0, "no program was produced", None
    _, err, stem = render_attempt((rid, code, out))
    if err:
        return -1.0, err, None
    r = score((gt_code, code))
    return (r["own"] if r and r["valid"] else -1.0), None, stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rich", default="data/rich/train")
    ap.add_argument("--n_parts", type=int, default=15000)
    ap.add_argument("--n_samples", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--view4_p", type=float, default=0.6)
    ap.add_argument("--max_chars", type=int, default=3000)
    ap.add_argument("--bad", type=float, default=0.9)
    ap.add_argument("--keep", type=float, default=0.98)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=max(1, usable_cpus() - 2))
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    import pandas as pd
    from pebble import ProcessPool
    pool = ProcessPool(max_workers=a.workers)            # before CUDA
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from mm.vlm_cad import _messages, compose_views, configure
    from train_vlm_cad import load_views
    print(f"[vlm] view grid {configure(a.model)} px", flush=True)

    rng = random.Random(a.seed)
    dfs = []
    for sh in sorted(os.listdir(a.rich)):
        pq = os.path.join(a.rich, sh, "parts.parquet")
        if os.path.exists(pq):
            dfs.append(pd.read_parquet(pq).assign(stem=lambda d, sh=sh: [os.path.join(a.rich, sh, "renders", u)
                                                                        for u in d.uid]))
    df = pd.concat(dfs, ignore_index=True)
    df = df[df.code.str.len() <= a.max_chars].sample(frac=1.0, random_state=a.seed).head(a.n_parts)
    df = df[[os.path.exists(f"{s}_3.png") for s in df.stem]].reset_index(drop=True)
    views = [[0, 1, 2, 3] if rng.random() < a.view4_p else [rng.randrange(4)] for _ in range(len(df))]
    print(f"[rich-rev] {len(df)} parts from {a.rich}; families {df.family.value_counts().to_dict()}", flush=True)

    processor = AutoProcessor.from_pretrained(a.model)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_len, limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=0.85, enable_prefix_caching=True, generation_config="vllm", seed=a.seed)
    stop = [processor.tokenizer.convert_tokens_to_ids("<|im_end|>"), processor.tokenizer.eos_token_id]
    reqs = []
    for s, v in zip(df.stem, views):
        im = compose_views(load_views("", s, v))
        reqs.append({"prompt": processor.apply_chat_template(_messages([im], None), tokenize=False,
                                                             add_generation_prompt=True),
                     "multi_modal_data": {"image": im}})
    t0 = time.time()
    greedy = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens, stop_token_ids=stop),
                          use_tqdm=False)
    samples = llm.generate(reqs, SamplingParams(temperature=a.temp, n=a.n_samples, max_tokens=a.max_new_tokens,
                                                stop_token_ids=stop, seed=a.seed), use_tqdm=False)
    codes = [[g.outputs[0].text.strip()] + [o.text.strip() for o in s.outputs] for g, s in zip(greedy, samples)]
    print(f"[rich-rev] generated {sum(map(len, codes))} programs in {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.join(a.out, "attempts"), exist_ok=True)
    futs = [[pool.schedule(work, args=((g, c, f"{u}_{k}".replace("/", "_"), a.out),), timeout=a.timeout)
             for k, c in enumerate(cs)] for g, u, cs in zip(df.code, df.uid, codes)]
    rows, t0 = [], time.time()
    for i, fs in enumerate(futs):
        res = []
        for fu in fs:
            try:
                res.append(fu.result())
            except Exception as e:  # noqa: BLE001
                res.append((-1.0, f"building it timed out ({type(e).__name__})", None))
        picks = [0]
        bad = [k for k in range(1, len(res)) if codes[i][k] and res[k][0] < a.bad]
        if bad:
            picks.append(rng.choice(bad))
        for k in picks:
            iou, err, stem = res[k]
            keep = iou >= a.keep
            rows.append({"rid": f"{df.uid[i]}_{k}".replace("/", "_"), "uid": df.uid[i], "family": df.family[i],
                         "mode": "view4" if len(views[i]) == 4 else "view1", "views": json.dumps(views[i]),
                         "prompt": "", "attempt": codes[i][k], "attempt_iou": iou,
                         "target": codes[i][k] if keep else df.code[i], "kind": "keep" if keep else "fix_gt",
                         "error": err, "attempt_stem": os.path.relpath(stem, a.out) if stem else None,
                         "input_stem": df.stem[i]})
        if i % 2000 == 0:
            print(f"[rich-rev] scored {i}/{len(futs)} parts {time.time() - t0:.0f}s", flush=True)
    pool.close()
    out = pd.DataFrame(rows)
    out.to_parquet(os.path.join(a.out, "revision.parquet"))
    g = out[out.rid.str.endswith("_0")]
    print(f"[rich-rev] wrote {len(out)} rows: kinds {out.kind.value_counts().to_dict()}, failed attempts "
          f"{int(out.error.notna().sum())}; greedy own IoU {100 * g.attempt_iou.clip(lower=0).mean():.1f}", flush=True)


if __name__ == "__main__":
    main()
