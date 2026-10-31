"""Zero-shot baseline with an open-weight model served by vLLM on the job's GPUs (e.g. Mistral Large 4, 1T MoE in
FP8 on 8x H200): same renders, same full-feature prompt as `scripts/claude_baseline.py submit --rich`.

  python scripts/mistral_real.py --model mistralai/Mistral-Large-4-1T-A52B --rich data/rich/real --modes view4 \
      --tp 8 --out results/mistral_real/preds.pkl
Writes preds.pkl in the rich layout (uids, family, gt_code, stems, preds["mistral:<mode>"]), scored by rich_iou.py.
Saves after every chunk, so a timeout keeps what is done.
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rich", required=True)
    ap.add_argument("--modes", nargs="*", default=["view4"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=24000)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--max_seqs", type=int, default=32)
    ap.add_argument("--gpu_mem", type=float, default=0.95)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    from scripts.claude_baseline import TASK_RICH, VIEW1, VIEWS, extract_code, grid_png
    from vllm import LLM, SamplingParams
    df = pd.concat([pd.read_parquet(os.path.join(a.rich, sh, "parts.parquet")).assign(shard=sh)
                    for sh in sorted(os.listdir(a.rich)) if os.path.exists(os.path.join(a.rich, sh, "parts.parquet"))],
                   ignore_index=True).head(a.n)
    stems = [os.path.join(a.rich, sh, "renders", u) for sh, u in zip(df.shard, df.uid)]
    P = {"uids": list(df.uid), "dup": [False] * len(df), "family": list(df.family), "gt_code": list(df.code),
         "stems": stems, "kind": "cadquery", "preds": {f"mistral:{m}": [None] * len(df) for m in a.modes},
         "raw": {f"mistral:{m}": [None] * len(df) for m in a.modes}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.time()
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_len, max_num_seqs=a.max_seqs,
              gpu_memory_utilization=a.gpu_mem, limit_mm_per_prompt={"image": 1})
    print(f"[mistral] loaded {a.model} in {time.time() - t0:.0f}s", flush=True)
    sp = SamplingParams(temperature=a.temp, top_p=0.95, max_tokens=a.max_tokens)
    for m in a.modes:
        head = (VIEWS if m.startswith("view4") else VIEW1) + "\n\n"
        for s in range(0, len(df), a.chunk):
            idx = list(range(s, min(s + a.chunk, len(df))))
            msgs = [[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + grid_png(stems[i], m)}},
                {"type": "text", "text": head + TASK_RICH}]}] for i in idx]
            t1 = time.time()
            outs = llm.chat(msgs, sp, use_tqdm=False)
            n_tok = 0
            for i, o in zip(idx, outs):
                text = o.outputs[0].text
                n_tok += len(o.outputs[0].token_ids)
                P["raw"][f"mistral:{m}"][i] = text
                P["preds"][f"mistral:{m}"][i] = extract_code(text)
            pickle.dump(P, open(a.out, "wb"))
            print(f"[mistral] {m} {idx[-1] + 1}/{len(df)} in {time.time() - t1:.0f}s, {n_tok / len(idx):.0f} tok/answer",
                  flush=True)
    print(f"[done] {a.out}", flush=True)


if __name__ == "__main__":
    main()
