"""Train the text -> CAD program model (mm/text_cad.py): Text2CAD prompts in, GenCAD/DeepCAD programs out.

Each step samples training parts, picks one of their prompt levels at random (--levels), and trains the decoder to
write the GT program while cross-attending to the prompt's tokens. Programs come from the cache that
train_ar_decoder.py writes (<emb_dir>/programs.npz, train + val) or from data/cad_vec. Every --eval_every steps
it greedily decodes --eval_n validation parts at each of --eval_levels and reports the eval_mm.py metrics.

  python train_text_cad.py --emb_dir data/emb --data data --name text_cad_v1
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from cadlib.macro import EOS_IDX
from mm.metrics import seq_metrics
from mm.text_cad import TextCAD
from train_ar_decoder import load_programs


def unpad(v):
    c = v[:, 0].tolist()
    return v[: c.index(EOS_IDX) + 1] if EOS_IDX in c else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="data/emb")
    ap.add_argument("--data", default="data")
    ap.add_argument("--name", default="text_cad")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--text_model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--text_layers", type=int, default=8)
    ap.add_argument("--max_text_len", type=int, default=512)
    ap.add_argument("--levels", nargs="*", default=["abstract", "beginner", "intermediate", "expert", "description"])
    ap.add_argument("--eval_levels", nargs="*", default=["expert", "beginner"])
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=1536)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--eval_n", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    device = a.device
    out = os.path.join(a.out_dir, a.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    with open(os.path.join(a.data, "filtered_data.json")) as f:
        splits = json.load(f)
    with open(os.path.join(a.data, "text_index.json")) as f:
        text_index = json.load(f)
    has = lambda u, lvls: u in text_index and any(text_index[u].get(lv) for lv in lvls)  # noqa: E731
    train = [u for u in splits["train"] if has(u, a.levels)][: a.max_items]
    val = [u for u in splits["validation"] if has(u, a.eval_levels)][: a.eval_n]
    vec = torch.from_numpy(load_programs(a.emb_dir, a.data, train + val).astype(np.int64))
    v_tr, v_va = vec[: len(train)], vec[len(train):]
    gt_va = [unpad(v) for v in v_va.numpy()]
    print(f"[data] train={len(train)} val={len(val)} levels={a.levels}", flush=True)

    model = TextCAD(a.text_model, a.text_layers, a.max_text_len, d_model=a.d_model, n_layers=a.n_layers,
                    n_heads=a.n_heads, d_ff=a.d_ff, dropout=a.dropout).to(device)
    params = [p for n, p in model.named_parameters() if not n.startswith("text.")]
    print(f"[model] trainable {sum(p.numel() for p in params) / 1e6:.1f}M, text model "
          f"{sum(p.numel() for p in model.text.parameters()) / 1e6:.0f}M frozen ({a.text_layers} layers)", flush=True)
    ema = {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("text.")}
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / a.steps))))
    amp = device.startswith("cuda")
    log = open(os.path.join(out, "log.jsonl"), "a")

    def prompt(u, levels):
        return text_index[u][random.choice([lv for lv in levels if text_index[u].get(lv)])]

    def swap_ema():
        """Swap live and EMA weights in place (call twice to restore)."""
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in ema:
                    tmp = p.detach().clone()
                    p.copy_(ema[n])
                    ema[n].copy_(tmp)

    def evaluate(step):
        model.eval()
        swap_ema()
        rec = {"step": step}
        for lv in a.eval_levels:
            idx = [i for i, u in enumerate(val) if text_index[val[i]].get(lv)]
            preds = []
            for s in range(0, len(idx), 128):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    preds += model.generate([text_index[val[i]][lv] for i in idx[s: s + 128]])
            m = seq_metrics(preds, [gt_va[i] for i in idx])
            rec[lv] = {k: round(float(v), 4) for k, v in m.items()} | {"n": len(idx)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        swap_ema()
        model.train()

    def save(step):
        swap_ema()
        model.save(os.path.join(out, "last.pt"), step=step, args=vars(a))
        swap_ema()

    model.train()
    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = random.sample(range(len(train)), a.bs)
        prompts = [prompt(train[i], a.levels) for i in idx]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss, logs = model.loss(prompts, v_tr[idx].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in ema:
                    ema[n].lerp_(p.detach(), 1 - a.ema)
        if step % 100 == 0:
            rec = {"step": step, "loss": round(float(loss), 5), **{k: round(float(v), 5) for k, v in logs.items()},
                   "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
        if step % a.eval_every == 0 or step == a.steps:
            evaluate(step)
        if step % a.save_every == 0 or step == a.steps:
            save(step)
    print(f"[done] {out}/last.pt", flush=True)


if __name__ == "__main__":
    main()
