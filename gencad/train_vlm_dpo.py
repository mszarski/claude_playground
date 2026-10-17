"""DPO for the CadQuery writer on preference pairs from rejection sampling (scripts/sample_rft.py chunks).

For each sampled DeepCAD train part: chosen = the best program by GT IoU (kept if >= --min_chosen), rejected = up to
--max_pairs programs at least --margin IoU worse or failing to build (the greedy program first when it qualifies, then
the worst samples). Both share the exact input (views, prompt) the programs were sampled from.
Loss = DPO (sigmoid, --beta) on sequence log-probs vs a frozen reference (the init model) + --nll_weight * mean token
NLL of the chosen program (keeps the chosen programs likely; plain DPO can push both down). Validation is
train_vlm_cq.py's (same 256 val parts, greedy decoding).

  python train_vlm_dpo.py --data data --init results/vlm_rft_v1/model --chunks results/rft_v1 --name vlm_dpo_v1
"""
import argparse
import glob
import json
import math
import os
import pickle
import random
import time

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mm.vlm_cad import VLMCad, compose_views, encode_batch, save_config  # noqa: E402
from train_vlm_cad import load_views  # noqa: E402
from train_vlm_cq import gt_vec, iou_scores, predict  # noqa: E402


def build_pairs(chunks, min_chosen, margin, max_pairs, max_chars):
    recs = []
    for fn in sorted(glob.glob(os.path.join(chunks, "chunk_*.pkl"))):
        with open(fn, "rb") as f:
            recs += pickle.load(f)
    pairs = []
    for r in recs:
        if r["scores"][0] is None:
            continue
        codes = [r["greedy"]] + list(r["samples"])
        v = [(s["gt"] if (c and s and s["valid"]) else -1.0) for c, s in zip(codes, r["scores"])]
        best = max(range(len(v)), key=lambda k: (v[k], -k))
        if v[best] < min_chosen or len(codes[best]) > max_chars:
            continue
        bad = sorted([k for k, x in enumerate(v) if x <= v[best] - margin and codes[k] and len(codes[k]) <= max_chars],
                     key=lambda k: (k != 0, v[k]))  # greedy first, then worst
        for k in bad[:max_pairs]:
            pairs.append({"uid": r["uid"], "views": r["views"], "prompt": r["prompt"], "chosen": codes[best],
                          "rejected": codes[k], "iou_c": v[best], "iou_r": v[k], "rej_is_greedy": k == 0})
    return pairs


def seq_logp(model, enc):
    """Sum and count of target-token log-probs per row (targets = labels != -100)."""
    labels = enc.pop("labels")
    logits = model(**enc).logits[:, :-1].float()
    tgt = labels[:, 1:]
    mask = tgt != -100
    lp = torch.log_softmax(logits, -1).gather(-1, tgt.clamp(min=0)[..., None]).squeeze(-1)
    return (lp * mask).sum(1), mask.sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--init", default="results/vlm_rft_v1/model")
    ap.add_argument("--chunks", default="results/rft_v1")
    ap.add_argument("--name", default="vlm_dpo")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--min_chosen", type=float, default=0.9)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--max_pairs", type=int, default=2, help="rejected programs per part")
    ap.add_argument("--max_chars", type=int, default=2500)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--nll_weight", type=float, default=0.2)
    ap.add_argument("--pairs_per_step", type=int, default=32)
    ap.add_argument("--micro", type=int, default=4, help="pairs per forward pass")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_n", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    import pandas as pd

    out = os.path.join(a.out_dir, a.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)
    render_dir = os.path.join(a.data, "renders")
    pairs = build_pairs(a.chunks, a.min_chosen, a.margin, a.max_pairs, a.max_chars)
    pd.DataFrame([{**p, "views": json.dumps(p["views"])} for p in pairs]).to_parquet(os.path.join(out, "pairs.parquet"))
    print(f"[data] {len(pairs)} pairs from {len({p['uid'] for p in pairs})} parts; rejected is greedy in "
          f"{sum(p['rej_is_greedy'] for p in pairs)}; mean IoU chosen {np.mean([p['iou_c'] for p in pairs]):.3f}, "
          f"rejected {np.mean([max(p['iou_r'], 0) for p in pairs]):.3f}, rejected invalid "
          f"{np.mean([p['iou_r'] < 0 for p in pairs]):.3f}", flush=True)

    # validation as in train_vlm_cq.py
    with open(os.path.join(a.data, "text_index.json")) as f:
        text_index = json.load(f)
    dc = pd.read_parquet(os.path.join(a.data, "deepcad_cq.parquet"))
    with open(os.path.join(render_dir, "status.json")) as f:
        rendered = {u for u, st in json.load(f).items() if st in ("ok", "skip")}
    val = dc[dc.ok & (dc.iou >= 0.95) & (dc.iou <= 1.001) & dc.uid.isin(rendered) & (dc.code.str.len() <= 2500)
             & (dc.split == "validation")]
    val_uids = [u for u in val.uid if text_index.get(u, {}).get("expert")][: a.eval_n]
    val_gt = [gt_vec(a.data, u) for u in val_uids]

    vlm = VLMCad.from_pretrained(a.init, dtype=torch.float32)
    vlm.model.to(a.device)
    ref = VLMCad.from_pretrained(a.init, dtype=torch.bfloat16).model.to(a.device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    params = [p for p in vlm.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95), fused=a.device.startswith("cuda"))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    amp = a.device.startswith("cuda")
    log = open(os.path.join(out, "log.jsonl"), "a")

    def evaluate(step):
        vlm.model.eval()
        vlm.model.gradient_checkpointing_disable()
        vlm.model.config.use_cache = True
        t0 = time.time()
        preds = predict(vlm, val_uids, render_dir, text_index, ["view1", "view4", "view1+expert"])
        rec = {"step": step, **{k: iou_scores(c, val_gt, workers=a.workers) for k, c in preds.items()},
               "eval_sec": round(time.time() - t0)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    def save(step):
        sd = {k: v.to(torch.bfloat16) for k, v in vlm.model.state_dict().items()}
        vlm.model.save_pretrained(os.path.join(out, "model"), state_dict=sd, safe_serialization=True)
        vlm.processor.save_pretrained(os.path.join(out, "model"))
        save_config(os.path.join(out, "model"))
        with open(os.path.join(out, "model", "step.json"), "w") as f:
            json.dump({"step": step, "args": vars(a)}, f)

    def batch(ps):
        images, prompts, targets = [], [], []
        for p in ps:
            im = compose_views(load_views("", os.path.join(render_dir, p["uid"]), p["views"]))
            for t in (p["chosen"], p["rejected"]):
                images.append(im)
                prompts.append(p["prompt"] or None)
                targets.append(t)
        return encode_batch(vlm.processor, images, prompts, targets)   # rows: c0, r0, c1, r1, ...

    order = []
    vlm.model.config.use_cache = False
    vlm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vlm.model.train()
    for step in range(1, a.steps + 1):
        t0 = time.time()
        if len(order) < a.pairs_per_step:
            order += random.sample(range(len(pairs)), len(pairs))
        idx, order = order[: a.pairs_per_step], order[a.pairs_per_step:]
        stats = {"loss": 0.0, "dpo": 0.0, "nll": 0.0, "acc": 0.0, "margin": 0.0}
        for s in range(0, len(idx), a.micro):
            ps = [pairs[i] for i in idx[s: s + a.micro]]
            enc = batch(ps).to(a.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                r_lp, _ = seq_logp(ref, dict(enc))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                p_lp, n_tok = seq_logp(vlm.model, dict(enc))
            z = a.beta * ((p_lp[0::2] - r_lp[0::2]) - (p_lp[1::2] - r_lp[1::2]))
            dpo = -F.logsigmoid(z).mean()
            nll = -(p_lp[0::2] / n_tok[0::2]).mean()
            loss = (dpo + a.nll_weight * nll) * len(ps) / len(idx)
            loss.backward()
            w = len(ps) / len(idx)
            stats["loss"] += loss.item()
            stats["dpo"] += dpo.item() * w
            stats["nll"] += nll.item() * w
            stats["acc"] += (z > 0).float().mean().item() * w
            stats["margin"] += (z / a.beta).mean().item() * w
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == 1:
            rec = {"step": step, **{k: round(v, 4) for k, v in stats.items()}, "gnorm": round(gnorm, 3),
                   "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % a.eval_every == 0 or step == a.steps:
            save(step)
            evaluate(step)
            vlm.model.config.use_cache = False
            vlm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            vlm.model.train()
    print(f"[done] {out}/model", flush=True)


if __name__ == "__main__":
    main()
