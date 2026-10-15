"""RL on IoU for the CadQuery writer (vlm_cq), after cadrille (arXiv 2505.22914, +6 IoU from RL).

On-policy group policy gradient (GRPO without std normalisation, "Dr. GRPO"): each step draws --prompts DeepCAD train
parts, samples --group programs per part at temperature --temp, executes them and scores each with the test metric
(scripts/cq_iou.score: volume IoU vs the GT solid, GT normalisation; code that fails = --invalid_reward). Advantage =
reward - group mean; groups with no reward spread are dropped (no signal). One gradient step per batch, so no
importance ratio or clipping is needed. Loss = -sum_t A * log p(token_t) / (n_seq * --len_norm).

Only DeepCAD parts are used as RL prompts (their GT solid is the metric's reference); CAD-Recode stays SFT-only.
Validation is train_vlm_cq.py's (same 256 val parts, greedy decoding), so numbers continue its log.

  python train_vlm_rl.py --data data --init results/vlm_cq_v1/model --name vlm_rl_v1
  python train_vlm_cq.py --data data --predict results/vlm_rl_v1/model --uids_from results/baselines_vlm_final/preds.pkl \
      --out_preds results/vlm_rl_v1_test/preds.pkl
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from mm.vlm_cad import VLMCad, compose_views, encode_batch
from train_vlm_cad import load_views
from train_vlm_cq import gt_vec, iou_scores, predict


def rewards(codes, gts, pool, invalid_reward, timeout):
    from scripts.cq_iou import score
    futs = [pool.schedule(score, args=((g, c, "cadquery"),), timeout=timeout) for g, c in zip(gts, codes)]
    out = []
    for fu in futs:
        try:
            r = fu.result()
        except Exception:  # noqa: BLE001 - timeout / crash -> invalid
            r = {"valid": False, "gt": 0.0}
        out.append(None if r is None else (r["gt"] if r["valid"] else invalid_reward))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--init", default="results/vlm_cq_v1/model")
    ap.add_argument("--name", default="vlm_rl")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--deepcad_cq", default="data/deepcad_cq.parquet")
    ap.add_argument("--mode_p", default="view1=0.4,view4=0.4,view1+expert=0.2")
    ap.add_argument("--prompts", type=int, default=16, help="parts per step")
    ap.add_argument("--group", type=int, default=8, help="samples per part")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--invalid_reward", type=float, default=-1.0)
    ap.add_argument("--len_norm", type=float, default=256.0, help="constant token normaliser (Dr. GRPO)")
    ap.add_argument("--micro", type=int, default=16, help="sequences per backward pass")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_n", type=int, default=256)
    ap.add_argument("--eval_first", type=int, default=1, help="evaluate the initial model at step 0")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--score_timeout", type=float, default=30.0)
    ap.add_argument("--optim", default="adamw", choices=["adamw", "sgd"], help="sgd: CPU smoke tests only")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    from pebble import ProcessPool

    out = os.path.join(a.out_dir, a.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)
    render_dir = os.path.join(a.data, "renders")
    with open(os.path.join(a.data, "text_index.json")) as f:
        text_index = json.load(f)

    # prompts: DeepCAD train parts with a render and a GT program, one per distinct program; val as train_vlm_cq.py
    dc = pd.read_parquet(a.deepcad_cq)
    with open(os.path.join(render_dir, "status.json")) as f:
        rendered = {u for u, st in json.load(f).items() if st in ("ok", "skip")}
    dc = dc[dc.ok & dc.uid.isin(rendered)]
    train_uids = [u for u in dc[dc.split == "train"].drop_duplicates("code").uid
                  if os.path.exists(os.path.join(a.data, "cad_vec", u + ".h5"))
                  and os.path.exists(os.path.join(render_dir, f"{u}_3.png"))]
    val = dc[(dc.split == "validation") & (dc.iou >= 0.95) & (dc.iou <= 1.001) & (dc.code.str.len() <= 2500)]
    val_uids = [u for u in val.uid if text_index.get(u, {}).get("expert")][: a.eval_n]
    val_gt = [gt_vec(a.data, u) for u in val_uids]
    modes = {k: float(v) for k, v in (kv.split("=") for kv in a.mode_p.split(","))}
    mode_names, mode_p = list(modes), np.array(list(modes.values())) / sum(modes.values())
    print(f"[data] {len(train_uids)} train parts, {len(val_uids)} val; modes {modes}", flush=True)

    pool = ProcessPool(max_workers=a.workers)          # before the model: workers fork without CUDA state
    vlm = VLMCad.from_pretrained(a.init, dtype=getattr(torch, a.dtype))
    vlm.model.to(a.device)
    tok = vlm.tok
    eos_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    params = [p for p in vlm.model.parameters() if p.requires_grad]
    if a.optim == "adamw":
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95),
                                fused=a.device.startswith("cuda"))
    else:
        opt = torch.optim.SGD(params, lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))
    amp = a.device.startswith("cuda")
    log = open(os.path.join(out, "log.jsonl"), "a")

    def evaluate(step):
        vlm.model.eval()
        t0 = time.time()
        preds = predict(vlm, val_uids, render_dir, text_index, ["view1", "view4", "view1+expert"])
        rec = {"step": step, **{k: iou_scores(c, val_gt, workers=a.workers) for k, c in preds.items()},
               "eval_sec": round(time.time() - t0)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        with open(os.path.join(out, f"val_samples_{step}.json"), "w") as f:
            json.dump({k: v[:20] for k, v in preds.items()}, f, indent=1)

    def save(step):
        sd = {k: v.to(torch.bfloat16) for k, v in vlm.model.state_dict().items()}
        vlm.model.save_pretrained(os.path.join(out, "model"), state_dict=sd, safe_serialization=True)
        vlm.processor.save_pretrained(os.path.join(out, "model"))
        with open(os.path.join(out, "model", "step.json"), "w") as f:
            json.dump({"step": step, "args": vars(a)}, f)

    def inputs(u, mode):
        p = text_index.get(u, {}).get("expert") if "expert" in mode else None
        if "expert" in mode and not p:
            mode = "view1"
        n = 4 if mode.startswith("view4") else 1
        views = [0, 1, 2, 3] if n == 4 else [random.randrange(4)]
        return compose_views(load_views("", os.path.join(render_dir, u), views)), p, mode

    if a.eval_first:
        evaluate(0)
    for step in range(1, a.steps + 1):
        t0 = time.time()
        uids = random.sample(train_uids, a.prompts)
        items = [inputs(u, mode_names[np.random.choice(len(mode_names), p=mode_p)]) for u in uids]
        images = [im for im, _, _ in items for _ in range(a.group)]
        prompts = [p for _, p, _ in items for _ in range(a.group)]
        gts = [g for g in (gt_vec(a.data, u) for u in uids) for _ in range(a.group)]

        # --- sample (no grad, cache on)
        vlm.model.eval()
        gen = []
        for s in range(0, len(images), 64):
            enc = encode_batch(vlm.processor, images[s: s + 64], prompts[s: s + 64]).to(a.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                o = vlm.model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=True, temperature=a.temp,
                                       top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id)
            gen += [row for row in o[:, enc["input_ids"].shape[1]:].cpu()]
        t_gen = time.time() - t0
        seqs = []                                         # generated ids up to and including the first eos
        for g in gen:
            ids = g.tolist()
            end = next((k + 1 for k, t in enumerate(ids) if t in eos_ids), len(ids))
            seqs.append(ids[:end])
        codes = [tok.decode(s, skip_special_tokens=True).strip() or None for s in seqs]

        # --- reward and group advantage
        r = rewards(codes, gts, pool, a.invalid_reward, a.score_timeout)
        t_score = time.time() - t0 - t_gen
        R = np.array([np.nan if x is None else x for x in r], dtype=float).reshape(a.prompts, a.group)
        adv = R - np.nanmean(R, axis=1, keepdims=True)
        keep = [i for i in range(len(seqs)) if np.isfinite(adv.flat[i]) and np.nanstd(R[i // a.group]) > 1e-6]

        # --- policy gradient on the kept sequences
        vlm.model.train()
        vlm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        vlm.model.config.use_cache = False
        n_seq = max(len(keep), 1)
        loss_sum = 0.0
        for s in range(0, len(keep), a.micro):
            idx = keep[s: s + a.micro]
            enc = encode_batch(vlm.processor, [images[i] for i in idx], [prompts[i] for i in idx])
            P = enc["input_ids"].shape[1]
            L = max(len(seqs[i]) for i in idx)
            gen_ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
            gen_mask = torch.zeros((len(idx), L), dtype=torch.long)
            for j, i in enumerate(idx):
                gen_ids[j, : len(seqs[i])] = torch.tensor(seqs[i])
                gen_mask[j, : len(seqs[i])] = 1
            for k, v in list(enc.items()):                # per-token tensors (e.g. mm_token_type_ids): text = 0
                if torch.is_tensor(v) and v.dim() == 2 and v.shape == (len(idx), P) and k != "input_ids":
                    enc[k] = torch.cat([v, gen_mask if k == "attention_mask" else torch.zeros_like(gen_ids)], 1)
            enc["input_ids"] = torch.cat([enc["input_ids"], gen_ids], 1)
            enc = enc.to(a.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = vlm.model(**enc).logits[:, P - 1: -1].float()
            lp = torch.log_softmax(logits, -1).gather(-1, enc["input_ids"][:, P:, None]).squeeze(-1)
            A = torch.tensor([adv.flat[i] for i in idx], dtype=torch.float32, device=lp.device)
            m = gen_mask.to(lp.device).float()
            loss = -((lp * m).sum(1) * A).sum() / (n_seq * a.len_norm)
            loss.backward()
            loss_sum += loss.item()
            del logits, lp
        vlm.model.gradient_checkpointing_disable()
        vlm.model.config.use_cache = True
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0)) if keep else 0.0
        if keep:
            opt.step()
            sched.step()
        opt.zero_grad(set_to_none=True)

        valid = [x for x in r if x is not None]
        rec = {"step": step, "reward": round(float(np.mean(valid)), 4),
               "iou_valid": round(float(np.mean([x for x in valid if x >= 0] or [0])), 4),
               "invalid": round(float(np.mean([x < 0 for x in valid])), 4), "kept": len(keep),
               "len": round(float(np.mean([len(s) for s in seqs])), 1), "loss": round(loss_sum, 5),
               "gnorm": round(gnorm, 4), "lr": sched.get_last_lr()[0], "t_gen": round(t_gen, 1),
               "t_score": round(t_score, 1), "sec": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % a.eval_every == 0 or step == a.steps:
            save(step)
            evaluate(step)
    pool.close()
    print(f"[done] {out}/model", flush=True)


if __name__ == "__main__":
    main()
