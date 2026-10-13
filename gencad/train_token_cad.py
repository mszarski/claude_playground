"""Train the image(+text) -> CAD program model (mm/token_cad.py) on renders, prompts and GT programs.

Per sample: a random render view, the GT program, and a random prompt level, which is dropped with probability
--p_no_text so the same model serves image-only and image+text requests. Every --eval_every steps it decodes
--eval_n validation parts from view 0 alone and from view 0 + the expert prompt, with the eval_mm.py metrics.

  python train_token_cad.py --emb_dir data/emb --data data --name token_cad_v1
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cadlib.macro import EOS_IDX
from mm.cond_encoders import load_rgb
from mm.metrics import seq_metrics
from mm.token_cad import TokenCAD
from train_ar_decoder import load_programs
from train_e2e import TF


def unpad(v):
    c = v[:, 0].tolist()
    return v[: c.index(EOS_IDX) + 1] if EOS_IDX in c else v


class Samples(Dataset):
    def __init__(self, uids, vec, img_dir, img_ids, text_index, levels, p_no_text, eval_level=None):
        self.uids, self.vec, self.img_dir, self.img_ids = uids, vec, img_dir, img_ids
        self.text_index, self.levels, self.p_no_text, self.eval_level = text_index, levels, p_no_text, eval_level

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        u = self.uids[i]
        views = self.img_ids[u]
        v = views[0] if self.eval_level else random.choice(views)
        pixels = TF(load_rgb(os.path.join(self.img_dir, f"{u}_{v}.png")))
        prompts = self.text_index.get(u, {})
        if self.eval_level:
            prompt = prompts.get(self.eval_level) or ""
        else:
            have = [lv for lv in self.levels if prompts.get(lv)]
            prompt = "" if not have or random.random() < self.p_no_text else prompts[random.choice(have)]
        return pixels, prompt, self.vec[i]


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] or None for b in batch], torch.stack([b[2] for b in batch])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="data/emb")
    ap.add_argument("--data", default="data")
    ap.add_argument("--name", default="token_cad")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--image_model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--text_model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--text_layers", type=int, default=8)
    ap.add_argument("--levels", nargs="*", default=["abstract", "beginner", "intermediate", "expert", "description"])
    ap.add_argument("--p_no_text", type=float, default=0.5)
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
    ap.add_argument("--workers", type=int, default=10)
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
    with open(os.path.join(a.data, "image_ids.json")) as f:
        img_ids = json.load(f)
    with open(os.path.join(a.data, "text_index.json")) as f:
        text_index = json.load(f)
    train = [u for u in splits["train"] if img_ids.get(u)][: a.max_items]
    val = [u for u in splits["validation"] if img_ids.get(u) and text_index.get(u, {}).get("expert")][: a.eval_n]
    vec = torch.from_numpy(load_programs(a.emb_dir, a.data, train + val).astype(np.int64))
    img_dir = os.path.join(a.data, "images")
    train_ds = Samples(train, vec[: len(train)], img_dir, img_ids, text_index, a.levels, a.p_no_text)
    gt_va = [unpad(v) for v in vec[len(train):].numpy()]
    print(f"[data] train={len(train)} val={len(val)}", flush=True)

    model = TokenCAD(a.image_model, a.text_model, a.text_layers, d_model=a.d_model, n_layers=a.n_layers,
                     n_heads=a.n_heads, d_ff=a.d_ff, dropout=a.dropout).to(device)
    names = [n for n, _ in model.named_parameters() if not n.startswith(TokenCAD.FROZEN)]
    params = [p for n, p in model.named_parameters() if n in names]
    print(f"[model] trainable {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    ema = {n: p.detach().clone() for n, p in model.named_parameters() if n in names}
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / a.steps))))
    amp = device.startswith("cuda")
    log = open(os.path.join(out, "log.jsonl"), "a")

    def swap_ema():
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
        ds = Samples(val, vec[len(train):], img_dir, img_ids, text_index, a.levels, 0.0, eval_level="expert")
        preds = {"image": [], "image+expert": []}
        for pixels, prompts, _ in DataLoader(ds, batch_size=128, num_workers=min(4, a.workers), collate_fn=collate):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                preds["image"] += model.generate(pixels=pixels)
                preds["image+expert"] += model.generate(pixels=pixels, prompts=prompts)
        for k, p in preds.items():
            rec[k] = {m: round(float(v), 4) for m, v in seq_metrics(p, gt_va).items()}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        swap_ema()
        model.train()

    def save(step):
        swap_ema()
        model.save(os.path.join(out, "last.pt"), step=step, args=vars(a))
        swap_ema()

    loader = DataLoader(train_ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                        collate_fn=collate, pin_memory=True, persistent_workers=a.workers > 0)
    model.train()
    step, t0 = 0, time.time()
    while step < a.steps:
        for pixels, prompts, v in loader:
            step += 1
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss, logs = model.loss(v.to(device), pixels=pixels.to(device, non_blocking=True), prompts=prompts)
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
                rec = {"step": step, "loss": round(float(loss), 5),
                       **{k: round(float(x), 5) for k, x in logs.items()},
                       "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
            if step % a.eval_every == 0 or step == a.steps:
                evaluate(step)
            if step % a.save_every == 0 or step == a.steps:
                save(step)
            if step >= a.steps:
                break
    print(f"[done] {out}/last.pt", flush=True)


if __name__ == "__main__":
    main()
