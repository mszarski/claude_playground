"""Fine-tune a pretrained VLM (default Qwen3-VL-2B-Instruct) to write CAD programs from renders and/or prompts
(mm/vlm_cad.py). Full fine-tuning, fp32 master weights + bf16 autocast, gradient checkpointing.

Each sample draws an input mode: one random view, a 2x2 grid of up to four random views, either of those plus a
random-level prompt, or the prompt alone. Exact duplicate programs in the training split are kept once by default
(DeepCAD repeats many parts; repeated programs teach memorisation, not reading the image). Every --eval_every
optimizer steps it greedily decodes --eval_n validation parts (1 view, 4 views, 1 view + expert prompt) and logs the
eval_mm.py metrics.

  python train_vlm_cad.py --emb_dir data/emb --data data --name vlm_cad_v1
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from mm.metrics import seq_metrics
from mm.vlm_cad import SIDE, VLMCad, compose_views, encode_batch, program_to_text
from train_ar_decoder import load_programs
from train_token_cad import unpad

MODES = ("view1", "view4", "view1+text", "view4+text", "text")


def load_views(img_dir, u, views):
    return [Image.open(os.path.join(img_dir, f"{u}_{v}.png")).convert("RGB") for v in views]


class Samples(Dataset):
    def __init__(self, uids, programs, img_dir, img_ids, text_index, levels, mode_p, fixed_cameras=False):
        self.uids, self.programs, self.img_dir, self.img_ids = uids, programs, img_dir, img_ids
        self.fixed_cameras = fixed_cameras
        self.text_index, self.levels = text_index, levels
        self.modes, self.mode_p = list(mode_p), np.array(list(mode_p.values())) / sum(mode_p.values())

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        u = self.uids[i]
        prompts = self.text_index.get(u, {})
        have = [lv for lv in self.levels if prompts.get(lv)]
        mode = self.modes[np.random.choice(len(self.modes), p=self.mode_p)]
        if "text" in mode and not have:
            mode = mode.split("+")[0] if "+" in mode else "view1"
        prompt = prompts[random.choice(have)] if "text" in mode else None
        image = None
        if mode.startswith("view"):
            views = list(self.img_ids[u])
            if mode.startswith("view4") and self.fixed_cameras:
                views = views[:4]                      # grid cell k is always camera k
            else:
                random.shuffle(views)
            image = compose_views(load_views(self.img_dir, u, views[: 1 if mode.startswith("view1") else 4]))
        return image, prompt, self.programs[i]


class Collate:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        return encode_batch(self.processor, [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch])


def evaluate_modes(vlm, uids, img_dir, img_ids, text_index, modes, bs=32, max_new_tokens=1024):
    """Greedy decode every uid under each mode -> {mode: [program or None]}."""
    out = {}
    for mode in modes:
        preds = []
        for s in range(0, len(uids), bs):
            chunk = uids[s: s + bs]
            images, prompts = [], []
            for u in chunk:
                p = text_index.get(u, {}).get("expert") if "expert" in mode else None
                if mode.startswith("view"):
                    n = int(mode[4])
                    images.append(compose_views(load_views(img_dir, u, img_ids[u][:n])))
                else:
                    images.append(None)
                prompts.append(p)
            keep = [i for i, (im, p) in enumerate(zip(images, prompts)) if im is not None or p]
            row = [None] * len(chunk)
            if keep:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=vlm.device.type == "cuda"):
                    got, _ = vlm.generate([images[i] for i in keep], [prompts[i] for i in keep],
                                          max_new_tokens=max_new_tokens)
                for i, v in zip(keep, got):
                    row[i] = v
            preds += row
        out[mode] = preds
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="data/emb")
    ap.add_argument("--data", default="data")
    ap.add_argument("--name", default="vlm_cad")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--levels", nargs="*", default=["abstract", "beginner", "intermediate", "expert", "description"])
    ap.add_argument("--mode_p", default="view1=0.3,view4=0.25,view1+text=0.15,view4+text=0.15,text=0.15")
    ap.add_argument("--images", default="renders", choices=["renders", "images"],
                    help="renders: our fixed-camera renders of the GT program (scripts/render_views.py); "
                         "images: GenCAD's renders, which do not all show the GT program (scripts/check_views.py)")
    ap.add_argument("--no_dedup", action="store_true", help="keep exact duplicate training programs")
    ap.add_argument("--bs", type=int, default=8, help="micro batch")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=15000, help="optimizer steps")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=2500)
    ap.add_argument("--eval_n", type=int, default=256)
    ap.add_argument("--save_every", type=int, default=2500)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    out = os.path.join(a.out_dir, a.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    with open(os.path.join(a.data, "filtered_data.json")) as f:
        splits = json.load(f)
    if a.images == "renders":
        with open(os.path.join(a.data, "renders", "status.json")) as f:
            img_ids = {u: [0, 1, 2, 3] for u, st in json.load(f).items() if st in ("ok", "skip")}
    else:
        with open(os.path.join(a.data, "image_ids.json")) as f:
            img_ids = json.load(f)
    with open(os.path.join(a.data, "text_index.json")) as f:
        text_index = json.load(f)
    train = [u for u in splits["train"] if img_ids.get(u)][: a.max_items]
    val = [u for u in splits["validation"] if img_ids.get(u) and text_index.get(u, {}).get("expert")][: a.eval_n]
    vec = load_programs(a.emb_dir, a.data, train + val).astype(np.int64)
    texts = [program_to_text(v) for v in vec[: len(train)]]
    if not a.no_dedup:
        seen, keep = set(), []
        for i, t in enumerate(texts):
            if t not in seen:
                seen.add(t)
                keep.append(i)
        print(f"[data] dedup: {len(train)} -> {len(keep)} training programs", flush=True)
        train, texts = [train[i] for i in keep], [texts[i] for i in keep]
    gt_va = [unpad(v) for v in vec[-len(val):]]
    img_dir = os.path.join(a.data, a.images)
    mode_p = {k: float(v) for k, v in (kv.split("=") for kv in a.mode_p.split(","))}
    assert set(mode_p) <= set(MODES), mode_p
    print(f"[data] train={len(train)} val={len(val)} modes={mode_p}", flush=True)

    vlm = VLMCad.from_pretrained(a.model, dtype=torch.float32)
    vlm.model.to(a.device)
    vlm.model.config.use_cache = False
    vlm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    n_tok = [len(vlm.tok(t)["input_ids"]) for t in texts[:2000]]
    print(f"[data] program tokens: mean {np.mean(n_tok):.0f}, p95 {np.percentile(n_tok, 95):.0f}, "
          f"max {max(n_tok)}", flush=True)
    params = [p for p in vlm.model.parameters() if p.requires_grad]
    print(f"[model] {a.model}: trainable {sum(p.numel() for p in params) / 1e6:.0f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95),
                            fused=a.device.startswith("cuda"))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    amp = a.device.startswith("cuda")
    log = open(os.path.join(out, "log.jsonl"), "a")

    def evaluate(step):
        vlm.model.eval()
        vlm.model.gradient_checkpointing_disable()     # otherwise generate() silently runs without a KV cache
        vlm.model.config.use_cache = True
        t0 = time.time()
        preds = evaluate_modes(vlm, val, img_dir, img_ids, text_index, ["view1", "view4", "view1+expert"])
        rec = {"step": step, "eval_sec": round(time.time() - t0)}
        for k, p in preds.items():
            rec[k] = {m: round(float(v), 4) for m, v in seq_metrics(p, gt_va).items()}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        vlm.model.config.use_cache = False
        vlm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        vlm.model.train()

    def save(step):
        sd = {k: v.to(torch.bfloat16) for k, v in vlm.model.state_dict().items()}
        vlm.model.save_pretrained(os.path.join(out, "model"), state_dict=sd, safe_serialization=True)
        vlm.processor.save_pretrained(os.path.join(out, "model"))
        with open(os.path.join(out, "model", "step.json"), "w") as f:
            json.dump({"step": step, "args": vars(a)}, f)

    ds = Samples(train, texts, img_dir, img_ids, text_index, a.levels, mode_p, fixed_cameras=a.images == "renders")
    loader = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                        collate_fn=Collate(vlm.processor), pin_memory=True, persistent_workers=a.workers > 0,
                        worker_init_fn=lambda w: np.random.seed(torch.initial_seed() % 2**32))
    vlm.model.train()
    step, micro, t0, acc_loss = 0, 0, time.time(), 0.0
    while step < a.steps:
        for enc in loader:
            enc = enc.to(a.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = vlm.model(**enc).loss / a.accum
            loss.backward()
            acc_loss += loss.item()
            micro += 1
            if micro % a.accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                rec = {"step": step, "loss": round(acc_loss / 50, 5), "lr": sched.get_last_lr()[0],
                       "sec": round(time.time() - t0, 1)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                acc_loss = 0.0
            if step % a.save_every == 0 or step == a.steps:
                save(step)
            if step % a.eval_every == 0 or step == a.steps:
                evaluate(step)
            if step >= a.steps:
                break
    print(f"[done] {out}/model", flush=True)


if __name__ == "__main__":
    main()
