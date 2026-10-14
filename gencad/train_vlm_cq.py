"""Fine-tune Qwen3-VL to write CadQuery programs (CAD-Recode style) from renders and/or prompts, on DeepCAD (converted
by scripts/convert_deepcad.py) plus CAD-Recode's ~1M procedural programs (scripts/render_cad_recode.py renders).

Same model, inputs and recipe as train_vlm_cad.py (DeepCAD-token output); what changes is the target language and the
data. Each sample comes from DeepCAD with probability --deepcad_p (else CAD-Recode); DeepCAD samples draw an input mode
like train_vlm_cad.py (1 view / 4-view grid / + random-level prompt / prompt only), CAD-Recode samples have no prompts,
so they use the image modes only. Validation: --eval_n DeepCAD validation parts, generated code executed and scored by
volume IoU against the GT solid (scripts/cq_iou.py).

  python train_vlm_cq.py --data data --name vlm_cq_v1
  python train_vlm_cq.py --data data --predict results/vlm_cq_v1/model --uids_from results/baselines_vlm_final/preds.pkl \
      --out_preds results/vlm_cq_v1/test_preds.pkl          # then: jobs/launch.sh cqiou (PREDS=...)
"""
import argparse
import json
import math
import os
import pickle
import random
import time

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from mm.vlm_cad import VLMCad, compose_views
from train_vlm_cad import Collate, load_views

IMAGE_MODES = ("view1", "view4")


class Samples(Dataset):
    """items: (image stem, code, uid with prompts or None)."""

    def __init__(self, items, text_index, levels, mode_p):
        self.items, self.text_index, self.levels = items, text_index, levels
        self.modes, self.mode_p = list(mode_p), np.array(list(mode_p.values())) / sum(mode_p.values())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        stem, code, uid = self.items[i]
        prompts = self.text_index.get(uid, {}) if uid else {}
        have = [lv for lv in self.levels if prompts.get(lv)]
        mode = self.modes[np.random.choice(len(self.modes), p=self.mode_p)]
        if "text" in mode and not have:
            mode = mode.split("+")[0] if "+" in mode else random.choice(IMAGE_MODES)
        prompt = prompts[random.choice(have)] if "text" in mode else None
        image = None
        if mode.startswith("view"):
            views = [0, 1, 2, 3] if mode.startswith("view4") else [random.randrange(4)]
            image = compose_views([load_views("", stem, [v])[0] for v in views])
        return image, prompt, code


def predict(vlm, uids, render_dir, text_index, modes, bs=32, max_new_tokens=1024):
    """Greedy-decode CadQuery text for each uid under each mode -> {mode: [code or None]}."""
    out = {}
    for mode in modes:
        rows = []
        for s in range(0, len(uids), bs):
            chunk = uids[s: s + bs]
            images, prompts = [], []
            for u in chunk:
                p = text_index.get(u, {}).get("expert") if "expert" in mode else None
                stem = os.path.join(render_dir, u)
                if mode.startswith("view") and os.path.exists(f"{stem}_0.png"):
                    images.append(compose_views(load_views("", stem, range(int(mode[4])))))
                else:
                    images.append(None)
                    p = p if not mode.startswith("view") else None
                prompts.append(p)
            keep = [i for i, (im, p) in enumerate(zip(images, prompts)) if im is not None or p]
            row = [None] * len(chunk)
            if keep:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=vlm.device.type == "cuda"):
                    _, texts = vlm.generate([images[i] for i in keep], [prompts[i] for i in keep],
                                            max_new_tokens=max_new_tokens)
                for i, t in zip(keep, texts):
                    row[i] = t.strip() or None
            rows += row
        out[mode] = rows
    return out


def iou_scores(codes, gt_vecs, workers=16, timeout=60):
    """Mean IoU (gt / own normalisation, %) and invalid rate of CadQuery code against GT DeepCAD programs."""
    from pebble import ProcessPool
    from scripts.cq_iou import score
    rows = []
    with ProcessPool(max_workers=workers) as pool:
        futs = [pool.schedule(score, args=((g, c, "cadquery"),), timeout=timeout) for g, c in zip(gt_vecs, codes)]
        for fu in futs:
            try:
                rows.append(fu.result())
            except Exception:  # noqa: BLE001
                rows.append({"valid": False, "gt": 0.0, "own": 0.0})
    rows = [r for r in rows if r is not None]
    return {"iou_gt": round(100 * float(np.mean([r["gt"] for r in rows])), 2),
            "iou_own": round(100 * float(np.mean([r["own"] for r in rows])), 2),
            "invalid": round(float(np.mean([not r["valid"] for r in rows])), 4), "n": len(rows)}


def gt_vec(data, u):
    with h5py.File(os.path.join(data, "cad_vec", u + ".h5"), "r") as f:
        return f["vec"][:].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--name", default="vlm_cq")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--deepcad_cq", default="data/deepcad_cq.parquet")
    ap.add_argument("--recode", default="data/cad_recode.parquet")
    ap.add_argument("--recode_renders", default="data/recode_renders")
    ap.add_argument("--min_fidelity", type=float, default=0.95, help="drop DeepCAD conversions below this IoU")
    ap.add_argument("--deepcad_p", type=float, default=0.3, help="share of samples drawn from DeepCAD")
    ap.add_argument("--levels", nargs="*", default=["abstract", "beginner", "intermediate", "expert", "description"])
    ap.add_argument("--mode_p", default="view1=0.3,view4=0.25,view1+text=0.15,view4+text=0.15,text=0.15")
    ap.add_argument("--max_code_chars", type=int, default=2500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--eval_n", type=int, default=256)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--predict", default=None, help="model dir: only write test predictions")
    ap.add_argument("--uids_from", default=None)
    ap.add_argument("--out_preds", default=None)
    ap.add_argument("--predict_modes", nargs="*", default=["view1", "view4", "view1+expert", "expert"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    render_dir = os.path.join(a.data, "renders")
    with open(os.path.join(a.data, "text_index.json")) as f:
        text_index = json.load(f)

    if a.predict:                                     # test-set predictions for scripts/cq_iou.py
        with open(a.uids_from, "rb") as f:
            P = pickle.load(f)
        vlm = VLMCad.from_pretrained(a.predict, dtype=torch.bfloat16)
        vlm.model.to(a.device).eval()
        preds = predict(vlm, P["uids"], render_dir, text_index, a.predict_modes, bs=64)
        os.makedirs(os.path.dirname(a.out_preds) or ".", exist_ok=True)
        with open(a.out_preds, "wb") as f:
            pickle.dump({"uids": P["uids"], "dup": P["dup"], "kind": "cadquery",
                         "preds": {f"vlm_cq:{m}": v for m, v in preds.items()}}, f)
        print(f"[done] {a.out_preds}")
        return

    out = os.path.join(a.out_dir, a.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    # --- DeepCAD: converted programs with a faithful conversion and a render, deduplicated by code
    dc = pd.read_parquet(a.deepcad_cq)
    with open(os.path.join(render_dir, "status.json")) as f:
        rendered = {u for u, st in json.load(f).items() if st in ("ok", "skip")}
    dc = dc[dc.ok & (dc.iou >= a.min_fidelity) & (dc.iou <= 1.001) & dc.uid.isin(rendered) & (dc.code.str.len() <= a.max_code_chars)]
    tr_dc = dc[dc.split == "train"].drop_duplicates("code")
    val = dc[dc.split == "validation"]
    val = val[val.uid.map(lambda u: bool(text_index.get(u, {}).get("expert")))].head(a.eval_n)
    items_dc = [(os.path.join(render_dir, u), c, u) for u, c in zip(tr_dc.uid, tr_dc.code)][: a.max_items]
    # --- CAD-Recode: programs that rendered
    rc = pd.read_parquet(a.recode)
    status = {}
    for fn in os.listdir(a.recode_renders):
        if fn.startswith("status_") and fn.endswith(".json"):
            with open(os.path.join(a.recode_renders, fn)) as f:
                status.update(json.load(f))
    rc = rc[(rc.split == "train") & rc.path.map(lambda p: status.get(p) in ("ok", "skip"))
            & (rc.code.str.len() <= a.max_code_chars)]
    items_rc = [(os.path.join(a.recode_renders, p[:-3]), c, None) for p, c in zip(rc.path, rc.code)][: a.max_items]
    items = items_dc + items_rc
    w = np.concatenate([np.full(len(items_dc), a.deepcad_p / max(len(items_dc), 1)),
                        np.full(len(items_rc), (1 - a.deepcad_p) / max(len(items_rc), 1))])
    print(f"[data] deepcad train {len(items_dc)} (of {len(dc[dc.split == 'train'])} faithful), cad-recode "
          f"{len(items_rc)}, val {len(val)}; deepcad share {a.deepcad_p}", flush=True)
    mode_p = {k: float(v) for k, v in (kv.split("=") for kv in a.mode_p.split(","))}
    val_uids = list(val.uid)
    val_gt = [gt_vec(a.data, u) for u in val_uids]

    vlm = VLMCad.from_pretrained(a.model, dtype=torch.float32)
    vlm.model.to(a.device)
    vlm.model.config.use_cache = False
    vlm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for name, its in [("deepcad", items_dc), ("cad-recode", items_rc)]:
        n_tok = [len(vlm.tok(c)["input_ids"]) for _, c, _ in its[:2000]]
        if n_tok:
            print(f"[data] {name} code tokens: mean {np.mean(n_tok):.0f}, p95 {np.percentile(n_tok, 95):.0f}, "
                  f"max {max(n_tok)}", flush=True)
    params = [p for p in vlm.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95), fused=a.device.startswith("cuda"))
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
        rec = {"step": step}
        for k, codes in preds.items():
            rec[k] = iou_scores(codes, val_gt)
        rec["eval_sec"] = round(time.time() - t0)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        with open(os.path.join(out, f"val_samples_{step}.json"), "w") as f:
            json.dump({k: v[:20] for k, v in preds.items()}, f, indent=1)
        vlm.model.config.use_cache = False
        vlm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        vlm.model.train()

    def save(step):
        sd = {k: v.to(torch.bfloat16) for k, v in vlm.model.state_dict().items()}
        vlm.model.save_pretrained(os.path.join(out, "model"), state_dict=sd, safe_serialization=True)
        vlm.processor.save_pretrained(os.path.join(out, "model"))
        with open(os.path.join(out, "model", "step.json"), "w") as f:
            json.dump({"step": step, "args": vars(a)}, f)

    ds = Samples(items, text_index, a.levels, mode_p)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=a.steps * a.bs * a.accum,
                                    replacement=True)
    loader = DataLoader(ds, batch_size=a.bs, sampler=sampler, num_workers=a.workers, drop_last=True,
                        collate_fn=Collate(vlm.processor), pin_memory=True, persistent_workers=a.workers > 0,
                        worker_init_fn=lambda _: np.random.seed(torch.initial_seed() % 2**32))
    vlm.model.train()
    step, micro, t0, acc_loss = 0, 0, time.time(), 0.0
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
        if step % a.eval_every == 0 or step == a.steps:
            save(step)
            evaluate(step)
        if step >= a.steps:
            break
    print(f"[done] {out}/model", flush=True)


if __name__ == "__main__":
    main()
