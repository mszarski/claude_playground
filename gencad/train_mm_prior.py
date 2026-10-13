"""Train the multi-condition (text / image / CCIP-image) diffusion prior on cached embeddings.

  python train_mm_prior.py --emb_dir data/emb --modalities image text --name mm_v1
  # GenCAD-equivalent baseline (CCIP image latent only):
  python train_mm_prior.py --emb_dir data/emb --modalities ccip_image --name baseline_dp

Each step: sample a CAD latent z0 plus whatever conditions that uid has (random render view, random prompt
level), randomly drop modalities (classifier-free guidance training), denoise. Periodically samples on a
fixed validation subset under every conditioning combination and logs cosine / retrieval metrics.
"""
import argparse
import copy
import itertools
import json
import os
import threading
import time

import torch
from torch.utils.data import DataLoader

from mm.data import cond_file, MultiModalLatentDataset, split_batch
from mm.diffusion import MultiCondDiffusionPrior
from mm.metrics import latent_metrics


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="data/emb")
    ap.add_argument("--split_json", default="data/filtered_data.json")
    ap.add_argument("--modalities", nargs="+", default=["image", "text"])
    ap.add_argument("--name", default="mm_prior")
    ap.add_argument("--out_dir", default="results")
    # model
    ap.add_argument("--d_cond", type=int, default=512)
    ap.add_argument("--fuser_layers", type=int, default=2)
    ap.add_argument("--d_main", type=int, default=1024)
    ap.add_argument("--d_hidden", type=int, default=2048)
    ap.add_argument("--n_blocks", type=int, default=10)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--cond_mode", default="adaln", choices=["adaln", "concat"])
    ap.add_argument("--objective", default="pred_x0", choices=["pred_x0", "pred_eps", "pred_v"])
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--p_drop", type=float, default=0.15, help="per-modality dropout prob")
    ap.add_argument("--p_drop_all", type=float, default=0.1)
    ap.add_argument("--contrastive_weight", type=float, default=0.1)
    # optim
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true", default=torch.cuda.is_available())
    # eval / io
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--eval_n", type=int, default=2048)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=1.0, help="guidance weight for every modality at eval")
    ap.add_argument("--save_every", type=int, default=10000)
    ap.add_argument("--hub_repo", default=None, help="push checkpoints + logs to this HF model repo (private)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max_items", type=int, default=None, help="debug: cap dataset size")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def cond_paths(a):
    return {m: cond_file(a.emb_dir, m) for m in a.modalities}


def push_async(folder, repo, step):
    def _go():
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo, private=True, exist_ok=True)
            api.upload_folder(folder_path=folder, repo_id=repo, commit_message=f"step {step}",
                              ignore_patterns=["*.tmp"])
            print(f"[hub] pushed step {step} to {repo}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[hub] push failed: {e}", flush=True)
    t = threading.Thread(target=_go, daemon=True)
    t.start()
    return t


@torch.no_grad()
def evaluate(model, val_ds, a, device):
    model.eval()
    n = min(a.eval_n, len(val_ds))
    batch = [val_ds[i] for i in range(n)]
    coll = torch.utils.data.default_collate([{k: v for k, v in b.items() if k != "uid"} for b in batch])
    z, embeds, present = split_batch(coll, device)
    results = {}
    mods = list(embeds)
    combos = [c for r in range(1, len(mods) + 1) for c in itertools.combinations(mods, r)]
    for combo in combos:
        rows = torch.stack([present[m] for m in combo], 0).all(0)  # only samples that have every modality
        if rows.sum() < 8:
            continue
        e = {m: embeds[m][rows] for m in combo}
        g = torch.Generator(device=device).manual_seed(0)
        zs = model.sample(e, guidance={m: a.guidance for m in combo}, steps=a.eval_steps, generator=g)
        results["+".join(combo)] = latent_metrics(zs, z[rows])
    # unconditional reference
    g = torch.Generator(device=device).manual_seed(0)
    zu = model.sample({}, steps=a.eval_steps, batch=min(n, 512), generator=g)
    results["uncond"] = latent_metrics(zu, z[: len(zu)])
    model.train()
    return results


def main():
    a = parse()
    device = torch.device(a.device)
    torch.manual_seed(0)
    run_dir = os.path.join(a.out_dir, a.name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    cad_path = os.path.join(a.emb_dir, "cad_latents.h5")
    train_ds = MultiModalLatentDataset(cad_path, cond_paths(a), "train", a.split_json, max_items=a.max_items)
    val_ds = MultiModalLatentDataset(cad_path, cond_paths(a), "validation", a.split_json,
                                     variants={m: 0 for m in a.modalities if m != "text"},
                                     max_items=a.eval_n)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} dims={train_ds.modality_dims}")
    # remember which frozen encoder produced each modality, so inference can rebuild it
    encoders = {m: {k: (v.decode() if isinstance(v, bytes) else v) for k, v in s.attrs.items()}
                for m, s in train_ds.stores.items()}

    model = MultiCondDiffusionPrior(
        train_ds.modality_dims, timesteps=a.timesteps, objective=a.objective,
        contrastive_weight=a.contrastive_weight,
        fuser_kwargs=dict(d_model=a.d_cond, n_layers=a.fuser_layers, p_drop={m: a.p_drop for m in a.modalities},
                          p_drop_all=a.p_drop_all),
        denoiser_kwargs=dict(d_main=a.d_main, d_hidden=a.d_hidden, n_blocks=a.n_blocks, dropout=a.dropout,
                             cond_mode=a.cond_mode),
    ).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))
    step = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        ema.load_state_dict(ck["ema_state_dict"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"[resume] from step {step}")

    dl = DataLoader(train_ds, batch_size=a.bs, shuffle=True, drop_last=True, num_workers=a.workers,
                    pin_memory=device.type == "cuda", persistent_workers=a.workers > 0)
    log_f = open(os.path.join(run_dir, "log.jsonl"), "a")
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=a.amp)
    t0, pusher = time.time(), None
    model.train()
    while step < a.steps:
        for batch in dl:
            z, embeds, present = split_batch(batch, device)
            with amp:
                loss, logs = model(z, embeds, present)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - a.ema)
            step += 1

            if step % 100 == 0:
                rec = {"step": step, "lr": sched.get_last_lr()[0], "grad_norm": gn.item(),
                       **{k: v.item() for k, v in logs.items()}, "sec": round(time.time() - t0, 1)}
                print(json.dumps(rec), flush=True)
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
            if step % a.eval_every == 0 or step == a.steps:
                res = evaluate(ema, val_ds, a, device)
                for k, v in res.items():
                    print(f"[eval {step}] {k:24s} " + " ".join(f"{m}={x:.4f}" for m, x in v.items()), flush=True)
                log_f.write(json.dumps({"step": step, "eval": res}) + "\n")
                log_f.flush()
            if step % a.save_every == 0 or step == a.steps:
                ck = {"config": model.config, "state_dict": model.state_dict(), "ema_state_dict": ema.state_dict(),
                      "opt": opt.state_dict(), "step": step, "args": vars(a), "encoders": encoders}
                torch.save(ck, os.path.join(run_dir, "last.pt.tmp"))
                os.replace(os.path.join(run_dir, "last.pt.tmp"), os.path.join(run_dir, "last.pt"))
                if a.hub_repo:
                    if pusher is not None:
                        pusher.join()
                    pusher = push_async(run_dir, a.hub_repo, step)
            if step >= a.steps:
                break
    if pusher is not None:
        pusher.join()
    print("[done]")


if __name__ == "__main__":
    main()
