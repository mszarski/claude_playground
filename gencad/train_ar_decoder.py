"""Train the autoregressive CAD decoder (mm/ar_decoder.py) on GenCAD's frozen CSR latents.

Pairs every train uid's CSR latent (data/emb/cad_latents.h5, from scripts/export_gencad_embeddings.py) with its
GT program (data/cad_vec/<uid>.h5). Programs are read once and cached next to the latents as programs.npz.
Every --eval_every steps it reconstructs --eval_n validation programs from their GT latents and reports the
same accuracies as eval_mm.py; with --csr_ckpt it first prints GenCAD's own CSR decoder on the same samples.

  python train_ar_decoder.py --emb_dir data/emb --data data --csr_ckpt data/ckpt/ae_ckpt_epoch1000.pth --name ar_v1
"""
import argparse
import copy
import json
import os
import time

import h5py
import numpy as np
import torch
from tqdm import tqdm

from cadlib.macro import EOS_IDX, MAX_TOTAL_LEN
from mm.ar_decoder import ARDecoder, pad_program
from mm.data import CadLatentStore
from mm.gencad_io import decode_latents, load_csr
from mm.metrics import seq_metrics


def load_programs(emb_dir, data, uids):
    """(N, MAX_TOTAL_LEN, 17) int16 programs for uids, cached in emb_dir/programs.npz."""
    cache = os.path.join(emb_dir, "programs.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        index = {u: i for i, u in enumerate(z["uids"].astype(str))}
        if all(u in index for u in uids):
            return z["vec"][[index[u] for u in uids]]
    vec = np.zeros((len(uids), MAX_TOTAL_LEN, 17), np.int16)
    for i, u in enumerate(tqdm(uids, desc="programs")):
        with h5py.File(os.path.join(data, "cad_vec", u + ".h5"), "r") as f:
            vec[i] = pad_program(f["vec"][:])
    try:
        np.savez(cache, uids=np.array(uids), vec=vec)
    except OSError as e:
        print(f"[data] could not cache programs: {e}")
    return vec


def unpad(v):
    """Padded program -> GT as stored (rows up to and including the first EOS), the form eval_mm.py scores."""
    cmds = v[:, 0].tolist()
    return v[: cmds.index(EOS_IDX) + 1] if EOS_IDX in cmds else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="data/emb")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split_json", default=None, help="default: <data>/filtered_data.json")
    ap.add_argument("--csr_ckpt", default=None, help="print GenCAD's CSR reconstruction on the eval set as a reference")
    ap.add_argument("--name", default="ar_decoder")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=1536)
    ap.add_argument("--n_mem", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--z_noise", type=float, default=0.0, help="std of Gaussian noise added to training latents")
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--eval_n", type=int, default=2000)
    ap.add_argument("--save_every", type=int, default=10000)
    ap.add_argument("--max_items", type=int, default=None, help="debug: cap the training set")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    device = a.device
    out = os.path.join(a.out_dir, a.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    cad = CadLatentStore(os.path.join(a.emb_dir, "cad_latents.h5"))
    with open(a.split_json or os.path.join(a.data, "filtered_data.json")) as f:
        splits = json.load(f)
    train = [u for u in splits["train"] if u in cad.index][: a.max_items]
    val = [u for u in splits["validation"] if u in cad.index][: a.eval_n]
    vec = torch.from_numpy(load_programs(a.emb_dir, a.data, train + val).astype(np.int64))
    z_all = torch.stack([cad.z[cad.index[u]] for u in train + val])
    z_tr, v_tr = z_all[: len(train)].to(device), vec[: len(train)].to(device)
    z_va, v_va = z_all[len(train):].to(device), vec[len(train):]
    gt_va = [unpad(v) for v in v_va.numpy()]
    print(f"[data] train={len(train)} val={len(val)}")

    if a.csr_ckpt:
        csr, _ = load_csr(a.csr_ckpt, device)
        recon = []
        for s in range(0, len(val), 256):
            recon += decode_latents(csr, z_va[s: s + 256])
        print(json.dumps({"step": 0, "csr_recon": seq_metrics(recon, gt_va)}), flush=True)
        del csr

    model = ARDecoder(d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads, d_ff=a.d_ff, n_mem=a.n_mem,
                      dropout=a.dropout).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / a.steps))))
    amp = device.startswith("cuda")
    log = open(os.path.join(out, "log.jsonl"), "a")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        model.train()
        idx = torch.randint(0, len(train), (a.bs,), device=device)
        z = z_tr[idx]
        if a.z_noise:
            z = z + a.z_noise * torch.randn_like(z)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss, logs = model.loss(z, v_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - a.ema)
        if step % 100 == 0:
            rec = {"step": step, "loss": float(loss), **{k: float(v) for k, v in logs.items()},
                   "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
        if step % a.eval_every == 0 or step == a.steps:
            recon = []
            for s in range(0, len(val), 256):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    recon += ema.generate(z_va[s: s + 256])
            rec = {"step": step, "ar_recon": seq_metrics(recon, gt_va)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % a.save_every == 0 or step == a.steps:
            model.save(os.path.join(out, "last.pt"), ema_state_dict=ema.state_dict(), step=step, args=vars(a))
    print(f"[done] {out}/last.pt")


if __name__ == "__main__":
    main()
