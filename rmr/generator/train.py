"""Train the plan -> motion generator.

Loss = masked flow-matching MSE + a velocity loss on the implied clean estimate ``x0_hat = x_t - t * v``
(frame-to-frame differences, weighted by ``1 - t``). Without the velocity term the model undershoots fast
motion (head and antennas came out about half as fast as real). 10% plan dropout trains the unconditional
branch for classifier-free guidance. The data is tiny, so the model overfits after a few thousand steps:
the checkpoint with the best held-out loss is kept.

Reference: ``generator/train.py`` in pham-tuan-binh/reachy-motion-generator (Apache-2.0).
"""
import json
import os
import time

import numpy as np
import torch

from .data import bucketize, fit_stats, samples_from_moves
from .model import MotionGenerator, device


def loss_fn(net, x0, Q, M, plan_drop=0.1, vel_w=1.0):
    B, dev = x0.shape[0], x0.device
    has = (torch.rand(B, 1, 1, device=dev) >= plan_drop).float()
    t = torch.sigmoid(torch.randn(B, device=dev) - 0.4)   # logit-normal, biased toward the data end
    t_ = t.view(-1, 1, 1)
    x1 = torch.randn_like(x0)
    xt = t_ * x1 + (1 - t_) * x0
    v = net(xt, t, Q, has, M == 0)
    w = M.unsqueeze(-1)
    loss = (((v - (x1 - x0)) ** 2) * w).sum() / (w.sum() * x0.shape[-1])
    if vel_w == 0:
        return loss
    x0_hat = xt - t_ * v
    d_hat, d_true = x0_hat[:, 1:] - x0_hat[:, :-1], x0[:, 1:] - x0[:, :-1]
    wv = (M[:, 1:] * M[:, :-1]).unsqueeze(-1) * (1 - t_)
    return loss + vel_w * ((d_hat - d_true) ** 2 * wv).sum() / ((d_true ** 2 * wv).sum() + 1e-6)


def train(moves, held_out, out="checkpoints/generator.pt", steps=5000, bs=16, lr=3e-4, eval_every=250, seed=0,
          model_kw=None, log=print):
    """``moves``: ``[(name, move)]``; clips named in ``held_out`` are used only for the held-out loss."""
    dev = device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_moves = [(n, m) for n, m in moves if n not in held_out]
    val_moves = [(n, m) for n, m in moves if n in held_out]
    S = samples_from_moves(train_moves)
    stats = fit_stats(S)
    buckets = bucketize(S, stats, dev)
    val_buckets = bucketize(samples_from_moves(val_moves), stats, dev) if val_moves else []
    net = MotionGenerator(**(model_kw or {})).to(dev)
    log(f"[generator] {len(train_moves)} train clips -> {len(S)} augmented | {len(val_moves)} held-out | "
        f"{sum(p.numel() for p in net.parameters()) / 1e6:.1f}M params | {dev}")
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=min(0.3, max(0.05, 2 / steps)))
    probs = np.array([len(b["X"]) for b in buckets], float)
    probs /= probs.sum()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    best, hist, history, t0 = (float("inf"), 0), [], [], time.time()
    for step in range(1, steps + 1):
        b = buckets[rng.choice(len(buckets), p=probs)]
        idx = torch.as_tensor(rng.integers(0, len(b["X"]), min(bs, len(b["X"]))), device=dev)
        loss = loss_fn(net, b["X"][idx], b["Q"][idx], b["M"][idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append(loss.item())
        if step % eval_every == 0 or step == steps:
            val = float("nan")
            if val_buckets:
                net.eval()
                torch.manual_seed(0)
                with torch.no_grad():
                    val = float(np.mean([loss_fn(net, v["X"], v["Q"], v["M"], plan_drop=0.0, vel_w=0.0).item()
                                         for v in val_buckets for _ in range(4)]))
                net.train()
            if not val_buckets or val < best[0]:
                best = (val, step)
                torch.save({"sd": net.state_dict(), "stats": stats, "step": step, "held_out": list(held_out),
                            "config": net.config}, out)
            train_loss = float(np.mean(hist[-eval_every:]))
            history.append({"step": step, "train": train_loss, "held_out": val})
            log(f"[generator] step {step:5d}/{steps} train {train_loss:.4f} held-out {val:.4f} "
                f"{(time.time() - t0) / step:.3f}s/it")
    log(f"[generator] kept step {best[1]} (held-out {best[0]:.4f}) -> {out}")
    with open(os.path.splitext(out)[0] + ".history.json", "w") as f:
        json.dump(history, f)
    return out
