"""Evaluation helpers for the latent prior and for decoded CAD sequences."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from cadlib.macro import ARC_IDX, CIRCLE_IDX, EXT_IDX, LINE_IDX, CMD_ARGS_MASK  # noqa: F401


@torch.no_grad()
def latent_metrics(z_pred, z_gt, ks=(1, 5, 10)):
    """Cosine similarity to GT and retrieval accuracy: is the GT latent among the k nearest (cosine) of the
    sample within this candidate set? Same protocol idea as GenCAD's retrieval eval, in CSR latent space."""
    zp, zg = F.normalize(z_pred.float(), dim=-1), F.normalize(z_gt.float(), dim=-1)
    cos = (zp * zg).sum(-1)
    sims = zp @ zg.t()
    rank = (sims > sims.diag()[:, None]).sum(1)  # 0 == top-1
    out = {"cos": cos.mean().item(), "mse": F.mse_loss(z_pred, z_gt).item()}
    for k in ks:
        out[f"R@{k}"] = (rank < k).float().mean().item()
    out["n"] = len(zp)
    return out


def seq_metrics(pred_vecs, gt_vecs, tolerance=3):
    """DeepCAD-style accuracy for decoded sequences.

    pred_vecs: list of (L, 17) arrays or None (no EOS -> invalid); gt_vecs: list of (L, 17) arrays.
    ACC_cmd: fraction of commands that match (over GT length).
    ACC_param: fraction of *used* parameters of correctly predicted commands within +-tolerance quantization
               bins (DeepCAD uses tolerance 3 of 256 levels).
    invalid: fraction with no EOS / unparseable.
    """
    n_cmd = n_cmd_ok = n_par = n_par_ok = invalid = 0
    for p, g in zip(pred_vecs, gt_vecs):
        if p is None:
            invalid += 1
            n_cmd += len(g)
            continue
        L = len(g)
        pc = np.full(L, -1)
        m = min(L, len(p))
        pc[:m] = p[:m, 0]
        ok = pc == g[:, 0]
        n_cmd += L
        n_cmd_ok += ok.sum()
        for j in np.nonzero(ok)[0]:
            mask = CMD_ARGS_MASK[g[j, 0]].astype(bool)
            if mask.sum() == 0:
                continue
            diff = np.abs(p[j, 1:][mask].astype(int) - g[j, 1:][mask].astype(int))
            n_par += mask.sum()
            n_par_ok += (diff < tolerance).sum()
    return {
        "acc_cmd": n_cmd_ok / max(1, n_cmd),
        "acc_param": n_par_ok / max(1, n_par),
        "invalid": invalid / max(1, len(gt_vecs)),
    }
