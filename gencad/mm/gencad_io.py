"""Glue to the original GenCAD code: load CSR / CCIP checkpoints, encode and decode CAD sequences."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cadlib.macro import EOS_IDX, MAX_TOTAL_LEN, SOL_IDX  # noqa: E402
from config import ConfigAE  # noqa: E402
from model import CLIP, ResNetImageEncoder, VanillaCADTransformer  # noqa: E402
from utils.model_utils import logits2vec  # noqa: E402


def _state_dict(ckpt):
    for k in ("model_state_dict", "state_dict", "model"):
        if isinstance(ckpt, dict) and k in ckpt:
            return ckpt[k]
    return ckpt


def load_csr(ckpt_path=None, device="cpu", config_overrides=None):
    cfg = ConfigAE(phase="test", device=torch.device(device), overwrite=False)
    for k, v in (config_overrides or {}).items():
        setattr(cfg, k, v)
    model = VanillaCADTransformer(cfg)
    if ckpt_path:
        model.load_state_dict(_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)))
    return model.to(device).eval(), cfg


def load_ccip(ckpt_path, csr_model, device="cpu", vision_network="resnet-18"):
    image_encoder = ResNetImageEncoder(network=vision_network)
    clip = CLIP(image_encoder=image_encoder, cad_encoder=csr_model, dim_latent=256)
    clip.load_state_dict(_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)))
    return clip.to(device).eval()


@torch.no_grad()
def encode_cad(csr, commands, args):
    """(B, 60) commands, (B, 60, 16) args -> (B, 256) CSR latents (the prior's target space)."""
    z = csr(commands, args, encode_mode=True)
    return z.reshape(z.shape[0], -1)


@torch.no_grad()
def decode_latents(csr, z):
    """(B, 256) latents -> list of CAD vectors (L_i, 17) int, truncated at EOS (None if no EOS found).

    Mirrors inference_gencad.py: the decoder predicts the sequence after an implicit leading SOL.
    """
    device = z.device
    out = csr(None, None, z=z.unsqueeze(1), return_tgt=False)
    vec = logits2vec(out, device=device if device.type == "cuda" else None)  # (B, S, 17)
    sol = np.full((vec.shape[0], 1, vec.shape[2]), -1, dtype=np.int64)
    sol[:, :, 0] = SOL_IDX
    vec = np.concatenate([sol, vec], axis=1)[:, :MAX_TOTAL_LEN]
    results = []
    for v in vec:
        cmds = v[:, 0].tolist()
        results.append(v[: cmds.index(EOS_IDX)] if EOS_IDX in cmds else None)
    return results


def vec_to_shape(cad_vec):
    """CAD vector -> OCC solid (needs pythonocc-core). Returns None if the sequence is not buildable."""
    import OCC.Core.BRepBndLib as bnd  # ImportError (no pythonocc-core) propagates to the caller
    if not hasattr(bnd, "brepbndlib_Add"):  # removed in pythonocc 7.8; upstream cadlib still imports it
        bnd.brepbndlib_Add = bnd.brepbndlib.Add
    import matplotlib
    use = matplotlib.use
    matplotlib.use = lambda *a, **k: None  # upstream cadlib/sketch.py forces TkAgg, which fails headless
    try:
        from cadlib.visualize import vec2CADsolid
    finally:
        matplotlib.use = use
    try:
        return vec2CADsolid(cad_vec.astype(float))
    except Exception:
        return None


def export_shape(shape, path):
    if path.endswith(".stl"):
        from OCC.Extend.DataExchange import write_stl_file
        write_stl_file(shape, path, mode="binary", linear_deflection=0.5, angular_deflection=0.3)
    elif path.endswith((".step", ".stp")):
        from OCC.Extend.DataExchange import write_step_file
        write_step_file(shape, path)
    else:
        raise ValueError(path)
