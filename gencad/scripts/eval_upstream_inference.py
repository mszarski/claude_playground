"""Score the released GenCAD pipeline exactly as upstream's inference_gencad.py runs it, on eval_mm.py's test samples.

Cross-checks our harness's `gencad_dp` row (scripts/export_gencad_embeddings.py + eval_mm.py). Per image, it follows
inference_gencad.main step for step: utils.process_image -> GenCADClipAdapter.embed_image(normalization=False) ->
GaussianDiffusion1D.sample(cond) -> CSR decode with a leading SOL row. The only change is batching. Outputs are
scored with the same metrics as eval_mm.py. Input variants (both use render view 0 of each uid):

  render   data/images/<uid>_<view>.png through process_image (Canny on the RGB render): upstream's inference path
  sketch   data/sketches/<uid>_<view>.png through the same Canny: what CCIPDataset(img_type="cad_sketch") trained on

  python scripts/eval_upstream_inference.py --data data --emb_dir data/emb --ckpt_dir data/ckpt --n 2000 \
      --out results/upstream/eval.json
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadlib.macro import EOS_IDX, MAX_TOTAL_LEN  # noqa: E402
from config import ConfigAE  # noqa: E402
from model import CLIP, GaussianDiffusion1D, GenCADClipAdapter, ResNetDiffusion, ResNetImageEncoder  # noqa: E402
from model import VanillaCADTransformer  # noqa: E402
from utils import logits2vec, process_image  # noqa: E402

from mm.data import MultiModalLatentDataset  # noqa: E402
from mm.metrics import latent_metrics, seq_metrics  # noqa: E402


def load_upstream(ckpt_dir, device):
    """Model construction copied from inference_gencad.main (same classes, hyper-parameters and checkpoints)."""
    net = ResNetDiffusion(d_in=256, n_blocks=10, d_main=2048, d_hidden=2048, dropout_first=0.1, dropout_second=0.1,
                          d_out=256)
    diffusion = GaussianDiffusion1D(net, z_dim=256, timesteps=500, objective="pred_x0", auto_normalize=False)
    ck = torch.load(os.path.join(ckpt_dir, "sketch_cond_diffusion_ckpt_epoch1000000.pt"), map_location="cpu",
                    weights_only=False)
    diffusion.load_state_dict(ck["model"])
    diffusion = diffusion.to(device).eval()

    cfg = ConfigAE(phase="test", device=device, overwrite=False)
    clip = CLIP(image_encoder=ResNetImageEncoder(network="resnet-18"), cad_encoder=VanillaCADTransformer(cfg),
                dim_latent=256)
    ck = torch.load(os.path.join(ckpt_dir, "ccip_sketch_ckpt_epoch300.pth"), map_location="cpu", weights_only=False)
    clip.load_state_dict(ck["model_state_dict"])
    clip.eval()
    adapter = GenCADClipAdapter(clip=clip).to(device)

    dcfg = ConfigAE(exp_name="test", phase="test", batch_size=1, device=device, overwrite=False)
    decoder = VanillaCADTransformer(dcfg).to(device)
    ck = torch.load(os.path.join(ckpt_dir, "ae_ckpt_epoch1000.pth"), map_location="cpu", weights_only=False)
    decoder.load_state_dict(ck["model_state_dict"])
    decoder.eval()
    return diffusion, adapter, decoder


@torch.no_grad()
def decode(decoder, latent, device):
    """inference_gencad.main's decode, batched: prepend SOL (command 4), cut at the first EOS (None if none)."""
    outputs = decoder(None, None, z=latent.unsqueeze(1), return_tgt=False)
    vec = logits2vec(outputs, device=device if device.type == "cuda" else None)  # upstream hard-codes .cuda()
    sol = np.full((vec.shape[0], 1, vec.shape[2]), -1, dtype=np.int64)
    sol[:, :, 0] = 4
    vec = np.concatenate([sol, vec], axis=1)[:, :MAX_TOTAL_LEN, :]
    out = []
    for v in vec:
        cmds = v[:, 0].tolist()
        out.append(v[: cmds.index(EOS_IDX)] if EOS_IDX in cmds else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--emb_dir", default="data/emb")
    ap.add_argument("--ckpt_dir", default="data/ckpt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--variants", nargs="*", default=["render", "sketch"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/upstream/eval.json")
    a = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # the same uids, in the same order, as eval_mm.py's gencad_dp row
    split_json = os.path.join(a.data, "filtered_data.json")
    ds = MultiModalLatentDataset(os.path.join(a.emb_dir, "cad_latents.h5"),
                                 {"ccip_image": os.path.join(a.emb_dir, "cond_ccip_image.h5")}, a.split, split_json,
                                 variants={"ccip_image": 0}, max_items=a.n)
    uids = [u for i, u in enumerate(ds.uids) if ds[i]["present/ccip_image"]]
    z_gt = torch.stack([ds.cad.z[ds.cad.index[u]] for u in uids]).to(device)
    gts = []
    for u in uids:
        with h5py.File(os.path.join(a.data, "cad_vec", u + ".h5"), "r") as f:
            gts.append(f["vec"][:])
    with open(os.path.join(a.data, "image_ids.json")) as f:
        img_ids = json.load(f)
    print(f"[upstream] {len(uids)} {a.split} uids")

    diffusion, adapter, decoder = load_upstream(a.ckpt_dir, device)
    report = {}
    for variant in a.variants:
        folder = {"render": "images", "sketch": "sketches"}[variant]
        torch.manual_seed(a.seed)
        zs, vecs = [], []
        for s in tqdm(range(0, len(uids), a.bs), desc=variant):
            chunk = uids[s: s + a.bs]
            # process_image's default img_type="image" applies Canny, as in inference_gencad.main
            img = torch.cat([process_image(os.path.join(a.data, folder, f"{u}_{img_ids[u][0]}.png")) for u in chunk])
            cond = adapter.embed_image(img.to(device), normalization=False)
            z = diffusion.sample(cond=cond)
            zs.append(z)
            vecs += decode(decoder, z, device)
        z = torch.cat(zs)
        m = latent_metrics(z, z_gt)
        m.update(seq_metrics(vecs, gts))
        name = f"upstream_inference:{variant}"
        print(f"{name:28s} " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()),
              flush=True)
        report[name] = m
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[upstream] wrote {a.out}")


if __name__ == "__main__":
    main()
