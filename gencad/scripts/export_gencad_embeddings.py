"""Export uid-aligned GenCAD embeddings for prior training.

Writes into --out:
  cad_latents.h5        CSR bottleneck latents for every uid in filtered_data.json (the prior's target)
  cond_ccip_image.h5    (optional, needs --ccip_ckpt) CCIP image latents for each render view of each uid

Why this exists: the upstream path (trainer/autoencoder_test.py::encode) builds latents with a *shuffled*
train dataloader and writes no uids, so the CAD latents and image latents fed to train_gencad.py dp are not
row-aligned. Everything here is keyed by uid.

Usage:
  python scripts/export_gencad_embeddings.py --data data --csr_ckpt data/ckpt/ae_ckpt_epoch1000.pth \
      --ccip_ckpt data/ckpt/ccip_ckpt_epoch300.pth --img_type cad_image --out data/emb
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cadlib.macro import EOS_VEC, MAX_TOTAL_LEN  # noqa: E402
from mm.data import write_store  # noqa: E402
from mm.gencad_io import encode_cad, load_ccip, load_csr  # noqa: E402
from utils.cad_dataset import CannyDetector  # noqa: E402


class SeqDataset(Dataset):
    def __init__(self, root, uids):
        self.root, self.uids = root, uids

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        with h5py.File(os.path.join(self.root, "cad_vec", self.uids[i] + ".h5"), "r") as f:
            v = f["vec"][:]
        v = np.concatenate([v, EOS_VEC[None].repeat(MAX_TOTAL_LEN - len(v), 0)], 0)
        return torch.tensor(v[:, 0], dtype=torch.long), torch.tensor(v[:, 1:], dtype=torch.long)


class ViewDataset(Dataset):
    """All (uid, view) render images, preprocessed exactly like utils/cad_dataset.CCIPDataset."""

    def __init__(self, root, uids, img_ids, img_type, n_views):
        self.dir = os.path.join(root, "images" if img_type == "cad_image" else "sketches")
        self.items = [(i, v) for i, u in enumerate(uids) for v in range(n_views)]
        self.uids, self.img_ids, self.img_type = uids, img_ids, img_type
        self.tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(),
                                      transforms.Normalize([0.5] * 3, [0.5] * 3)])
        self.canny = CannyDetector()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, v = self.items[k]
        uid = self.uids[i]
        views = self.img_ids.get(uid, [])
        path = os.path.join(self.dir, f"{uid}_{views[v]}.png") if v < len(views) else None
        if path is None or not os.path.exists(path):
            return torch.zeros(3, 256, 256), i, v, False
        img = Image.open(path).convert("RGB")
        if self.img_type == "cad_sketch":
            e = self.canny(np.array(img), 100, 200)[:, :, None].repeat(3, 2)
            img = Image.fromarray(e)
        return self.tf(img), i, v, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--csr_ckpt", required=True)
    ap.add_argument("--ccip_ckpt", default=None)
    ap.add_argument("--img_type", default="cad_image", choices=["cad_image", "cad_sketch"])
    ap.add_argument("--n_views", type=int, default=5)
    ap.add_argument("--out", default="data/emb")
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N uids")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    with open(os.path.join(a.data, "filtered_data.json")) as f:
        splits = json.load(f)
    uids = [u for s in ("train", "validation", "test") for u in splits[s]]
    uids = [u for u in uids if os.path.exists(os.path.join(a.data, "cad_vec", u + ".h5"))][: a.limit]
    print(f"[export] {len(uids)} uids with cad_vec")

    csr, _ = load_csr(a.csr_ckpt, device)
    zs = []
    for cmd, args in tqdm(DataLoader(SeqDataset(a.data, uids), batch_size=a.bs, num_workers=a.workers),
                          desc="CSR latents"):
        zs.append(encode_cad(csr, cmd.to(device), args.to(device)).float().cpu().numpy())
    zs = np.concatenate(zs)
    with h5py.File(os.path.join(a.out, "cad_latents.h5"), "w") as f:
        f.create_dataset("uids", data=np.array(uids, dtype="S"))
        f.create_dataset("z", data=zs.astype(np.float32))
    print(f"[export] cad_latents.h5 {zs.shape}  range [{zs.min():.3f}, {zs.max():.3f}]")

    if a.ccip_ckpt:
        with open(os.path.join(a.data, "image_ids.json")) as f:
            img_ids = json.load(f)
        clip = load_ccip(a.ccip_ckpt, csr, device)
        emb = np.zeros((len(uids), a.n_views, 256), np.float32)
        valid = np.zeros((len(uids), a.n_views), bool)
        dl = DataLoader(ViewDataset(a.data, uids, img_ids, a.img_type, a.n_views), batch_size=a.bs,
                        num_workers=a.workers)
        with torch.no_grad():
            for img, i, v, ok in tqdm(dl, desc="CCIP image latents"):
                # same as GenCADClipAdapter.embed_image(normalization=False), which inference_gencad.py feeds to the prior
                e = clip.to_visual_latent(clip.image_encoder(img.to(device))).float().cpu().numpy()
                emb[i.numpy(), v.numpy()] = e
                valid[i.numpy(), v.numpy()] = ok.numpy()
        write_store(os.path.join(a.out, "cond_ccip_image.h5"), uids, emb, valid,
                    source="gencad_ccip", img_type=a.img_type)
        print(f"[export] cond_ccip_image.h5 {emb.shape}, {valid.mean():.1%} views found")


if __name__ == "__main__":
    main()
