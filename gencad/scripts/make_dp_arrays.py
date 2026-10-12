"""Turn uid-keyed stores into the row-aligned 'zs' arrays that upstream `train_gencad.py dp` expects.

One row per (uid, render view) of the chosen split, so CAD latent row i and image latent row i always belong
to the same model. Upstream's Trainer1D also hard-codes data/cad_embeddings.h5 for validation, so that file
is written too.

  python scripts/make_dp_arrays.py --emb_dir data/emb --split train --out_dir data/dp
  python train_gencad.py dp -name repl -gpu 0 -cad_emb data/dp/cad_zs.h5 -img_emb data/dp/img_zs.h5
"""
import argparse
import json
import os
import shutil
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.data import CadLatentStore, EmbeddingStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="data/emb")
    ap.add_argument("--cond", default="ccip_image")
    ap.add_argument("--split_json", default="data/filtered_data.json")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_dir", default="data/dp")
    ap.add_argument("--gt_copy", default="data/cad_embeddings.h5")
    a = ap.parse_args()

    cad = CadLatentStore(os.path.join(a.emb_dir, "cad_latents.h5"))
    cond = EmbeddingStore(os.path.join(a.emb_dir, f"cond_{a.cond}.h5"))
    with open(a.split_json) as f:
        uids = json.load(f)[a.split]
    cz, iz = [], []
    for u in uids:
        if u not in cad.index or u not in cond.index:
            continue
        i = cond.index[u]
        for k in np.nonzero(cond.valid[i].numpy())[0]:
            cz.append(cad.z[cad.index[u]].numpy())
            iz.append(cond.emb[i, k].float().numpy())
    cz, iz = np.stack(cz).astype(np.float32), np.stack(iz).astype(np.float32)
    os.makedirs(a.out_dir, exist_ok=True)
    for name, arr in (("cad_zs.h5", cz), ("img_zs.h5", iz)):
        with h5py.File(os.path.join(a.out_dir, name), "w") as f:
            f.create_dataset("zs", data=arr)
    if a.gt_copy:
        shutil.copy(os.path.join(a.out_dir, "cad_zs.h5"), a.gt_copy)
    print(f"[dp] {len(cz)} aligned rows from {len(uids)} {a.split} uids -> {a.out_dir}")


if __name__ == "__main__":
    main()
