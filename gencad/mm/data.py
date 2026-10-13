"""Cached-embedding datasets keyed by DeepCAD uid (e.g. "0067/00675619").

On-disk format (all HDF5, written by scripts/export_gencad_embeddings.py and scripts/embed_conditions.py):

  cad_latents.h5      uids (N,) str      z    (N, 256) float32     CSR bottleneck latents (prior target)
  cond_<name>.h5      uids (N,) str      emb  (N, K, d) float16    K variants per uid (render views or
                                         valid (N, K) bool          prompt levels); attrs: dim, variants

<name> is the modality name used by the prior ("image", "text", "ccip_image", ...).
"""
from __future__ import annotations

import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_uids(ds):
    return [u.decode() if isinstance(u, bytes) else str(u) for u in ds[:]]


def write_store(path, uids, emb, valid=None, **attrs):
    """emb: (N, K, d) or (N, d). Stored as float16 to keep 150k x 5 x 1024 in a few GB."""
    emb = np.asarray(emb)
    if emb.ndim == 2:
        emb = emb[:, None]
    if valid is None:
        valid = np.ones(emb.shape[:2], dtype=bool)
    with h5py.File(path, "w") as f:
        f.create_dataset("uids", data=np.array(uids, dtype="S"))
        f.create_dataset("emb", data=emb.astype(np.float16), chunks=True)
        f.create_dataset("valid", data=np.asarray(valid, dtype=bool))
        f.attrs["dim"] = emb.shape[-1]
        for k, v in attrs.items():
            f.attrs[k] = json.dumps(v) if not isinstance(v, (int, float, str)) else v


class EmbeddingStore:
    def __init__(self, path):
        with h5py.File(path, "r") as f:
            self.uids = _read_uids(f["uids"])
            self.emb = torch.from_numpy(f["emb"][:])           # float16 (N, K, d)
            self.valid = torch.from_numpy(f["valid"][:]).bool()  # (N, K)
            self.attrs = dict(f.attrs)
        self.index = {u: i for i, u in enumerate(self.uids)}
        self.dim = self.emb.shape[-1]

    def get(self, uid, variant=None, rng=None):
        """Returns (embedding float32 (d,), present bool). variant=None -> random valid variant."""
        i = self.index.get(uid)
        if i is None or not self.valid[i].any():
            return torch.zeros(self.dim), False
        ok = torch.nonzero(self.valid[i]).flatten().tolist()
        if variant is None:
            k = ok[int(torch.randint(len(ok), (1,)))]  # torch RNG is reseeded per DataLoader worker
        else:
            if variant not in ok:
                return torch.zeros(self.dim), False
            k = variant
        return self.emb[i, k].float(), True


class PooledViewStore(EmbeddingStore):
    """Multi-view condition: the mean of several variants (render views) of one uid, read from the per-view store.
    variant=None (training) -> mean of a random non-empty subset of the valid views, so one model serves any number
    of input views; variant=k -> mean of the first k+1 valid views (eval: k = n_views - 1)."""

    def get(self, uid, variant=None, rng=None):
        i = self.index.get(uid)
        if i is None or not self.valid[i].any():
            return torch.zeros(self.dim), False
        ok = torch.nonzero(self.valid[i]).flatten()
        if variant is None:
            ok = ok[torch.randperm(len(ok))[: int(torch.randint(1, len(ok) + 1, (1,)))]]
        else:
            ok = ok[: variant + 1]
        return self.emb[i, ok].float().mean(0), True


POOLED_SUFFIX = "_mv"


def cond_file(emb_dir, name):
    """Embedding file for a modality name: '<m>_mv' pools the views of cond_<m>.h5."""
    base = name[: -len(POOLED_SUFFIX)] if name.endswith(POOLED_SUFFIX) else name
    return os.path.join(emb_dir, f"cond_{base}.h5")


def open_store(name, path):
    return PooledViewStore(path) if name.endswith(POOLED_SUFFIX) else EmbeddingStore(path)


class CadLatentStore:
    def __init__(self, path):
        with h5py.File(path, "r") as f:
            self.uids = _read_uids(f["uids"])
            self.z = torch.from_numpy(f["z"][:]).float()
        self.index = {u: i for i, u in enumerate(self.uids)}


class MultiModalLatentDataset(Dataset):
    """Yields {"z": (256,), "uid", "emb/<m>": (d_m,), "present/<m>": bool} for every modality store."""

    def __init__(self, cad_path, cond_paths: dict, split=None, split_json=None, variants: dict | None = None,
                 require_any=True, max_items=None):
        self.cad = CadLatentStore(cad_path)
        self.stores = {m: open_store(m, p) for m, p in cond_paths.items()}
        uids = self.cad.uids
        if split is not None:
            with open(split_json) as f:
                keep = set(json.load(f)[split])
            uids = [u for u in uids if u in keep]
        if require_any:  # drop CAD models with no condition at all
            uids = [u for u in uids if any(u in s.index and s.valid[s.index[u]].any() for s in self.stores.values())]
        if max_items:
            uids = uids[:max_items]
        self.uids = uids
        self.variants = variants or {}  # {modality: fixed variant index} for deterministic eval

    @property
    def modality_dims(self):
        return {m: s.dim for m, s in self.stores.items()}

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        item = {"z": self.cad.z[self.cad.index[uid]], "uid": uid}
        for m, s in self.stores.items():
            e, ok = s.get(uid, self.variants.get(m))
            item[f"emb/{m}"] = e
            item[f"present/{m}"] = torch.tensor(ok)
        return item


def split_batch(batch, device):
    """batch dict -> (z, embeds, present) on device."""
    z = batch["z"].to(device, non_blocking=True)
    embeds = {k[4:]: v.to(device, non_blocking=True) for k, v in batch.items() if k.startswith("emb/")}
    present = {k[8:]: v.to(device, non_blocking=True) for k, v in batch.items() if k.startswith("present/")}
    return z, embeds, present


def default_paths(root):
    return os.path.join(root, "cad_latents.h5"), lambda name: os.path.join(root, f"cond_{name}.h5")
