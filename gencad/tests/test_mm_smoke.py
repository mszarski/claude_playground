"""Smoke tests for the multimodal extension. CPU-only, synthetic data, ~1-2 minutes.

  python -m pytest tests/test_mm_smoke.py -q        (or: python tests/test_mm_smoke.py)

Real-encoder test (downloads SigLIP2 + Qwen3-Embedding) runs only with MM_TEST_ENCODERS=1.
"""
import json
import os
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mm.data import MultiModalLatentDataset, write_store  # noqa: E402
from mm.diffusion import MultiCondDiffusionPrior  # noqa: E402
from mm.fusion import ConditionFuser  # noqa: E402
from mm.metrics import latent_metrics, seq_metrics  # noqa: E402

DIMS = {"image": 16, "text": 24}


def toy_problem(n=512, seed=0):
    """z is a fixed nonlinear function of (image, text) so a conditional model can beat an unconditional one."""
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(n, DIMS["image"], generator=g)
    txt = torch.randn(n, DIMS["text"], generator=g)
    Wi = torch.randn(DIMS["image"], 256, generator=g) / 4
    Wt = torch.randn(DIMS["text"], 256, generator=g) / 5
    z = torch.tanh(img @ Wi + txt @ Wt)
    return z, {"image": img, "text": txt}


def tiny_prior(**kw):
    return MultiCondDiffusionPrior(DIMS, timesteps=200, contrastive_weight=kw.pop("contrastive_weight", 0.1),
                                   fuser_kwargs=dict(d_model=64, n_layers=1, n_heads=4),
                                   denoiser_kwargs=dict(d_main=128, d_hidden=256, n_blocks=3, dropout=0.0), **kw)


def test_fuser_null_tokens_and_dropout():
    f = ConditionFuser(DIMS, d_model=32, n_layers=1, n_heads=4, p_drop={"image": 1.0, "text": 0.0}, p_drop_all=0.0)
    f.eval()
    B = 4
    e = {"image": torch.randn(B, 16), "text": torch.randn(B, 24)}
    present = {m: torch.ones(B, dtype=torch.bool) for m in DIMS}
    keep = f.sample_keep_mask(present, B, "cpu")
    assert not keep["image"].any() and keep["text"].all()
    # dropping a modality == not providing it
    a = f(e, keep)
    b = f({"text": e["text"]}, {"text": keep["text"]})
    assert torch.allclose(a, b, atol=1e-5)
    # nothing provided -> unconditional vector, identical for every row
    u = f({}, batch=B)
    assert u.shape == (B, 32) and torch.allclose(u[0], u[1])


def test_prior_learns_conditioning():
    torch.manual_seed(0)
    z, e = toy_problem(512)
    model = tiny_prior()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    first = None
    for step in range(600):
        idx = torch.randint(0, len(z), (128,))
        loss, logs = model(z[idx], {m: v[idx] for m, v in e.items()})
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else logs["diff_loss"].item()
    last = logs["diff_loss"].item()
    assert last < first * 0.7, (first, last)

    model.eval()
    sub = torch.arange(128)
    g = torch.Generator().manual_seed(0)
    zc = model.sample({m: v[sub] for m, v in e.items()}, steps=25, generator=g)
    zu = model.sample({}, batch=128, steps=25, generator=torch.Generator().manual_seed(0))
    mc, mu = latent_metrics(zc, z[sub]), latent_metrics(zu, z[sub])
    print("cond", mc, "\nuncond", mu)
    assert mc["cos"] > mu["cos"] + 0.1, (mc, mu)
    assert zc.abs().max() <= 1.0 + 1e-6


def test_guidance_modes_and_subsets():
    model = tiny_prior().eval()
    _, e = toy_problem(6)
    for mode in ("chain", "independent"):
        for sub in ({"image"}, {"text"}, {"image", "text"}):
            out = model.sample({m: e[m] for m in sub}, guidance={"image": 2.0, "text": 3.0}, mode=mode, steps=5)
            assert out.shape == (6, 256) and torch.isfinite(out).all()
    # chain guidance with all weights 1 telescopes to the plain fully-conditioned prediction
    order = ["image", "text"]
    bank = []
    for s in model._cond_sets(order, "chain"):
        keep = {m: torch.full((6,), m in s) for m in model.modalities}
        bank.append(model.fuser(e, keep, batch=6))
    x, t = torch.randn(6, 256), torch.full((6,), 50)
    guided = model._guided_x0(x, t, bank, order, {"image": 1.0, "text": 1.0}, "chain")
    plain = model.denoiser(x, t, bank[-1])
    assert torch.allclose(guided, plain, atol=1e-4)


def test_save_load_roundtrip(tmp_path=None):
    tmp = tmp_path or tempfile.mkdtemp()
    model = tiny_prior()
    p = os.path.join(str(tmp), "p.pt")
    model.save(p)
    m2, _ = MultiCondDiffusionPrior.load(p)
    for (k, v), (k2, v2) in zip(model.state_dict().items(), m2.state_dict().items()):
        assert k == k2 and torch.equal(v, v2)


def make_stores(root, n=300):
    z, e = toy_problem(n)
    uids = [f"{i // 100:04d}/{i:08d}" for i in range(n)]
    with h5py.File(os.path.join(root, "cad_latents.h5"), "w") as f:
        f.create_dataset("uids", data=np.array(uids, dtype="S"))
        f.create_dataset("z", data=z.numpy())
    # image: 3 views (jittered), the last view missing for every 4th item
    views = np.stack([e["image"].numpy() + 0.05 * np.random.randn(n, 16) for _ in range(3)], 1)
    valid = np.ones((n, 3), bool)
    valid[::4, 2] = False
    write_store(os.path.join(root, "cond_image.h5"), uids, views, valid, backend="toy", model="toy")
    # text: 2 levels, missing entirely for every 5th item (Text2CAD does not cover every DeepCAD model)
    tv = np.stack([e["text"].numpy(), e["text"].numpy() * 0.9], 1)
    tvalid = np.ones((n, 2), bool)
    tvalid[::5] = False
    write_store(os.path.join(root, "cond_text.h5"), uids, tv, tvalid, backend="toy", model="toy",
                variants=["beginner", "expert"])
    splits = {"train": uids[:240], "validation": uids[240:270], "test": uids[270:]}
    with open(os.path.join(root, "filtered_data.json"), "w") as f:
        json.dump(splits, f)
    return uids


def test_dataset_presence_masks():
    root = tempfile.mkdtemp()
    make_stores(root)
    ds = MultiModalLatentDataset(os.path.join(root, "cad_latents.h5"),
                                 {m: os.path.join(root, f"cond_{m}.h5") for m in DIMS},
                                 "train", os.path.join(root, "filtered_data.json"))
    assert len(ds) == 240 and ds.modality_dims == DIMS
    it = ds[0]
    assert it["present/image"] and not it["present/text"]  # uid 0 has no text
    assert it["emb/text"].abs().sum() == 0
    it = ds[1]
    assert it["present/image"] and it["present/text"]


def test_train_script_end_to_end():
    root = tempfile.mkdtemp()
    make_stores(root)
    cmd = [sys.executable, os.path.join(ROOT, "train_mm_prior.py"), "--emb_dir", root,
           "--split_json", os.path.join(root, "filtered_data.json"), "--modalities", "image", "text",
           "--out_dir", root, "--name", "run", "--steps", "60", "--bs", "32", "--workers", "0",
           "--d_cond", "64", "--fuser_layers", "1", "--d_main", "128", "--d_hidden", "256", "--n_blocks", "2",
           "--timesteps", "100", "--eval_every", "60", "--eval_n", "30", "--eval_steps", "5",
           "--save_every", "60", "--warmup", "10", "--device", "cpu"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    print(r.stdout[-3000:], r.stderr[-3000:])
    assert r.returncode == 0
    ck = os.path.join(root, "run", "last.pt")
    assert os.path.exists(ck)
    model, meta = MultiCondDiffusionPrior.load(ck)
    assert meta["encoders"]["text"]["backend"] == "toy"
    assert "image+text" in open(os.path.join(root, "run", "log.jsonl")).read()


def _fake_cad_vec():
    from cadlib.macro import EOS_IDX, EXT_IDX, LINE_IDX, N_ARGS, SOL_IDX
    rows = [[SOL_IDX] + [-1] * N_ARGS]
    for x, y in [(10, 10), (200, 10), (200, 200), (10, 200)]:
        rows.append([LINE_IDX, x, y] + [-1] * (N_ARGS - 2))
    rows.append([EXT_IDX] + [-1] * 5 + [128, 128, 128, 100, 100, 100, 50, 180, 128, 0, 0])
    rows.append([EOS_IDX] + [-1] * N_ARGS)
    return np.array(rows[:-1])  # stored sequences exclude the padding EOS


def test_seq_metrics():
    g = _fake_cad_vec()
    perfect = seq_metrics([g.copy()], [g])
    assert perfect["acc_cmd"] == 1.0 and perfect["acc_param"] == 1.0 and perfect["invalid"] == 0
    bad = seq_metrics([None], [g])
    assert bad["invalid"] == 1.0


def test_csr_glue_untrained():
    """Encode/decode through GenCAD's CSR (random weights) to check shapes and the SOL/EOS handling."""
    from cadlib.macro import EOS_VEC, MAX_TOTAL_LEN
    from mm.gencad_io import decode_latents, encode_cad, load_csr
    csr, _ = load_csr(None, "cpu")
    v = _fake_cad_vec()
    v = np.concatenate([v, EOS_VEC[None].repeat(MAX_TOTAL_LEN - len(v), 0)])
    cmd = torch.tensor(v[:, 0])[None].long().repeat(2, 1)
    args = torch.tensor(v[:, 1:])[None].long().repeat(2, 1, 1)
    z = encode_cad(csr, cmd, args)
    assert z.shape == (2, 256) and z.abs().max() <= 1
    out = decode_latents(csr, z)
    assert len(out) == 2 and all(o is None or o.shape[1] == 17 for o in out)


def test_real_encoders():
    if os.environ.get("MM_TEST_ENCODERS") != "1":
        print("skip real encoder test (set MM_TEST_ENCODERS=1)")
        return
    from PIL import Image
    from mm.cond_encoders import ImageEncoder, TextEncoder
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ie = ImageEncoder(device=dev)
    img = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
    e = ie([img, img])
    assert e.shape == (2, ie.dim) and torch.allclose(e.norm(dim=-1), torch.ones(2, device=e.device), atol=1e-3)
    te = TextEncoder(device=dev)
    t = te(["a plate with four holes", "a cylinder"])
    assert t.shape == (2, te.dim)
    st = TextEncoder("siglip", "google/siglip2-base-patch16-224", device=dev)
    assert st(["a cube"]).shape[-1] == ie.dim  # shared text-image space


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        print(f"--- {fn.__name__}", flush=True)
        fn()
    print(f"ALL {len(fns)} TESTS PASSED")
