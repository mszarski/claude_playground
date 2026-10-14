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


def test_best_of_n_picks_highest_scoring_decodable_candidate():
    import mm.rerank as rerank
    from mm.gencad_io import load_csr
    torch.manual_seed(0)
    prior = tiny_prior().eval()
    csr, _ = load_csr(None, "cpu")
    embeds = {m: torch.randn(3, d) for m, d in DIMS.items()}
    real_decode = rerank.decode_latents
    try:  # only odd candidates decode; the winner must be one of them
        rerank.decode_latents = lambda c, z: [_fake_cad_vec() if i % 2 else None for i in range(len(z))]
        vecs, z, scores = rerank.best_of_n(prior, csr, embeds, n=4, steps=5)
    finally:
        rerank.decode_latents = real_decode
    assert len(vecs) == 3 and z.shape == (3, 256)
    assert all(v is not None for v in vecs) and torch.isfinite(scores).all()


def test_ar_decoder_learns_a_program_and_decodes_through_gencad_io():
    from mm.ar_decoder import ARDecoder, pad_program
    from mm.gencad_io import decode_latents
    torch.manual_seed(0)
    g = _fake_cad_vec()
    model = ARDecoder(d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0)
    z, v = torch.randn(4, 256), torch.tensor(np.stack([pad_program(g)] * 4))
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    for _ in range(150):
        loss, _ = model.loss(z, v)
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = decode_latents(model.eval(), z)  # dispatches to ARDecoder.generate
    assert all(o is not None and o.tolist() == g.tolist() for o in out)


def test_train_ar_decoder_script_end_to_end():
    root = tempfile.mkdtemp()
    uids = make_stores(root, n=60)
    data = os.path.join(root, "data")
    for u in uids:
        os.makedirs(os.path.dirname(os.path.join(data, "cad_vec", u + ".h5")), exist_ok=True)
        with h5py.File(os.path.join(data, "cad_vec", u + ".h5"), "w") as f:
            f["vec"] = _fake_cad_vec()
    os.replace(os.path.join(root, "filtered_data.json"), os.path.join(data, "filtered_data.json"))
    cmd = [sys.executable, os.path.join(ROOT, "train_ar_decoder.py"), "--emb_dir", root, "--data", data,
           "--out_dir", root, "--name", "ar", "--d_model", "64", "--n_layers", "1", "--n_heads", "4",
           "--d_ff", "128", "--bs", "16", "--steps", "20", "--warmup", "5", "--eval_every", "20", "--eval_n", "5",
           "--save_every", "20", "--device", "cpu"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    print(r.stdout[-2000:], r.stderr[-2000:])
    assert r.returncode == 0 and '"ar_recon"' in r.stdout
    from mm.ar_decoder import ARDecoder
    model, ck = ARDecoder.load(os.path.join(root, "ar", "last.pt"))
    assert ck["step"] == 20 and os.path.exists(os.path.join(root, "programs.npz"))


def test_pooled_view_store_averages_views():
    from mm.data import PooledViewStore, cond_file, open_store
    root = tempfile.mkdtemp()
    make_stores(root)
    path = cond_file(root, "image_mv")
    assert path.endswith("cond_image.h5")
    s = open_store("image_mv", path)
    assert isinstance(s, PooledViewStore)
    u = s.uids[1]                                     # uid 1 has all 3 views
    i = s.index[u]
    e1, ok = s.get(u, 0)
    assert ok and torch.allclose(e1, s.emb[i, 0].float())
    e3, _ = s.get(u, 2)
    assert torch.allclose(e3, s.emb[i, :3].float().mean(0))
    u0 = s.uids[0]                                    # uid 0: view 2 missing -> at most 2 views averaged
    j = s.index[u0]
    assert torch.allclose(s.get(u0, 4)[0], s.emb[j, :2].float().mean(0))
    r, ok = s.get(u, None)                            # random subset: some mean of 1..3 views
    assert ok and r.shape == (s.dim,)


class _CharTokenizer:
    """Offline stand-in for a HF tokenizer: one id per character, right-padded."""

    def __call__(self, texts, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        from transformers import BatchEncoding
        ids = [[3 + (ord(c) % 250) for c in t[:max_length]] for t in texts]
        n = max(len(i) for i in ids)
        return BatchEncoding({"input_ids": torch.tensor([i + [0] * (n - len(i)) for i in ids]),
                              "attention_mask": torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i in ids])})


def test_text_cad_reads_prompts_and_writes_programs():
    from transformers import Qwen3Config, Qwen3Model
    from mm.ar_decoder import pad_program
    from mm.text_cad import TextCAD
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=260, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=256)
    model = TextCAD(text_model=Qwen3Model(cfg), tokenizer=_CharTokenizer(), text_layers=2,
                    d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0)
    assert len(model.text.layers) == 2
    g1 = _fake_cad_vec()
    g2 = g1.copy()
    g2[1:5, 1:3] //= 2
    prompts = ["a big square plate", "a small square plate"]
    vec = torch.tensor(np.stack([pad_program(g1), pad_program(g2)]))
    opt = torch.optim.Adam([p for n, p in model.named_parameters() if not n.startswith("text.")], 3e-3)
    model.train()
    for _ in range(150):
        loss, _ = model.loss(prompts, vec)
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = model.eval().generate(prompts)
    assert [o.tolist() for o in out] == [g1.tolist(), g2.tolist()]   # different prompts -> their own programs
    assert all(not k.startswith("text.") for k in model.trainable_state_dict())


def test_token_cad_uses_image_and_optional_text_tokens():
    from transformers import Qwen3Config, Qwen3Model, SiglipVisionConfig, SiglipVisionModel
    from mm.ar_decoder import pad_program
    from mm.token_cad import TokenCAD
    torch.manual_seed(0)
    vision = SiglipVisionModel(SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                                  num_attention_heads=2, image_size=32, patch_size=8))
    text = Qwen3Model(Qwen3Config(vocab_size=260, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                  num_attention_heads=2, num_key_value_heads=1, head_dim=16))
    model = TokenCAD(image_model=vision, text_model=text, tokenizer=_CharTokenizer(), text_layers=1,
                     d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0)
    g1 = _fake_cad_vec()
    g2 = g1.copy()
    g2[1:5, 1:3] //= 2
    pixels = torch.randn(1, 3, 32, 32).repeat(2, 1, 1, 1)          # the same image twice ...
    prompts = ["big", "small"]                                        # ... told apart only by the prompt
    vec = torch.tensor(np.stack([pad_program(g1), pad_program(g2)]))
    opt = torch.optim.Adam([p for n, p in model.named_parameters() if not n.startswith(TokenCAD.FROZEN)], 3e-3)
    model.train()
    for _ in range(150):
        loss, _ = model.loss(vec, pixels=pixels, prompts=prompts)
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = model.eval().generate(pixels=pixels, prompts=prompts)
    assert [o.tolist() for o in out] == [g1.tolist(), g2.tolist()]     # text tokens are actually read
    assert all(o is not None for o in model.generate(pixels=pixels))  # image-only path works
    assert all(o is not None for o in model.generate(pixels=pixels, prompts=["big", None]))  # mixed batch
    assert all(not k.startswith(TokenCAD.FROZEN) for k in model.trainable_state_dict())


def _tiny_qwen3_vl(path):
    """Random 2-layer Qwen3-VL with the real Qwen3-VL-2B processor saved to `path` (None if offline)."""
    try:
        from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
        proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        cfg = Qwen3VLConfig.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    except Exception as e:  # noqa: BLE001 - no network / no cache
        print(f"[skip] Qwen3-VL processor unavailable: {e}")
        return None
    cfg.text_config.update(dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                                num_key_value_heads=2, head_dim=16))
    cfg.text_config.rope_parameters["mrope_section"] = [2, 3, 3]
    cfg.vision_config.update(dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                                  deepstack_visual_indexes=[0]))
    torch.manual_seed(0)
    Qwen3VLForConditionalGeneration(cfg).save_pretrained(path)
    proc.save_pretrained(path)
    return path


def test_vlm_cad_text_roundtrip_and_script_end_to_end():
    from PIL import Image
    from mm.vlm_cad import compose_views, program_to_text, text_to_program
    vec = _fake_cad_vec()
    assert (text_to_program(program_to_text(vec)) == vec).all()
    assert text_to_program("S\nL 1") is None and text_to_program("X 1 2") is None and text_to_program("") is None
    ims = [Image.new("RGB", (64, 64), c) for c in ["red", "green", "blue"]]
    assert compose_views(ims[:1]).size == (448, 448) and compose_views(ims).getpixel((300, 300)) == (255, 255, 255)

    root = tempfile.mkdtemp()
    model = _tiny_qwen3_vl(os.path.join(root, "tiny_vlm"))
    if model is None:
        return
    uids = make_stores(root, n=300)[:40] + make_stores(root, n=300)[240:250]
    data = os.path.join(root, "data")
    img_ids, text_index = {}, {}
    for k, u in enumerate(uids):
        os.makedirs(os.path.dirname(os.path.join(data, "cad_vec", u + ".h5")), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.join(data, "images", u)), exist_ok=True)
        v = _fake_cad_vec()
        v[1:5, 1] = k                       # distinct programs, so dedup keeps them all
        with h5py.File(os.path.join(data, "cad_vec", u + ".h5"), "w") as f:
            f["vec"] = v
        for view in range(3):
            Image.new("RGB", (64, 64), (k, view * 50, 0)).save(os.path.join(data, "images", f"{u}_{view}.png"))
        img_ids[u] = [0, 1, 2]
        text_index[u] = {"expert": f"a plate {k} units wide", "beginner": "a plate"}
    splits = {"train": uids[:40], "validation": uids[40:], "test": []}
    for name, obj in [("filtered_data.json", splits), ("image_ids.json", img_ids), ("text_index.json", text_index)]:
        with open(os.path.join(data, name), "w") as f:
            json.dump(obj, f)
    cmd = [sys.executable, os.path.join(ROOT, "train_vlm_cad.py"), "--emb_dir", root, "--data", data,
           "--out_dir", root, "--name", "vlm", "--model", model, "--bs", "4", "--accum", "2", "--steps", "4",
           "--warmup", "1", "--images", "images", "--eval_every", "4", "--eval_n", "4", "--save_every", "4", "--workers", "0",
           "--device", "cpu"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    print(r.stdout[-3000:], r.stderr[-3000:])
    assert r.returncode == 0 and '"view1+expert"' in r.stdout
    assert os.path.exists(os.path.join(root, "vlm", "model", "step.json"))


def test_deepcad_to_cadquery_matches_cadlib_geometry():
    try:
        import cadquery  # noqa: F401
    except ImportError:
        print("[skip] cadquery not installed")
        return
    from mm.cadquery_io import deepcad_reference, deepcad_to_cadquery, iou, normalise, run_cadquery
    vec = _fake_cad_vec()                     # 190 x 190 square plate, one extrude
    code = deepcad_to_cadquery(vec)
    assert code.startswith("import cadquery as cq") and "Workplane(" in code and ".extrude(" in code
    shape, ref = run_cadquery(code), deepcad_reference(vec)
    assert shape is not None and ref is not None
    assert iou(normalise(ref), normalise(shape)) > 0.99
    assert run_cadquery("r = 1/0") is None and run_cadquery("import cadquery as cq\nr=cq.Workplane('XY')") is None


def test_vlm_cq_script_end_to_end():
    """train_vlm_cq.py on 6 DeepCAD + 4 CAD-Recode samples with a tiny random Qwen3-VL; val IoU is computed."""
    try:
        import cadquery  # noqa: F401
        import pandas as pd
        from PIL import Image
    except ImportError:
        print("[skip] cadquery not installed")
        return
    from mm.cadquery_io import deepcad_to_cadquery
    root = tempfile.mkdtemp()
    model = _tiny_qwen3_vl(os.path.join(root, "tiny_vlm"))
    if model is None:
        return
    data = os.path.join(root, "data")
    uids = [f"0000/{i:08d}" for i in range(8)]
    rows, status = [], {}
    for k, u in enumerate(uids):
        v = _fake_cad_vec()
        v[2:4, 1] = 150 + k                          # distinct plates
        os.makedirs(os.path.dirname(os.path.join(data, "cad_vec", u)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.join(data, "renders", u)), exist_ok=True)
        with h5py.File(os.path.join(data, "cad_vec", u + ".h5"), "w") as f:
            f["vec"] = v
        for cam in range(4):
            Image.new("RGB", (64, 64), (k * 20, cam * 40, 0)).save(os.path.join(data, "renders", f"{u}_{cam}.png"))
        status[u] = "ok"
        rows.append((u, "train" if k < 6 else "validation", deepcad_to_cadquery(v), True, 1.0))
    with open(os.path.join(data, "renders", "status.json"), "w") as f:
        json.dump(status, f)
    pd.DataFrame(rows, columns=["uid", "split", "code", "ok", "iou"]).to_parquet(os.path.join(data, "deepcad_cq.parquet"))
    rc_rows, rc_status = [], {}
    for k in range(4):
        p = f"train/batch_00/{k}.py"
        rc_rows.append(("train", p, f"import cadquery as cq\nr=cq.Workplane('XY').box({10 + k},20,30)"))
        os.makedirs(os.path.join(data, "recode_renders", "train", "batch_00"), exist_ok=True)
        for cam in range(4):
            Image.new("RGB", (64, 64), (0, 0, k * 50)).save(os.path.join(data, "recode_renders", p[:-3] + f"_{cam}.png"))
        rc_status[p] = "ok"
    pd.DataFrame(rc_rows, columns=["split", "path", "code"]).to_parquet(os.path.join(data, "cad_recode.parquet"))
    with open(os.path.join(data, "recode_renders", "status_0.json"), "w") as f:
        json.dump(rc_status, f)
    with open(os.path.join(data, "text_index.json"), "w") as f:
        json.dump({u: {"expert": f"a plate {k}", "beginner": "a plate"} for k, u in enumerate(uids)}, f)
    cmd = [sys.executable, os.path.join(ROOT, "train_vlm_cq.py"), "--data", data, "--out_dir", root, "--name", "cq",
           "--model", model, "--deepcad_cq", os.path.join(data, "deepcad_cq.parquet"),
           "--recode", os.path.join(data, "cad_recode.parquet"),
           "--recode_renders", os.path.join(data, "recode_renders"), "--bs", "4", "--accum", "1", "--steps", "3",
           "--warmup", "1", "--eval_every", "3", "--eval_n", "2", "--workers", "0", "--device", "cpu"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    print(r.stdout[-3000:], r.stderr[-3000:])
    assert r.returncode == 0 and '"iou_gt"' in r.stdout and "cad-recode 4" in r.stdout
