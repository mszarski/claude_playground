"""Generate CAD from text and/or images with the multimodal prior.

  # text only
  python infer_mm.py --prior results/mm_v1/last.pt --csr_ckpt data/ckpt/ae_ckpt_epoch1000.pth \
      --text "A rectangular plate with four corner holes" --n 4 --out out/plate
  # image only
  python infer_mm.py ... --image sketch.png --w_image 2.0
  # both: the image fixes the overall shape, the text steers details (guidance applied in --order)
  python infer_mm.py ... --image photo.png --text "add a central through hole" --w_image 1.5 --w_text 3.0
  # best-of-N: draw 8 candidates per output, keep the decoded program that best matches the condition
  python infer_mm.py ... --image photo.png --best_of 8

Outputs per sample: <out>/<i>_<j>.h5 (CAD vector), and .step/.stl/.png if pythonocc-core is installed.
"""
import argparse
import json
import os

import h5py
import torch

from mm.diffusion import MultiCondDiffusionPrior
from mm.gencad_io import decode_latents, export_shape, load_csr, vec_to_shape
from mm.rerank import best_of_n


def build_embeds(prior, ck, a, device):
    enc_info = ck.get("encoders", {})
    embeds, n_inputs = {}, None
    if a.text and "text" in prior.modalities:
        from mm.cond_encoders import TextEncoder
        info = enc_info.get("text", {})
        enc = TextEncoder(info.get("backend", "hf-embed"), info.get("model", "Qwen/Qwen3-Embedding-0.6B"),
                          device=device)
        embeds["text"] = enc(a.text)
        n_inputs = len(a.text)
    if a.image and "image" in prior.modalities:
        from mm.cond_encoders import ImageEncoder
        info = enc_info.get("image", {})
        enc = ImageEncoder(info.get("backend", "siglip"), info.get("model", "google/siglip2-base-patch16-224"),
                           device=device)
        embeds["image"] = enc(a.image)
        n_inputs = len(a.image)
    if a.image and "image_mv" in prior.modalities:  # multi-view model: all --image paths are views of one part
        from mm.cond_encoders import ImageEncoder
        info = enc_info.get("image_mv", {})
        enc = ImageEncoder(info.get("backend", "siglip"), info.get("model", "google/siglip2-base-patch16-224"),
                           device=device)
        embeds["image_mv"] = enc(a.image).float().mean(0, keepdim=True)
        n_inputs = 1
    if a.image and "ccip_image" in prior.modalities:
        assert a.ccip_ckpt, "prior uses ccip_image; pass --ccip_ckpt"
        from mm.gencad_io import load_ccip
        from utils import process_image
        csr, _ = load_csr(a.csr_ckpt, device)
        clip = load_ccip(a.ccip_ckpt, csr, device)
        img_type = enc_info.get("ccip_image", {}).get("img_type", "cad_image")
        # upstream naming is inverted: process_image(..., "image") applies Canny (for the sketch-trained CCIP),
        # process_image(..., "sketch") passes the RGB through unchanged (for the render-trained CCIP)
        mode = "image" if img_type == "cad_sketch" else "sketch"
        imgs = torch.cat([process_image(p, img_type=mode) for p in a.image]).to(device)
        with torch.no_grad():
            embeds["ccip_image"] = clip.to_visual_latent(clip.image_encoder(imgs)).float()
        n_inputs = len(a.image)
    assert embeds, f"no usable condition; prior modalities = {prior.modalities}"
    sizes = {v.shape[0] for v in embeds.values()}
    if len(sizes) > 1:  # broadcast a single text/image against several of the other
        n = max(sizes)
        embeds = {k: v.expand(n, -1) if v.shape[0] == 1 else v for k, v in embeds.items()}
        n_inputs = n
    return embeds, n_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True)
    ap.add_argument("--csr_ckpt", required=True)
    ap.add_argument("--ccip_ckpt", default=None)
    ap.add_argument("--text", nargs="*", default=[])
    ap.add_argument("--image", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=4, help="samples per input")
    ap.add_argument("--w_text", type=float, default=3.0)
    ap.add_argument("--w_image", type=float, default=2.0)
    ap.add_argument("--w_ccip_image", type=float, default=1.0)
    ap.add_argument("--order", nargs="*", default=["image", "image_mv", "ccip_image", "text"])
    ap.add_argument("--mode", default="chain", choices=["chain", "independent"])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--best_of", type=int, default=1,
                    help="draw N candidates per output and keep the one whose decoded program the prior scores "
                         "closest to the condition (needs a prior trained with --contrastive_weight > 0)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--export", nargs="*", default=["step", "stl"], help="any of step stl png")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prior, ck = MultiCondDiffusionPrior.load(a.prior)
    prior = prior.to(device).eval()
    csr, _ = load_csr(a.csr_ckpt, device)
    embeds, n_inputs = build_embeds(prior, ck, a, device)

    embeds = {k: v.repeat_interleave(a.n, 0) for k, v in embeds.items()}
    g = torch.Generator(device=device).manual_seed(a.seed)
    weights = {"text": a.w_text, "image": a.w_image, "image_mv": a.w_image, "ccip_image": a.w_ccip_image}
    kw = dict(guidance=weights, order=a.order, mode=a.mode, steps=a.steps, eta=a.eta, generator=g)
    if a.best_of > 1:
        vecs = best_of_n(prior, csr, embeds, n=a.best_of, **kw)[0]
    else:
        z = prior.sample(embeds, **kw)
        with torch.no_grad():
            vecs = decode_latents(csr, z)

    os.makedirs(a.out, exist_ok=True)
    summary = []
    for k, vec in enumerate(vecs):
        i, j = divmod(k, a.n)
        stem = os.path.join(a.out, f"{i}_{j}")
        rec = {"input": i, "sample": j, "valid_seq": vec is not None,
               "text": a.text[i] if i < len(a.text) else (a.text[0] if len(a.text) == 1 else None),
               "image": a.image[i] if i < len(a.image) else (a.image[0] if len(a.image) == 1 else None)}
        if vec is not None:
            with h5py.File(stem + ".h5", "w") as f:
                f.create_dataset("out_vec", data=vec, dtype=int)
            rec["n_commands"] = len(vec)
            if a.export:
                try:
                    shape = vec_to_shape(vec)
                except ImportError:
                    shape, a.export = None, []
                    print("[warn] pythonocc-core not installed; skipping STEP/STL export")
                rec["valid_solid"] = shape is not None
                if shape is not None:
                    for ext in a.export:
                        if ext in ("step", "stl"):
                            export_shape(shape, f"{stem}.{ext}")
                        elif ext == "png":
                            from inference_gencad import save_view
                            save_view(shape, "iso", f"{stem}.png")
        summary.append(rec)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    ok = sum(r["valid_seq"] for r in summary)
    solid = sum(r.get("valid_solid", False) for r in summary)
    print(f"[infer] {len(summary)} samples for {n_inputs} inputs: {ok} valid sequences, {solid} valid solids -> {a.out}")


if __name__ == "__main__":
    main()
