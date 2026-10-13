"""Fine-tune the image encoder end to end with the multimodal prior.

Every other prior here trains on embeddings computed once by a frozen encoder (as GenCAD's own diffusion prior
does with its CCIP latents). This script starts from a trained prior (e.g. results/mm_v1/last.pt) and its SigLIP2
vision tower and trains both on raw renders, so the image features are shaped by the prior's denoising loss
instead of by the encoder's original pretraining. Text embeddings stay the precomputed ones.

At the end it writes, under <out_dir>/<name>/:
  last.pt        the prior (loadable with MultiCondDiffusionPrior.load; its 'encoders' entry points at encoder/)
  encoder/       the fine-tuned SigLIP2 model (transformers save_pretrained)
  emb/           cond_image.h5 re-embedded with the fine-tuned encoder for the val + test uids (all views). Score it
                 with scripts/baselines.py --prior <out>/last.pt --override image=<out>/emb/cond_image.h5
                 (everything else is read from the usual --emb_dir; bucket mounts do not keep symlinks).

  python train_e2e.py --init_prior results/mm_v1/last.pt --emb_dir data/emb --data data --name e2e_v1
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from mm.cond_encoders import _as_tensor, load_rgb
from mm.data import CadLatentStore, EmbeddingStore, write_store
from mm.diffusion import MultiCondDiffusionPrior
from mm.metrics import latent_metrics

# SigLIP2 (patch16-224) preprocessing: bilinear resize to 224x224, scale to [0, 1], normalise with mean = std = 0.5
TF = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
                         transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])


class RenderDataset(Dataset):
    """(render, CAD latent, text embedding) per uid; a random view in training, view 0 for eval."""

    def __init__(self, uids, img_dir, img_ids, cad, text, eval_text_level=None):
        """eval_text_level=None: training (random view, random text level); an int: eval (view 0, that level)."""
        self.uids = [u for u in uids if img_ids.get(u) and u in cad.index]
        self.img_dir, self.img_ids, self.cad, self.text = img_dir, img_ids, cad, text
        self.level = eval_text_level

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        u = self.uids[i]
        views = self.img_ids[u]
        v = random.choice(views) if self.level is None else views[0]
        img = TF(load_rgb(os.path.join(self.img_dir, f"{u}_{v}.png")))
        t, t_ok = self.text.get(u, self.level)
        return {"pixels": img, "z": self.cad.z[self.cad.index[u]], "text": t, "text_ok": torch.tensor(t_ok)}


def embed(model, pixels):
    return F.normalize(_as_tensor(model.get_image_features(pixel_values=pixels)).float(), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_prior", required=True)
    ap.add_argument("--encoder", default="google/siglip2-base-patch16-224")
    ap.add_argument("--emb_dir", default="data/emb")
    ap.add_argument("--data", default="data")
    ap.add_argument("--name", default="e2e")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr_prior", type=float, default=1e-4)
    ap.add_argument("--lr_enc", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=2500)
    ap.add_argument("--eval_n", type=int, default=1024)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    device = a.device
    out = os.path.join(a.out_dir, a.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    from transformers import AutoModel
    prior, ck = MultiCondDiffusionPrior.load(a.init_prior)
    assert "image" in prior.modalities, f"init prior has no 'image' modality: {prior.modalities}"
    prior = prior.to(device).train()
    enc = AutoModel.from_pretrained(a.encoder).to(device)
    enc.requires_grad_(False)
    enc.vision_model.requires_grad_(True)            # get_image_features only uses the vision tower (+ its head)
    enc.train()

    with open(os.path.join(a.data, "filtered_data.json")) as f:
        splits = json.load(f)
    with open(os.path.join(a.data, "image_ids.json")) as f:
        img_ids = json.load(f)
    cad = CadLatentStore(os.path.join(a.emb_dir, "cad_latents.h5"))
    text = EmbeddingStore(os.path.join(a.emb_dir, "cond_text.h5"))
    expert = json.loads(text.attrs["variants"]).index("expert")
    img_dir = os.path.join(a.data, "images")
    train = RenderDataset(splits["train"][: a.max_items], img_dir, img_ids, cad, text)
    val = RenderDataset(splits["validation"][: a.eval_n], img_dir, img_ids, cad, text, eval_text_level=expert)
    print(f"[data] train={len(train)} val={len(val)}", flush=True)
    loader = DataLoader(train, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                        pin_memory=True, persistent_workers=a.workers > 0)

    opt = torch.optim.AdamW([{"params": prior.parameters(), "lr": a.lr_prior},
                             {"params": [p for p in enc.parameters() if p.requires_grad], "lr": a.lr_enc}],
                            weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / a.steps))))
    amp = device.startswith("cuda")
    log = open(os.path.join(out, "log.jsonl"), "a")

    @torch.no_grad()
    def evaluate(step):
        prior.eval()
        enc.eval()
        vl = DataLoader(val, batch_size=128, num_workers=min(4, a.workers))
        z, img, txt, tok = [], [], [], []
        for b in vl:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                img.append(embed(enc, b["pixels"].to(device)))
            z.append(b["z"]), txt.append(b["text"]), tok.append(b["text_ok"])
        z, img = torch.cat(z).to(device), torch.cat(img)
        txt, tok = torch.cat(txt).to(device), torch.cat(tok).to(device)
        rec = {"step": step}
        for name, emb, rows in [("image", {"image": img}, torch.ones_like(tok)),
                                ("image+text", {"image": img, "text": txt}, tok)]:
            g = torch.Generator(device=device).manual_seed(0)
            e = {m: v[rows] for m, v in emb.items()}
            zs = prior.sample(e, guidance={m: a.guidance for m in e}, steps=a.eval_steps, generator=g)
            rec[name] = {k: round(float(v), 4) for k, v in latent_metrics(zs, z[rows]).items()}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        prior.train()
        enc.train()

    evaluate(0)
    step, t0 = 0, time.time()
    while step < a.steps:
        for b in loader:
            step += 1
            z = b["z"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                img = embed(enc, b["pixels"].to(device, non_blocking=True))
                embeds = {"image": img, "text": b["text"].to(device)}
                present = {"image": torch.ones(len(z), dtype=torch.bool, device=device),
                           "text": b["text_ok"].to(device)}
                loss, logs = prior(z, embeds, present)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(prior.parameters()) + list(enc.vision_model.parameters()), 1.0)
            opt.step()
            sched.step()
            if step % 100 == 0:
                rec = {"step": step, **{k: round(float(v), 5) for k, v in logs.items()},
                       "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
            if step % a.eval_every == 0 or step == a.steps:
                evaluate(step)
            if step >= a.steps:
                break

    # save the encoder and the prior (its 'encoders' entry now points at the fine-tuned encoder)
    enc.save_pretrained(os.path.join(out, "encoder"))
    encoders = dict(ck.get("encoders", {}))
    encoders["image"] = {"backend": "siglip", "model": os.path.abspath(os.path.join(out, "encoder"))}
    prior.save(os.path.join(out, "last.pt"), encoders=encoders, step=step, args=vars(a), init_prior=a.init_prior)

    # re-embed val + test renders (every view) with the fine-tuned encoder
    enc.eval()
    uids = [u for u in splits["validation"] + splits["test"] if img_ids.get(u)]
    K = max(len(img_ids[u]) for u in uids)
    emb = np.zeros((len(uids), K, enc.config.vision_config.hidden_size), np.float16)
    valid = np.zeros((len(uids), K), bool)
    items = [(i, k, os.path.join(img_dir, f"{u}_{v}.png")) for i, u in enumerate(uids)
             for k, v in enumerate(img_ids[u][:K]) if os.path.exists(os.path.join(img_dir, f"{u}_{v}.png"))]

    class Items(Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, j):
            i, k, p = items[j]
            return TF(load_rgb(p)), i, k

    with torch.no_grad():
        for x, i, k in tqdm(DataLoader(Items(), batch_size=256, num_workers=a.workers), desc="export"):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                e = embed(enc, x.to(device)).cpu().numpy()
            emb[i.numpy(), k.numpy()] = e
            valid[i.numpy(), k.numpy()] = True
    emb_out = os.path.join(out, "emb")
    os.makedirs(emb_out, exist_ok=True)
    write_store(os.path.join(emb_out, "cond_image.h5"), uids, emb, valid, backend="siglip-e2e",
                model=os.path.abspath(os.path.join(out, "encoder")), variants=list(range(K)))
    print(f"[done] {out}: last.pt, encoder/, emb/ ({len(uids)} val+test uids re-embedded)", flush=True)


if __name__ == "__main__":
    main()
