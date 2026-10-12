# GenCAD → text + image conditioned CAD generation

Everything new lives in `mm/`, `scripts/`, `jobs/`, `tests/` and three top-level entry points
(`train_mm_prior.py`, `eval_mm.py`, `infer_mm.py`). Upstream GenCAD is fetched by `setup.sh` and untouched except `train_gencad.py`
(added CLI overrides for epochs / batch size / lr / steps / `-img_type`).

## How GenCAD works (and where the hook goes)

```
image ──ResNet-18──► CCIP image latent (256) ──diffusion prior──► CSR latent (256) ──CSR decoder──► CAD command sequence
                    (contrastively aligned                         (bottleneck of a
                     with CSR latents)                              transformer autoencoder)
```

The CSR latent space is the right place to inject other modalities. The autoencoder already turns any 256-d
latent into a CAD sequence, so a new modality only needs a prior that maps *condition → CSR latent*. The
decoder and the CAD kernel side stay untouched.

## What was built

**Multi-condition prior** (`mm/diffusion.py`, `mm/fusion.py`, `mm/denoiser.py`)
- Frozen pretrained encoders produce the condition embeddings (`mm/cond_encoders.py`):
  - images: SigLIP2 (`google/siglip2-base-patch16-224`), which handles renders, sketches and photos
  - text: Qwen3-Embedding-0.6B. Text2CAD prompts are long and full of dimensions, and CLIP/SigLIP text towers
    truncate at 64–77 tokens. `--backend siglip` gives a shared text-image space instead.
  - GenCAD's own CCIP image latent can be added as a third modality (`ccip_image`)
- `ConditionFuser`: one token per modality plus a CLS token, in a small transformer. A missing modality gets a
  learned null token, so text-only, image-only, both and unconditional are the same model.
  Modality dropout during training (15% each, 10% all) enables classifier-free guidance.
- Denoiser: residual MLP over the 256-d latent with AdaLN-zero conditioning in every block (GenCAD's
  `ResNetDiffusion` only concatenates the condition at the input). `--cond_mode concat` is available for ablations.
- Sampling: DDIM with **per-modality guidance weights**, either `chain` (image first, then text refines; the
  InstructPix2Pix style) or `independent` (composable diffusion). All condition sets go through one batched forward.
- Optional auxiliary contrastive loss between fused condition and target latent (`--contrastive_weight`).

**Data plumbing**
- `scripts/export_gencad_embeddings.py`: CSR latents + CCIP image latents **keyed by uid**. The upstream
  path (`trainer/autoencoder_test.py::encode`) uses a *shuffled* dataloader with no ids, so CAD latents and
  image latents fed to `train_gencad.py dp` are not row-aligned; worth checking before trusting a from-scratch replication.
- `scripts/build_text_index.py`: Text2CAD annotations → `uid → {abstract, beginner, intermediate, expert, …}`,
  with column auto-detection. It joins to GenCAD's splits on DeepCAD uid and prints the coverage.
- `scripts/embed_conditions.py`: caches SigLIP2 / Qwen3 embeddings to HDF5 (all views / all prompt levels;
  one is sampled per step as augmentation).
- `scripts/make_dp_arrays.py`: correctly aligned arrays for the *upstream* `train_gencad.py dp` (replication).

**Evaluation** (`eval_mm.py`): on the test split, for `csr_recon` (autoencoder ceiling), `gencad_dp`
(released GenCAD prior: the replication number) and the new prior under every modality combination × guidance
weight: latent cosine, retrieval R@1/5/10 in CSR space, DeepCAD-style command accuracy, parameter accuracy
(±3 of 256) and invalid ratio.

**Compute** (`jobs/launch.sh`, `jobs/entry.sh`): every stage is one Hugging Face Job. The repo is mounted at
`/code`, and a private HF bucket at `/work` persists data, embeddings, checkpoints and results between jobs.

## Status

| | |
|---|---|
| CPU tests (`tests/test_mm_smoke.py`, 9 tests) | pass. On a toy task the conditioned prior reaches cosine 0.89 / R@1 1.0 vs ~0 unconditional |
| End-to-end through real GenCAD code (synthetic CAD files, random CSR/CCIP weights) | export → train → eval → infer all run |
| Real data / GPU | **blocked**: this workspace can't reach huggingface.co or drive.google.com and has no HF token |
| STEP/STL export | needs `pythonocc-core` (conda); jobs install it with `OCC=1` |
| Upstream `train_gencad.py csr/ccip/dp` | not run here (hard-coded CUDA); wired into `jobs/entry.sh` |

## Plan (run order)

Prereqs: `export HF_TOKEN=...` (write + Jobs), `./setup.sh`, run from `gencad/`.

| # | Command | What | Hardware / rough cost |
|---|---|---|---|
| 0 | `jobs/launch.sh fetch` | Drive dataset + released ckpts + Text2CAD → bucket | CPU, ~1–3 h |
| 1 | `jobs/launch.sh test` | tests on GPU incl. real SigLIP2 / Qwen3 encoders | T4, minutes |
| 2 | `jobs/launch.sh export` | uid-aligned CSR + CCIP latents with **released** ckpts | A10G, <1 h |
| 3 | `RUN=released jobs/launch.sh eval` | **replication check**: released GenCAD prior on the test split | A10G, ~1 h |
| 4 | `jobs/launch.sh embed` | SigLIP2 image + Qwen3 text embeddings (~800k images, ~600k prompts) | A10G, ~2–4 h |
| 5 | `RUN=mm_v1 jobs/launch.sh mm --modalities image text` | train the multimodal prior | A100, ~4–8 h for 200k steps |
| 6 | `RUN=mm_v1 jobs/launch.sh eval` | compare against step 3 | A10G, ~1 h |
| 7 (opt.) | `RUN=repl jobs/launch.sh csr` → `ccip` → `export` → `dp` → `eval` | full from-scratch replication | A100, multi-day |

Ablations worth running after step 6: `--modalities ccip_image` (an apples-to-apples rerun of GenCAD's prior with
the new denoiser), `--modalities image text ccip_image`, `--cond_mode concat`, `--contrastive_weight 0`, and text
backend `siglip` vs `hf-embed`.

## Known risks / open questions
- **Drive download**: `gdown --folder` caps at 50 files per folder. If the dataset folder holds loose files rather
  than archives, fetch needs a manual download plus `hf upload` into the bucket.
- **Text2CAD layout**: column names vary across releases. `build_text_index.py` prints what it detected and the
  join coverage; check that before embedding.
- **CSR decoder is the ceiling**: if `csr_recon` invalid/accuracy is poor, no prior will fix it.
- **Domain gap**: GenCAD images are clean renders. For real photos, add augmentation or render-style
  transfer; SigLIP2 should degrade more gracefully than the ResNet-18 CCIP encoder.
- **Next step beyond a prior**: fine-tune the CSR decoder to cross-attend directly to text tokens (Text2CAD-style)
  for dimension-level control, keeping the prior for global shape.

## Quick reference
```bash
python -m pytest tests/test_mm_smoke.py -q                       # CPU, ~20 s
python train_mm_prior.py --emb_dir data/emb --modalities image text --name mm_v1
python eval_mm.py --emb_dir data/emb --csr_ckpt data/ckpt/ae_ckpt_epoch1000.pth --prior results/mm_v1/last.pt \
    --gencad_dp_ckpt data/ckpt/sketch_cond_diffusion_ckpt_epoch1000000.pt
python infer_mm.py --prior results/mm_v1/last.pt --csr_ckpt data/ckpt/ae_ckpt_epoch1000.pth \
    --image part.png --text "add four M6 holes in the corners" --w_image 1.5 --w_text 3 --n 4 --out out/
```
