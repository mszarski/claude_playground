# GenCAD multimodal: handoff notes for Claude Code

Goal: (1) replicate GenCAD (image → CAD, https://github.com/ferdous-alam/GenCAD), then (2) condition it on text
and images by hooking in pretrained encoders. Read `MULTIMODAL.md` for the design, the full plan and the risks.

## Layout: run `./setup.sh` first
Upstream GenCAD has no license, so it is **not committed** here. `./setup.sh` clones it into `upstream/` at a
pinned commit, applies `patches/train_gencad.patch` (CLI overrides for epochs/bs/lr/steps/img_type), and symlinks
its files (`cadlib/`, `model/`, `config/`, `utils/`, `trainer/`, `data/`, `train_gencad.py`, ...) into this folder.
Those symlinks and `upstream/` are gitignored. Run every command **from this `gencad/` folder**.

Tracked here (ours): `mm/`, `scripts/`, `jobs/`, `tests/`, `train_mm_prior.py`, `eval_mm.py`, `infer_mm.py`,
`patches/`, `setup.sh`. To change upstream code, edit it in `upstream/`, then regenerate the patch:
`git -C upstream diff > patches/<name>.patch`, and make `setup.sh` apply it.

## State (2026-10-12, second session)
- Local: `pip install torch torchvision -r jobs/requirements-job.txt && ./setup.sh`; tests pass (9, CPU). GPU
  `test` stage passes on a T4 with the real SigLIP2/Qwen3 encoders.
- Bucket `mszarski/gencad-work` holds: `data/{cad_vec,images,sketches}.tar`, split JSONs, `text_index.json`
  (Text2CAD v1.1, 98.6% of train uids), all 7 released checkpoints in `ckpt/`,
  `emb/{cad_latents,cond_ccip_image,cond_image,cond_text}.h5` (168,674 uids; text = 6 levels: abstract,
  beginner, intermediate, expert, description, keywords), and `results/<run>/{last.pt,log.jsonl,eval*.json}`.
- All priors: 200k steps, ~1.5 h on one A100. Pull results with
  `hf buckets cp hf://buckets/mszarski/gencad-work/results/<run>/eval.json <local>`.

### Results: 2000 test samples, guidance w=2 unless noted
| row | cos | R@1 | R@10 | acc_cmd | acc_param |
|---|---|---|---|---|---|
| csr_recon (upper bound) | 1.000 | 1.000 | 1.000 | 0.918 | 0.957 |
| gencad_dp (released, sketch) | 0.535 | 0.220 | 0.421 | 0.599 | 0.659 |
| upstream inference code, sketch input (`RUN=upstream jobs/launch.sh upstream`) | 0.534 | 0.227 | 0.424 | 0.597 | 0.660 |
| upstream inference code, RGB render input (what `inference_gencad.py` does) | 0.511 | 0.188 | 0.384 | 0.589 | 0.647 |
| mm_ccip: ccip_image (GenCAD features, new denoiser) | 0.641 | 0.461 | 0.637 | 0.666 | 0.748 |
| mm_img: image (SigLIP2) | 0.635 | 0.448 | 0.603 | 0.671 | 0.743 |
| mm_v1: image | 0.626 | 0.435 | 0.599 | 0.665 | 0.740 |
| mm_v1: text, random level | 0.490 | 0.162 | 0.311 | 0.557 | 0.634 |
| mm_v1: text, abstract / beginner / intermediate / expert | 0.438 / 0.515 / 0.550 / 0.570 | 0.100 / 0.203 / 0.245 / 0.254 | 0.201 / 0.360 / 0.447 / 0.478 | 0.498 / 0.576 / 0.616 / 0.650 | 0.614 / 0.633 / 0.651 / 0.671 |
| mm_v1: image+text, random level | 0.647 | 0.468 | 0.645 | 0.681 | 0.751 |
| mm_v1: image+text, expert | 0.666 | 0.503 | 0.693 | 0.695 | 0.759 |

Takeaways: our harness reproduces the released GenCAD pipeline (upstream's own inference code on the same
sketch inputs gives the same numbers within sampling noise; on RGB renders it is a bit worse). The gain over GenCAD comes from the new denoiser/training (mm_ccip ≈ mm_img), not SigLIP2. Text
helps on top of an image; more detailed prompts help more; expert text alone beats GenCAD's image prior on
acc_cmd. w=3 ≈ w=2 everywhere, w=1 a bit lower. Per-level evals: `eval_mm.py --text_level <level>`.

### Visual check (`OCC=1 RUN=visual_v1 jobs/launch.sh visual`, grids in bucket `results/visual_v1/`)
12 random test uids + 6 free-form prompts, guidance 2. Valid solids: gt 12, gencad 9, image 10, text:beginner 12,
text:expert 11, image+text:expert 10; prompts 18/18. Simple parts (plates, cylinders, pegged plates, frames) come
out right under every condition. Text recovers detailed parts (holed brackets) that image-only and GenCAD miss,
but text gets proportions wrong (boxes come out as cubes or tall blocks); the image fixes proportions.
image+text sometimes fragments a part that text alone gets right. Prompts: washer, hex nut, cylinder with a
concentric hole are clean; "plate with four holes" and "stepped shaft" are partial; "L-bracket" comes out as
disconnected plates.

### Skewed rectangles (`scripts/diagnose_rectangles.py`, report in bucket `results/rect/report.json`)
300 test models whose GT is one axis-aligned rectangle extrude (SOL, 4 lines, Ext), one sample each:
rectangle / skewed quad / other program = csr_recon 100/0/0, gencad 91.3/4.0/4.7, mm_v1 image 98.3/1.7/0,
mm_v1 image+expert 97.3/2.7/0, mm_v1 expert 98.0/2.0/0, mm_ccip 97.7/2.0/0.3, mm_img 96.0/3.0/1.0 (%).
Mechanism (from the visual-check trapezoid, uid 0068/00682829, 8 seeds): the image alone is ambiguous about the
plate's width, so mm_v1:image samples land on two modes (x=211, the GT, or x=169). The CSR decoder picks every
corner coordinate as an independent argmax token, so a latent between the modes gives corner 1 x=211 and corner
2 x=169: a trapezoid (3/8 seeds). Nothing in decoding ties consecutive corners together. Adding expert text
removes the ambiguity (8/8 exact). Fix ideas: snap near-axis-aligned edges after decoding, decode several samples
and keep the one whose render best matches the input, or constrained decoding.

### Best-of-N (`--best_of N`; `mm/rerank.py`)
Samples N latents, decodes each, re-encodes the decoded program with the CSR encoder and keeps the one the
prior's contrastive head (`MultiCondDiffusionPrior.score`) rates closest to the condition. On the 2000 test
samples, mm_v1 at w=2 (text = expert), single sample -> best-of-8: image R@1 0.435 -> 0.433, acc_cmd 0.665 ->
0.668; text R@1 0.254 -> 0.269, acc_cmd 0.650 -> 0.662; image+text R@1 0.503 -> 0.515, acc_cmd 0.695 -> 0.700.
Small gains for 8x the sampling cost, none for image only. It does not fix skewed rectangles (300-plate
diagnostic, `results/rect/report_bo8.json`: skew rate unchanged or slightly up): the contrastive head separates
different parts, not a corner that is off by a few of 256 units. Note the rectangle numbers move 2-3 points
between runs on different hardware (different noise), so treat differences that size as noise. Default stays 1.

### Stochastic sampling (`--eta`, `--samplers eta:steps`)
Root cause of skew: the CSR decoder is non-autoregressive (its inputs are constant positional embeddings, never
earlier tokens), so every coordinate is an independent argmax given z. Test of whether pred_x0 DDIM averaging
(eta=0) produces the in-between latents: all 1685 test rectangle plates (`results/rect/report_eta.json`),
skew rate eta0/s100 -> eta0.5/s100 -> eta1/s100 -> eta1/s1000: image 1.5 / 1.2 / 1.1 / 1.3 %, image+expert
1.8 / 1.1 / 1.2 / 1.5 %, expert 3.3 / 2.9 / 2.7 / 1.8 %; gencad 1.8 % skewed + 3.1 % other program. Standard
error at these rates is ~0.3 points. Full eval at eta1/s100 vs eta0 (`results/mm_v1/eval_eta1.json`): image
R@1 0.430 vs 0.435, text 0.263 vs 0.254, image+text 0.501 vs 0.503; acc_cmd/acc_param 0.5-0.8 points lower.
So averaging is a minor contributor: stochastic sampling trims skew by about a third at best and costs a
little accuracy. Default stays eta=0. The remaining lever is the decoder (autoregressive decoding, below).

### Autoregressive decoder (`mm/ar_decoder.py`, `train_ar_decoder.py`, `RUN=ar_v1 jobs/launch.sh ar`)
17.8M-param transformer that writes the program row by row from the frozen CSR latent (100k steps, ~2 h A100;
checkpoint `results/ar_v1/last.pt`; use with `--ar_decoder`). New whole-program metrics in `mm/metrics.py`:
`prog_match` (same commands, used params within +-3) and `cmd_edit` (command edit distance per GT row), because
acc_cmd compares rows by position and punishes a decoder that can insert/drop a row. 2000 test samples, w=2,
text = expert, CSR decoder vs AR decoder:
| row | prog_match CSR / AR | cmd_edit CSR / AR | acc_cmd CSR / AR | acc_param CSR / AR |
|---|---|---|---|---|
| GT latent (recon) | 0.775 / 0.794 | 0.005 / 0.014 | 0.918 / 0.853 | 0.957 / 0.952 |
| gencad_dp | 0.102 / 0.102 | 0.185 / 0.181 | 0.599 / 0.598 | 0.659 / 0.660 |
| mm_v1 image | 0.178 / 0.178 | 0.133 / 0.134 | 0.665 / 0.650 | 0.740 / 0.733 |
| mm_v1 text | 0.136 / 0.136 | 0.154 / 0.153 | 0.650 / 0.635 | 0.671 / 0.670 |
| mm_v1 image+text | 0.216 / 0.215 | 0.110 / 0.111 | 0.695 / 0.674 | 0.759 / 0.751 |
Rectangle skew with the AR decoder (step 40k, all 1685 plates, `results/rect/report_ar40k.json`): gencad 3.9 %,
image 2.7 %, image+expert 2.9 %, expert 6.4 % (vs 1.8 / 1.5 / 1.8 / 3.3 % with the CSR decoder). Conclusion: the
AR decoder reconstructs GT latents slightly better as whole programs but is no better on prior samples and skews
more. Most skew therefore comes from the prior placing latents that genuinely encode a skewed shape, which the
AR decoder follows more literally; the mode-mixing trapezoid was a special case. Default stays the CSR decoder.
Also from these metrics: whole programs right (prog_match) is 0.102 for GenCAD vs 0.178 image / 0.216
image+text for mm_v1.

### Duplicates and nearest-neighbour baselines (`scripts/baselines.py`, bucket `results/baselines/report.json`)
Of the 2000 eval test programs, 29.4 % occur verbatim in the training set and 38.5 % within +-3 units ("dup",
n=770; the rest "novel", n=1230). Nearest-neighbour baselines answer with the GT program of the closest training
model (image: view 0 vs all training views; text: expert level; oracle: closest GT latent). prog_match /
acc_cmd / acc_param:
| method | all | dup (n=770) | novel (n=1230) |
|---|---|---|---|
| knn oracle (GT latent) | 0.318 / 0.815 / 0.794 | 0.826 / 0.902 / 0.992 | 0.000 / 0.782 / 0.704 |
| knn image (SigLIP2) | 0.257 / 0.631 / 0.768 | 0.666 / 0.894 / 0.962 | 0.000 / 0.534 / 0.624 |
| knn ccip_image | 0.256 / 0.660 / 0.770 | 0.665 / 0.895 / 0.963 | 0.000 / 0.572 / 0.637 |
| knn text (expert) | 0.056 / 0.558 / 0.589 | 0.144 / 0.698 / 0.742 | 0.000 / 0.506 / 0.498 |
| gencad | 0.102 / 0.600 / 0.658 | 0.265 / 0.765 / 0.822 | 0.000 / 0.538 / 0.556 |
| mm_v1 image | 0.179 / 0.667 / 0.740 | 0.464 / 0.887 / 0.923 | 0.001 / 0.585 / 0.620 |
| mm_v1 text | 0.137 / 0.652 / 0.670 | 0.355 / 0.773 / 0.828 | 0.000 / 0.607 / 0.587 |
| mm_v1 image+text | 0.215 / 0.694 / 0.759 | 0.553 / 0.892 / 0.942 | 0.003 / 0.620 / 0.648 |
Readings: (1) whole-program matches come almost only from duplicated test parts, for every method; (2) on
duplicates, plain image retrieval beats every generator (0.666 vs 0.464-0.553 prog_match); (3) on novel parts,
mm_v1 beats GenCAD clearly (image+text acc_cmd 0.620 vs 0.538, acc_param 0.648 vs 0.556) and beats retrieval on
commands, roughly ties on parameters. Report the novel split as the headline from now on; consider a hybrid
(retrieve when the nearest training match is very close, generate otherwise).

### Multi-view prior `mm_mv` (`--modalities image_mv text`; bucket `results/baselines_mm_mv/report.json`)
image_mv = mean SigLIP2 embedding of a random 1-5 views in training (mm.data.PooledViewStore); scored at 1/3/5 views.
prog_match / acc_cmd / acc_param, same 2000 test uids:
| condition | all | dup (n=770) | novel (n=1230) |
|---|---|---|---|
| image, 1 view | 0.166 / 0.671 / 0.731 | 0.430 / 0.891 / 0.915 | 0.001 / 0.590 / 0.610 |
| image, 3 views | 0.214 / 0.673 / 0.752 | 0.555 / 0.893 / 0.942 | 0.001 / 0.592 / 0.629 |
| image, 5 views | 0.227 / 0.672 / 0.755 | 0.587 / 0.892 / 0.947 | 0.001 / 0.591 / 0.629 |
| image 1 view + text | 0.205 / 0.691 / 0.751 | 0.529 / 0.894 / 0.937 | 0.002 / 0.616 / 0.636 |
| image 5 views + text | 0.241 / 0.696 / 0.765 | 0.621 / 0.896 / 0.955 | 0.003 / 0.622 / 0.648 |
| text only | 0.129 / 0.654 / 0.669 | 0.332 / 0.778 / 0.824 | 0.001 / 0.608 / 0.587 |
More views help mostly on duplicated parts (image prog_match 0.430 -> 0.587; R@1 0.426 -> 0.494). On novel parts
they add ~2 points of acc_param for image alone and nothing once text is present (5 views + text 0.622 / 0.648
vs mm_v1 image+text 0.620 / 0.648). At 1 view mm_mv is slightly below mm_v1. Viewpoint ambiguity is therefore
not what limits novel parts.

### Decoder ceiling on the split (`RUN=baselines_ceiling jobs/launch.sh baselines --skip_knn --ar_decoder ...`)
Decoding each test part's own GT latent (the best latent a prior could produce), prog_match / acc_cmd / acc_param:
CSR all 0.775 / 0.918 / 0.957, dup 0.977 / 0.902 / 0.997, **novel 0.649 / 0.923 / 0.942**; AR decoder novel
0.666 / 0.834 / 0.931. So the latent space represents novel parts well (65 % of them exactly), while the best
prior gets 0.3 % of novel programs exactly (acc_param 0.648). The gap is entirely in mapping image/text to
the right latent: the prior (and its conditioning) is where to invest, not the autoencoder or decoder.

### GenCAD paper on frozen / staged training (arXiv 2409.16294, read 2026-10-13)
Staging is motivated by scalability and reuse ("to make our approach scalable for large-scale datasets and
models, we utilize the pre-trained CAD encoder ... instead of training from scratch"), not accuracy. CCIP trains a
ResNet-18 from scratch against the frozen CSR encoder; encoder variants (ResNet-10/34, ViT) are compared on
retrieval only; the diffusion prior is not ablated and no end-to-end or pretrained encoder is tried. Also: the
training set "contains several scaled versions of the same CAD model" (fits our 38 % duplication) and the paper
calls the CSR decoder autoregressive although it is fed constant embeddings.

### End-to-end image encoder (`train_e2e.py`, `RUN=e2e_v1 jobs/launch.sh e2e`; ~2 h A100)
Starts from mm_v1 and trains SigLIP2's vision tower (lr 1e-5) with the prior (lr 1e-4) on raw renders, 20k steps,
bs 256; re-embeds val/test renders to `results/e2e_v1/emb/cond_image.h5`; scored with
`baselines.py --override image=...` (bucket `results/baselines_e2e/report.json`). Val R@1 dipped to 0.388 at 2.5k
then recovered: image 0.463 -> 0.467, image+text 0.550 -> 0.571. Test split, frozen mm_v1 -> e2e, prog_match /
acc_cmd / acc_param:
| condition | dup (n=770) | novel (n=1230) |
|---|---|---|
| image | 0.464 / 0.887 / 0.923 -> 0.419 / 0.883 / 0.909 | 0.001 / 0.585 / 0.620 -> 0.002 / 0.595 / 0.625 |
| image+text | 0.553 / 0.892 / 0.942 -> 0.539 / 0.888 / 0.937 | 0.003 / 0.620 / 0.648 -> 0.007 / 0.629 / 0.648 |
Novel parts move by about +1 point acc_cmd and +0-0.5 acc_param (within noise for n=1230); duplicated parts get
slightly worse. So the frozen encoder is not the bottleneck either: frozen SigLIP2, GenCAD's CAD-trained CCIP and
an end-to-end-tuned SigLIP2 all land within ~1 point on novel parts.

### Text -> program from prompt tokens (`mm/text_cad.py`, `train_text_cad.py`, `RUN=text_cad_v1 jobs/launch.sh textcad`)
The AR program decoder cross-attends to per-token hidden states of a frozen Qwen3-Embedding-0.6B cut to 8 layers;
no pooled embedding, no latent, no prior. 18.2M trainable params; 40k steps, bs 128, random prompt level per
sample (abstract/beginner/intermediate/expert/description); ~3.6 h A100. Test split (`baselines.py --text_cad`,
bucket `results/baselines_text_cad/report.json`), prog_match / acc_cmd / acc_param:
| model, prompt | all | dup (n=770) | novel (n=1230) |
|---|---|---|---|
| text_cad, expert | 0.242 / 0.673 / 0.786 | 0.588 / 0.786 / 0.919 | **0.026 / 0.631 / 0.718** |
| text_cad, intermediate | 0.176 / 0.607 / 0.730 | 0.451 / 0.739 / 0.888 | 0.005 / 0.558 / 0.644 |
| text_cad, beginner | 0.128 / 0.563 / 0.685 | 0.329 / 0.705 / 0.838 | 0.002 / 0.510 / 0.596 |
| text_cad, abstract | 0.054 / 0.487 / 0.637 | 0.142 / 0.659 / 0.767 | 0.000 / 0.423 / 0.548 |
| mm_v1 text (pooled, prior), expert | 0.137 / 0.652 / 0.670 | 0.355 / 0.773 / 0.828 | 0.000 / 0.607 / 0.587 |
| best image+text prior (mm_mv 5 views + expert) | 0.241 / 0.696 / 0.765 | 0.621 / 0.896 / 0.955 | 0.003 / 0.622 / 0.648 |
First change that moves novel parts: from expert prompts, novel acc_param 0.587 -> 0.718 (+13 points), whole
programs right 0 -> 2.6 % (32 of 1230), and expert text alone now beats the best image+text prior on novel
parts on every metric. Duplicated parts also improve (0.355 -> 0.588). Less detailed prompts degrade as expected;
abstract prompts carry no dimensions. Command structure is the weaker half (novel acc_cmd 0.631 vs ceiling
0.923), so mistakes are now mostly wrong sketch structure, not wrong numbers.

### Image (+ text) -> program from patch tokens (`mm/token_cad.py`, `train_token_cad.py`, `RUN=token_cad_v1 jobs/launch.sh tokencad`)
Same AR decoder, cross-attending to the 196 SigLIP2 patch states of a render (frozen) plus, half the time, Qwen
prompt tokens. 18.5M trainable; 40k steps bs 128, ~4.2 h A100. Test split (`baselines.py --token_cad`, bucket
`results/baselines_token_cad/report.json`), prog_match / acc_cmd / acc_param:
| model, input | dup (n=770) | novel (n=1230) |
|---|---|---|
| token_cad, image | 0.373 / 0.789 / 0.870 | 0.002 / 0.581 / 0.627 |
| token_cad, image + expert | 0.603 / 0.812 / 0.927 | 0.018 / 0.647 / 0.712 |
| token_cad, expert only (never trained without an image) | 0.171 / 0.706 / 0.808 | 0.005 / 0.597 / 0.662 |
| text_cad, expert | 0.588 / 0.786 / 0.919 | 0.026 / 0.631 / 0.718 |
| mm_v1 prior, image (pooled) | 0.464 / 0.887 / 0.923 | 0.001 / 0.585 / 0.620 |
| mm_v1 prior, image + expert (pooled) | 0.553 / 0.892 / 0.942 | 0.003 / 0.620 / 0.648 |
Patch tokens do not help images: image-only novel parts are where every image model has landed (0.58-0.60 /
0.62-0.63), and on duplicates image-only is worse than the pooled prior (recognition of known parts suffers).
With a prompt, image + expert ~= text_cad expert on novel parts (+1.6 acc_cmd, -0.6 acc_param, -0.8 prog_match):
the numbers come from the text, the image adds a little structure. Conclusion: the renders do not carry enough
information to pin down novel programs (isometric view, unknown scale, hidden features); detailed text does.

### !! GenCAD's renders mostly do not show the GT program (`scripts/check_views.py`, 2026-10-13)
Every uid has 5 renders `<uid>_{0..4}.png`. They are NOT 5 camera angles: the camera is the same isometric view and
the part has different proportions in each. Silhouette test on 80 uids (GT solid projected along every isometric
diagonal, both crops normalised to remove position/scale, `results/check_views/report.json`): IoU with file `_0`
0.988 on average (best of the 5 for 78/80 uids, >0.95 for 97.5%); files `_1.._4` 0.74-0.87 (>0.95 for only 7-21%).
`image_ids.json` lists the ids **shuffled** (e.g. `[4, 2, 3, 0, 1]`), and every script here (embeddings variant 0,
eval_mm, baselines, token_cad, visual_check, upstream inference) took the first listed id: **the true render for
only 19.4% of test parts**. Training sampled a random one of the 5 (20% correct), as upstream GenCAD does. So every
image result above was scored, ~80% of the time, on an image of a differently proportioned part, and trained with
the same label noise. This is the likely reason image models never got novel parts right ("renders do not carry
enough information" was wrong). Fixes: `baselines.py --true_view` (query with file `_0` / its stored variant);
`scripts/render_views.py` re-renders every GT program from 4 fixed cameras (iso top, iso bottom, front, top) into
`/work/data/renders.tar`, used by default by `train_vlm_cad.py`.

### Geometric metrics (`scripts/geom_eval.py`) and the VLM program writer (`mm/vlm_cad.py`, `train_vlm_cad.py`)
prog_match counts a correct shape built by a different (or slightly off) program as wrong; the field reports IoU,
median Chamfer distance and invalidity ratio (CAD-Recode / cadrille protocol: GT normalised to [-0.5, 0.5]^3, the
prediction gets the same transform; CD on 8192 points x1e3). `baselines.py --save_preds` + `jobs/launch.sh geom`.
Reference (cadrille, arXiv 2505.22914, DeepCAD test): single image IoU 81.7 / CD 0.21 (SFT, trained on CAD-Recode
procedural data); 4-view grid trained on DeepCAD only 78.2 IoU, with CAD-Recode data 86.1, + RL 92.2. Their model is
Qwen2-VL-2B fully fine-tuned to write CadQuery code. `mm/vlm_cad.py` does the same with Qwen3-VL-2B-Instruct writing
our DeepCAD program as text (`S`, `L x y`, `A x y alpha f`, `R x y r`, `E <11 args>`), inputs 1 view or a 2x2 grid of
4 views (448 px, 196 visual tokens) plus an optional prompt. `RUN=vlm_cad_v1 jobs/launch.sh vlmcad` (H200).

### Baseline in geometric terms (`results/baselines_true/{report,geom}.json`, test 2000, true view `_0`)
Re-scoring the existing models with the correct render (`--true_view`) changes nothing (mm_v1 image novel acc_cmd /
acc_param 0.584 / 0.623 vs 0.585 / 0.620): trained on 80% wrong views, they learned to ignore proportions. IoU %
(invalid = 0) / median CD x1e3 / invalid %:
| method | dup (n=770) | novel (n=1229) |
|---|---|---|
| csr_recon (GT latent decoded) | 99.0 / 0.14 / 0.0 | 84.0 / 0.20 / 2.0 |
| knn image (retrieval) | 73.0 / 0.21 / 0.0 | 18.4 / 98.7 / 0.0 |
| gencad (released) | 39.2 / 18.2 / 3.2 | 11.1 / 124 / 10.0 |
| mm_v1 image | 59.6 / 0.49 / 0.8 | 15.0 / 110 / 10.6 |
| mm_v1 image+expert | 70.9 / 0.27 / 0.6 | 16.5 / 91.8 / 9.4 |
| text_cad expert | 69.4 / 0.27 / 0.4 | 23.8 / 77.3 / 4.5 |
| token_cad image | 53.2 / 3.2 / 0.8 | 17.7 / 93.6 / 3.6 |
| token_cad image+expert | 70.8 / 0.24 / 0.9 | 23.7 / 70.8 / 5.0 |
On novel parts every model is at 11-24 % IoU with median CD ~70-130: wrong shapes, not near misses. cadrille reports
78-82 % IoU / CD ~0.2 on the whole DeepCAD test set from images. That is the gap to close.

### token_cad retrained on the true render (`--true_view`, `results/token_cad_true`, test `results/baselines_token_true`)
Same model and recipe as token_cad_v1, but trained and evaluated on render file `_0` only. Test, image only:
prog_match dup 0.356 -> **0.590**, novel 0.002 -> 0.006; IoU dup 53.2 -> **69.5**, novel 17.7 -> **21.1**
(image + expert: IoU dup 75.4, novel 23.8). The data fix makes recognition of known parts much better and novel parts
only slightly better: a small decoder trained from scratch on DeepCAD recalls training parts rather than reading new
geometry. cadrille's two levers against exactly this are a pretrained VLM (vlm_cad_v1, running) and ~1M procedurally
generated programs (CAD-Recode v1.5, `filapro/cad-recode-v1.5`, CadQuery code, CC BY-NC 4.0; cadrille's weights are
`maksimko123/cadrille`).

### vlm_cad_v1: Qwen3-VL-2B writing programs from our renders (`results/vlm_cad_v1`, H200, 15k steps x 32)
Full fine-tune, dedup'd train (152k -> 113k programs), our fixed-camera renders, mixed inputs (1 view / 2x2 grid of
4 views / + random-level prompt / prompt only). ~2.1 s/step on an H200. Test at the **step-5k checkpoint**
(`results/baselines_vlm5k`), novel parts (n=1230): prog_match / acc_cmd / acc_param, IoU / median CD / IR:
| input | novel program | novel IoU / CD / IR | dup IoU |
|---|---|---|---|
| 1 view | 0.011 / 0.557 / 0.636 | 24.5 / 65.9 / 2.9 | 52.0 |
| 4 views | 0.013 / 0.595 / 0.658 | 27.0 / 51.8 / 2.4 | 52.7 |
| 1 view + expert | **0.104 / 0.758 / 0.777** | **36.2 / 33.5 / 0.9** | 68.1 |
Previous best on novel parts: text_cad expert 0.026 / 0.631 / 0.718, IoU 23.8; token_cad true view + expert IoU 23.8.
The pretrained VLM is the first model that generalises: 4x the novel exact-program rate with a prompt and +12 IoU.
Image-only novel IoU 24-27 (vs 17.7-21.1 for token_cad). Lower dup scores than token_cad are expected (no
duplicates in training).

**Final checkpoint (step 15k, `results/baselines_vlm_final/{report,geom}.json`)**, test 2000:
| input | novel program (prog / cmd / param) | novel IoU / CD / IR | dup IoU | all IoU |
|---|---|---|---|---|
| 1 view | 0.052 / 0.622 / 0.701 | 31.9 / 41.2 / 0.4 | 64.4 | 44.4 |
| 4 views | 0.076 / 0.650 / 0.731 | 33.5 / 37.1 / 0.8 | 65.1 | 45.7 |
| 1 view + expert | **0.180 / 0.805 / 0.817** | **42.8 / 16.8 / 0.6** | 74.6 | 55.1 |
| expert only | 0.160 / 0.804 / 0.811 | 40.3 / 23.3 / 0.4 | 71.6 | 52.4 |
5k -> 15k steps: novel IoU 24.5 -> 31.9 (1 view), 27.0 -> 33.5 (4 views), 36.2 -> 42.8 (+ expert); invalid
outputs < 1 %. vs GenCAD (all 21.9, novel 11.1) and the best earlier model (token_cad true view + expert: all
43.7, novel 23.8). Still far from cadrille (~78-82 IoU on DeepCAD, whole test set) and from the decoder ceiling
(novel 84.0). Text still carries most of the signal: expert prompt alone 40.3 novel IoU vs 4 views 33.5.

### vlm_cq_v1: Qwen3-VL-2B writing CadQuery, DeepCAD (converted) + CAD-Recode (`train_vlm_cq.py`, H200, 30k x 32)
Data: DeepCAD programs converted to CAD-Recode-style CadQuery (`scripts/convert_deepcad.py`, `/work/data/deepcad_cq
.parquet`; builds 99 %, test fidelity vs cadlib mean IoU 0.977, 87 % > 0.99) - 98,729 unique train programs with
fidelity >= 0.95; CAD-Recode v1.5 (`/work/data/cad_recode.parquet`, 982,847 programs; rendered with our cameras by
`scripts/render_cad_recode.py`, 99.98 % build, `/work/data/recode_renders_{0..3}.tar`) - 981,667 train programs.
Sampling 30 % DeepCAD / 70 % CAD-Recode; prompts only on DeepCAD samples. ~1.8 s/step. Validation = 256 DeepCAD val
parts (dup and novel mixed), generated code executed, IoU vs GT (`scripts/cq_iou.py` protocol):
| step | 1 view IoU gt/own, IR | 4 views | 1 view + expert |
|---|---|---|---|
| 5k | 65.8 / 66.0, 4.7 % | 72.4 / 72.0, 5.5 % | 72.8 / 73.8, 7.0 % |
| 10k | 74.0 / 73.9, 3.1 % | 79.1 / 79.7, 4.3 % | 80.5 / 81.6, 3.9 % |
| 15k | 79.1 / 79.3, 4.3 % | 84.8 / 84.7, 2.7 % | 85.5 / 86.0, 2.7 % |
For scale, vlm_cad_v1 (DeepCAD tokens, no CAD-Recode) final checkpoint on the test set, all parts: 1 view 44.4,
4 views 45.7, + expert 55.1 (different split, so read as indicative until the test eval). Image-only jumps by
~20-27 points: the procedural CAD-Recode data is what teaches reading geometry from renders. Invalid code 5-7 %.
Final validation (step 30k): 1 view 86.0, 4 views 91.1, 1 view + expert 90.0 IoU, invalid 0.8-2.0 %.

**Test split, final checkpoint** (superseded: see the re-scored table below) (`results/vlm_cq_v1_test/{preds.pkl,cq_iou.json}`; 2000 test uids, 3 whose GT does
not build dropped, so n = 1997 / dup 770 / novel 1227). IoU % with GT normalisation (gt) / each solid fitted to its own
box (own), invalid code IR %; invalid = IoU 0:
| input | all IoU gt / own, IR | dup IoU gt, IR | novel IoU gt / own, IR | vlm_cad_v1 novel / all IoU |
|---|---|---|---|---|
| 1 view | 78.2 / 78.0, 8.0 | 85.1, 6.8 | 73.9 / 73.5, 8.8 | 31.9 / 44.4 |
| 4 views | **86.6** / 86.5, 4.0 | 92.4, 3.0 | **82.9** / 82.6, 4.6 | 33.5 / 45.7 |
| 1 view + expert | 85.2 / 85.6, 4.5 | 92.0, 3.1 | 81.0 / 81.5, 5.4 | 42.8 / 55.1 |
| expert only | 76.0 / 77.9, 5.4 | 86.2, 2.7 | 69.6 / 71.7, 7.0 | 40.3 / 52.4 |
Novel-part IoU goes from 33.5 (vlm_cad_v1, 4 views) to 82.9: +49 points from CAD-Recode data plus CadQuery output.
The test numbers match validation (4 views 91.1 val vs 86.6 test all; val mixes dup and novel). 4 views now beat
1 view + expert prompt: geometry comes from the images, not the text, a reversal of every earlier model. Level with
cadrille's DeepCAD-only + CAD-Recode SFT on 4 views (86.1) and above its single-image SFT (81.7), though cadrille
scores the whole DeepCAD test set and we score our 2000-uid subset; still below cadrille + RL (92.2). Remaining
loss is mostly invalid code (4-9 % at IoU 0): with 4 views, valid outputs average ~90 IoU. Next levers: retry /
best-of-N on code that fails to build, RL on IoU (cadrille's +6), longer training (val still rising at 30k).

### !! vlm_cq_v1 test numbers above are 2-5 points too low (scoring job lost parts)
Re-scoring the same `results/vlm_cq_v1_test/preds.pkl` locally (4 CPUs, `cq_iou_local{,2}.json`) gives, IoU % gt / own,
IR % (all | dup | novel):
| input | all | dup | novel |
|---|---|---|---|
| 1 view | 83.0 / 82.9, 2.3 | 90.2 / 90.4, 1.0 | 78.6 / 78.2, 3.1 |
| 4 views | **89.1 / 89.1, 1.3** | 94.9 / 95.2, 0.5 | **85.5 / 85.2, 1.7** |
| 1 view + expert | 87.9 / 88.4, 1.6 | 94.6 / 94.9, 0.4 | 83.7 / 84.2, 2.3 |
| expert only | 78.1 / 80.1, 2.8 | 88.3 / 89.9, 0.3 | 71.7 / 73.9, 4.4 |
The cloud `cqiou` job had marked an extra 2-6 % of parts invalid; the local invalid rates match training-time
validation (view4 1.2 %), and Claude's predictions score identically in two local runs, so the job most likely
lost parts to worker timeouts/crashes under load. `cq_iou.py` now prints `timeout/crash=<n>` per method; if it is
not ~0, rerun with fewer `--workers`. Use the local numbers from now on.

### Claude zero-shot baseline (`scripts/claude_baseline.py`, bucket `results/claude_baseline{,_v1e}/`)
claude-opus-5-5, effort high, Message Batches API, same inputs as vlm_cq (2x2 grid of our 4 fixed-camera renders, or
1 view + the expert prompt), plus the camera layout and a CadQuery style example in the prompt; no fine-tuning, no
DeepCAD data. view4: 2000/2000 succeeded in ~70 min, ~$18; view1+expert: ~3.5 h, ~$27. Scored locally with
`scripts/cq_iou.py` (n = 1997 / dup 770 / novel 1227), no timeouts. IoU % gt / own, IR %:
| method | all | dup | novel |
|---|---|---|---|
| Claude, 4 views | 83.2 / **83.4**, 3.2 | 87.8 / 87.8, 2.2 | 80.4 / **80.6**, 3.7 |
| vlm_cq_v1, 4 views | 89.1 / **89.1**, 1.3 | 94.9 / 95.2, 0.5 | 85.5 / **85.2**, 1.7 |
| Claude, 1 view + expert | 80.2 / **80.5**, 1.4 | 80.9 / 81.0, 0.9 | 79.8 / **80.2**, 1.6 |
| vlm_cq_v1, 1 view + expert | 87.9 / **88.4**, 1.6 | 94.6 / 94.9, 0.4 | 83.7 / **84.2**, 2.3 |
| cadrille (DeepCAD test, whole set), 4 views | SFT DeepCAD + CAD-Recode 86.1; + RL 92.2 | | |
Readings: (1) vlm_cq leads by 5.7 (4 views) and 7.9 (1 view + expert) points own IoU overall, 4.6 / 4.0 on novel
parts; the dup gap is larger (7-14) since vlm_cq has seen those shapes. (2) gt ~= own for Claude: from the camera
description it puts parts in the right frame. (3) Text barely helps Claude (novel 80.6 with 4 views vs 80.2 with
1 view + expert), while it lifts vlm_cq's 1 view from 78.2 to 84.2. (4) Claude's code is short (view4 median 239
chars; 44 % circles, 11 % arcs, 40 % cuts) and fails to build 1.4-3.7 % of the time.

**Fallback** (`scripts/fallback.py`, per-part rows from `cq_iou.py --rows`; bucket `results/fallback/`): vlm_cq's
part, or Claude's when vlm_cq's code fails to build. Own IoU all / novel, IR:
| input | vlm_cq alone | + Claude fallback | oracle best of the two per part |
|---|---|---|---|
| 4 views | 89.1 / 85.2, 1.3 % | **89.8 / 86.2, 0.1 %** (Claude used on 25 parts) | 92.6 / 90.1 |
| 1 view + expert | 88.4 / 84.2, 1.6 % | **89.3 / 85.7, 0.1 %** (31 parts) | 94.2 / 92.5 |
The fallback is a free +0.7-0.9 (removes almost all invalid outputs). The oracle (+3.5-5.8) says the two models err on
different parts, so a selector that picks between candidates (e.g. render agreement with the input) could gain more.

### RL on IoU (`train_vlm_rl.py`, `RUN=vlm_rl_v2 jobs/launch.sh vlmrl`, H200; done, small gain)
Starts from `results/vlm_cq_v1/model`. On-policy group policy gradient (Dr. GRPO: advantage = reward - group mean,
no std normalisation, constant token normaliser, no KL, one update per batch so no ratio/clipping). Each step: 16
DeepCAD train parts (104,625 distinct programs), mode view1 / view4 / view1+expert = 0.4 / 0.4 / 0.2, 8 samples each
at temperature 1.0, reward = `cq_iou.score` IoU (GT normalisation) or -1 if the code fails. lr 1e-6 AdamW, 300 steps,
save + validate (same 256 val parts and greedy decoding as train_vlm_cq.py) every 100 steps. ~60-115 s/step, mostly
sampling (HF generate). Step-0 validation (SFT model, this harness): view1 85.8, view4 90.5, view1+expert 90.2 IoU gt.
Test after: `train_vlm_cq.py --predict results/vlm_rl_v2/model --uids_from results/baselines_vlm_final/preds.pkl
--out_preds results/vlm_rl_v2_test/preds.pkl`, then `cq_iou.py` locally (see the timeout note above).
Result (300 steps; validation every 100 steps flat: view4 90.3 / 90.7 / 90.8 vs 91.1 SFT). Test, scored locally
(`results/vlm_rl_v2_test/{cq_iou_local.json,rows.pkl}`, no timeouts), IoU gt / own, IR (all | novel), SFT -> RL:
view1 83.0 -> **84.5** / 84.5, IR 1.7 | novel 78.6 -> 79.7; view4 89.1 -> 89.2, IR 1.5 | novel 85.5 -> 85.2;
view1 + expert 87.9 -> 88.0 | novel 83.7 -> 83.9; expert 78.1 -> 78.8 | novel 71.7 -> 72.6. So +1.5 for 1 view and
nothing elsewhere: not worth more H200 time in this form.

### Render-agreement selection (`scripts/select_render.py`, bucket `results/select/{view4,view1+expert}/`)
Pick one program per part among candidates (greedy SFT, greedy RL, Claude, 8 samples of the SFT model at temp 0.8 from
`results/vlm_cq_samples/preds.pkl`) by how well the candidate's render matches the **input** views (4 views for view4,
view 0 only for view1 modes): mean silhouette IoU (`sil`) or mean grey-level difference (`pix`) at 128 px, same
cameras/framing as `scripts/render_views.py`. 2000 test parts x 11 candidates ran locally in ~30 min per mode on 4
CPUs (~0.08 s per candidate including the GT IoU); 2-4 timeouts per mode. Own IoU %, all / novel (IR ~0 when selecting):
| candidates, selector | view4 | view1 + expert |
|---|---|---|
| SFT greedy alone | 89.1 / 85.2 | 88.4 / 84.2 |
| one SFT sample alone (temp 0.8) | 86.4 / 82.0 | 86.8 / 82.0 |
| SFT + Claude fallback (previous best) | 89.8 / 86.2 | 89.3 / 85.7 |
| SFT + RL + Claude, pick by pix | 91.4 / 88.8 | 93.0 / 91.5 |
| **local only**: SFT + 8 samples, pick by pix | **91.7 / 89.3** | 91.2 / 88.3 |
| all 11, pick by sil | 91.0 / 89.1 | 91.6 / 90.0 |
| all 11, pick by pix | **91.9 / 89.8** | **93.2 / 91.8** |
| oracle: SFT + 8 samples | 94.3 / 92.1 | 93.3 / 90.8 |
| oracle: all 11 | 95.0 / 93.1 | 95.7 / 94.3 |
Readings: (1) selection works: +2.8 (view4) / +4.8 (view1+expert) over SFT greedy, +2.1 / +3.9 over the Claude
fallback; `pix` beats `sil` everywhere (shading sees holes and steps a silhouette misses). (2) Fully local best-of-9
(SFT + its own samples) gets view4 to 91.7 / 89.3 with no Claude; with 1 view + text Claude adds more (91.2 -> 93.2):
from one view the model's samples are less diverse in the right direction and Claude reads the prompt differently.
(3) Picks (all 11, pix), view4: sft 440, claude 373, rl 65, samples ~125-166 each; view1+expert: claude 613, sft 510.
(4) Own samples beat greedy by > 0.02 IoU on 29.5 % of parts and by > 0.10 on 13 % (both modes); every part whose
greedy code fails to build has a valid sample (25/25, 30/31). The oracle over own samples (94.3 vs 89.1) is the
headroom for rejection-sampling fine-tuning (sample on train parts, keep the best by GT IoU, fine-tune): the next step.
Caveat: the selector compares against renders from our own renderer, so on real photos `pix` would need a learned or
edge-based comparison. Not run: extra Claude samples as candidates (kept the model local by choice).

### Rejection-sampling fine-tuning (RFT; `scripts/sample_rft.py`, `scripts/build_rft.py`, `train_vlm_cq.py --rft`; round 1 done)
Goal: teach the model to produce its best-of-N answer greedily, so it stays one local model with no selector or Claude.
1. **Sampling** (`RUN=rft_v1 INIT=results/vlm_cq_v1/model jobs/launch.sh rftsample --n_parts 30000`, H200, 1.6 h):
   vLLM 0.31 (needs gcc in the image; FlashInfer's sampler is disabled because its JIT needs nvcc; 4096-token
   context, 10 over-long expert prompts dropped). 30k random distinct DeepCAD *train* programs, one input each
   (view1 with a random camera 0.4 / view4 0.4 / view1+expert 0.2), greedy + 8 samples at T 0.8, each scored vs GT in a
   CPU pool overlapping the next chunk's generation. Check on the 256 val parts (`results/rft_check`): vLLM greedy
   view4 90.7 IoU (HF generate 90.5-91.1), best-of-9 94.9. Train parts: greedy 0.76, best-of-9 0.87 mean IoU (lower
   than test because a third use a single random camera; front/top alone are ambiguous); a sample beats greedy by
   > 0.02 on ~42 % of parts. Timeouts 2-14 per 18k programs.
2. **Targets** (`results/rft_v1/rft.parquet`, stats in `rft_stats.json`): best program per part by GT IoU, kept if >= 0.9
   (ties -> greedy, then shortest): 20,680 targets, mean IoU 0.978; 12,787 are samples, 5,857 improve greedy by > 0.02,
   2,468 by > 0.10, 160 fix invalid greedy code; view4 9,667 / view1 6,535 / view1+expert 4,478.
3. **Fine-tune** (`RUN=vlm_rft_v1 STEPS=3000 jobs/launch.sh vlmcq --model results/vlm_cq_v1/model --rft
   results/rft_v1/rft.parquet --rft_p 0.5 --lr 2e-5 --warmup 100 --eval_every 1000`): half of each batch RFT targets
   with their exact input, half the original DeepCAD/CAD-Recode mix. ~2.1 s/step on an H200. Validation (gt IoU,
   steps 1000 / 2000 / 3000; SFT 86.0 / 91.1 / 90.0): view1 85.4 / 86.0 / 87.0, view4 89.3 / 90.6 / 90.8,
   view1+expert 90.6 / 89.1 / 90.6.
4. **Test** (`results/vlm_rft_v1_test/{preds.pkl,cq_iou_local.json,rows.pkl}`, scored locally, no timeouts). Own IoU,
   all / novel, and the paired difference to SFT on the same parts (+- standard error):
| input | SFT | RFT round 1 | paired diff all | paired diff novel |
|---|---|---|---|---|
| 1 view | 82.9 / 78.2 | **85.1 / 79.3** | **+2.18 +- 0.40** | +1.16 +- 0.53 |
| 4 views | 89.1 / 85.2 | 89.6 / 85.8 | +0.55 +- 0.31 | +0.58 +- 0.46 |
| 1 view + expert | 88.4 / 84.2 | 88.2 / 83.8 | -0.17 +- 0.28 | -0.39 +- 0.43 |
| expert only | 80.1 / 73.9 | 80.4 / 74.5 | +0.33 +- 0.36 | +0.64 +- 0.54 |
   Readings: a real gain only for 1 view (+2.2, like RL's +1.5 there); 4 views +0.5 is marginal; text modes flat.
   Many parts move both ways (view4: 185 better / 134 worse by > 0.05), so the update reshuffles as much as it
   improves. Likely reasons: 7.9k of the 20.7k targets are the greedy program itself (self-distillation, no new
   information) and the 5.9k real improvements are a small share of the batches; much of the test oracle gap
   (94.3 vs 89.1) is sampling luck that one greedy answer cannot capture. Cost: sampling ~1.6 h H200, fine-tune
   ~2.6 h H200, test predictions ~3 h A100 (HF generate; switch `vlmpred` to vLLM next time).

### DPO on rejection-sampling pairs (`train_vlm_dpo.py`, `RUN=... INIT=... CHUNKS=results/rft_v1 jobs/launch.sh vlmdpo`)
Pairs from the RFT samples (no new sampling): chosen = best program by GT IoU (>= 0.9), rejected = up to 2 programs
>= 0.15 IoU worse or failing to build, greedy first: 13,147 pairs from 7,569 parts (rejected is greedy in 2,045;
chosen IoU 0.964, rejected 0.389, 11 % invalid). Sigmoid DPO (beta 0.1) vs a frozen reference (= init) + nll_weight x
token NLL of the chosen program; 32 pairs/step, ~5 s/step on an H200. CPU check: with reference = policy the loss is
exactly log 2.
**v1** (init RFT round 1, lr 1e-6, nll 0.2, 500 steps): pair accuracy 0.68 -> 0.85, margin up to ~15 nats, but
validation got worse and the invalid rate rose (gt IoU view1 / view4 / view1+expert, invalid on view1+expert):
RFT 87.0 / 90.8 / 90.6, 0.8 % -> step 250 86.0 / 89.8 / 87.2, 5.5 % -> step 500 86.0 / 88.8 / 84.5, 7.8 %. Classic DPO
likelihood displacement: it lowers the rejected programs faster than it raises the chosen ones, mass moves to unseen
(broken) code; chosen NLL rose 0.047 -> 0.061. Not test-scored.
**v2** (init RFT, nll 1.0, lr 3e-7, 300 steps, `results/vlm_dpo_v2{,_test}`): pair accuracy ~0.8, margin ~1 nat (the NLL
anchor keeps it close to the reference); validation step 100 / 200 / 300 view1 87.2 / 86.6 / 88.0, view4 90.4 / 91.2 /
91.0, view1+expert 90.4 / 90.8 / 90.5, invalid normal. Test (vLLM predictions, local scoring), own IoU all / novel:
view1 85.3 / 79.7, view4 89.5 / 85.6, view1+expert 88.1 / 83.9, expert 80.5 / 74.6. Paired vs RFT round 1: +0.30 +- 0.17,
-0.09 +- 0.18, +0.02 +- 0.19, +0.03 +- 0.19 (all), i.e. no change; vs SFT +2.4 (view1), +0.5 (view4), flat with text.
Conclusion: post-training on the model's own samples (RL, RFT, DPO) has plateaued at ~89.5 own IoU single-answer for
4 views; the 5-point best-of-N headroom does not transfer into the greedy answer with these methods at this scale.

### vLLM test predictions (`scripts/predict_vllm.py`, `INIT=<model dir> RUN=<name> jobs/launch.sh vllmpred`)
Same inputs/prompts/output format as `train_vlm_cq.py --predict`. 4 modes x 2000 test parts in ~5 min on an A100
(HF generate: ~3 h). Validated on the RFT model: local cq_iou own IoU all view1 / view4 / view1+expert / expert
85.0 / 89.6 / 88.1 / 80.5 vs HF-generate predictions 85.1 / 89.6 / 88.2 / 80.4. Use it for all test predictions.

### Online RL (GRPO; `train_vlm_grpo.py`, `RUN=vlm_grpo_v1 STEPS=1000 jobs/launch.sh vlmgrpo`, H200; best single model)
vLLM (in-process, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, 30 % of the GPU) samples from the current policy; the HF model
(fp32 AdamW) takes one step per batch on exactly the sampled tokens (on-policy, no ratio/clipping, no KL); new weights
are loaded into vLLM after every step (`model_runner.get_model().load_weights`, then `reset_prefix_cache`). Sync check:
log-prob of a fixed program by vLLM vs HF -0.751 / -0.780 before, -0.757 / -0.755 after step 1. Each step 32 DeepCAD
train parts (view1 random camera 0.4 / view4 0.4 / view1+expert 0.2) x 8 samples at T 1.0, reward GT IoU (invalid
-0.5), GRPO advantage (r - mean) / (std + 1e-4) per group, loss -mean(A x mean token log-prob), lr 2e-6 constant, init
RFT round 1. ~25 s/step (sampling 3 s, scoring 4-20 s, update 14 s); validation through vLLM in ~35 s. 1000 steps
(~7 h, ~$35): train reward 0.728 -> 0.76-0.88, invalid samples 3.1 % -> 2.0 %; validation noisy-flat to slightly up
(view1 87.2 -> 87.4-87.7, view4 91.3 -> 91.0-92.3). **Test** (`results/vlm_grpo_v1_test`, vLLM greedy, local scoring),
own IoU all / novel, invalid %, and paired difference to SFT and to RFT round 1 (all parts, +- s.e.):
| input | GRPO | IR | vs SFT | vs RFT |
|---|---|---|---|---|
| 1 view | 85.3 / 80.4 | 1.3 | +2.39 +- 0.41 | +0.30 +- 0.33 |
| 4 views | **90.5 / 87.2** | 1.1 | **+1.45 +- 0.35** (novel +1.97) | +0.89 +- 0.32 |
| 1 view + expert | 89.3 / 85.7 | 1.1 | +0.97 +- 0.30 | +1.21 +- 0.26 |
| expert only | 81.3 / 75.7 | 1.8 | +1.26 +- 0.34 | +0.85 +- 0.31 |
First post-training that helps every mode on test, mostly on novel parts (+1.5-2.2 vs SFT) and through fewer invalid
programs (IR 1.1-1.8 % vs 1.3-2.8 %). Validation (256 parts) hid it: test differences of ~1 point are below its noise.
The RFT start matters less than the RL itself: earlier tiny on-policy RL (300 steps x 16 parts, lr 1e-6, HF
generate) moved nothing; this run saw 32k part-samples with std-normalised advantages.

## Handoff (2026-10-16 ~07:20 UTC)
- RL scored: +1.5 for 1 view, flat elsewhere; stopped. Render selection done (above): local best-of-9 view4 91.7.
- **Next (agreed direction: improve the model itself, no cloud at inference): rejection-sampling fine-tuning.**
  Sample ~8 programs per DeepCAD *train* part (weighted to parts where greedy IoU < 0.9 or invalid), keep the best by
  GT IoU when it beats greedy by a margin, fine-tune `vlm_cq_v1` on those mixed with the original SFT data, score on
  test locally. Generation is the cost (HF generate: ~45 min per 2000 parts x 8 samples on an H200), so switch the
  sampler to vLLM first. Never use test-derived outputs for training.
- Data for local runs: `data/renders.tar` and `data/cad_vec.tar` from the bucket, extract only the test uids
  (2000 from `results/baselines_vlm_final/preds.pkl`, 4 renders each); `pip install cadquery pebble h5py trimesh`
  and `./setup.sh` (cadlib).
- Launch gotcha: `jobs/launch.sh` uploads everything under `data/` except the big extracted folders, so local
  `data/renders.tar`, `cad_vec.tar`, `text_index.json` would be uploaded too: move `data/` aside and leave a
  placeholder file (an empty `data/` is dropped, which breaks `entry.sh`).
- Claude API key: stored as `GENCAD_API_KEY` (`ANTHROPIC_*` names are reserved); `scripts/claude_baseline.py`
  reads `ANTHROPIC_API_KEY`, so run it as `ANTHROPIC_API_KEY=$GENCAD_API_KEY python scripts/claude_baseline.py ...`.
- Bucket housekeeping: `results/vlm_cq_v1/model/` holds six stale `.tmp*` files of 4.9 GB each (interrupted saves).

## Next steps (ideas, none started)
- Product path: text_cad (or token_cad image+text) for prompts with dimensions, retrieval for near-duplicates,
  mm_v1 for image-only; a small demo / API around that.
- text_cad tuning: more Qwen layers (8 now), longer training (val still rising slowly at 40k), beam search.
- The novel-part gap (prior 0.3-0.7 % exact vs decoder ceiling 65 %) survives encoder, view-count, decoder and
  sampler changes. Untested levers: prior capacity/objective (a bigger denoiser, pred_v, longer training) and
  exact numbers from text. Or accept that image -> exact program is underdetermined on this data and target
  "close enough" (cmd_edit / acc_param) rather than prog_match.
- Hybrid retrieval + generation: return the nearest training program when the match is very close.
- mm with `image text ccip_image`; `--cond_mode concat`; `--contrastive_weight 0`; longer training (val
  metrics flatten after ~120k steps, so probably little to gain).
- Qualitative check: `infer_mm.py` on a few test uids / free-form prompts, export STEP with `OCC=1`.
- Jobs: `jobs/launch.sh <stage>` (`DRY=1` prints the command); monitor with `hf jobs logs <id>` /
  `hf jobs inspect <id> --format json`.

## Gotchas
- Upstream `trainer/autoencoder_test.py::encode` writes CSR latents from a *shuffled* loader with no uids, so
  they don't line up with image latents. Use `scripts/export_gencad_embeddings.py` (uid-keyed) and
  `scripts/make_dp_arrays.py` instead.
- The released CCIP/prior checkpoints are sketch-conditioned (Canny edges of renders); `jobs/entry.sh` picks
  `IMG_TYPE=cad_sketch` automatically when the CCIP checkpoint name contains "sketch".
- `utils/image_processor.process_image` naming is inverted: `img_type="image"` applies Canny, `"sketch"` doesn't.
- Upstream `config/configAE.py` defaults are debug-sized (batch_size=2); pass `-bs/-epochs/-lr` overrides.
- Upstream training code hard-codes CUDA.
- Job image must have torch>=2.6 (transformers 5.x uses `torch.accelerator`).
- Text2CAD is gated (CC BY-NC-SA 4.0, auto-approved); the HF account behind `HF_TOKEN` has accepted it.
  Use only `text2cad_v1.1/*.csv` (v1.0 is noisier). Never commit its data here.
- `jobs/launch.sh` uploads a dereferenced snapshot of this folder (symlinks followed, data/results excluded).
