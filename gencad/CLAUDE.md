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
