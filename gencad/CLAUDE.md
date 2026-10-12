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
| mm_ccip: ccip_image (GenCAD features, new denoiser) | 0.641 | 0.461 | 0.637 | 0.666 | 0.748 |
| mm_img: image (SigLIP2) | 0.635 | 0.448 | 0.603 | 0.671 | 0.743 |
| mm_v1: image | 0.626 | 0.435 | 0.599 | 0.665 | 0.740 |
| mm_v1: text, random level | 0.490 | 0.162 | 0.311 | 0.557 | 0.634 |
| mm_v1: text, abstract / beginner / intermediate / expert | 0.438 / 0.515 / 0.550 / 0.570 | 0.100 / 0.203 / 0.245 / 0.254 | 0.201 / 0.360 / 0.447 / 0.478 | 0.498 / 0.576 / 0.616 / 0.650 | 0.614 / 0.633 / 0.651 / 0.671 |
| mm_v1: image+text, random level | 0.647 | 0.468 | 0.645 | 0.681 | 0.751 |
| mm_v1: image+text, expert | 0.666 | 0.503 | 0.693 | 0.695 | 0.759 |

Takeaways: the gain over GenCAD comes from the new denoiser/training (mm_ccip ≈ mm_img), not SigLIP2. Text
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

## Next steps (ideas, none started)
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
