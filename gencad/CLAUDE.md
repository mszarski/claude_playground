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
- Bucket `mszarski/gencad-work` holds: `data/{cad_vec,images,sketches}.tar` + split JSONs, all 7 released
  checkpoints in `ckpt/`, `emb/{cad_latents,cond_ccip_image,cond_image}.h5` (168,674 uids).
- Replication baseline (`results/released/eval.json`, 2000 test samples, sketch CCIP + sketch DP):
  csr_recon acc_cmd 0.918 / acc_param 0.957; gencad_dp cos 0.535, R@1 0.220, R@10 0.421, acc_cmd 0.599,
  acc_param 0.659.
- **Text is blocked**: `SadilKhan/Text2CAD` is gated and the account has not been granted access. Once it is,
  run `jobs/launch.sh text`, then embed only the text (`embed` would redo the 45-min image pass):
  `jobs/launch.sh shell -- python scripts/embed_conditions.py text --text_index /work/data/text_index.json --out /work/emb/cond_text.h5`, then
  `RUN=mm_v1 jobs/launch.sh mm --modalities image text` and `RUN=mm_v1 jobs/launch.sh eval`.
- Image-only prior `RUN=mm_img jobs/launch.sh mm --modalities image` launched; evaluate with
  `RUN=mm_img jobs/launch.sh eval` and compare to `results/released/eval.json`.
- Pull results locally with `hf buckets cp hf://buckets/mszarski/gencad-work/results/<run>/eval.json ...`.

## Next steps
1. Evaluate `mm_img`; ablation `RUN=mm_ccip ... mm --modalities ccip_image` (GenCAD's conditioning, new denoiser).
2. Text, once access is granted (see above).
3. Jobs: `jobs/launch.sh <stage>` (`DRY=1` prints the command); monitor with `hf jobs logs <id>` /
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
- Text2CAD is gated: the HF account behind `HF_TOKEN` must be granted access on the dataset page.
- `jobs/launch.sh` uploads a dereferenced snapshot of this folder (symlinks followed, data/results excluded).
