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

## State at handoff (2026-10-12)
- All code is written. `./setup.sh && python -m pytest tests/test_mm_smoke.py -q` passes (9 tests, CPU, ~1 min).
  The full export → train → eval → infer chain ran on synthetic CAD files with random GenCAD weights: the code
  runs, but no real numbers exist yet.
- Nothing has touched real data or a GPU. The previous cloud session had no network access to huggingface.co or
  drive.google.com and no HF token. That is the only reason it stopped.

## Next steps
1. Env: `pip install torch torchvision -r jobs/requirements-job.txt`, then `./setup.sh` and run the tests.
   STEP/STL export needs `pythonocc-core` (conda-forge only); it's optional locally, jobs install it with `OCC=1`.
2. `hf auth whoami` must work (needs `HF_TOKEN` with write + Jobs). Compute runs on HF Jobs via
   `jobs/launch.sh <stage>`; use `DRY=1` first to print the command. Stages in order: `fetch`, `test`, `export`,
   `RUN=released eval` (replication baseline), `embed`, `RUN=mm_v1 mm --modalities image text`, `RUN=mm_v1 eval`.
   Monitor with `hf jobs logs <id> --follow`.
3. `fetch` is the riskiest step: `gdown --folder` caps at 50 files per Drive folder, and the Text2CAD column
   layout is auto-detected (`scripts/build_text_index.py` prints what it found and how many GenCAD uids join).
   Check both outputs before moving on.

## Gotchas
- Upstream `trainer/autoencoder_test.py::encode` writes CSR latents from a *shuffled* loader with no uids, so
  they don't line up with image latents. Use `scripts/export_gencad_embeddings.py` (uid-keyed) and
  `scripts/make_dp_arrays.py` instead.
- The released CCIP/prior checkpoints are sketch-conditioned (Canny edges of renders); `jobs/entry.sh` picks
  `IMG_TYPE=cad_sketch` automatically when the CCIP checkpoint name contains "sketch".
- `utils/image_processor.process_image` naming is inverted: `img_type="image"` applies Canny, `"sketch"` doesn't.
- Upstream `config/configAE.py` defaults are debug-sized (batch_size=2); pass `-bs/-epochs/-lr` overrides.
- Upstream training code hard-codes CUDA.
- `jobs/launch.sh` uploads a dereferenced snapshot of this folder (symlinks followed, data/results excluded).
