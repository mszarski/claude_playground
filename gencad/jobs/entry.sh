#!/usr/bin/env bash
# Entry point for every HF Job. Launched by jobs/launch.sh as:  bash /code/jobs/entry.sh <stage> [extra args]
#
# Mounts (set by launch.sh):
#   /code  this repo (read-only snapshot of your local checkout)
#   /work  read-write HF bucket that persists between jobs:
#          /work/data/{cad_vec,images,sketches}.tar, /work/data/*.json   dataset
#          /work/ckpt/                                                   upstream pretrained checkpoints
#          /work/emb/                                                    cached embeddings
#          /work/results/                                                all training runs
#
# Env knobs: CSR_CKPT, CCIP_CKPT, DP_CKPT (default: upstream pretrained in /work/ckpt), RUN (run name),
#            IMG_TYPE (cad_image|cad_sketch), OCC=1 to install pythonocc-core for STEP/STL export.
set -euo pipefail
STAGE=${1:?stage}; shift || true
RUN=${RUN:-$STAGE}
echo "=== stage=$STAGE run=$RUN $(date -u) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cp -r /code /app && cd /app
pip install -q -r jobs/requirements-job.txt
if [[ "${OCC:-0}" == "1" ]]; then conda install -y -q -c conda-forge pythonocc-core=7.9.0; fi
mkdir -p /work/data /work/ckpt /work/emb /work/results
rm -rf results && ln -s /work/results results
ln -sfn /work/emb data/emb
ln -sfn /work/ckpt data/ckpt
rm -rf model/ckpt && ln -sfn /work/ckpt model/ckpt   # upstream inference_gencad.py looks here
for f in /work/data/*.json; do [[ -e "$f" ]] && cp -f "$f" data/; done

# bucket FUSE is slow for 100k+ small files: unpack the archives this stage needs onto local disk
need() { for d in "$@"; do
  if [[ -f /work/data/$d.tar ]]; then echo "[data] unpack $d"; tar -xf /work/data/$d.tar -C data; else echo "[data] missing /work/data/$d.tar"; fi
done; }

# CadQuery (OCP) needs system GL libraries the pytorch image does not ship
need_cq() {
  python -c "import cadquery" 2>/dev/null && return
  apt-get update -qq && apt-get install -y -qq libgl1 libglu1-mesa libxrender1 libxext6 >/dev/null
  pip install -q cadquery pebble
  python -c "import cadquery; print('[cadquery]', cadquery.__version__)"
}

first() { for p in "$@"; do [[ -e "$p" ]] && { echo "$p"; return; }; done; echo "${1}"; }
CSR_CKPT=${CSR_CKPT:-$(first data/ckpt/ae_ckpt_epoch1000.pth $(ls data/ckpt/*ae*.pth 2>/dev/null))}
CCIP_CKPT=${CCIP_CKPT:-$(first data/ckpt/ccip_sketch_ckpt_epoch300.pth $(ls data/ckpt/*ccip*.pth 2>/dev/null))}
# the released CCIP/DP checkpoints come in sketch (Canny edges of renders) and image flavours; keep them paired
if [[ "$CCIP_CKPT" == *sketch* ]]; then IMG_TYPE=${IMG_TYPE:-cad_sketch}; DP_KIND=sketch; else IMG_TYPE=${IMG_TYPE:-cad_image}; DP_KIND=image; fi
DP_CKPT=${DP_CKPT:-$(shopt -s nullglob; first data/ckpt/${DP_KIND}_cond_diffusion*.pt data/ckpt/*diffusion*.pt none)}
echo "[ckpt] CSR=$CSR_CKPT CCIP=$CCIP_CKPT DP=$DP_CKPT IMG_TYPE=$IMG_TYPE"

case "$STAGE" in
  fetch)      # Drive + Text2CAD -> bucket. CPU flavor is enough.
    python scripts/fetch_data.py --data /tmp/d
    for d in cad_vec images sketches multi_view_images; do
      [[ -d /tmp/d/$d ]] && { echo "[fetch] tar $d"; tar -cf /work/data/$d.tar -C /tmp/d $d; }
    done
    cp -f /tmp/d/*.json /work/data/ 2>/dev/null || true
    # fetch_data.py symlinks checkpoints into model/ckpt, which is /work/ckpt here: replace the links with files
    for f in /tmp/d/ckpt/*; do rm -f "/work/ckpt/$(basename "$f")"; cp -f "$f" /work/ckpt/; done
    ls -la /work/data /work/ckpt ;;

  text)       # Text2CAD annotations -> text_index.json (needs access to the gated SadilKhan/Text2CAD)
    # v1.1 only: the repo also has v1.0, whose prompts are noisier, and merging both would mix them per uid
    python scripts/build_text_index.py --hf_repo SadilKhan/Text2CAD --hf_patterns "text2cad_v1.1/*.csv" \
      --split_json data/filtered_data.json \
      --out /work/data/text_index.json "$@" ;;

  test)       # unit + smoke tests on synthetic data, real encoders included
    MM_TEST_ENCODERS=1 python -m pytest tests/test_mm_smoke.py -q -s "$@" ;;

  csr)        # replication stage 1: CSR autoencoder
    need cad_vec
    python train_gencad.py csr -name "$RUN" -gpu 0 -epochs ${EPOCHS:-1000} -bs ${BS:-512} -lr ${LR:-1e-3} "$@" ;;

  ccip)       # replication stage 2: contrastive CAD-image pretraining
    need cad_vec images sketches
    python train_gencad.py ccip -name "$RUN" -gpu 0 -cad_ckpt "$CSR_CKPT" -img_type "$IMG_TYPE" \
      -epochs ${EPOCHS:-300} -bs ${BS:-128} "$@" ;;

  export)     # uid-aligned CSR latents (+ CCIP image latents)
    need cad_vec images sketches
    python scripts/export_gencad_embeddings.py --data data --csr_ckpt "$CSR_CKPT" --ccip_ckpt "$CCIP_CKPT" \
      --img_type "$IMG_TYPE" --out data/emb "$@" ;;

  embed)      # frozen SigLIP2 image + Qwen3 text embeddings
    need images
    python scripts/embed_conditions.py image --data data --out data/emb/cond_image.h5 "$@"
    if [[ -f data/text_index.json ]]; then
      python scripts/embed_conditions.py text --text_index data/text_index.json --out data/emb/cond_text.h5
    else echo "[embed] no data/text_index.json -> skipping text"; fi ;;

  dp)         # replication stage 3: upstream diffusion prior, on correctly aligned arrays
    python scripts/make_dp_arrays.py --emb_dir data/emb --split train --out_dir data/dp
    python train_gencad.py dp -name "$RUN" -gpu 0 -cad_emb data/dp/cad_zs.h5 -img_emb data/dp/img_zs.h5 \
      -steps ${STEPS:-1000000} -save_every ${SAVE_EVERY:-50000} "$@" ;;

  ar)         # autoregressive CAD decoder on the frozen CSR latents (mm/ar_decoder.py)
    need cad_vec
    python train_ar_decoder.py --emb_dir data/emb --data data --csr_ckpt "$CSR_CKPT" --name "$RUN" \
      --steps ${STEPS:-100000} "$@" ;;

  e2e)        # fine-tune the SigLIP2 image encoder end to end with a trained prior (train_e2e.py)
    need images
    python train_e2e.py --init_prior "${INIT_PRIOR:-results/mm_v1/last.pt}" --emb_dir data/emb --data data \
      --name "$RUN" --steps ${STEPS:-20000} "$@" ;;

  textcad)    # text -> CAD program decoded from prompt tokens (train_text_cad.py; needs programs.npz in /work/emb)
    need cad_vec
    python train_text_cad.py --emb_dir data/emb --data data --name "$RUN" --steps ${STEPS:-40000} "$@" ;;

  tokencad)   # image (+ text) -> CAD program decoded from patch/prompt tokens (train_token_cad.py)
    need cad_vec images
    python train_token_cad.py --emb_dir data/emb --data data --name "$RUN" --steps ${STEPS:-40000} "$@" ;;

  vlmcad)     # Qwen3-VL fine-tuned to write CAD programs from renders / prompts (train_vlm_cad.py)
    need cad_vec
    if [[ " $* " == *" --images images "* ]]; then need images; else need renders; fi
    python train_vlm_cad.py --emb_dir data/emb --data data --name "$RUN" --steps ${STEPS:-15000} "$@" ;;

  mm)         # the new multimodal prior
    python train_mm_prior.py --emb_dir data/emb --name "$RUN" ${HUB_REPO:+--hub_repo $HUB_REPO} "$@" ;;

  eval)
    need cad_vec
    ARGS=(--emb_dir data/emb --data data --csr_ckpt "$CSR_CKPT" --out "results/$RUN/eval.json")
    [[ -f "results/$RUN/last.pt" ]] && ARGS+=(--prior "results/$RUN/last.pt")
    [[ "$DP_CKPT" != none && -f "$DP_CKPT" ]] && ARGS+=(--gencad_dp_ckpt "$DP_CKPT")
    python eval_mm.py "${ARGS[@]}" "$@" ;;

  upstream)   # released GenCAD through upstream's own inference path, scored like eval_mm (cross-check of gencad_dp)
    need cad_vec images sketches
    python scripts/eval_upstream_inference.py --data data --emb_dir data/emb --ckpt_dir data/ckpt \
      --out "results/$RUN/eval.json" "$@" ;;

  baselines)  # train/test duplication + nearest-neighbour baselines, and priors split into duplicated vs novel parts
    need cad_vec
    [[ " $* " == *" --token_cad "* ]] && need images
    [[ " $* " == *" --vlm_cad "* ]] && need renders
    python scripts/baselines.py --emb_dir data/emb --data data --csr_ckpt "$CSR_CKPT" \
      --out "results/$RUN/report.json" "$@" ;;

  geom)       # IoU / Chamfer / invalidity of saved predictions (scripts/geom_eval.py; installs OCC itself)
    [[ "${OCC:-0}" == "1" ]] || conda install -y -q -c conda-forge pythonocc-core=7.9.0
    need cad_vec
    python scripts/geom_eval.py --data data --preds "${PREDS:-results/$RUN/preds.pkl}" --out "results/$RUN/geom.json" "$@" ;;

  render)     # our own fixed-camera renders of every GT program -> /work/data/renders.tar (scripts/render_views.py)
    [[ "${OCC:-0}" == "1" ]] || conda install -y -q -c conda-forge pythonocc-core=7.9.0
    pip install -q pebble
    need cad_vec
    python scripts/render_views.py --data data --out data/renders "$@"
    if [[ " $* " == *" --limit "* ]]; then mkdir -p "results/$RUN" && cp -r data/renders "results/$RUN/"   # preview
    else tar -cf /work/data/renders.tar -C data renders && ls -la /work/data/renders.tar; fi ;;

  fetchrecode) # CAD-Recode v1.5 (~1M CadQuery programs) -> /work/data/cad_recode.parquet
    command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git >/dev/null; }
    python scripts/fetch_cad_recode.py --out /work/data/cad_recode.parquet "$@" ;;

  renderrecode) # CAD-Recode programs -> renders with our cameras; one shard per job -> /work/data/<RUN>.tar
    need_cq
    cp /work/data/cad_recode.parquet data/
    python scripts/render_cad_recode.py --parquet data/cad_recode.parquet --out data/recode_renders "$@"
    tar -cf "/work/data/$RUN.tar" -C data recode_renders && ls -la "/work/data/$RUN.tar" ;;

  convertdc)  # DeepCAD programs -> CadQuery source + fidelity vs cadlib -> /work/data/deepcad_cq.parquet
    need_cq
    need cad_vec
    python scripts/convert_deepcad.py --data data --out /work/data/deepcad_cq.parquet "$@" ;;

  cqiou)      # IoU of saved predictions (DeepCAD programs or CadQuery code) vs GT, in CadQuery (scripts/cq_iou.py)
    need_cq
    need cad_vec
    python scripts/cq_iou.py --data data --preds "${PREDS:-results/$RUN/preds.pkl}" --out "results/$RUN/cq_iou.json" "$@" ;;

  checkviews) # do all 5 renders of a uid show its GT program? (scripts/check_views.py; installs OCC itself)
    [[ "${OCC:-0}" == "1" ]] || conda install -y -q -c conda-forge pythonocc-core=7.9.0
    need cad_vec images
    python scripts/check_views.py --data data --out "results/$RUN/report.json" "$@" ;;

  visual)     # rendered side-by-side check (needs OCC=1): GT vs GenCAD vs the prior under each condition
    need cad_vec images
    PRIOR=${PRIOR:-results/mm_v1/last.pt}
    ARGS=(--prior "$PRIOR" --csr_ckpt "$CSR_CKPT" --emb_dir data/emb --data data --out "results/$RUN")
    [[ "$DP_CKPT" != none && -f "$DP_CKPT" ]] && ARGS+=(--gencad_dp_ckpt "$DP_CKPT")
    python scripts/visual_check.py "${ARGS[@]}" "$@" ;;

  shell)      # arbitrary command, e.g. launch.sh shell -- python infer_mm.py ...
    [[ "${1:-}" == "--" ]] && shift
    "$@" ;;

  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "=== done $STAGE $(date -u) ==="
