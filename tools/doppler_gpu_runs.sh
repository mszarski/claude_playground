#!/usr/bin/env bash
# The Doppler measurements a CPU cannot finish (docs/tasks/DOPPLER.md D3.1 items 1 and 6, table 9.5), for the GPU job.
#
# Usage: tools/doppler_gpu_runs.sh <out-dir>      run by tools/gpu_job.sh (GPUSONAR_JOB=doppler) after its CUDA build;
#                                                 from the repository root of a built checkout (python/gpusonar)
# Sub-runs (GPUSONAR_DOPPLER_RUNS, comma separated, default tone,falsealarm,timing), in this order, each under its own
# time limit (coreutils timeout), each writing its JSON into <out-dir> as it goes and its log to <out-dir>/<run>.log:
#   tone        tools/doppler_tone_eval.py flank_patrol_doppler_tone --background 4 (the scene's 6 pings, both engines,
#               the background's 4 pings), simulator and chain on the device -> doppler_tone_eval.json
#   falsealarm  tools/doppler_bank_threshold.py --at-default: the live chain's false detections at the default budget
#               (1e-3 a ping) on the same noise-only pings of the long-range array, bank off, combine cfar and combine
#               max at 3 m/s (17 hypotheses a beam), in that order -> doppler_bank_at_default.json
#   timing      tools/live_chain_timing.py: a ping's cost with the sweep bank off, max and cfar (the Popoto trial pings,
#               1 and 3 rows; flank_patrol, popoto_far_mover, popoto_12km) and with the tone path (popoto_12km_doppler;
#               flank_patrol_doppler_tone when the tone run left its records in the cache) -> timing_*.json
#   ls12        (only when named) the large submerged class at 12 km: tools/doppler_tone_eval.py on
#               popoto_12km_large_submerged_tone (tone then sweep in one ping) and on popoto_12km_large_submerged with
#               alternating pings (12) -> ls12_tone_train.json, ls12_tone_alternate12.json; and the train scene again
#               with the tone path's grating-lobe notch off on the same cached pings -> ls12_tone_train_lobe_off.json
#               (before and after the notch in one job; each JSON attributes every false detection)
#               (GPUSONAR_DOPPLER_TIMEOUT_LS12, default 80m; GPUSONAR_DOPPLER_LS12_ARGS appended)
# Variables:
#   GPUSONAR_DOPPLER_BACKEND       the chain's and the simulator's backend (default cuda; host for a check on a CPU)
#   GPUSONAR_DOPPLER_QUICK=1       small sizes, to check the plumbing on a CPU: the tone run on popoto_12km_doppler
#                                  (2 pings, 1 background ping), 3 noise pings at 1 m/s, the 1-row trial once,
#                                  popoto_12km on 2 pings (C++ engine), the tone path on popoto_12km_doppler
#   GPUSONAR_DOPPLER_FA_PINGS      noise pings a row of the false-alarm count (default 4000; see below)
#   GPUSONAR_DOPPLER_FA_ROW_SECONDS  each row of that count stops after this long (default 780)
#   GPUSONAR_DOPPLER_TIMEOUT_TONE, _FALSEALARM, _TIMING   time limits (timeout(1) durations; default 80m, 30m, 40m)
#   GPUSONAR_DOPPLER_TONE_ARGS, _FA_ARGS, _TIMING_ARGS    appended to that sub-run's tool (the timing args to each call)
#   GPUSONAR_UPLOAD_REPO           a Hugging Face dataset: <out-dir> is uploaded to it after every sub-run, into the
#                                  folder GPUSONAR_UPLOAD_PATH (default doppler)
# The eigenray scenes need GPUSONAR_BELLHOP (gpu_job.sh: GPUSONAR_ORACLES=1). At 1e-3 a ping, n pings expect n / 1000
# false detections; the exact 95 % interval of a count at the budget spans a factor of 0.27 to 2.6 of it at 4000 pings
# (4 expected), 0.48 to 1.8 at 10 000, 0.71 to 1.4 at 40 000, and a zero count bounds the rate under 3.7 / n a ping
# (9.2e-4 at 4000). On this CPU a ping of that count took 0.86 s without the bank and 14.3 s with it (17 a beam); on the
# device the rows with the bank are expected to stop at their time limit first (the pings done are in the JSON).
# The exit status is the first failing sub-run's (124 for one stopped by its limit).
case "${1:-}" in -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0;; esac
if [ $# -ne 1 ] || [ "${1#-}" != "$1" ]; then
  echo "usage: tools/doppler_gpu_runs.sh <out-dir>   (tools/doppler_gpu_runs.sh --help)" >&2
  exit 2
fi
set -uo pipefail
set -f                                      # no globbing of the argument lists below (they are split on spaces)
out=$1
cd "$(dirname "$0")/.."
mkdir -p "$out"
out=$(cd "$out" && pwd)
cache="$out/cache"
runs=",${GPUSONAR_DOPPLER_RUNS:-tone,falsealarm,timing},"
be=${GPUSONAR_DOPPLER_BACKEND:-cuda}
quick=${GPUSONAR_DOPPLER_QUICK:-0}
py=${PYTHON:-python3}
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
[ -n "${GPUSONAR_BELLHOP:-}" ] || echo "doppler: GPUSONAR_BELLHOP is not set: the eigenray scenes will fail (set GPUSONAR_ORACLES=1 in the job)"
status=0

upload() {
  if [ -n "${GPUSONAR_UPLOAD_REPO:-}" ]; then
    # The record cache stays behind: it is large and the JSON holds what was measured.
    hf upload "$GPUSONAR_UPLOAD_REPO" "$out" "${GPUSONAR_UPLOAD_PATH:-doppler}" --type dataset --exclude "cache/*" > /dev/null \
      || echo "doppler: upload failed"
  fi
}

run() {   # run <name> <limit> <command...>: logged, limited, its status kept
  local name=$1 limit=$2 rc
  shift 2
  echo "=== doppler $name (limit $limit): $*" | tee -a "$out/$name.log"
  timeout -k 60 "$limit" "$@" 2>&1 | tee -a "$out/$name.log"
  rc=${PIPESTATUS[0]}
  echo "=== doppler $name: exit $rc" | tee -a "$out/$name.log"
  [ "$rc" -eq 0 ] || [ "$status" -ne 0 ] || status=$rc
}

if [ "$quick" = "1" ]; then
  # Free field and no boundary reverberation (--set): the scenes' eigenray simulation takes more than half an hour a
  # ping on a CPU; this checks the plumbing and the outputs, it measures nothing.
  quick_set="--set fidelity=free_field --set reverberation=[]"
  tone_scene=popoto_12km_doppler; tone_args="--pings 2 --background 1 $quick_set"
  fa_args="--pings 3 --max-rates 1"
  trial_args="--rows 1 --repeats 1"; scenes="popoto_12km"; scene_args="--scene-pings 2 $quick_set"; engines="cpp"
else
  tone_scene=flank_patrol_doppler_tone; tone_args="--background 4"
  fa_args="--pings ${GPUSONAR_DOPPLER_FA_PINGS:-4000} --max-rates 3 --max-seconds ${GPUSONAR_DOPPLER_FA_ROW_SECONDS:-780}"
  trial_args="--rows 1 3 --repeats 3"; scenes="flank_patrol popoto_far_mover popoto_12km"; scene_args="--scene-pings 3"; engines="cpp python"
fi

if [[ $runs == *,tone,* ]]; then
  # shellcheck disable=SC2086
  run tone "${GPUSONAR_DOPPLER_TIMEOUT_TONE:-80m}" "$py" tools/doppler_tone_eval.py "$tone_scene" $tone_args --backend "$be" \
      --sim-backend "$be" --cache "$cache" --json "$out/doppler_tone_eval.json" ${GPUSONAR_DOPPLER_TONE_ARGS:-}
  upload
fi
if [[ $runs == *,falsealarm,* ]]; then
  # shellcheck disable=SC2086
  run falsealarm "${GPUSONAR_DOPPLER_TIMEOUT_FALSEALARM:-30m}" "$py" tools/doppler_bank_threshold.py --at-default --sonar long_range \
      --combines off,cfar,max $fa_args --backend "$be" --json "$out/doppler_bank_at_default.json" ${GPUSONAR_DOPPLER_FA_ARGS:-}
  upload
fi
if [[ $runs == *,timing,* ]]; then
  # One limit for the whole sub-run: the calls in turn (one failing does not stop the rest), each its own JSON (every row
  # written as it is measured).
  timing() {
    local s e cached rc=0
    set -f
    # shellcheck disable=SC2086
    "$py" tools/live_chain_timing.py $trial_args --engine cpp python --doppler --doppler-combine off max cfar --backend "$be" \
        --json "$out/timing_trial.json" ${GPUSONAR_DOPPLER_TIMING_ARGS:-} || rc=$?
    for e in $engines; do                   # every scene with the C++ engine first, then the Python engine (records cached)
      for s in $scenes; do
        # shellcheck disable=SC2086
        "$py" tools/live_chain_timing.py --scene "$s" $scene_args --repeats 1 --engine "$e" --doppler --doppler-combine off max cfar \
            --backend "$be" --sim-backend "$be" --cache "$cache" --json "$out/timing_${s}_$e.json" ${GPUSONAR_DOPPLER_TIMING_ARGS:-} || rc=$?
      done
    done
    # shellcheck disable=SC2086
    "$py" tools/live_chain_timing.py --scene popoto_12km_doppler $scene_args --repeats 1 --engine $engines --doppler-tone \
        --backend "$be" --sim-backend "$be" --cache "$cache" --json "$out/timing_popoto_12km_doppler.json" ${GPUSONAR_DOPPLER_TIMING_ARGS:-} || rc=$?
    set +f
    cached=$(ls "$cache"/flank_patrol_doppler_tone_scene_6_* 2> /dev/null | head -1)
    set -f
    if [ "$quick" != "1" ] && [ -n "$cached" ]; then
      # shellcheck disable=SC2086
      "$py" tools/live_chain_timing.py --scene flank_patrol_doppler_tone --scene-pings 6 --repeats 1 --engine $engines --doppler-tone \
          --backend "$be" --sim-backend "$be" --cache "$cache" --json "$out/timing_flank_patrol_doppler_tone.json" ${GPUSONAR_DOPPLER_TIMING_ARGS:-} || rc=$?
    else
      echo "doppler timing: flank_patrol_doppler_tone left out (no cached records of the tone run; its JSON has the cost a ping)"
    fi
    return "$rc"                            # a failed call does not stop the others; the last failure's status
  }
  export -f timing
  export py out be cache trial_args scenes scene_args engines quick
  run timing "${GPUSONAR_DOPPLER_TIMEOUT_TIMING:-40m}" bash -c timing
  upload
fi
if [[ $runs == *,ls12,* ]]; then
  # Not in the default runs: the large submerged class at 12 km (DOPPLER.md "Large submerged at 12 km"): the tone above
  # the band then the sweep in one ping, and alternating tone and sweep pings on the sweep-only scene. 58 minutes a
  # ping to simulate on a CPU.
  ls12() {
    local rc=0
    set -f
    # shellcheck disable=SC2086
    "$py" tools/doppler_tone_eval.py popoto_12km_large_submerged_tone --max-rate 8 --truth-at reflection --gate-range-m 100 \
        --background 4 --backend "$be" --sim-backend "$be" --cache "$cache" --json "$out/ls12_tone_train.json" ${GPUSONAR_DOPPLER_LS12_ARGS:-} || rc=$?
    # The same pings (the cache) with the grating-lobe notch off: the tone path as before the notch (DOPPLER.md GL).
    # shellcheck disable=SC2086
    "$py" tools/doppler_tone_eval.py popoto_12km_large_submerged_tone --max-rate 8 --truth-at reflection --gate-range-m 100 \
        --background 4 --backend "$be" --sim-backend "$be" --cache "$cache" --tone-lobe-notch off \
        --json "$out/ls12_tone_train_lobe_off.json" ${GPUSONAR_DOPPLER_LS12_ARGS:-} || rc=$?
    # shellcheck disable=SC2086
    "$py" tools/doppler_tone_eval.py popoto_12km_large_submerged --alternate 6.5e+3 --pings 12 --max-rate 8 --truth-at reflection \
        --gate-range-m 100 --backend "$be" --sim-backend "$be" --cache "$cache" --json "$out/ls12_tone_alternate12.json" ${GPUSONAR_DOPPLER_LS12_ARGS:-} || rc=$?
    set +f
    return "$rc"
  }
  export -f ls12
  export py out be cache
  run ls12 "${GPUSONAR_DOPPLER_TIMEOUT_LS12:-80m}" bash -c ls12
  upload
fi
exit "$status"
