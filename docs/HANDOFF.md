# Handoff notes

Refreshed 2026-11-24 at commit `933972b` on branch `claude/gpu-sonar-suite-u289p5`
(decisions to D-181; the version before this one was written 2026-11-19 at `2160fe8` and
stopped at D-144), and brought to D-201 and the opt-in 3-D propagation tier on 2026-11-26, then to
D-225 (the merge `b88c458`) the same day: only the state lines and the statements that merge made false; then
to D-234 (the merge `4057d65` of the third batch) the same day, the same way; then to D-240 (the merge `8581533` of the
fourth batch) on 2026-11-27, the same way; then to D-245 (the merge `f40d4c3` of the fifth batch) the same day,
the same way, with the rule for device code learned in D-242 in section 11; then to D-250 (the merge `bda7c7d` of the
sixth batch) the same day, the same way, with three.js in section 2; then to D-254 (the merge `e06fac8` of the
seventh batch and the regenerated gallery `082dd21`) the same day, the same way; then to D-261 (the merge `08ddfab` of
the eighth batch) on 2026-11-28, the same way, with the `doppler` job's new sub-runs in section 4; then to D-263 (the
merge `e16e214` of the ninth batch and the regenerated gallery `4720050`) the same day, the same way; then to D-267
(the merge `cd620b4` of the tenth batch) the same day, the same way, with the `depthrange` job mode in section 4.
**Read first: the deployed stacked-row depth estimate states wrong depths confidently in shallow water (D-266 F1):
a ship 4 m deep at 1.7 km in 100 m of water reads 95.7 +- 27.4 m and is called submerged on 4 of 4 pings; the
depth is dishonest on 22 of 28 depths at 1 to 2 km there; not fixed (issue `sonar-4e2h`; the fix moves a default
and waits on the owner); held by five strict expected-failure tests. And as deployed, depth cannot tell a submerged
target from a ship at 12 km** (the sound-speed gradient prior: 2400 m of sigma there; the rows' 1.5 m aperture).
Shown to work: about 1 km with the short pulse in the elevation scenes' geometry, and with the water known from a
deep own depth out to the ranges of `docs/tasks/DEPTH_RANGE.md` DR.5; a case for three rows from depth at 12 km is
not shown (section 10; `docs/STATUS.md` open items 101 to 107). **Then: since D-262 the 12 km long-range baseline in
the tree is a 2 s hyperbolic sweep at 197 dB with the plain filter (the owner's decision, D-256). What got worse
with it: nothing is seen inside 1.5 km; the elevation scenes keep the 1 s sweep; two face self-checks fail (one
passes with the opt-in range correction of D-264); a mover's range reads 6 m per m/s
off, with no range rate. Measured at the deep site it holds every target at sea state 3 on one seed, the resolved
hulls by +1.28 / +0.73 dB (thin), not at sea state 4; sea state 5 needs a larger transmitter not yet
identified**: section 10, and `docs/STATUS.md` open items 96 to 100. These notes are for whoever picks the work up next, a person or a coding
model. `docs/STATUS.md` is the authoritative list of what works and what is wrong; this
file tells you how to work here and where the edges are. The wider handoff set is in
`docs/handoff/` (start at `docs/handoff/README.md`).

**Everything here is simulation and analysis. Nothing has run on a vehicle, on the Popoto
boards, on a DGX Spark, a Jetson, any Sensor Bridge hardware, or on real data.**

## 1. What this is

One GPU sonar suite for a forward-looking sonar (FLS), synthetic aperture sonar (SAS) and a
long-range flank array ("Popoto"), in this repository beside the unrelated
`trajectory_classifier` package. C++20/CUDA core (`libs/`), Python package
(`python/gpusonar/`), tools (`tools/`), YAML configuration (`configs/`), tests
(`tests/cpp`, `tests/python`), documents (`docs/`).

The owner's brief asked for: Holoscan capture, a reimplementation of ASASIN
(arXiv:2101.05888), a simulator for all modes, one record format, one CUDA backprojection
kernel, YAML config, GoogleTest and pytest behind one command (`tools/test.sh`), and the
documents `PAPER_NOTES`, `HOLOSCAN_NOTES`, `OPEN_SOURCE`, `DECISIONS`, `STATUS` (kept
current), `RESULTS`, `PHYSICS_NOTES` and a README.

## 2. Rules that are in force

From the owner's brief:

- Tests before or alongside components. Never mark a milestone done with failing tests.
  Never weaken a test to make it pass without saying so plainly.
- **Stop and ask the owner before**: adding a large or copyleft dependency; changing the
  record schema (frozen since milestone 2); changing any FPGA code; acting when the paper
  and a cited reference disagree in a way that changes results.
- GPL and LGPL code may be used only as a separate test oracle or optional tool, never
  linked into the core. Bellhop, Bellhop3D and KRAKEN run as separate processes; PyRAM is
  imported by tests only (D-149). Bellhop is the `eigenray` tier's program and, since
  2026-11-26 with the owner's approval, Bellhop3D the opt-in `eigenray3d` tier's
  (`python/gpusonar/sim/bellhop3d.py`; nothing linked or copied).
- Outside code vendored with the owner's approval: three.js 0.170.0 (MIT), one unmodified file of 691,648
  bytes under `third_party/three/`, embedded only into the generated 3-D scene pages
  (`python/gpusonar/viz/page3d.py`) and never imported by the suite, the simulator or the chains; the owner
  approved the viewer dependency ("dependency is fine") and, on 2026-11-27, taking it from the npm registry
  after the two CDNs were refused by the network policy ("Yes its acceptable"; D-248; `docs/OPEN_SOURCE.md`
  2g, 3.14). Bring any further browser library to the owner first.
- Do not copy code from any closed or GPL source into the core.
- Do not invent numbers that a paper reports.
- Unconfirmed hardware facts live in config, not code.
- Record every open-source decision in `docs/OPEN_SOURCE.md`.

Set by the owner since (each is in `docs/DECISIONS.md`):

- **The owner approves each work package before it starts, and any change that moves a
  default result before it is merged.** Bring before -> after numbers on a representative
  case, worst first. New behaviour is opt-in unless the owner says otherwise.
- **Defaults the owner chose, not to be moved without asking:** `auto` is the default
  backend everywhere and `cuda` never falls back (D-147); the Sensor Bridge loopback test
  runs at real time and a failure is reported, not retried (D-153); the gallery's FLS
  classification uses `row_snr_db: 6` (D-154); echo grouping and what the gallery scores
  (D-155, D-172); detections at their reflection time (D-156); the classifier fits the
  measured positions (D-157); the leading edge is the live chain's default range (D-160)
  and the gallery's (D-178); the Sensor Bridge receive thread count (D-162); ping 0's
  height free in motion estimation and the window energy gate (D-164, D-171); the
  detector's constants stay as they are (D-168); a per-board stream's missing span drops its
  ping (`missing: drop`, D-184); the Sensor Bridge receiver's queue holds 4 s of stream
  (D-187); a frame damaged in the first 0.25 s of a Popoto-rate loopback is start-up loss
  (D-188); `flank_patrol`'s small vehicle stays a hard target (D-198); the long-range baseline is the
  plain filter with every Doppler option off and the tone path off (D-244); its waveform is a 2 s hyperbolic
  sweep at 197 dB (D-256 decided, D-262 done; the 1 s linear sweep until then): blind inside 1.5 km, a mover's
  range 6 m per m/s off, no range rate. The scenes and tests made for the 1 s linear sweep (the three elevation
  scenes, the bank's scene `popoto_12km_large_submerged`, the 0.1 s builds, the sea-state-5 placeholder) state it.
- **Left opt-in on purpose** (each measured better on some scenes and worse on others):
  the per-axis outlier test `classify.outlier_alpha` (D-159), the classification rules from
  detections (D-152), the sub-grid Popoto bearing (D-176), the LFM range-Doppler correction
  `range_doppler` (D-181), merging FLS detections before tracking (D-135), a board clock's
  rate and its estimate's discipline (`source.clock_ppm`, `source.clock`, D-185), the K
  reverberation shape from a density (D-196), image sources off a tilted bed and walls as any
  plane (D-201), the `eigenray3d` tier (2026-11-26), the range integration
  `cfar.integrate_cells` (D-97). Turning one on by default is the owner's call.
- **The two live-chain engines are kept equal.** A behaviour change in the Python engine
  (`python/gpusonar/live.py`, `detect.py`, `classify.py`) is made in the C++ engine
  (`libs/live`) in the same package and held by the parity tests in
  `tests/python/test_live_cpp.py`.
- **Every change names its source**: a paper and equation, a datasheet, an opened
  implementation, or "ours" with the reason, in the status words of `docs/REFERENCES.md`
  (confirmed, via secondary, ours, unconfirmed).

Privacy and access, set during the work:

- The owner's email and project details are private. The **only** approved external
  destination for project content is Hugging Face: the private dataset
  `mszarski/gpusonar-ci` and HF Jobs. Literature and product searches use generic terms
  only. No contact, quote or registration forms, no accounts, no emails.
- **Nothing is deleted from the Hugging Face dataset without the owner's say.** A job
  uploads under its own tarball name and its own folder (`GPUSONAR_UPLOAD_PATH`) and
  touches no other file there.
- **APL-UW TR 9407 is distribution-limited. A download of it was denied. Do not download
  or read it from anywhere, and do not route around the denial** (not through a subagent
  either). The owner's ruling: "Use anything from arxiv that cites things you need".
- Paywalled, login-gated or bot-challenged pages: move on. No sci-hub.
- Licensed downloaded data (UACR, Watermark, Smart Ocean channel soundings) is never
  committed or redistributed; only derived tables and figures, with attribution.
- No `LD_PRELOAD` or capability tricks in HF containers. Never paste or print the HF token.

Repository and git:

- Push only to `claude/gpu-sonar-suite-u289p5`. No pull request unless the owner asks.
- Commit messages end with the co-author and session trailer lines the harness specifies.
  No model identifier in any committed artifact.
- Issue tracking is `bd` (beads), prefix `sonar-`. Put it on the path first:
  `export PATH=$PATH:$HOME/.local/bin:$HOME/go/bin`. Every `bd` write re-exports
  `.beads/issues.jsonl`, which then needs its own commit; a stop hook refuses to end a turn
  with uncommitted changes. Only the orchestrating session runs `bd`; an agent in a
  worktree does not, and lists follow-ups in its report instead.
- `.claude/worktrees/` is excluded locally. Remove an agent worktree with
  `git worktree remove -f -f <path>` and delete its branch after merging.

Writing style the documents hold to: plain statements, what failed or got worse is stated
first and in bold, nothing is called fixed that was not measured, every number names the
tool and commit it came from.

## 3. How to build and test

The container has 15 GB of RAM and 4 cores and no GPU, and a few GB of free disk. More than
one test process at a time, or a wide build, exhausts it.

```bash
cd /home/user/claude_playground
export OMP_NUM_THREADS=2 PYTHONPATH=$PWD/python
export GPUSONAR_BELLHOP=<path to bellhop.exe>     # see below
cmake --build build -j2                            # build/ is the CPU build
ctest --test-dir build --output-on-failure         # the entry named `pytest` is the whole pytest run
```

`tools/test.sh` does configure, build and ctest in one command (`cmake -S . -B build -G
Ninja`, `cmake --build build`, `ctest`; `SONAR_BUILD_DIR`, `CMAKE_BUILD_TYPE` and
`SONAR_CMAKE_ARGS` override), but builds at full width; on this machine run the two steps
above instead. The whole suite took about two hours here when last timed (2026-11-19). For
a quick check run single pytest files, one at a time:
`python3 -m pytest tests/python/test_live.py -q -p no:cacheprovider`.

A build for a worktree (checked against `CMakeLists.txt`: the options exist with these
names; `SONAR_BUILD_TESTS` and `SONAR_BUILD_PYTHON` default ON, `SONAR_BUILD_HOLOSCAN`
OFF, `SONAR_ENABLE_CUDA` AUTO, `SONAR_CERES_DIR` `/opt/ceres-nosuitesparse`):

```bash
cmake -S <worktree> -B <build dir outside the worktree> -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSONAR_BUILD_TESTS=ON -DSONAR_BUILD_PYTHON=ON -DSONAR_ENABLE_CUDA=OFF -DSONAR_BUILD_HOLOSCAN=OFF \
  -DSONAR_CERES_DIR=/opt/ceres-nosuitesparse -DSONAR_ALLOW_SUITESPARSE=OFF
cmake --build <build dir> -j1          # one job when other builds run at the same time
export PYTHONPATH=<worktree>/python OMP_NUM_THREADS=1
```

The Python module is written into `<worktree>/python/gpusonar/` (ignored by git). **The
built module in the main checkout can be older than the C++ sources**: on 2026-11-24 it
dated from 2026-11-22 and `tools/sonar_live.py --scenario ...` failed there with
`KeyError: 'classify_max_fit_rms_bearing'`. Rebuild before trusting a local failure.

Bellhop (GPL, run as a separate program) was built in the session scratchpad at
`<scratchpad>/at/Bellhop/bellhop.exe`. The scratchpad is not durable. If it is gone,
`tools/build_oracles.sh` rebuilds Bellhop, Bellhop3D and KRAKEN.
Without `GPUSONAR_BELLHOP` the Bellhop-dependent tests skip, and the run ends with a line counting
those skips; with `GPUSONAR_REQUIRE_ORACLES=1` they fail instead. Other switches (the full table is
`docs/handoff/TESTS.md` section 1): `GPUSONAR_BELLHOP3D` (Bellhop3D; else `bellhop3d.exe` beside
Bellhop), `GPUSONAR_SLOW_ORACLES=1` (the 8-minute acceptance of the `eigenray3d` tier, not part of
the suite; its skip names no oracle variable, so `GPUSONAR_REQUIRE_ORACLES=1` does not fail it),
`GPUSONAR_HSB_STUDY=1` and `GPUSONAR_HSB_TIMING=1` (Sensor Bridge measurements, not tests),
`GPUSONAR_DOPPLER_STUDY=1` (the margin tests of the Doppler scenes) and `GPUSONAR_DOPPLER_CHAIN=1` (the
chain's acceptance on `flank_patrol_doppler_tone`; not run yet: the scene's simulation did not finish in
4 h 40 min on this machine, D-237); in the GPU job,
`GPUSONAR_SHARD=I/N` (one shard of the test files, D-239).

The Holoscan and Sensor Bridge tests cannot run here (the SDK cannot be imported in this
container); they skip locally and run in the GPU job.

A container restart kills background jobs (it happened once, mid-suite). Committed and
pushed work is safe; anything only in the scratchpad or a running process is not.

## 4. GPU work goes through Hugging Face Jobs

The owner's instruction: "Use huggingface to test all gpu things" and "You can use hf to
build cuda". The `hf` CLI is installed and logged in. The pattern:

1. `git -C <worktree> archive --format=tar.gz -o <name>.tar.gz HEAD` (your own tarball name)
2. `hf upload mszarski/gpusonar-ci <name>.tar.gz <name>.tar.gz --type dataset`
3. Launch a job that downloads the archive and runs `tools/gpu_job.sh`.
4. Remove the local tarball afterwards (the disk is small). Leave the dataset as it is.

The launcher script lives in the scratchpad (`hf/launch_named.sh`; not durable); its whole
content is:

```bash
#!/usr/bin/env bash
# launch_named.sh <tarball-name.tar.gz> <flavor> <timeout> [-e KEY=VAL ...]
tb=$1; flavor=$2; to=$3; shift 3
hf jobs run --detach --flavor "$flavor" --timeout "$to" --secrets HF_TOKEN "$@" \
  nvidia/cuda:12.6.3-devel-ubuntu24.04 bash -c "apt-get update -qq && apt-get install -y -qq python3-venv python3-pip > /dev/null && python3 -m venv /opt/b && /opt/b/bin/pip install -q huggingface_hub && export PATH=/opt/b/bin:\$PATH && hf download mszarski/gpusonar-ci $tb --type dataset --local-dir /w > /dev/null && mkdir -p /w/s && tar -xzf /w/$tb -C /w/s && cd /w/s && (bash tools/gpu_job.sh; echo JOB_EXIT \$?)" \
  2>&1 | grep -o "mszarski/[0-9a-f]\{24\}" | head -1
```

The full suite before a merge (what "2278 passed, 102 skipped, 1 xfailed" was run with):

```bash
launch_named.sh <name>.tar.gz l4x1 3h -e GPUSONAR_ORACLES=1 -e GPUSONAR_CERES=1 \
  -e GPUSONAR_HOLOSCAN_CPP=1 -e GPUSONAR_HSB=1 -e GPUSONAR_REQUIRE_ORACLES=1
```

The same suite in shards (D-239): three jobs at once, the same switches plus `GPUSONAR_SHARD`; each shard
runs its share of the test files (`tools/test_shards.py`, weights in `tests/python/shard_weights.json`;
the C++ suites in shard 1). **The suite has passed when every shard has.** Each shard pays the whole build,
so three cost more GPU time than one job; the first trial was done about 60 minutes after launch, against
85 minutes of tests alone unsharded (D-239).

```bash
for i in 1 2 3; do
  launch_named.sh <name>.tar.gz l4x1 3h -e GPUSONAR_ORACLES=1 -e GPUSONAR_CERES=1 \
    -e GPUSONAR_HOLOSCAN_CPP=1 -e GPUSONAR_HSB=1 -e GPUSONAR_REQUIRE_ORACLES=1 -e GPUSONAR_SHARD=$i/3
done
```

Job modes of `tools/gpu_job.sh` (its header, `bash tools/gpu_job.sh --help`, lists every
switch and variable; uploads happen only with `-e GPUSONAR_UPLOAD_REPO=<dataset>`, into the
folder `GPUSONAR_UPLOAD_PATH`, which defaults to the mode's name):

| Mode | How | Typical time on `l4x1` | Output |
| --- | --- | --- | --- |
| whole test suite (default, `GPUSONAR_JOB` unset) | the command above; `GPUSONAR_HSB=1` builds Sensor Bridge 2.7.0 and implies the Holoscan C++ operators | about 45 to 50 min (49 min 55 s for D-201's); timeout 3h | log only; grep for `passed` and `JOB_EXIT` |
| `gallery` | add `-e GPUSONAR_JOB=gallery -e GPUSONAR_ORACLES=1 -e GPUSONAR_CERES=1` (`GPUSONAR_GALLERY_ARGS` appended to `tools/gallery.py --backend cuda`) | about 40 min when last recorded; the 26 scenes themselves take 489 s (`docs/SCENARIOS.md`) | 57 files in the folder, default `gallery` |
| `clseval` | `-e GPUSONAR_JOB=clseval -e GPUSONAR_ORACLES=1` (`GPUSONAR_CLSEVAL_ARGS`, e.g. `--seeds 10 --commit <hash>`) | about 50 min for 12 scenes x 10 seeds | `classify_rules_eval.json`, `.md` and `captures/` |
| `falsealarm` | `-e GPUSONAR_JOB=falsealarm` (`GPUSONAR_FA_ARGS`, e.g. `--maps long_range`) | not recorded; the three-row static count timed out at about 43 000 pings (D-168) | `at_default.json` |
| `benchall` | `-e GPUSONAR_JOB=benchall -e GPUSONAR_ORACLES=1 -e GPUSONAR_CERES=1` (`GPUSONAR_SCALE` default 1, `GPUSONAR_BENCHALL_ARGS`, `GPUSONAR_BENCHALL_PYTEST`) | timeout 150m | `gpu_benchmark_all.json` and `.md` |
| `motion` | `-e GPUSONAR_JOB=motion -e GPUSONAR_CERES=1` | not recorded | log only: `tests/python/test_motion*.py`, then `tools/motion_study.py timing` |
| `elevbench` | `-e GPUSONAR_JOB=elevbench` (`GPUSONAR_ELEVBENCH_ARGS`) | timeout 60m | `popoto_stage2.json` |
| `liveablate` | `-e GPUSONAR_JOB=liveablate` (`GPUSONAR_LIVEABLATE_ARGS`) | not recorded | `live_chain_ablation.json` |
| `benchmark` | `-e GPUSONAR_JOB=benchmark` (`GPUSONAR_SCALE` 2, `GPUSONAR_PIXEL` 0.05 m, `GPUSONAR_MODES` `sas,fls,flank`) | not recorded | log only: one `BENCHMARK_JSON {...}` line |
| `doppler` (D-241) | `-e GPUSONAR_JOB=doppler -e GPUSONAR_ORACLES=1` (Bellhop): `tools/doppler_gpu_runs.sh /work/doppler`, simulator and chain on the device. Sub-runs `GPUSONAR_DOPPLER_RUNS` (default `tone,falsealarm,timing`; since D-258 `tone` and `ls12` take a second pass on the same cached pings with the tracker taking the tone's range rate; only when named: `ls12`, the 12 km large-submerged scenes of D-243, since D-246 with the train scene run again with the grating-lobe notch off on the same cached pings, `hfm12`, D-247: `popoto_12km_large_submerged` with the linear and the hyperbolic sweep at 1 s and 4 s, plain and with the bank, `deep12`, D-260: `popoto_12km_deep_sea_state_5` at sea states 3, 4, 5 case by case (`GPUSONAR_DOPPLER_DEEP12_CASES`; any `<ss>_<lfm\|hfm><seconds>[p<level>][_cfar]`, e.g. the energy candidates of DOPPLER.md DM.5), and `regen`, D-260: the outputs D-251 made stale through `tools/regen_stale.py` (`GPUSONAR_DOPPLER_REGEN_PARTS`, default `bank,d3,ls,baseline,p12km`; `GPUSONAR_DOPPLER_REGEN_TREE` names the tree)), each under its own limit `GPUSONAR_DOPPLER_TIMEOUT_TONE`, `_FALSEALARM`, `_TIMING`, `_LS12`, `_HFM12`, `_DEEP12` (a case's), `_REGEN` (a part's) (80m, 30m, 40m, 110m since D-258 (80m before), 150m, 45m, 60m); `GPUSONAR_DOPPLER_DEEP12_ARGS` is appended to each `deep12` call; `GPUSONAR_DOPPLER_FA_PINGS` (4000) and `GPUSONAR_DOPPLER_FA_ROW_SECONDS` (780) size the false-alarm count; `GPUSONAR_DOPPLER_TONE_ARGS`, `_FA_ARGS`, `_TIMING_ARGS`, `_LS12_ARGS`, `_HFM12_ARGS` are appended; `GPUSONAR_DOPPLER_BACKEND=host` and `GPUSONAR_DOPPLER_QUICK=1` are a plumbing check on a CPU that measures nothing (`tests/python/test_doppler_gpu_job.py` runs it with `GPUSONAR_DOPPLER_JOB_QUICK=1`). `bash tools/doppler_gpu_runs.sh --help` is the authority | not recorded as a whole; the tone scene's 6 pings simulated in 189 s; the bank's false-alarm rows stopped at their one-hour limit (D-241) | `doppler_tone_eval.json`, `doppler_bank_at_default.json`, `timing_*.json`, with `ls12` `ls12_tone_train.json`, `ls12_tone_train_lobe_off.json` and `ls12_tone_alternate12.json`, with `hfm12` `hfm12_<wave><T>[eq]_*.json`, with `deep12` `deep12_<case>.json` (summarised by `tools/deep12_eval.py`), with `regen` the committed files' names with the stale parts replaced, with the D5 passes `doppler_tone_eval_rate.json`, `ls12_tone_train_rate.json`, `ls12_tone_alternate12_rate.json`, one log a sub-run, uploaded after each sub-run (default folder `doppler`); exit status the first failing sub-run's (124: stopped by its limit) |
| `depthrange` (D-266) | `-e GPUSONAR_JOB=depthrange`: `tools/depth_range_sweep.py --backend cuda`, the depth estimate against range on the three-row face with the 2 s hyperbolic baseline, three sweeps (deep water at own depth 225 m and 30 m, 100 m of water at 30 m), 1.7 to 12 km, targets at 4, 30, 60, 150 and 300 m, two seeds, 3 pings; `GPUSONAR_DEPTHRANGE_ARGS` replaces the sweeps' common arguments (to split the ranges over two jobs); then the pooled tables. `docs/tasks/DEPTH_RANGE.md` DR.7 | not recorded in the documents (job 6b0aaa826c5d61fff18c3ee2 ran it; a case costs 2 to 8 min on the CPU) | `deep_own225.json`, `deep_own30.json`, `shelf100_own30.json` and their logs, uploaded after each sweep (default folder `depthrange`), then `tables.md` and `pooled.json`; the tables of that job are not yet committed (`sonar-1l0m`) |

Flavours and pitfalls:

- **`l4x1` is the reference flavour.** Every timing and every Sensor Bridge number in the
  documents is from an L4 unless it says otherwise.
- L4 capacity can be unavailable for hours (the job sits in `SCHEDULING`). `a10g-large`
  is an acceptable fallback for tests. **`a10g-small` is not**: its 16 GB ran out in the
  Ceres build and the job hung at 0 % CPU, and it has a 3-CPU cgroup quota the affinity
  does not show (D-173).
- `hf jobs wait <id>` returned early while a job was still `SCHEDULING`. Poll
  `hf jobs inspect <id> --format json` instead, then `hf jobs logs <id>`.
- One GPU job at a time per agent; timeout 3 h at most.
- Results come back with
  `hf download mszarski/gpusonar-ci --type dataset --include "<folder>/*" --local-dir <dir>`.
  Copy gallery files into `docs/figures/`. `tests/python/test_realistic_scenes.py::test_documented_full_size_classes`
  reads `docs/figures/gallery.json`, so a new gallery can break it; update its expectations
  honestly, with a comment saying what got worse.

## 5. Where things are

| Need | Read |
| --- | --- |
| The handoff set: entry point, hardware, software, tests, outputs | `docs/handoff/README.md`, `HARDWARE.md`, `SOFTWARE.md`, `TESTS.md`, `OUTPUTS.md` |
| What works, what is wrong, questions for the owner | `docs/STATUS.md` (open items 1 to 95, owner questions 1 to 68) |
| Why anything is the way it is | `docs/DECISIONS.md` (D-1 to D-267; later entries amend earlier ones) |
| Every measured number and its source | `docs/RESULTS.md` (the Sensor Bridge numbers of D-153 to D-192 are summarised there, and D-193 to D-201 in a summary table before "Not yet measured"; D-160, D-161, D-176 and D-181 are measured in their `DECISIONS.md` entries only) |
| Scenes, with figures and per-scene results | `docs/SCENARIOS.md`, `docs/figures/` (gallery of D-253) |
| Work waiting on owner input, written as briefs for an agent | `docs/tasks/VEHICLE_ADAPTERS.md` (`sonar-o7m`), `docs/tasks/REAL_DATA.md` (`sonar-sul`); `docs/tasks/PROPAGATION_3D.md` (`sonar-rcq`: the scope of 3-D propagation, its packages P1 to P6 done, owner questions in section 6); `docs/tasks/DOPPLER.md` (`sonar-u1fu`: packages D2, D4, D3 and D8 done, "D3 done" says what was not measured, "D8" the device port and the defect of D8.5, "Large submerged at 12 km" the 12 km scenes, "D5" the tracker's range rate, "Deep site at 12 km" the measurement that failed the 1 s baseline for the resolved hull (DM.1 to DM.7), "The baseline becomes the 2 s hyperbolic sweep" the baseline since D-262 (HB.1 to HB.4); its question 1 answered, D-231) |
| The Popoto array design, budget, builds | `docs/DESIGN.md` (generated by `tools/design_report.py`), `docs/REFERENCE_DESIGN.md`, `docs/DEPLOYMENT.md` (since D-257 also "A forward-looking sonar"), `docs/design/` (the 12 km trade study per site and class: `docs/design/popoto_trade_study.md`, D-233; its sections 0 and 0a carry the deep site's measured table since D-260) |
| Config keys | `docs/CONFIGURATION.md`; files in `configs/{sonars,scenarios,environments,deploy,builds,chains,nav}` |
| Physics and paper reading | `docs/PHYSICS_NOTES.md`, `docs/PAPER_NOTES.md`, `docs/REFERENCES.md` |
| Record format (frozen) | `docs/RECORD_FORMAT.md` |

Code worth knowing first:

- `libs/live/` and `python/gpusonar/live.py`: the long-range live chain (beamform, detect,
  track, classify, elevation and depth). Two engines, C++ and Python, held equal by tests.
- `python/gpusonar/buildspec.py` and `tools/build_from_spec.py`: the build-spec generator.
  Six Popoto builds in `configs/builds/`; it writes sonar and deployment files and runs
  per-build self-tests (`--check-shipped` verifies the committed files match).
- `tools/gallery.py`: runs 26 scenes end to end and writes `docs/figures/gallery.json` (with `--quick`: `build/gallery_quick/`, not committed).
- `tools/classify_rules_eval.py`: the same scenes over several seeds, with saved captures
  that can be re-scored offline; the ten-seed numbers of D-159 to D-181 come from it.
- `tools/sonar_live.py`, `tools/sonar_live_hsb.py`, `apps/sonar_live.cpp`: the live chain
  as a program (Python, through Sensor Bridge, standalone C++).
- `tools/hsb_host_check.py`: a read-only check of a host for Sensor Bridge's socket receiver
  at a Popoto build's rate, and the command that measures it there (`DEPLOYMENT.md`,
  "Checking a host and making the measurement"; not run on such a host yet).
- `tools/hsb_frame_timing.py` (the receive path's cost a frame, D-190), `tools/live_chain_timing.py`
  (chain and synchronisation time a ping on the Popoto trials, D-191), `tools/eigenray_tl_check.py`
  (each eigenray scenario's target loss by each route, D-197), `tools/board_merge.py` (a relay for
  per-board streams; every receiver merges itself since D-200).
- `python/gpusonar/sim/bellhop3d.py`: the opt-in `eigenray3d` tier (Bellhop3D as a separate
  process; `docs/tasks/PROPAGATION_3D.md`).
- `tools/popoto_12km_study.py`, `tools/fjord_rehearsal.py`, `tools/design_report.py`: the
  studies behind the design documents.
- `tools/doppler_tone_eval.py` (D-237: the tone path end to end through both engines, per target and
  ping, with the background's false detections) and `tools/doppler_bank_threshold.py --chain cfar` (the
  CFAR combiner's false detections on noise); `tools/test_shards.py` (D-239: the shard a GPU job runs);
  `tools/doppler_gpu_runs.sh` (D-241: the `doppler` job mode's sub-runs; `--help` lists its variables).
- `tools/deep12_eval.py` (D-260: each deep-site case set against the planner's prediction from the trade study's
  own functions; `--predict CASE`, `--predict-cases`), `tools/regen_stale.py` (D-260: which parts of each output
  D-251 made stale, `--list`, and their rerun), `tools/gallery.py --evidence` and `--reverb` (D-259, opt-in: each
  track's classifier evidence; an old reverberation law for comparison).
- `tools/depth_range_sweep.py` (D-266: depth against range on the three-row face through both engines; `--physics`,
  `--table`; the GPU job mode `depthrange`; `docs/tasks/DEPTH_RANGE.md`), `tools/short_range_study.py` (D-265: the
  short-range mode's model figures; `docs/design/short_range_mode.md`).
- `libs/live/include/sonar/live/doppler_batched.hpp`, `libs/live/src/doppler_batched.cpp`,
  `libs/live/src/doppler_cuda.cu` (D-242): the sweep bank and the tone bank batched over an executor
  (`doppler.batched`), on CPU threads or the device; the shared kernels are launched through their own
  files' launchers (section 11).

## 6. State of verification (to D-267)

- **GPU, latest (NVIDIA L4, three shards, HF jobs 6b0aaa8054072e8b716d27fd, 6b0aaa8054072e8b716d27ff, 6b0aaa816c5d61fff18c3ede, the tenth batch's tree `904c744` of D-264 to D-266, merged as `cd620b4`, the same switches as D-263): 3567 passed, 126 skipped, 9 xfailed, none failed** (D-267). The five more expected failures are D-266's strict ones (`tests/python/test_depth_range_sweep.py`: the shallow-water depth dishonesty F1 and the timing stage's floor F2; they hold defects, not fixed). One conflict in building the batch (`docs/handoff/SIMULATION.md`: the timing rows of the last sweep kept, the two new scenes' rows added). At this tree: 102 scenario files; 179 Python test files (2147 `def test_` functions); GoogleTest 220 cases by the sources' count (219 `TEST`, 1 `TEST_P`).
- **The suite before it (NVIDIA L4, three shards, HF jobs 6b0a8a766c5d61fff18bd22e, 6b0a8a766c5d61fff18bd232, 6b0a8a7754072e8b716cb7a3, the ninth batch's tree `dfe9733` of D-262, merged as `e16e214`, the same switches as D-261): 3507 passed, 124 skipped, 4 xfailed, none failed** (D-263). The suite takes longer with the 2 s pulse: the slowest shard 60 minutes of tests (40 before). No merge conflict.
- **The suite before it (NVIDIA L4, three shards, HF jobs 6b0a30146c5d61fff18ac1ab, 6b0a301554072e8b716b9b00, 6b0a30156c5d61fff18ac1ae, the eighth batch's tree `3fb6bf4` of D-257 to D-260, merged as `08ddfab`, the same switches as D-254): 3494 passed, 124 skipped, 4 xfailed, none failed** (D-261). **The first run of the batch (at `ff04b58`) failed 3 tests**, 3488 passed: all from the deep-site package's new files in test files its agent had not run (a list of shipped sonars; D-227's "a scene states its site once"; a no-bottom warning), fixed in the scenes and the list, records unchanged (hashes; D-260). One conflict in building the batch (`docs/tasks/DOPPLER.md`: both appended sections kept). The two more skips than D-254's 122 are not itemised in D-261 (`docs/handoff/TESTS.md` section 1). The container was restarted during the batch; two agents' background runs were lost, no committed work.
- **The suite before it (NVIDIA L4, three shards, HF jobs 6b0947c254072e8b71697a54, 6b0947c254072e8b71697a58, 6b0947c36c5d61fff188c461, the seventh batch's tree `81364e0` of D-251 to D-253, merged as `e06fac8`, the same switches as D-250): 3392 passed, 122 skipped, 4 xfailed, none failed** (D-254). D-250's fifth expected failure, `test_hsb.py`'s load-dependent frame-loss limit, did not occur. The tracker-view branch alone: 3357 passed. One conflict in building the batch (`docs/handoff/SIMULATION.md`: the fix branch's row kept).
- **Before that (NVIDIA L4, three shards, HF jobs 6b0935f554072e8b71696d01, 6b0935f66c5d61fff188b937, 6b0935f754072e8b71696d05, the sixth batch's tree `67d962b` of D-246 to D-248, merged as `bda7c7d`, the same switches as D-245): 3321 passed, 121 skipped, 5 xfailed, none failed** (D-250). The fifth expected failure is `test_hsb.py`'s measured frame-loss limit at 102.4 kHz (D-186), which reports as an expected failure when the job's machine loses frames; two of the four more skips are the visualization's browser checks ("Node is not installed"). A first launch was cancelled early: two visualization tests still expected 96 scenario files (corrected in `67d962b`). The merge differs from the tested tree only in `docs/DECISIONS.md` and the issue export.
- **Before that (NVIDIA L4, three shards, HF jobs 6b090cd36c5d61fff188820e, 6b090cd354072e8b71692fa9, 6b090cd46c5d61fff1888214, the fifth batch's fixed tree `a37e0be` of D-241 to D-243 with D-242's repair, merged as `f40d4c3`, the same switches as D-240): 3002 passed, 117 skipped, 4 xfailed, none failed** (D-245). **The first suite on the batch (`batch5` at `d54deb9`) failed 4 tests** (2996 passed): bit-identity comparisons between two programs on cuda with every Doppler option off, broken by a device kernel instantiated in two `.cu` files (D-242; section 11); fixed without changing the four tests, and merged only after the suite on the fix passed.
- **Before that (NVIDIA L4, HF job 6b08b9f554072e8b71681cd2, the fourth 2026-11-26 batch's tree `da015ba` of D-236 to D-239, merged as `8581533`, with `GPUSONAR_SLOW_ORACLES=1`): 2933 passed, 113 skipped, 4 xfailed, none failed** (D-240; the two more skips than D-234's 111 are the chain acceptance on `flank_patrol_doppler_tone`, which needs `GPUSONAR_DOPPLER_CHAIN=1`, and one `--help` check of a new tool). The sharding (D-239) was added on top and passed as three shards: 2941 passed, 114 skipped, 4 xfailed (jobs 6b08c1c16c5d61fff1879ac2, 6b08c1c254072e8b71683e13, 6b08c1c36c5d61fff1879ac7; the eight tests of `test_shards.py` and one more `--help` skip). Before it, the third batch's tree `0a8b477`: 2886 passed, 111 skipped, 4 xfailed (job 6b084a4cd91f621b930b6331, D-234); the second batch's tree `de3717b`: 2823 passed, 103 skipped, 4 xfailed (job 6b080457d91f621b930a652a, D-229); the merged tree `b88c458`: 2763 passed, 102 skipped, 4 xfailed (job 6b07ea74d91f621b930a343d; D-225, which lists the six failures of that batch's earlier suites and their fixes). Before them, D-202's tree: 2301 passed, 102 skipped (job 6b079ab0d91f621b93098915); D-201's: 2278 passed, 102 skipped, 1 xfailed (job 6b078a26d91f621b93096bc7).
  CUDA build with Bellhop, Bellhop3D, KRAKEN, PyRAM, Ceres, the Holoscan C++ operators and
  Sensor Bridge 2.7.0, `GPUSONAR_REQUIRE_ORACLES=1`. The skips are data and tools not in the
  job and the opt-in measurements. The 3-D work after D-201 (`eigenray3d`, `flank_seamount_3d`)
  is covered by a later job. Every decision from D-183 to D-201 names its own job and count
  (`docs/handoff/TESTS.md`, "Current counts", lists them).
- **CPU: the last whole-suite run on record is the tree of D-157** (`605a4e1`, 2026-11-22: 157 of
  157 ctest entries; `docs/STATUS.md`, the paragraphs before "Open questions for the project
  owner"). Since then each package ran its affected test files locally and the whole suite on the
  GPU job. A whole CPU run at HEAD is not on record.
- The gallery (`docs/figures/`, `gallery.json`) is from the L4 job 6b0a8a776c5d61fff18bd237 on the ninth
  batch's tree `dfe9733` (D-262, committed whole as `4720050`), with the 2 s hyperbolic baseline. **Worse against
  the file before it:** `popoto_env_winter_sand_200m` the ship's track 2.8 -> 9.3 m from its outline (one more
  uncertain track); `popoto_env_deep_silt_1000m` confirmed tracks 3 -> 2, the ship's track 301 -> 305 m off (wrong
  before and after); `popoto_12km` one multipath track now uncertain (its ship 2.1 -> 1.1 m); `popoto_env_gravel_30m`
  10.0 -> 8.4 m, tracks 2 -> 3. FLS, SAS, flank and pool scenes numerically as before. Stale with the baseline:
  `docs/tasks/doppler_deep12_eval.json` (the baseline's own cases not merged, `sonar-n8ho`),
  `docs/handoff/examples/live_popoto_12km_*`, `docs/handoff/hardware_builds.json` at 1000 m (inside the blind
  range), `docs/design/popoto_12km.json` (`docs/STATUS.md` item 99). The gallery before it was from the L4 job
  6b0947c46c5d61fff188c465 on the
  seventh batch's tree `81364e0` (D-253, committed whole as `082dd21`), under the per-frequency
  reverberation Doppler (D-251). **Worse against the file before that:** `fls_170k_traffic` one mover not on a
  moving object; `popoto_env_gravel_30m` confirmed tracks 3 -> 2. Explained since (D-259, `sonar-4pon` closed):
  the first is a wake track that appears in 5 of 8 full-size seeds under the default law and 3 of 8 under the
  old (a rate, not shown to be D-251's), the second a track on no object; `flank_patrol`'s difference is D-216's.
  `sonar-pli3` is closed (its one-detection differences superseded, not explained). The outputs D-251 made stale
  were regenerated (D-260, `sonar-rar0` closed; worse: the closing hull untracked at the last alternating ping,
  `popoto_12km_doppler`'s default chain 4 false detections on 4 target-free pings). Still stale: `sonar-tmvu`'s
  (`docs/STATUS.md`, the list of stale generated files; still open at D-263).
- Not in either suite: the generator's self-tests. **With the 2 s baseline two fail (D-262, not loosened):** the
  three-row rehearsal's depth honesty at 1.7 km (28 m, 35 m and none for 60 m true; the geometry, it fails with the
  1 s sweep there too; explained by D-266 as F1, the rows' depth dishonest there: `sonar-4e2h`) and the one- and
  two-row "free" range check, which misses its 3 m limit: -3.06 m (1x32) and -3.31 m (2x32) by D-264 (the offset
  depends on the echo's SNR, not on the pulse length, D-264 correcting D-262; with the opt-in
  `cfar.range_edge_correction: pulse` both pass, within 0.06 m; by default they still fail); the three-row "free" range check now
  compares with the truth displaced by the predicted bias. Before: 42 of 42 on both engines for the six builds
  (D-143, 2026-11-19, the 1 s sweep); in between only checked that every deploy and build file is accepted by both
  engines (D-166, D-177).
- Nothing has run on real hardware or real data. Nothing has run on DGX Spark or Jetson.

## 7. What was done in D-158 to D-201, and what each broke or left open

Numbers are in the `DECISIONS.md` entries. Issues named here are open in `bd` unless said.

**Live chain correctness**

| Decisions | What | Worse, or left open |
| --- | --- | --- |
| D-158 | The detector's first scale is checked to be the single sample; pings whose navigation sent no velocity are counted (`pings_without_velocity`) | A counter, not a refusal: such pings are processed with velocity 0 |
| D-159 | Bearing sigma kept; opt-in per-axis outlier test; the fit limit grows with range (on in the live chain) | The 12 km cross-range error is not reduced (mean -23.3 m, seeds 51-60); false movers under the grown limit unmeasured (`sonar-kal`); the outlier test has little power on six points (`sonar-69f`) |
| D-160 | **The leading edge is the default range** (`cfar.range_estimate: first`) | A detection no elevation places reads 1 to 3.6 m short for a strong point echo; the ship's depth at 12 km with three rows is 8-35 m (true 4 m); blanking and burst filter untested with it (`sonar-9hy`) |
| D-165 | No fitted velocity from echo delays alone | A source without timestamps gives speed 0, never a mover; its tracker has no clock unless `tracker.ping_interval_s` is set (D-175; `sonar-y7k`) |
| D-166, D-177 | Every options dict in both engines refuses a key it does not read, naming the nearest | An out-of-repo config with a stale key now stops; the stages' own key=value readers (`tonal_`, `multipath_depth_`, `coverage_`) are not covered |
| D-167 | Navigation unit to array: `nav.lever_arm_m`, `nav.mount_rpy_deg` | Zero until surveyed; the array's velocity is not corrected (`sonar-0ca`); generated deploy files do not carry the keys (`sonar-vjq`) |
| D-172, D-175 | Echo groups within one gate of their object; bounces never a group's object; an earlier detection is taken at the track's time; `pings_without_time` | Grouping gate sensitive to one point (`sonar-36z`); bounce grouping not measured on captures with multipath on (`sonar-ome`); two-ship scene's ship B is a later arrival on 2 seeds (`sonar-i28`) |
| D-181 | Opt-in LFM range-Doppler correction `range_doppler: off / ownship / full` | Off by default; `far_mover` and the closing FLS boat get worse with it; in the live chain held by tests only, not measured on the scenes |

**Sensor Bridge path**

| Decisions | What | Worse, or left open |
| --- | --- | --- |
| D-161, D-177 | Flank stream receive: blocks placed by sample counter, pings lost whole counted, repeats told from restarts | Still miscounts an index skip that also lost samples, and a real loss with jitter beyond half an interval; `reorder_blocks` and `max_held_bytes` are not exposed as options and the passive assembler does not reorder (`sonar-thu`) |
| D-162 | `receive --chain-omp-threads`: the chain's OpenMP below the core count | About +10 % chain time on L4; assumes the single-thread scheduler (`sonar-vmm`) |
| D-163 | `receive` always stops (`hsb.ReceiveStop`); exit status 3 when stopped short | An idle stop after the board has gone ends with `shutdown_error` and `os._exit`; the idle path only ran in runs set up for it (`sonar-3wd`) |
| D-173 | Loopback tests for `serve --scenario` and `engine: cpp`; thread count from the CPU quota | Pace table measured on a10g-small, not repeated on L4 (`sonar-e2u`); `serve` has no `--backend` (`sonar-l6j`) |
| D-180 | `sonar_live_hsb receive` builds navigation with `NavListener.from_config` | none recorded |

**Motion estimation**

| Decisions | What | Worse, or left open |
| --- | --- | --- |
| D-164 | Ping 0's height is an unknown tied to its DVL sample | Some seeds a little worse, all inside the bounds; the bounds are ours, the paper states none |
| D-171 | Range windows with no seabed dropped by energy (`min_window_energy_db: -20`); the two SAS gallery scenes' seabed widened | The scenes are new draws, not paired with before; `wreck_sas` sway 0.045 -> 0.053 wavelengths; the `minefield_sas` benchmark row in RESULTS is of the old scene (`sonar-egj`); delays read 0.2-0.9 % below the model in every window (`sonar-oii`) |

**Gallery and evaluation**

| Decisions | What | Worse, or left open |
| --- | --- | --- |
| D-168 | False alarms counted at the default budget: 0.82 of it pooled; constants kept | Two-row cell mean runs 1.26 times its share (`sonar-031`); three-row static count unfinished (`sonar-4xn`); the 9 dB floor drops most stacked-row noise detections (`sonar-xl7`) |
| D-169, D-170 | The closing boat's 4.6 m is its propeller; what remains of the FLS class symptoms | `fls_120k_rough_surface` glint movers (78 over ten seeds) kept as a known limit; the winter-sand stray detection (`sonar-2o5`); FLS slant-range bias (`sonar-m1r`) |
| D-172 | Gallery places Popoto echoes from the array at reception | `winter_sand_200m` reads ahead of the ship's centre |
| D-174, D-179 | Gallery regenerated twice | `wreck_sas` and `minefield_sas` numbers are new draws; `gravel_30m` 10.0 m short of the hull; `popoto_12km` 9.8 m from the ship's centre |
| D-176 | Opt-in sub-grid Popoto bearing (`POPOTO_BEARING = "parabola"`) | Worse on three measures, so off; gallery tracker bearing sigma 1 degree against 0.05-0.3 of scatter (`sonar-bu3`) |
| D-178 | The gallery's range in the multipath scenes is the leading edge; `flank_patrol` is classified | `gravel_30m` 34.3 m beyond -> 10.1 m short, mover -> uncertain on seed 52 (explained in D-181, not cured; the scene's lower peak since D-137 is `sonar-9b8`); `popoto_12km` seed 60 3.0 -> 65.4 m (`sonar-tlc`); `small_vehicle` never detected (explained by D-198: below the scene's reverberation, kept as a hard target) |

**Since D-182** (the handoff set is D-182)

| Decisions | What | Worse, or left open |
| --- | --- | --- |
| D-183 | `source.sound_speed`: the receive window's layout speed in every receiver; `sonar_live` reports an error on its UDP path | The key both lays out the window and is the record's speed until navigation replaces it (`sonar-qei`); `receive` against `serve --scenario` multi-board at a non-nominal speed not verified (`sonar-ui5`) |
| D-184, D-185, D-200 | One stream per board, merged in every receiver (Python source, `sonar_live` with and without `--listen`, the Holoscan source operator, `sonar_live_hsb.py receive`); a board clock's rate configured or estimated | A wire-format use of two reserved header fields (the owner accepted it); capture starts must agree with true time to about 0.8 ms; ranges not corrected for a clock's rate (`sonar-9a2`); through Sensor Bridge all boards share one data channel and must have equal channel counts; the Holoscan application and `receive` ignore the clock-rate keys of a whole-array stream (`sonar-pnyv`); never run against a real board or network |
| D-186 to D-189 | Sensor Bridge at Popoto rates: the loss is at the start of a run; the queue holds 4 s of stream; damaged frames named; a host check (`tools/hsb_host_check.py`) and runbook | A board must stream 0.25 s before the first transmit that matters, nothing enforces it (`sonar-a75`); the host with Sensor Bridge's own socket setup and the RoCE receiver are unmeasured (`sonar-02c`); the containers refuse every such setting |
| D-190 to D-192, D-195 | The frame path decodes only what a window takes; boards resampled by board in both engines; weights from three trig calls; the decimate-first correction by board. C++ 0.048 / 0.054 / 0.065 s a ping at 32 / 64 / 96 channels on an L4 (from 1.05 to 1.83 s at 64 and 96) | Outputs the same to the bit on every case measured; a sample could round one float32 step the other way on other data (D-192). The first 32-channel loopback of a job shows 6 to 11 ms worst delay (`sonar-6rs`) |
| D-193 | The boundary models against Gallaudet 2001 at 68 kHz | The surface model lies 3.4 and 13.0 dB above the data at 3 m/s ("a lower bound" withdrawn against data; `sonar-64sm`) |
| D-194 | Stale simulator statements corrected (the TR 0702 matrix, RESULTS, SCENARIOS, SIMULATION, REFERENCES) | none |
| D-196 | The K shape of reverberation from a density per cell (opt-in) | No density is known for any seabed; texture correlation between cells and pings not modelled (`sonar-6fc`) |
| D-197, D-199 | The eigenray tier against Bellhop3D's N × 2-D mode explained | The fjord beds kept; whether the 2-row scenario keeps its 12-arrival limit is the owner's (`sonar-2e9e`) |
| D-198 | `flank_patrol`'s small vehicle is below the scene's reverberation; left as a hard target | The link budget and coverage model had no reverberation term; since D-221 the term exists, and since D-227 the shipped planning sites and face deployment files state their scattering (illustrative values) |
| D-201 | Image sources off a tilted bed and walls as any plane (opt-in); four multipath scenes | The FLS chain tracks multipath ghosts as objects (`sonar-izys`); SAS sharpness falls to a third with bed multipath, and every other SAS scene is free field (`sonar-ya2c`); wall coefficients are placeholders; no diagrams of walls (`sonar-ttr1`) |
| (2026-11-26) | The opt-in `eigenray3d` tier (Bellhop3D), the cross-slope warning, one top-level `bathymetry`, `flank_seamount_3d` (`docs/tasks/PROPAGATION_3D.md` P1 to P6) | No reverberation over relief at that tier; its acceptance test takes 8 minutes and runs only with `GPUSONAR_SLOW_ORACLES=1` (the entry is D-202) |
| D-204 to D-225 (2026-11-26) | Five 3-D scenes; the general image construction in C++; the conformance alignments applied (grain-size relations, the fit's small-slope and prior, elastic rock in the simulator, Bellhop and the live chain, `dahl_1997` and the geometric-mean bistatic rule, the large-scale slope law); `penetration: fluid` (the two buried scenes) and `rough`; the engine never silent and recorded; the link budget's reverberation term; four papers and five site rows | Surface reverberation about 1 dB lower in multipath scenes; SPACE'08 steep tails 1.89 measured over model, the Kirchhoff tail rejected as the cause; rock shear unconfirmed; `rough` costly on large scenes; design budgets noise-only (no site states scattering); three fit tests and two other tests weakened (D-216, D-223, D-225); gallery and `sst_validation` stale |
| D-226 to D-229 (2026-11-26) | Williams-Jackson (1998) and Jackson-Briggs (1992) as opt-ins; the sites' boundary scattering in the depth plan and the coverage pictures; the Doppler workstream scoped (`docs/tasks/DOPPLER.md`); the owner's answers 36 to 42 | The +5 dB class over the 100 m shelf reaches 2.56 km in the plan (8.02 to 9.58 before); every site value illustrative; `flank_patrol` worst margin error 3.4 dB; simulated surface reverberation has no Doppler spread at long range |
| D-264 to D-267 (2026-11-28) | Depth against range on the three-row face measured (D-266, `tools/depth_range_sweep.py`, `docs/tasks/DEPTH_RANGE.md`, GPU job mode `depthrange`); an opt-in pulse reference for the leading-edge range (D-264, `cfar.range_edge_correction: pulse`, both engines); a short-range mode, opt-in (D-265, the sonar's `short_range` block; `popoto_short_range_mode`, `popoto_elevation_short_range`); the merge (D-267: 3567 passed, 126 skipped, 9 xfailed) | **The deployed stacked-row depth is dishonest in 100 m of water at 1 to 2 km (22 of 28 depths; a ship 4 m deep at 1.7 km reads 95.7 +- 27.4 m, called submerged on 4 of 4 pings); not fixed (`sonar-4e2h`, moves a default, the owner's); five strict expected failures hold it; as deployed (the gradient prior) depth tells nothing from the surface beyond about 2 km and cannot tell a submerged target from a ship at 12 km**; the opt-in timing stage's own dishonest cases (`sonar-1l0m`); the GPU grid's tables not committed (job 6b0aaa826c5d61fff18c3ee2); with the range correction on two tracks end 0.006 and 0.019 m further from the truth, a weak highlight within 3 dB of a stronger skirt read as that skirt, judgement constants, no clutter test (`sonar-82nx`, the default the owner's); the short-range mode costs the long mode 18 % of its update rate, its far echoes in the next long window are not simulated, tracks seen only by short pings are dropped if they come less often than every other ping, no build-spec key, no device run (`sonar-qsna`); the engines disagree on `popoto_face_2row_shallow` without `elevation` (`sonar-hygi`) |
| D-262, D-263 (2026-11-28) | The 12 km baseline transmits a hyperbolic sweep of 2 s at 197 dB, plain filter (D-262, `sonar-7h6q` closed; a default moved; `docs/tasks/DOPPLER.md` HB.1 to HB.4); `projector.waveform.type: hfm` in a build spec; `range_doppler: ownship \| full` refused on a sampled pulse in both engines; the Python multipath-depth window timed as the C++ engine's; the gallery regenerated (job 6b0a8a776c5d61fff18bd237, `4720050`); the merge (D-263) | **Nothing seen inside 1.5 km; the elevation scenes keep the 1 s sweep, no depth at 12 km measured; two face self-checks fail (depth at 1.7 km, `sonar-9035`, explained by D-266 as F1; the 3 m range check by 0.06 m, `sonar-8h09`, corrected by D-264: -3.06 / -3.31 m, set by the echo's SNR, not the pulse length) and one's criterion changed; a mover's range 6 m per m/s off, no range rate; the winter-sand ship's track 2.8 -> 9.3 m; deep silt confirmed tracks 3 -> 2; the projector's longest pulse at that drive on no sheet; the simulator 5 to 17 times slower on a CPU**; the baseline holds sea state 3 at the deep site thinly (+1.28 / +0.73 dB on the hulls, one seed), not sea state 4; `doppler_deep12_eval.json` lacks the baseline's own cases (`sonar-n8ho`); not merged then: `depthrange`, `rangeedge`, `shortmode` (merged since, D-267) |
| D-255 to D-261 (2026-11-27 and 28) | The owner's decisions: the baseline at sea state 3, an alternate at sea state 5, an FLS deployment block (D-255); after the deep site's measurement, the baseline's waveform a 2 s hyperbolic sweep at 197 dB and sea state 5 as needing a larger transmitter not yet identified (D-256; done in D-262, merged D-263). FLS deployment files and `look: boresight`, opt-in (D-257); tracks use the measured range rate, `tracker.range_rate`, opt-in (D-258); the gallery's differences explained, gallery `--evidence` and `--reverb` (D-259); the deep site at 12 km measured, the scenes `popoto_12km_deep` and `popoto_12km_deep_sea_state_5`, the placeholder alternate build `popoto_face_3x32_ss5`, `tools/deep12_eval.py`, the stale outputs of D-251 regenerated with `tools/regen_stale.py`, the `deep12` and `regen` sub-runs (D-260); the merge (D-261) | **The shipped baseline (1 s, 197 dB, linear, plain filter) at the deep site at sea state 3 holds neither the closing nor the oblique `large_submerged` hull nor the ship (0 of 6 pings each); the alternate fails at sea state 5 for both hulls**; the planner's margins hold for an unresolved, non-moving target only; the decided fix holds the hulls at sea state 3 by +1.28 / +0.73 dB on one seed, the oblique hull not tracked, not at sea state 4; regenerating D-251's outputs: the closing hull untracked at the last alternating ping, `popoto_12km_doppler` 4 false detections on 4 target-free pings; `fls_170k_traffic`'s wake false mover a rate of 5 of 8 seeds, a rule would move a default (`sonar-ya9s`); the range rate: the oblique target's class flickers on alternating pings, the navigation accuracy assumed; FLS: no 300k or 750k files, the engines disagree on the pool scenes with the defaults, one test narrowed (`sonar-w3fq`); the first suite on the batch failed 3 tests (fixed, D-260) |
| D-251 to D-254 (2026-11-27) | The reverberation's Doppler per frequency as the default (`doppler: per_frequency \| centre` on reverberation items; both engines and the CUDA sum; D-251); the tone path at 12 km counted again with it (D-252); the live chain in the viewer (`tools/scenario_view.py --chain`) with opt-in read-only taps in both engines (tracker, classifier, detector), the audio default by record (time stretch under 0.5 s), the gallery regenerated (D-253); the merge (D-254) | **On the regenerated gallery `fls_170k_traffic` has one mover not on a moving object** (`sonar-4pon`) and `popoto_env_gravel_30m` one confirmed track fewer (3 -> 2); outputs made before D-251 with a moving platform and boundary reverberation are stale (`sonar-rar0`); two tests changed with the default (a bound 13 -> 15, one test sums six seeds; no tolerance loosened); the generator 1.0 to 1.14 times slower in C++, 1.1 to 1.4 in NumPy; the tone path: 0 false detections with the notch, 17 without, on 4 target-free pings only, still off by default; **the chain's default `look: starboard` mirrors an FLS's echoes astern** (the viewer sets `look: port`; an FLS deployment block is the owner's, `sonar-tpex`); the pool scenes have no chain view; `popoto_env_summer_mud_60m` no detection; nobody has listened |
| D-246 to D-250 (2026-11-27) | The tone path's grating-lobe and endfire notch, opt-in with the tone path (`doppler.tone_lobe_notch`, `tone_lobe_floor_db`), its coverage of both paths and `docs/design/doppler_tradeoffs.md`, the attribution of false detections (D-246); transmit options by sea state and the plain filter on a mover, linear against hyperbolic sweep (`popoto_trade_study.py --transmit`, sections 0 and 0a; `source: hfm` through both engines, `tools/hfm_doppler_eval.py`, the `hfm12` sub-run; D-247); the visualization: 2-D pages, a 3-D viewer and sound for every scenario (`tools/scenario_view.py`, `python/gpusonar/viz/`, three.js vendored; D-248); the simulator's reverberation Doppler found (D-249); the merge (D-250) | **The simulator's boundary reverberation applied Doppler as one shift at the record's centre frequency** (D-249; fixed by D-251; the owner: "FIx it", `sonar-l8yp`); **D-243's 22 false tone detections were mostly that artefact** (D-246: 25 with the notch, at its edge; recounted by D-252); far-sidelobe false detections remain (`sonar-wntj`); **the hyperbolic sweep's gain on the real 12 km scene is small** (1 to 2.4 dB for four times the energy) and the noise-limited case is not run (`sonar-rctz`); **the shipped transmitter holds 12 km at the deep site only to sea state 3** (the model's, for a point target; measured by D-260, not even there for the resolved hull); the visualization: four scenes without rays or sound, six without sound, nobody has listened, FLS and SAS sound a click (time stretch agreed; the default since D-253) |
| D-241 to D-245 (2026-11-27) | `GPUSONAR_JOB=doppler` (`tools/doppler_gpu_runs.sh`; sub-runs `tone`, `falsealarm`, `timing`, opt-in `ls12`) and its first runs (D-241); `doppler.batched` and `doppler.batch`: the sweep bank and the tone bank batched on the device (`libs/live/src/doppler_cuda.cu`, `doppler_batched.cpp`, `include/sonar/live/doppler_batched.hpp`; D-242); two opt-in 12 km large-submerged scenes and the ls12 sub-run (D-243); the owner's decisions on the baseline, the tone path and the transmitter (D-244); the merge (D-245) | **With the tone path on, 22 false detections on 4 target-free pings at 12 km at the default budget**, all the tone's, against 0 off (put on the grating lobes beyond ±32.4 degrees by D-243; mostly a simulator artefact by D-246); the tone path stays off by default (D-244; the notch is branch `notch`, `sonar-bc5r`; the far-sidelobe choice `sonar-wntj`); **the default chain is blind on `flank_patrol_doppler_tone`** (0 of 6 pings for all three targets, 26 false detections on 4 target-free pings; D-241); **D-242's first device code broke the default chain on cuda** (four bit-identity tests failed; a kernel instantiated in two `.cu` files; fixed, guarded by a test); the bank still costs 3.4 to 6.0 times the chain a ping on the device (`sonar-j1f9`); tracks ignore the measured rate (`sonar-evd4`); one device test loosened to 1e-5 m/s; the bank's false-alarm rows stopped at their one-hour limit |
| D-235 to D-240 (2026-11-26 and 27) | The owner's answers on the 12 km requirement and the sea-state labels (D-235); the rows' gain against reverberation (`reverberation_rows_gain`), `reach_limited_by` and the `penetration: rough` cost cap `roughness.max_pairs`, opt-in (D-236); Doppler D3, both engines, opt-in: `doppler.combine: cfar` and `doppler.tone: bank`, the scene `flank_patrol_doppler_tone` (D-237); POP-3 for `large_submerged` first and `ship` second, per site (D-238); the GPU suite in shards, `GPUSONAR_SHARD=I/N` (D-239); the merge (D-240) | **The tone above the band is not measured end to end** (its simulation did not finish in 4 h 40 min on a CPU; `sonar-iu67`); the tone path's leakage rule was designed on the records that showed 43 to 50 false detections a ping (not an independent test); the default chain finds the ship on `popoto_12km_doppler` on 0 of 6 pings (a pulse train compressed whole); **POP-3 reads "partly met"** (not at the sea-state-5 and summer-mud sites); `cfar` costs the still wreck 0.48 dB; the Python chain's nonsense on a short CW record; three shards cost more GPU time than one job; not merged: `gpujob`, `scene12`, `viz`, the CUDA port (`sonar-ca0o`) |
| D-230 to D-234 (2026-11-26) | The library sediment fit takes a row (D-230); pulse trains and four Doppler scenes (D-231); the sweep's Doppler bank in both engines, opt-in (D-232); the Popoto 12 km trade study (D-233); the merge (D-234) | **The bank lowers every target's margin 3.4 to 5.3 dB and stays off** (`sonar-094p`, `sonar-ef64`, `sonar-itzy`); a tone in the sweep's band is masked by the sweep's reverberation (10.9 against 39.8 dB; `sonar-3in1` for the ±3 dB scatter); **the +5 dB class is not reached at 12 km at any illustrative site**, POP-3 names no class (`sonar-bh5c`, `sonar-b1p0`); not merged: the tone path (D3) and branch `a-rows` (both merged since, D-240) |

**Tooling and configuration**

| Decisions | What | Worse, or left open |
| --- | --- | --- |
| D-177 | The beads pre-commit hook skips linked worktrees | Agents still commit with `-c core.hooksPath=/dev/null` |
| D-180 | YAML numbers written with a signed exponent (`38.0e+3`); `tests/python/test_yaml_numbers.py` | Five readers still use plain `yaml.safe_load` on user-written files (`sonar-1yq`) |

## 8. Open work (bd issues)

From `.beads/issues.jsonl` as of 2026-11-26 (the tree of D-201 and the 3-D work): 77 not closed of
176. P1: 6 (3 in progress, 3 open); P2: 7, all open; P3: 39 (1 in progress); P4: 25 (1 in progress).
Recounted from the export at `76e01a5` (after D-234's follow-ups were filed; counted from the file, not
with `bd`): 96 not closed of 208; P1: 8 (5 in progress, 3 open); P2: 11, all open; P3: 50, all open; P4:
27 (1 in progress). Recounted from the export at `02bce49` (after the fourth merge and the gallery
difference were filed; counted from the file, not with `bd`): 101 not closed of 214; P1: 12 (6 in progress,
6 open); P2: 12, all open; P3: 50, all open; P4: 27 (1 in progress). Recounted from the export at `b7c38d0`
(after the fifth merge and its follow-ups were filed; counted from the file, not with `bd`): 104 not closed
of 221; P1: 11 (5 in progress, 6 open); P2: 15, all open; P3: 51, all open; P4: 27 (1 in progress). Recounted
from the export at `0fb640c` (after the sixth merge and its follow-up were filed; counted from the file, not with
`bd`): 105 not closed of 224; P1: 10 (5 in progress, 5 open); P2: 16, all open; P3: 52, all open; P4: 27 (1 in
progress). Recounted from the export at `bec83e5` (after the seventh merge and its follow-ups were filed;
counted from the file, not with `bd`): 106 not closed of 227; P1: 9 (5 in progress, 4 open); P2: 18, all open; P3:
52, all open; P4: 27 (1 in progress). Recounted from the export at `138ee99` (after the eighth merge and its follow-ups
were filed; counted from the file, not with `bd`): 106 not closed of 232; P1: 10 (6 in progress, 4 open); P2: 17, all
open; P3: 52, all open; P4: 27 (1 in progress). Recounted from the export at `050b568` (after the ninth merge and its
follow-ups were filed; counted from the file, not with `bd`): 108 not closed of 235; P1: 11 (5 in progress, 6 open); P2:
18 (1 in progress, 17 open); P3: 52, all open; P4: 27 (1 in progress). Recounted from the export at `c431dbd` (after
the tenth merge and its follow-ups were filed; counted from the file, not with `bd`): 110 not closed of 240; P1: 11
(5 in progress, 6 open); P2: 20, all open; P3: 52, all open; P4: 27 (1 in progress). The themes below were made from the first count; the rows "Doppler" and "The
12 km requirement" were added at D-234, "Visualization" and the fourth merge's issues at D-240, the fifth merge's
issues at D-245, the sixth merge's at D-250 (the row "Simulator" new), the seventh merge's at D-254, the eighth merge's at
D-261 (the row "FLS chain" new), the ninth merge's at D-263, the tenth merge's at D-267.
"In progress" on a milestone issue means the milestone is not closed, not that someone is working on
it now. Milestone 12 (`sonar-dzu`), `sonar-k7m` (closer to SST) and this documentation set
(`sonar-4mu`) were closed on 2026-11-24 and 25.

| Theme | Issues (priority) |
| --- | --- |
| Waiting on the owner | `sonar-o7m` (P1) vehicle adapters: board samples and navigation; `sonar-sul` (P1) calibration and the live chain on real data; `sonar-9a2` (P2) board timing: capture-start stamping, clock rates, the channel map; `sonar-68e` (P2) the grating lobe beyond ±42° at 6 kHz; `sonar-2e9e` (P3) the fjord beds' 12-arrival limit; `sonar-ya2c` (P2) whether SAS scenes carry bed multipath |
| Milestones not closed | `sonar-l59` (P1) M11 Holoscan integration; `sonar-3s7` (P1) M13 performance model and Jetson; `sonar-k29` (P1) stacked rows for elevation |
| Link budget | `sonar-w2w3` (P1; closed 2026-11-26 in the export): the reverberation term exists since D-221 (`coverage.reverberation`, `SonarEquation(reverberation_db=...)`); since D-227 the depth plan and the coverage picture include the sites' scattering (illustrative values; the +5 dB class over the 100 m shelf 8.0 to 9.6 → 2.56 km in the plan); the free-field budgets stay noise-only and say so |
| Doppler | `sonar-u1fu` (P1, in progress) the workstream: D2, D4, D3 and D8 merged (D-231, D-232, D-237, D-242: `doppler.combine: cfar`, `doppler.tone: bank`, `doppler.batched`, opt-in); closed with the fifth merge: `sonar-iu67` (the GPU job mode and its runs, D-241), `sonar-ca0o` (the device port, D-242), `sonar-094p` (taken into `sonar-iu67`); `sonar-bc5r` the tone path's grating-lobe notch (closed, merged D-246, D-250; the tone path still off by default); `sonar-evd4` the tracker uses the measured rate (D5; closed, merged D-258, D-261, opt-in); `sonar-wntj` (P2) the tone's far-sidelobe false detections, the owner to choose; `sonar-j1f9` (P2) batch the backprojection across hypotheses, cache the filters, no map re-upload; `sonar-r8nr` (P2) the Python chain's nonsense on a short CW record; `sonar-ef64` (P3) the bank's sidelobe detections under very strong echoes; `sonar-itzy` (P3) the winning hypothesis for elevation and SBL; `sonar-3in1` (P3) the tone margins' scatter (more background pings, the clipped window) |
| The 12 km requirement | `sonar-7h6q` the baseline's waveform, a 2 s hyperbolic sweep at 197 dB (closed: D-262, merged D-263); **`sonar-4e2h` (P1) the stacked-row depth dishonest in shallow water (D-266 F1): widen or withhold when no echo explains the reading (moves a default, the owner's)**; `sonar-1l0m` (P2) the multipath timing stage's floor, pairing and fringe slips, the surface echo as its own detection (D-266 F2 to F4), and the GPU depth grid's tables committed; `sonar-82nx` (P2) the leading-edge range correction's device run and the owner's decision on the default (D-264); `sonar-qsna` (P2) the short-range mode's follow-ups (D-265); closed with the tenth merge: `sonar-9035` (depth against range, D-266), `sonar-8h09` (the range offset, D-264), `sonar-9ed5` (the short-range mode, D-265); `sonar-n8ho` (P2) the baseline's own deep-site and shelf cases into `doppler_deep12_eval.json`, the 12 km examples regenerated; `sonar-2bjq` (P2) the resolved hull in the detector (a test summed over highlights or pings; the oblique hull detected and not tracked; several seeds; the closing hull untracked at the last alternating ping); `sonar-9ed5` a short-range mode for the long-range sonar (closed: built opt-in, D-265, merged D-267; the projector's longest pulse at full drive still an owner question, `STATUS.md` question 66); `sonar-bh5c` (P1) the owner: what the 12 km requirement is for, the operating area's wind and bottom backscatter, whether 32 a row is fixed, the "sea state 5" label; `sonar-3983` (P1, in progress) the trade study; `sonar-b1p0` (P2) its follow-ups: own depth by reach, the rows measurement at more sites and pings, the ship's 4.8 dB cross-check gap; `sonar-qbrg` scenes with `large_submerged` at 12 km (closed, merged D-243, D-245); `sonar-vjq7` transmit options by sea state and the requirement's wording (closed, merged D-247, D-250; the proposed POP-3 sentence is in `STATUS.md`, not yet in `tools/design_report.py`); `sonar-rctz` the noise-limited chain run at the deep site (closed, merged D-260, D-261: the baseline fails for the resolved hull); `sonar-8fzu` (P3) regenerate `REFERENCE_DESIGN.md` and the planner JSON for the new fields. Since D-235 the owner has said whom 12 km is for (`large_submerged` first, `ship` second); POP-3 reads "partly met" (D-238); since D-244 the baseline is the plain sweep with every Doppler option off, the requirement to be worded with its conditions and the transmitter specified by energy; since D-255 the baseline at sea state 3 and an alternate at sea state 5 |
| Visualization | `sonar-6k79` (P1) rays, bounces and the received signal at each hydrophone for every scenario: the 2-D pages, the 3-D viewer with three.js vendored from the npm registry and the sound merged (D-248, D-250; `docs/tasks/VISUALIZATION.md`), and since D-253 the chain view with opt-in taps in both engines, the audio fixes and time stretch by default for records under 0.5 s (merged D-254); left: page clutter, the short scenes' sound remade, full-ping chain runs on the GPU, the four scenes without rays; `sonar-tpex` a deployment block for the FLS (closed: the files and `look: boresight`, D-257, D-261; the viewer takes the look from the file); `sonar-bbqr` (P3) the software diagrams miss `doppler_cuda.cu` |
| FLS chain | `sonar-ya9s` (P2) the owner: a wake or hop rule for the FLS classifier (moves a default; D-259); `sonar-w3fq` (P2) the Python chain refusing sub-three-cell windows as C++ does, FLS 300k and 750k files, FLS tracker tuning (D-257); `sonar-4pon` (closed, D-259: the wake false mover explained) |
| Simulator | `sonar-l8yp` (P1) the boundary reverberation's Doppler at the centre frequency: closed, fixed by D-251 (merged D-254); `sonar-rar0` regenerate the outputs it made stale (closed, D-260, D-261) |
| Hardware design | `sonar-mu3` (P2) sync reference generator, input protection, power, mechanical and depth rating, FLS and SAS front ends, hydrophones; `sonar-68e` (P2) above |
| Detector and false alarms | `sonar-031` (P3) two-row cell mean 1.26 times its share; `sonar-4xn` (P3) three-row static count; `sonar-xl7` (P3) 9 dB floor on stacked rows; `sonar-gg6` (P4) constants against budget; `sonar-9hy` (P3) leading edge with blanking and burst filter |
| Tracking and classification at long range | `sonar-tlc` (P3) association among many arrivals in one gate; `sonar-36z` (P3) grouping velocity gate; `sonar-69f` (P3) outlier test power; `sonar-kal` (P3) long-range clutter scene; `sonar-ome` (P3) bounce grouping with multipath on; `sonar-dqc` (P3) parabola fallback; `sonar-2o5` (P3) winter-sand stray detection; `sonar-9b8` (P3) gravel peak and position; `sonar-t4u` (P3) function defaults differ from the chain's; `sonar-i28` (P4) two-ship scene; `sonar-7q0` (P4) SNR-weighted fit; `sonar-bu3` (P4) gallery bearing sigma; `sonar-y7k` (P4) `tracker.ping_interval_s` in configs |
| FLS and SAS scenes | `sonar-izys` (P2) the FLS chain tracks multipath ghosts as objects; `sonar-ya2c` (P2) SAS with bed multipath; `sonar-mib` (P3) FLS detection merge; `sonar-m1r` (P4) slant-range bias; `sonar-a2s` (P4) echo placement in the FLS runner; `sonar-n18` (P4) assert flank classes; `sonar-mgt` (P4) rerun full-size clseval at HEAD; `sonar-18n` (P4) captures lack ping times; `sonar-s0l` (P4) quick-mode class of the boat under way; `sonar-txl` (P4) the ten-line API example |
| Depth and elevation | `sonar-4e2h` (P1) and `sonar-1l0m` (P2), above; `sonar-hygi` (P1) the engines disagree on `popoto_face_2row_shallow` without `elevation` (C++ 2 detections, Python 1; D-264); `sonar-d91` (P3) two-row depth sigma too wide for shallow targets at 1 km |
| Navigation | `sonar-0ca` (P3) lever-arm correction of velocity; `sonar-vjq` (P4) generated deploy files and the lever arm |
| Motion estimation (SAS) | `sonar-0bc` (P3) retune lambda2; `sonar-dix` (P3) bathymetry also moves the DVL seabed; `sonar-7k2` (P3) along-track surge; `sonar-oii` (P4) delays below the model; `sonar-b6r` (P4) CUDA delay estimation slower than host |
| Sensor Bridge, streams and receive | `sonar-02c` (P2) a host with Sensor Bridge's own socket setup, and RoCE; `sonar-pnyv` (P2) clock-rate keys ignored by the Holoscan application and `receive` for a whole-array stream; `sonar-a75` (P3) unexplained a10g-large losses, the 0.25 s lead not enforced; `sonar-6rs` (P3) the first loopback case of a job; `sonar-4zx` (P3) `flank_32` 5 ms bound on a10g-large; `sonar-57w` (P3) no issue for a real board on that path; `sonar-ui5` (P3) `receive` at a non-nominal speed, multi-board; `sonar-thu` (P3) receive follow-ups; `sonar-3wd` (P3) idle stop on GPU; `sonar-l6j` (P3) `serve --backend cuda`; `sonar-e2u` (P4) pace table on L4; `sonar-vmm` (P4) thread count and scheduler; `sonar-qei` (P4) window layout speed against the record's speed |
| Performance | `sonar-1m4` (P3) keep the fine map on the device; `sonar-egj` (P4) rerun the `minefield_sas` benchmark row |
| Generated outputs | `sonar-tmvu` (P2) regenerate on GPU what is stale (`sst_validation`, `sst_pursuit.png`, `live_popoto_12km_report.jsonl`; the gallery done at D-253); `sonar-rar0` closed (the outputs D-251 made stale, regenerated, D-260); `sonar-pli3` closed (the whole gallery regenerated, D-253) |
| Simulator | `sonar-rcq` (P3, in progress) 3-D propagation; `sonar-7iy` (P3) relief reverberation's bounce legs; `sonar-9le` (P3) high-frequency surface; `sonar-3k1` (P3) bottom as published, layered seafloor; `sonar-64sm` (P3) surface model at low wind; `sonar-yus6` (P3) rough-surface loss in range-dependent water; `sonar-6t5` (P3) ping-to-ping coherence; `sonar-f3mi` (P3) `fls_170k_transit` reverberation; `sonar-8qn9` (P3) old-form walls; `sonar-6fc` (P4, in progress) K texture correlation; `sonar-ugc` (P4) per-element patterns; `sonar-pup` (P4) water per bearing; `sonar-5bf2` (P4) small items from the M12 audit; `sonar-ttr1` (P4) diagrams for walls and a tilted bed |
| Configuration and tools | `sonar-1yq` (P3) five readers on plain `yaml.safe_load`; `sonar-ess` (P4) build diagram annotation |

Also known, with no issue: `docs/design/popoto_depth_planner.json`, `docs/figures/fjord_depth_plan.md`
and the depth sections of `DESIGN.md` predate the detector recount (STATUS open item 37);
`docs/design/popoto_12km.json` was regenerated on an L4 under D-251 (D-260) and not with the 2 s baseline (D-262),
so it is of the 1 s sweep; `docs/handoff/hardware_builds.json` evaluates the face builds at 1000 m, inside the 1.5 km
blind range;
`RESULTS.md`'s "Sensor Bridge loopback" section keeps its first record (the loopback test not yet
passed) beside a "Since then" paragraph that says it has passed since D-153.

## 9. Waiting on the owner

Five lists. Nothing in them can be settled from the repository.

| List | Where | Count | About |
| --- | --- | --- | --- |
| Open questions for the project owner | `docs/STATUS.md`, section of that name | 75 numbered (35 when this table was written); 1 to 12 are the Popoto array, and 17, 19, 22, 28, 29, 35 to 43, 50, 54, 56, 58 to 60 and 63 are marked applied or answered, 51 in part (its worst sea state answered by D-255) | rows, band, converter, projector, static frame, navigation unit, target classes, waters, dive, recording, receive windows, coverage site; then wire format (13-15), defaults not applied (16, 18, 20, 21), reading and older items (23-27); raised since D-182 (30-34): board timing, the grating lobe, the fjord beds, SAS with bed multipath, 3-D propagation; Doppler (43-49, from `docs/tasks/DOPPLER.md` section 5; 43, the transmitter, answered: free to choose along with the Popoto, D-231); the default bistatic roughness model (50, answered); what the 12 km requirement is for, the operating area, 32 a row, the "sea state 5" label (51-54, from the trade study; `sonar-bh5c`; 51 answered in part, D-235: `large_submerged` first and `ship` second, still open the worst sea state, the detection and false-alarm probabilities and the slowest radial speed; 54 answered, the labels corrected); whether `--check` fails on "partly met" (55, D-238); the owner's decisions of 2026-11-27 (56, answered, D-244); the far-sidelobe remedy for the tone path (57, `sonar-wntj`); three.js vendored from the npm registry (58, answered, D-248: yes); time stretch for short records (59, answered, D-248: agreed); the simulator's reverberation Doppler (60, answered, D-249: fix it); the projector's response across 4-6 kHz for a hyperbolic sweep (61); the POP-3 wording the transmit study supports (62; measured, it does not hold for the resolved hull, D-260); an FLS deployment block (63, answered by D-255, built by D-257); the worst sea state (answered by D-255: the baseline sea state 3, an alternate sea state 5) and, after the deep-site measurement, the baseline's waveform and sea state 5 (answered by D-256: a 2 s hyperbolic sweep at 197 dB; sea state 5 needs a larger transmitter not yet identified); raised by the eighth merge: a wake or hop rule for the FLS classifier (64, `sonar-ya9s`), FLS tracker tuning (65, `sonar-w3fq`), a short-range mode and the projector's longest pulse at full drive (66, `sonar-9ed5`; the mode answered "do it", D-263, built opt-in, D-265; the pulse still open), the resolved hull in the detector (67, `sonar-2bjq`), a larger transmitter for sea state 5 (68); raised by the ninth merge: whether the leading-edge range correction becomes the default (69, `sonar-82nx`; built opt-in, D-264); raised by the tenth merge: fix the shallow-water depth dishonesty of the stacked rows (70, `sonar-4e2h`; moves a default; recommended), fix and deploy the multipath timing stage (71, `sonar-1l0m`), a source for the sound speed through the water column that needs no dive (72), whether the projector and drive board can alternate 2 s and 0.25 s pulses, the receiver's recovery time, and whether 82 % of the long update rate is acceptable (73 to 75, `sonar-qsna`) |
| Vehicle adapters | `docs/tasks/VEHICLE_ADAPTERS.md` section 3 (issue `sonar-o7m`) | 17 | the boards' output format, timing and transmit signal; the navigation unit's interface; the clock between them; the lever arm |
| Real data | `docs/tasks/REAL_DATA.md` section 3 (issue `sonar-sul`) | 11 | which data exists (FLS log, SAS image, Popoto recording), its instruments, levels, geometry and navigation |
| Doppler | `docs/tasks/DOPPLER.md` section 5 (issue `sonar-u1fu`; the same as `STATUS.md` questions 43 to 49) | 7; 1 answered (the transmitter is free to choose along with the Popoto, 2026-11-26, D-231) | the slowest target speed of interest, whether stationary targets must stay detectable, the 12 km class and aspect (now `STATUS.md` question 51), the navigation unit's velocity accuracy, a `range_rate_m_s` report field, real data with a CW transmit |
| 3-D propagation | `docs/tasks/PROPAGATION_3D.md` section 6 (issue `sonar-rcq`) | 9; 1 answered (Bellhop3D as an opt-in provider, a separate process, 2026-11-26), 5 answered by `sonar-6c8`, 6 done (the warning), 8 settled as built (the Fortran program), 9 narrowed by D-199 to the 2-row fjord scenario's 12-arrival limit | relief across the look in the owner's waters, the ranges and slopes that matter, run time and disk, walls and banks in the FLS and SAS use |

Each task file states first what is unknown, then the questions to ask before any code, the
current state of the code, a plan and the rules.

Already decided by the owner, besides the list in section 2: two rows recommended on the
face; the FLS merge stays off; the deployment defaults were accepted (D-118 to D-120,
D-137 to D-143); the tracker takes the most probable pairs first (D-141); the per-board header
field and `missing: drop` (D-184); the receiver queue of 4 s (D-187); damaged frames at the
start accepted when the index confirms it (D-188); the resampler's weights from three trig calls and the
decimate-first correction by board (asked for by the owner, D-192, D-195); the K density as an opt-in input (D-196);
`flank_patrol`'s small vehicle left as a hard target (D-198); Bellhop3D as an opt-in provider
run as a separate process (2026-11-26); the answers to `STATUS.md` questions 36 to 42 (D-229; the
library fit's row of 36 built, D-230); the transmitter (projector, drive board, waveform) free to
choose along with the Popoto (2026-11-26, D-231); the 12 km requirement is for `large_submerged` first and
`ship` second, and the sea-state labels corrected (D-235); the GPU suite in shards (D-239); the Doppler
GPU job mode and the CUDA port agreed, and the visualization workstream asked for (2026-11-26, as the
issues `sonar-iu67`, `sonar-ca0o` and `sonar-6k79` record); the baseline is the plain sweep on the
three-row face with every Doppler option off, the 12 km requirement worded with its conditions, the
transmitter specified by energy (level and pulse length) and chosen for the sea state needed with a table
of transmit options by sea state, the present default kept; the tone path off by default, the notch to be
implemented with test coverage and documents for both (D-244); three.js vendored from the npm registry and time
stretch as the default sound of short records (D-248; done for records under 0.5 s, D-253, "this includes fls,
yes of course that makes sense"); the simulator's reverberation Doppler to be fixed, the per-frequency law the
default with the old law an option (D-249; done, D-251); the baseline at sea state 3 for testing and prototyping and an
alternate at sea state 5, and an FLS deployment block (D-255; the block done, D-257); after the deep-site measurement the
baseline's waveform a 2 s hyperbolic sweep at 197 dB with the plain filter, done as its own package with every
long-range result regenerated, and sea state 5 recorded as needing a larger transmitter not yet identified (D-256;
done, D-262); the depth estimate against range, the leading-edge range offset and a short-range mode, shown the two
failing self-checks and the blind range: "do all 3" (2026-11-28, D-263; done, D-264 to D-266, merged D-267; the
range correction and the short-range mode opt-in).

## 10. Design numbers to keep straight

**First (D-266, 2026-11-28): depth.** The deployed stacked-row depth (every stacked-row deployment file) is
dishonest in 100 m of water at own depth 30 m on 22 of the 28 depths given at 1 to 2 km: a ship 4 m deep at 1.7 km
reads 95.7 +- 27.4 m and is called submerged on 4 of 4 pings (the 2 s sweep 99.8 +- 30.4 m); in deep water from
30 m a 30 m target at 1.7 km reads 3.4 +- 10.2 m, called surface on 4 of 4 (`docs/tasks/DEPTH_RANGE.md` DR.4,
`docs/tasks/depth_range_sweep.json`; not fixed, `sonar-4e2h`). As deployed the gradient prior (0.05 /s) adds 48 m
of sigma at 1.7 km and 2400 m at 12 km: depth cannot tell a submerged target from a ship at 12 km. With the water
known, from own depth 225 m in deep water the rows tell 30 m from the surface to 2 km, 60 m to 5 km, 150 m to 8 km,
300 m to 12 km; the opt-in timing stage 30 to 300 m from 1.7 to 12 km with 0.2 to 1.2 m of sigma (its own defects,
`sonar-1l0m`). The elevation scenes' depths (targets near 1 km, the short pulse) are the case where the rows were
shown to work. A case for three rows from depth at 12 km is not shown. The leading edge's offset is set by the
echo's SNR, not by the pulse length (D-264; about (c / 2 pi B) sqrt(S / F) before the peak; -1.4 to -1.9 m at
37 dB for 0.1 to 4 s pulses); the opt-in `cfar.range_edge_correction: pulse` removes it (point echoes within 2 mm
on average). The short-range mode (D-265, opt-in): a cycle of 23.2 s for 18.93 s (82 % of the long update rate),
the 0.25 s pulse 9.0 dB less energy, blind inside 187.5 m.

**Then (D-262, 2026-11-28): the 12 km baseline is a 2 s hyperbolic sweep at 197 dB with the plain filter, and
every margin in this section is the budget's or the planner's, for a steady, unresolved, non-moving point echo.**
What the 2 s pulse costs: nothing is seen inside 1.5 km (c T / 2; 0.75 km at 1 s); a mover's range reads 6 m per
m/s of range rate off (f_stop T / B), which the plain filter cannot correct; no range rate; the elevation scenes
keep the 1 s sweep (depth at 12 km measured since, D-266: above). Measured with it through the simulator and both chain
engines at the illustrative deep site at sea state 3 with the apl_1994 bubble law (one seed, 6 pings; job
6b0a8a7854072e8b716cb7a8): every target 6 of 6, closing hull +1.28 dB, crossing +5.13, point +6.12, ship +6.26,
the oblique hull +0.73 dB as D-260's `ss3_hfm2` (the same transmission; detected and not tracked there): the two
resolved hulls thin. Sea state 4: the two hulls 0 of 6. Sea state 5: nothing held (it needs a larger transmitter
not yet identified). The planner gives the baseline +5.4 / +6.3 dB there at sea state 3 (`large_submerged` /
`ship`, `DESIGN.md` POP-3): the class's hull, five highlights that resolve in range against a detector testing
0.375 m and 2 m cells, came 3.8 to 6.0 dB under the point in the same case (D-260). The 1 s linear sweep before it
held neither the closing nor the oblique hull nor the ship there (0 of 6, D-260). On the reverberation-limited
12 km shelf the baseline gives 9.84 / 10.0 / 12.61 dB (closing / oblique / crossing), all 6 of 6 and tracked as
movers (job 6b0a8a7954072e8b716cb7aa; 7.98 / 9.22 / 11.41 with the 1 s sweep, D-243).

Checked 2026-11-24 against `docs/design/derived.json` (requirement `POP-3`, sonar
`popoto_long_range`), `docs/design/popoto_12km.json` and `docs/design/popoto_stage2.json`; the first bullet
re-read from `derived.json` at D-263 (2026-11-28). `popoto_12km.json` and `popoto_stage2.json` are of the 1 s
sweep (not regenerated by D-262).

- 12 km link-budget margins for one, two and three rows at 197 dB with the 2 s baseline: +1.9 / +4.4 / +5.9 dB;
  single-ping detection probability 0.94 / 1.00 / 1.00 (`derived.json`, `POP-3`: one row has 17.6 dB at 12 km and
  needs 15.8, 13.2 and 11.8 dB). With the 1 s sweep: −1.1 / +1.4 / +2.9 dB, 0.22 / 0.87 / 0.99 (14.6 dB in one
  row). The range study's "one row of 32 hydrophones needs 198 dB at 4.2 kHz" is for the 1 s pulse. These are for
  a steady point echo of +25 dB (a placeholder) in noise, spherical spreading.
- In simulation the ship at 12 km is detected on 6 of 6 pings and tracked from ping 3 in
  100 m over sand with one, two and three rows; not detected in the deep-silt and
  summer-mud waters (`popoto_12km.json`, keys `cases.*.{1,2,3}.pings_detected` and
  `track_confirmed_at_ping`). Regenerated on an L4 under D-251 (D-260): every case's pings detected, median
  SNR and track ping as before. Of the 1 s sweep: not regenerated with the 2 s baseline.
- Full-rate GPU ping (0.5 to 13 km at 102.4 kHz, C++ chain, L4): 0.48 / 1.03 / 1.49 s for
  one, two and three rows against the 1 s sweep's 17.9 s ping interval (`popoto_stage2.json`,
  `decimation.timing.cuda`; timed on a loaded machine, so upper figures; not re-timed with the 2 s pulse). With
  the 2 s baseline `derived.json` gives `min_ping_interval_s` 18.93 and the window opens at 1.5 km.
- The live chain on the device at the working rate: 12.7 / 25.9 / 37.5 ms a ping for one,
  two and three rows (L4, `docs/benchmarks/live_chain_ablation_l4_b2a918e.json`, D-150), on
  pings with ideal clocks: that benchmark and the full-rate one leave board synchronisation
  out. With it, on the Popoto trials' pings: C++ 0.048 / 0.054 / 0.065 s and Python 0.084 /
  0.18 / 0.25 s a ping at 32 / 64 / 96 channels (L4, D-192, D-195; `tools/live_chain_timing.py`).
- With the Doppler options, on the scenes' pings (L4, C++ chain, bank off -> `max` -> `cfar`, same tool):
  `flank_patrol` 0.115 -> 0.393 -> 0.418 s, `popoto_far_mover` 0.062 -> 0.349 -> 0.374 s, `popoto_12km`
  0.019 -> 0.097 -> 0.108 s after D-242 (job 6b090cd56c5d61fff1888217); with the per-hypothesis loop before
  it 3.558, 2.410 and 0.372 s with `max` (job 6b08c9d954072e8b71685cd6, D-241). The bank still costs 3.4 to
  6.0 times the chain a ping (D-245's correction); it is off by default.
- The 12 km transmitter by sea state (D-247; `docs/design/popoto_trade_study.md` section 0; illustrative deep
  site, assumed target strengths, the shipped 3 x 32 face; **the model's, for a point target**): the 1 s 197 dB
  shipped until D-262 held `large_submerged` and `ship` up to sea state 3 (+2.4 / +3.3 dB; measured, not for the
  resolved hull or the moving ship, above); the 2 s 197 dB baseline +5.4 / +6.3 dB at sea state 3 and -0.9 / +1.1 dB
  at sea state 4 (`DESIGN.md` POP-3; measured, sea state 3 held thinly, sea state 4 not, above); the least energy a ping that holds both at sea states 2 to 6 with the
  apl_1994 bubble law is 197, 197, 203, 206, 212 dB re 1 uPa^2 s at 1 m. With the plain filter the linear
  sweep's peak loss on a mover is 8.8 / 14.8 dB at 2 / 8 m/s for 1 s and grows with the pulse.
- **The link budget's margins above are for noise.** The term exists (`coverage.reverberation`,
  `SonarEquation(reverberation_db=...)`, D-221), and since D-227 the depth plan and the coverage
  picture include the shipped sites' scattering (illustrative values). The free-field budgets stay
  noise-only and say so. Where reverberation dominates they are optimistic:
  `flank_patrol`'s −3 dB small vehicle at 2.2 km had about 30 dB by the noise-only budget (D-198);
  with the term it is predicted at 10.3 dB over the background against 6.9 dB measured (D-227; 9.7
  against 6.6 at D-221).
- **With the sites' reverberation, per class** (the trade study, D-233, `docs/design/popoto_trade_study.md`;
  illustrative sites, assumed strengths): the +5 dB `xl_auv` is not met at 12 km by the shipped 3 x 32 at
  any of 6 sites (−4.6 to −116.1 dB); `ship` is met at four (+8.8 to +29.8 dB) and fails at sea state 5
  (−6.1) and over summer mud (−96.1); one row of 32 holds 12 km wherever three do. Not robust to one sea
  state or 3 dB of bottom scattering. The planner's figures for a point target, and the grid's of the 1 s
  linear sweep: at the deep site, one of the four, the 1 s baseline measured did not hold the resolved hull, and
  the 2 s baseline holds it by +1.28 / +0.73 dB on one seed (above).

## 11. How to work here

The method that kept the tree sound from D-158 to D-201:

1. **One `bd` issue per package.** Claim it, and commit the claim before creating any
   worktree.
2. **A git worktree per package** under `.claude/worktrees/<name>` on its own branch. An
   agent works only there, with a written brief (lead the report with what failed; do not
   edit STATUS, RESULTS, DECISIONS; list stale generated files and follow-ups).
3. **Tests first, both engines.** Every behaviour change has a test that fails before and
   passes after. A change to one live-chain engine is made in the other and held by the
   parity tests in `tests/python/test_live_cpp.py`.
4. **The orchestrator checks before merging**: re-reads the diff, reruns the affected test
   files, trial-merges on a side branch, runs the full GPU suite on Hugging Face Jobs
   (section 4), and only then fast-forwards the session branch and pushes. Agents' reports
   contained errors more than once; check their numbers against the files before quoting
   them.
5. **Ask before a default moves.** Anything that changes a default result goes to the
   owner with before -> after numbers first.
6. **Write it down.** `DECISIONS.md` gets an entry per package (what got worse first),
   `STATUS.md`'s "(now to D-xxx)" line is updated, follow-ups are filed in `bd`, and the
   worktrees, build directories and uploaded tarballs are removed.

False failures and other pitfalls:

- The program tests in `tests/python/test_live_cpp.py` need `<worktree>/build/bin/sonar_live`.
  With the build directory elsewhere, make a temporary `build` symlink and remove it
  afterwards. While it exists,
  `tests/python/test_study_resume.py::test_gallery_quick_does_not_default_to_the_committed_gallery`
  fails (it compares an unresolved path) and the stop hook reports an untracked file.
  Neither is a real failure.
- Since D-177 the beads pre-commit hook skips linked worktrees (before, a worktree commit
  exported into the main checkout's `issues.jsonl`). Agents still commit with
  `git -c core.hooksPath=/dev/null commit`, staging explicit paths, never `.beads/`.
- The main checkout's built Python module can be stale (section 3).
- YAML: PyYAML reads `38.0e3` as a string. Write `38.0e+3` (D-180;
  `tests/python/test_yaml_numbers.py` fails on any such value in a tracked file).
- The container's disk is small: remove build directories and tarballs after use.
- The Sensor Bridge loopback test runs at real time with a socket buffer worth about 20 ms
  of `flank_32`'s stream (4.4 to 12.8 ms at a Popoto build's rate); on a loaded machine it can
  lose a frame. It is reported, not retried (D-153). The Popoto-rate cases lose their first
  frames at start-up (likely Sensor Bridge's receiver allocating its queue: read in its source, not
  traced) and pass when only the first 0.25 s is lost
  (D-187, D-188); one of six ending as an expected failure (xfail) in a full suite is the known
  state (D-201's job), not a regression.
- Hugging Face Jobs pitfalls are in section 4.

Device code (learned in D-242; `docs/tasks/DOPPLER.md` D8.5):

- **One device instantiation per kernel template in the whole build.** The project's CUDA files are
  compiled without relocatable device code, so a kernel template (`parallel_for_kernel<K>`) instantiated
  in two `.cu` files registers two device bodies under one weak host stub, and the program's link and
  load order picks one. When the files' flags differ (`doppler_cuda.cu` is compiled with `--fmad=false`,
  `backproject_cuda.cu` and `beamform_cuda.cu` are not), two programs built from one library give
  different last digits: the first suite on the fifth batch failed four bit-identity tests that way with
  every Doppler option off (D-245). Call a shared kernel through its own file's launcher
  (`launch_on_device` in `backproject_cuda.cu` and `detector_cuda.cu`, `farfield_cuda_device` in
  `beamform_cuda.cu`); never instantiate it again in a new file.
- The guard: `tests/python/test_doppler_device.py::test_no_kernel_template_is_instantiated_for_the_device_in_two_cuda_files`
  (`nm` on every built `build/**/*.cu.o`; it runs in the GPU job and skips without a CUDA build), with the
  CUDA-gated `::test_the_default_chain_on_the_device_gives_the_same_lines_in_fresh_processes`. A new `.cu`
  file also needs a row in `tools/software_diagrams.py` (it fails on a `.cu` file no row names; at this tree
  `libs/live/src/doppler_cuda.cu` has none, `docs/handoff/SOFTWARE.md` section 10).
- Bit-identity comparisons between programs are not to be loosened when they fail on cuda only: look for
  a duplicated device instantiation first.
