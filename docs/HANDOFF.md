# Handoff notes

Refreshed 2026-11-24 at commit `933972b` on branch `claude/gpu-sonar-suite-u289p5`
(decisions to D-181; the version before this one was written 2026-11-19 at `2160fe8` and
stopped at D-144), and brought to D-201 and the opt-in 3-D propagation tier on 2026-11-26. These notes are for whoever picks the work up next, a person or a coding
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
  (D-188); `flank_patrol`'s small vehicle stays a hard target (D-198).
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
`GPUSONAR_HSB_STUDY=1` and `GPUSONAR_HSB_TIMING=1` (Sensor Bridge measurements, not tests).

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
| What works, what is wrong, questions for the owner | `docs/STATUS.md` (open items 1 to 45, owner questions 1 to 35) |
| Why anything is the way it is | `docs/DECISIONS.md` (D-1 to D-201; later entries amend earlier ones) |
| Every measured number and its source | `docs/RESULTS.md` (the Sensor Bridge numbers of D-153 to D-192 are summarised there, and D-193 to D-201 in a summary table before "Not yet measured"; D-160, D-161, D-176 and D-181 are measured in their `DECISIONS.md` entries only) |
| Scenes, with figures and per-scene results | `docs/SCENARIOS.md`, `docs/figures/` (gallery of D-179) |
| Work waiting on owner input, written as briefs for an agent | `docs/tasks/VEHICLE_ADAPTERS.md` (`sonar-o7m`), `docs/tasks/REAL_DATA.md` (`sonar-sul`); `docs/tasks/PROPAGATION_3D.md` (`sonar-rcq`: the scope of 3-D propagation, its packages P1 to P6 done, owner questions in section 6) |
| The Popoto array design, budget, builds | `docs/DESIGN.md` (generated by `tools/design_report.py`), `docs/REFERENCE_DESIGN.md`, `docs/DEPLOYMENT.md`, `docs/design/` |
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

## 6. State of verification (to D-201 and the 3-D work)

- **GPU, latest (NVIDIA L4, HF job 6b079ab0d91f621b93098915, the tree of D-202, with `GPUSONAR_SLOW_ORACLES=1`): 2301 passed, 102 skipped, none failed.** Before it, D-201's tree: 2278 passed, 102 skipped, 1 xfailed (job 6b078a26d91f621b93096bc7).
  CUDA build with Bellhop, Bellhop3D, KRAKEN, PyRAM, Ceres, the Holoscan C++ operators and
  Sensor Bridge 2.7.0, `GPUSONAR_REQUIRE_ORACLES=1`. The skips are data and tools not in the
  job and the opt-in measurements. The 3-D work after D-201 (`eigenray3d`, `flank_seamount_3d`)
  is covered by a later job. Every decision from D-183 to D-201 names its own job and count
  (`docs/handoff/TESTS.md`, "Current counts", lists them).
- **CPU: the last whole-suite run on record is the tree of D-157** (`605a4e1`, 2026-11-22: 157 of
  157 ctest entries; `docs/STATUS.md`, the paragraphs before "Open questions for the project
  owner"). Since then each package ran its affected test files locally and the whole suite on the
  GPU job. A whole CPU run at HEAD is not on record.
- The gallery (`docs/figures/`, `gallery.json`) is from the L4 job 6b054768c97b8ed4fcd78015
  on tree `4a80876` (D-179). D-180 to D-201 change no default result of a gallery scene, so it
  stands.
- Not in either suite: the generator's self-tests. Last recorded result 42 of 42 on both
  engines for the six builds (D-143, 2026-11-19); since then only checked that every
  deploy and build file is accepted by both engines (D-166, D-177).
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
| D-198 | `flank_patrol`'s small vehicle is below the scene's reverberation; left as a hard target | The link budget and coverage model have no reverberation term (`sonar-w2w3`, P1) |
| D-201 | Image sources off a tilted bed and walls as any plane (opt-in); four multipath scenes | The FLS chain tracks multipath ghosts as objects (`sonar-izys`); SAS sharpness falls to a third with bed multipath, and every other SAS scene is free field (`sonar-ya2c`); wall coefficients are placeholders; no diagrams of walls (`sonar-ttr1`) |
| (2026-11-26) | The opt-in `eigenray3d` tier (Bellhop3D), the cross-slope warning, one top-level `bathymetry`, `flank_seamount_3d` (`docs/tasks/PROPAGATION_3D.md` P1 to P6) | No reverberation over relief at that tier; its acceptance test takes 8 minutes and runs only with `GPUSONAR_SLOW_ORACLES=1`; the decision entry is to be written |

**Tooling and configuration**

| Decisions | What | Worse, or left open |
| --- | --- | --- |
| D-177 | The beads pre-commit hook skips linked worktrees | Agents still commit with `-c core.hooksPath=/dev/null` |
| D-180 | YAML numbers written with a signed exponent (`38.0e+3`); `tests/python/test_yaml_numbers.py` | Five readers still use plain `yaml.safe_load` on user-written files (`sonar-1yq`) |

## 8. Open work (bd issues)

From `.beads/issues.jsonl` as of 2026-11-26 (the tree of D-201 and the 3-D work): 77 not closed of
176. P1: 6 (3 in progress, 3 open); P2: 7, all open; P3: 39 (1 in progress); P4: 25 (1 in progress).
"In progress" on a milestone issue means the milestone is not closed, not that someone is working on
it now. Milestone 12 (`sonar-dzu`), `sonar-k7m` (closer to SST) and this documentation set
(`sonar-4mu`) were closed on 2026-11-24 and 25.

| Theme | Issues (priority) |
| --- | --- |
| Waiting on the owner | `sonar-o7m` (P1) vehicle adapters: board samples and navigation; `sonar-sul` (P1) calibration and the live chain on real data; `sonar-9a2` (P2) board timing: capture-start stamping, clock rates, the channel map; `sonar-68e` (P2) the grating lobe beyond ±42° at 6 kHz; `sonar-2e9e` (P3) the fjord beds' 12-arrival limit; `sonar-ya2c` (P2) whether SAS scenes carry bed multipath |
| Milestones not closed | `sonar-l59` (P1) M11 Holoscan integration; `sonar-3s7` (P1) M13 performance model and Jetson; `sonar-k29` (P1) stacked rows for elevation |
| Link budget | `sonar-w2w3` (P1) the link budget and the coverage model have no reverberation term |
| Hardware design | `sonar-mu3` (P2) sync reference generator, input protection, power, mechanical and depth rating, FLS and SAS front ends, hydrophones; `sonar-68e` (P2) above |
| Detector and false alarms | `sonar-031` (P3) two-row cell mean 1.26 times its share; `sonar-4xn` (P3) three-row static count; `sonar-xl7` (P3) 9 dB floor on stacked rows; `sonar-gg6` (P4) constants against budget; `sonar-9hy` (P3) leading edge with blanking and burst filter |
| Tracking and classification at long range | `sonar-tlc` (P3) association among many arrivals in one gate; `sonar-36z` (P3) grouping velocity gate; `sonar-69f` (P3) outlier test power; `sonar-kal` (P3) long-range clutter scene; `sonar-ome` (P3) bounce grouping with multipath on; `sonar-dqc` (P3) parabola fallback; `sonar-2o5` (P3) winter-sand stray detection; `sonar-9b8` (P3) gravel peak and position; `sonar-t4u` (P3) function defaults differ from the chain's; `sonar-i28` (P4) two-ship scene; `sonar-7q0` (P4) SNR-weighted fit; `sonar-bu3` (P4) gallery bearing sigma; `sonar-y7k` (P4) `tracker.ping_interval_s` in configs |
| FLS and SAS scenes | `sonar-izys` (P2) the FLS chain tracks multipath ghosts as objects; `sonar-ya2c` (P2) SAS with bed multipath; `sonar-mib` (P3) FLS detection merge; `sonar-m1r` (P4) slant-range bias; `sonar-a2s` (P4) echo placement in the FLS runner; `sonar-n18` (P4) assert flank classes; `sonar-mgt` (P4) rerun full-size clseval at HEAD; `sonar-18n` (P4) captures lack ping times; `sonar-s0l` (P4) quick-mode class of the boat under way; `sonar-txl` (P4) the ten-line API example |
| Depth and elevation | `sonar-d91` (P3) two-row depth sigma too wide for shallow targets at 1 km |
| Navigation | `sonar-0ca` (P3) lever-arm correction of velocity; `sonar-vjq` (P4) generated deploy files and the lever arm |
| Motion estimation (SAS) | `sonar-0bc` (P3) retune lambda2; `sonar-dix` (P3) bathymetry also moves the DVL seabed; `sonar-7k2` (P3) along-track surge; `sonar-oii` (P4) delays below the model; `sonar-b6r` (P4) CUDA delay estimation slower than host |
| Sensor Bridge, streams and receive | `sonar-02c` (P2) a host with Sensor Bridge's own socket setup, and RoCE; `sonar-pnyv` (P2) clock-rate keys ignored by the Holoscan application and `receive` for a whole-array stream; `sonar-a75` (P3) unexplained a10g-large losses, the 0.25 s lead not enforced; `sonar-6rs` (P3) the first loopback case of a job; `sonar-4zx` (P3) `flank_32` 5 ms bound on a10g-large; `sonar-57w` (P3) no issue for a real board on that path; `sonar-ui5` (P3) `receive` at a non-nominal speed, multi-board; `sonar-thu` (P3) receive follow-ups; `sonar-3wd` (P3) idle stop on GPU; `sonar-l6j` (P3) `serve --backend cuda`; `sonar-e2u` (P4) pace table on L4; `sonar-vmm` (P4) thread count and scheduler; `sonar-qei` (P4) window layout speed against the record's speed |
| Performance | `sonar-1m4` (P3) keep the fine map on the device; `sonar-egj` (P4) rerun the `minefield_sas` benchmark row |
| Simulator | `sonar-rcq` (P3, in progress) 3-D propagation; `sonar-7iy` (P3) relief reverberation's bounce legs; `sonar-9le` (P3) high-frequency surface; `sonar-3k1` (P3) bottom as published, layered seafloor; `sonar-64sm` (P3) surface model at low wind; `sonar-yus6` (P3) rough-surface loss in range-dependent water; `sonar-6t5` (P3) ping-to-ping coherence; `sonar-f3mi` (P3) `fls_170k_transit` reverberation; `sonar-8qn9` (P3) old-form walls; `sonar-6fc` (P4, in progress) K texture correlation; `sonar-ugc` (P4) per-element patterns; `sonar-pup` (P4) water per bearing; `sonar-5bf2` (P4) small items from the M12 audit; `sonar-ttr1` (P4) diagrams for walls and a tilted bed |
| Configuration and tools | `sonar-1yq` (P3) five readers on plain `yaml.safe_load`; `sonar-ess` (P4) build diagram annotation |

Also known, with no issue: `docs/design/popoto_depth_planner.json`, `docs/figures/fjord_depth_plan.md`
and the depth sections of `DESIGN.md` predate the detector recount (STATUS open item 37);
`docs/design/popoto_12km.json` was last written 2026-11-20, before D-156 to D-201;
`RESULTS.md`'s "Sensor Bridge loopback" section keeps its first record (the loopback test not yet
passed) beside a "Since then" paragraph that says it has passed since D-153.

## 9. Waiting on the owner

Four lists. Nothing in them can be settled from the repository.

| List | Where | Count | About |
| --- | --- | --- | --- |
| Open questions for the project owner | `docs/STATUS.md`, section of that name | 35 numbered; 1 to 12 are the Popoto array, and 17, 19, 22, 28, 29 and 35 are marked applied or answered | rows, band, converter, projector, static frame, navigation unit, target classes, waters, dive, recording, receive windows, coverage site; then wire format (13-15), defaults not applied (16, 18, 20, 21), reading and older items (23-27); raised since D-182 (30-34): board timing, the grating lobe, the fjord beds, SAS with bed multipath, 3-D propagation |
| Vehicle adapters | `docs/tasks/VEHICLE_ADAPTERS.md` section 3 (issue `sonar-o7m`) | 17 | the boards' output format, timing and transmit signal; the navigation unit's interface; the clock between them; the lever arm |
| Real data | `docs/tasks/REAL_DATA.md` section 3 (issue `sonar-sul`) | 11 | which data exists (FLS log, SAS image, Popoto recording), its instruments, levels, geometry and navigation |
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
run as a separate process (2026-11-26).

## 10. Design numbers to keep straight

Checked 2026-11-24 against `docs/design/derived.json` (requirement `POP-3`, sonar
`popoto_long_range`), `docs/design/popoto_12km.json` and `docs/design/popoto_stage2.json`.
All three statements still hold.

- 12 km link-budget margins for one, two and three rows at 197 dB: −1.1 / +1.4 / +2.9 dB;
  single-ping detection probability 0.22 / 0.87 / 0.99 (`derived.json`, `POP-3`: one row
  has 14.6 dB at 12 km and needs 15.8, 13.2 and 11.8 dB). One row of 32 hydrophones needs
  198 dB at 4.2 kHz. These are for a steady point echo of +25 dB (a placeholder) in noise,
  spherical spreading.
- In simulation the ship at 12 km is detected on 6 of 6 pings and tracked from ping 3 in
  100 m over sand with one, two and three rows; not detected in the deep-silt and
  summer-mud waters (`popoto_12km.json`, keys `cases.*.{1,2,3}.pings_detected` and
  `track_confirmed_at_ping`). **That file is from 2026-11-20, before the leading-edge
  default (D-160) and everything after; it has not been regenerated.**
- Full-rate GPU ping (0.5 to 13 km at 102.4 kHz, C++ chain, L4): 0.48 / 1.03 / 1.49 s for
  one, two and three rows against a 17.9 s ping interval (`popoto_stage2.json`,
  `decimation.timing.cuda`; `derived.json` `min_ping_interval_s` 17.93); timed on a loaded
  machine, so upper figures.
- The live chain on the device at the working rate: 12.7 / 25.9 / 37.5 ms a ping for one,
  two and three rows (L4, `docs/benchmarks/live_chain_ablation_l4_b2a918e.json`, D-150), on
  pings with ideal clocks: that benchmark and the full-rate one leave board synchronisation
  out. With it, on the Popoto trials' pings: C++ 0.048 / 0.054 / 0.065 s and Python 0.084 /
  0.18 / 0.25 s a ping at 32 / 64 / 96 channels (L4, D-192, D-195; `tools/live_chain_timing.py`).
- **The link budget has no reverberation term** (`SonarEquation`, the coverage model; D-198,
  `sonar-w2w3`): the margins above are for noise. Where reverberation dominates they are
  optimistic: `flank_patrol`'s −3 dB small vehicle at 2.2 km has about 30 dB by the budget and
  3.8 to 5.1 dB over its background in the simulated data.

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
