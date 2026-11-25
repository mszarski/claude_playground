# Handoff notes

Refreshed 2026-11-24 at commit `933972b` on branch `claude/gpu-sonar-suite-u289p5`
(decisions to D-181; the version before this one was written 2026-11-19 at `2160fe8` and
stopped at D-144). These notes are for whoever picks the work up next, a person or a coding
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
  imported by tests only (D-149).
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
  detector's constants stay as they are (D-168).
- **Left opt-in on purpose** (each measured better on some scenes and worse on others):
  the per-axis outlier test `classify.outlier_alpha` (D-159), the classification rules from
  detections (D-152), the sub-grid Popoto bearing (D-176), the LFM range-Doppler correction
  `range_doppler` (D-181), merging FLS detections before tracking (D-135). Turning one on
  by default is the owner's call.
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

Bellhop (GPL, test oracle only) was built in the session scratchpad at
`<scratchpad>/at/Bellhop/bellhop.exe`. The scratchpad is not durable. If it is gone,
`tools/build_oracles.sh` rebuilds Bellhop and KRAKEN.
Without `GPUSONAR_BELLHOP` the Bellhop-dependent tests skip, and the run ends with a line counting
those skips; with `GPUSONAR_REQUIRE_ORACLES=1` they fail instead.

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

The full suite before a merge (what "2180 passed, 101 skipped" was run with):

```bash
launch_named.sh <name>.tar.gz l4x1 3h -e GPUSONAR_ORACLES=1 -e GPUSONAR_CERES=1 \
  -e GPUSONAR_HOLOSCAN_CPP=1 -e GPUSONAR_HSB=1 -e GPUSONAR_REQUIRE_ORACLES=1
```

Job modes of `tools/gpu_job.sh` (its header, `bash tools/gpu_job.sh --help`, lists every
switch and variable; uploads happen only with `-e GPUSONAR_UPLOAD_REPO=<dataset>`, into the
folder `GPUSONAR_UPLOAD_PATH`, which defaults to the mode's name):

| Mode | How | Typical time on `l4x1` | Output |
| --- | --- | --- | --- |
| whole test suite (default, `GPUSONAR_JOB` unset) | the command above; `GPUSONAR_HSB=1` builds Sensor Bridge 2.7.0 and implies the Holoscan C++ operators | about 45 min; timeout 3h | log only; grep for `passed` and `JOB_EXIT` |
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
| What works, what is wrong, questions for the owner | `docs/STATUS.md` (open items 1 to 45, owner questions 1 to 29) |
| Why anything is the way it is | `docs/DECISIONS.md` (D-1 to D-181; later entries amend earlier ones) |
| Every measured number and its source | `docs/RESULTS.md` (the Sensor Bridge numbers of D-153 to D-173 are summarised there; D-160, D-161, D-176 and D-181 are measured in their `DECISIONS.md` entries only) |
| Scenes, with figures and per-scene results | `docs/SCENARIOS.md`, `docs/figures/` (gallery of D-179) |
| Work waiting on owner input, written as briefs for an agent | `docs/tasks/VEHICLE_ADAPTERS.md` (`sonar-o7m`), `docs/tasks/REAL_DATA.md` (`sonar-sul`) |
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
- `tools/popoto_12km_study.py`, `tools/fjord_rehearsal.py`, `tools/design_report.py`: the
  studies behind the design documents.

## 6. State of verification at `933972b`

- **GPU, latest (NVIDIA L4, HF job 6b07440eb6a90bbe997e6027, the tree of D-191): 2180 passed, 101 skipped, none failed.**
- **GPU before it (NVIDIA L4, HF job 6b060a4b1460ffe1d5e93111, the merged tree of D-183; before it job 6b055ee4c97b8ed4fcd799db on the tree of D-181 with 2078 passed, 93 skipped): 2102
  passed, 95 skipped, none failed.** CUDA build with Bellhop, Bellhop3D, KRAKEN, PyRAM,
  Ceres, the Holoscan C++ operators and Sensor Bridge 2.7.0. The skips are data and tools
  not in the job. The commits after it (`169826f`, `933972b`) change documents and the
  issue export only.
- **CPU: the last whole-suite run on record is the tree of D-157** (`605a4e1`, 2026-11-22: 157 of
  157 ctest entries; `docs/STATUS.md`, the paragraphs before "Open questions for the project
  owner"). Since then each package ran its affected test files locally and the whole suite on the
  GPU job. A whole CPU run at HEAD is not on record.
- The gallery (`docs/figures/`, `gallery.json`) is from the L4 job 6b054768c97b8ed4fcd78015
  on tree `4a80876` (D-179). D-180 and D-181 change no default, so it stands.
- Not in either suite: the generator's self-tests. Last recorded result 42 of 42 on both
  engines for the six builds (D-143, 2026-11-19); since then only checked that every
  deploy and build file is accepted by both engines (D-166, D-177).
- Nothing has run on real hardware or real data. Nothing has run on DGX Spark or Jetson.

## 7. What was done in D-158 to D-181, and what each broke or left open

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
| D-178 | The gallery's range in the multipath scenes is the leading edge; `flank_patrol` is classified | `gravel_30m` 34.3 m beyond -> 10.1 m short, mover -> uncertain on seed 52 (explained in D-181, not cured; the scene's lower peak since D-137 is `sonar-9b8`); `popoto_12km` seed 60 3.0 -> 65.4 m (`sonar-tlc`); `small_vehicle` never detected (`sonar-80b`) |

**Tooling and configuration**

| Decisions | What | Worse, or left open |
| --- | --- | --- |
| D-177 | The beads pre-commit hook skips linked worktrees | Agents still commit with `-c core.hooksPath=/dev/null` |
| D-180 | YAML numbers written with a signed exponent (`38.0e+3`); `tests/python/test_yaml_numbers.py` | Five readers still use plain `yaml.safe_load` on user-written files (`sonar-1yq`) |

## 8. Open work (bd issues)

From `.beads/issues.jsonl` at `933972b`: 49 not closed. P1: 8 (6 in progress, 2 open);
P2: none; P3: 25; P4: 16. "In progress" on a milestone issue means the milestone is not
closed, not that someone is working on it now.

| Theme | Issues (priority) |
| --- | --- |
| Waiting on the owner | `sonar-o7m` (P1) vehicle adapters: board samples and navigation; `sonar-sul` (P1) calibration and the live chain on real data |
| Milestones not closed | `sonar-l59` (P1) M11 Holoscan integration; `sonar-dzu` (P1) M12 realistic propagation and reverberation; `sonar-3s7` (P1) M13 performance model and Jetson; `sonar-k29` (P1) stacked rows for elevation; `sonar-k7m` (P1) closer to SST; `sonar-4mu` (P1) this documentation set |
| Detector and false alarms | `sonar-031` (P3) two-row cell mean 1.26 times its share; `sonar-4xn` (P3) three-row static count; `sonar-xl7` (P3) 9 dB floor on stacked rows; `sonar-gg6` (P4) constants against budget; `sonar-9hy` (P3) leading edge with blanking and burst filter |
| Tracking and classification at long range | `sonar-tlc` (P3) association among many arrivals in one gate; `sonar-36z` (P3) grouping velocity gate; `sonar-69f` (P3) outlier test power; `sonar-kal` (P3) long-range clutter scene; `sonar-ome` (P3) bounce grouping with multipath on; `sonar-dqc` (P3) parabola fallback; `sonar-2o5` (P3) winter-sand stray detection; `sonar-9b8` (P3) gravel peak and position; `sonar-i28` (P4) two-ship scene; `sonar-7q0` (P4) SNR-weighted fit; `sonar-bu3` (P4) gallery bearing sigma; `sonar-y7k` (P4) `tracker.ping_interval_s` in configs |
| FLS and flank scenes | `sonar-80b` (P3) `small_vehicle` never detected; `sonar-mib` (P3) FLS detection merge; `sonar-m1r` (P4) slant-range bias; `sonar-a2s` (P4) echo placement in the FLS runner; `sonar-n18` (P4) assert flank classes; `sonar-mgt` (P4) rerun full-size clseval at HEAD; `sonar-18n` (P4) captures lack ping times |
| Depth and elevation | `sonar-d91` (P3) two-row depth sigma too wide for shallow targets at 1 km |
| Navigation | `sonar-0ca` (P3) lever-arm correction of velocity; `sonar-vjq` (P4) generated deploy files and the lever arm |
| Motion estimation (SAS) | `sonar-0bc` (P3) retune lambda2; `sonar-dix` (P3) bathymetry also moves the DVL seabed; `sonar-7k2` (P3) along-track surge; `sonar-oii` (P4) delays below the model; `sonar-b6r` (P4) CUDA delay estimation slower than host |
| Sensor Bridge and receive | `sonar-thu` (P3) receive follow-ups; `sonar-3wd` (P3) idle stop on GPU; `sonar-l6j` (P3) `serve --backend cuda`; `sonar-e2u` (P4) pace table on L4; `sonar-vmm` (P4) thread count and scheduler |
| Performance | `sonar-1m4` (P3) keep the fine map on the device; `sonar-egj` (P4) rerun the `minefield_sas` benchmark row |
| Simulator | `sonar-6c8` (P3) Bellhop 2-D against Bellhop3D differ by 1-2.5 dB |
| Configuration | `sonar-1yq` (P3) five readers on plain `yaml.safe_load` |

Also known, with no issue: `docs/design/popoto_depth_planner.json`, `docs/figures/fjord_depth_plan.md`
and the depth sections of `DESIGN.md` predate the detector recount (STATUS open item 37);
`docs/design/popoto_12km.json` was last written 2026-11-20, before D-156 to D-181;
`RESULTS.md`'s "Sensor Bridge loopback" section still says the loopback test has not
passed in a job (it has since D-153).

## 9. Waiting on the owner

Three lists. Nothing in them can be settled from the repository.

| List | Where | Count | About |
| --- | --- | --- | --- |
| Open questions for the project owner | `docs/STATUS.md`, section of that name | 29 numbered; 1 to 12 are the Popoto array, and 17, 19, 22, 28 and 29 are marked applied or answered | rows, band, converter, projector, static frame, navigation unit, target classes, waters, dive, recording, receive windows, coverage site; then wire format (13-15), defaults not applied (16, 18, 20, 21), reading and older items (23-27) |
| Vehicle adapters | `docs/tasks/VEHICLE_ADAPTERS.md` section 3 (issue `sonar-o7m`) | 17 | the boards' output format, timing and transmit signal; the navigation unit's interface; the clock between them; the lever arm |
| Real data | `docs/tasks/REAL_DATA.md` section 3 (issue `sonar-sul`) | 11 | which data exists (FLS log, SAS image, Popoto recording), its instruments, levels, geometry and navigation |

Each task file states first what is unknown, then the questions to ask before any code, the
current state of the code, a plan and the rules.

Already decided by the owner, besides the list in section 2: two rows recommended on the
face; the FLS merge stays off; the deployment defaults were accepted (D-118 to D-120,
D-137 to D-143); the tracker takes the most probable pairs first (D-141).

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
  two and three rows (L4, `docs/benchmarks/live_chain_ablation_l4_b2a918e.json`, D-150).

## 11. How to work here

The method that kept the tree sound from D-158 to D-181:

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
  of stream; on a loaded machine it can lose a frame. It is reported, not retried (D-153).
- Hugging Face Jobs pitfalls are in section 4.
