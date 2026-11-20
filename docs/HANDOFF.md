# Handoff notes

Written 2026-11-19 at commit `2160fe8` on branch `claude/gpu-sonar-suite-u289p5`. Everything
is committed and pushed; the working tree is clean. These notes are for whoever picks the
work up next. `docs/STATUS.md` is the authoritative, current list of what works and what is
wrong; this file tells you how to work here and where the edges are.

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
- **Stop and ask the owner before**: adding a large or copyleft dependency; changing the
  record schema (frozen since milestone 2); changing any FPGA code; acting when the paper
  and a cited reference disagree in a way that changes results.
- GPL and LGPL code may be used only as a separate test oracle or optional tool, never
  linked into the core. Bellhop and KRAKEN run as separate processes.
- Do not copy code from any closed or GPL source into the core.
- Do not invent numbers that a paper reports.
- Unconfirmed hardware facts live in config, not code.
- Record every open-source decision in `docs/OPEN_SOURCE.md`.

Privacy and access, set during the work:

- The owner's email and project details are private. The **only** approved external
  destination for project content is Hugging Face: the private dataset
  `mszarski/gpusonar-ci` and HF Jobs. Literature and product searches use generic terms
  only. No contact, quote or registration forms, no accounts, no emails.
- **APL-UW TR 9407 is distribution-limited. A download of it was denied. Do not download
  or read it from anywhere, and do not route around the denial** (not through a subagent
  either). The owner's ruling: "Use anything from arxiv that cites things you need".
- Paywalled, login-gated or bot-challenged pages: move on. No sci-hub.
- Licensed downloaded data (UACR, Watermark, Smart Ocean channel soundings) is never
  committed or redistributed; only derived tables and figures, with attribution.
- No `LD_PRELOAD` or capability tricks in HF containers. Never paste the HF token.

Repository and git:

- Push only to `claude/gpu-sonar-suite-u289p5`. No pull request unless the owner asks.
- Commit messages end with the co-author and session trailer lines the harness specifies.
  No model identifier in any committed artifact.
- Issue tracking is `bd` (beads), prefix `sonar-`. Put it on the path first:
  `export PATH=$PATH:$HOME/.local/bin:$HOME/go/bin`. Every `bd` write re-exports
  `.beads/issues.jsonl`, which then needs its own commit; a stop hook refuses to end a turn
  with uncommitted changes.
- `.claude/worktrees/` is excluded locally. Remove an agent worktree with
  `git worktree remove -f -f <path>` and delete its branch after merging.

Writing style the documents hold to: plain statements, what failed or got worse is stated
first and in bold, nothing is called fixed that was not measured, every number names the
tool and commit it came from.

## 3. How to build and test

The container has 15 GB of RAM and 4 cores and no GPU. More than one test process at a
time, or a wide build, exhausts it.

```bash
cd /home/user/claude_playground
export OMP_NUM_THREADS=2 PYTHONPATH=$PWD/python
export GPUSONAR_BELLHOP=<path to bellhop.exe>     # see below
cmake --build build -j2                            # build/ is the CPU build
ctest --test-dir build --output-on-failure         # 138 entries; entry 1 is the whole pytest run
```

`tools/test.sh` does configure, build and ctest in one command, but builds at full width;
on this machine run the two steps above instead. The whole suite takes about two hours
here. For a quick check run single pytest files:
`python -m pytest tests/python/test_live.py -q`.

Bellhop (GPL, test oracle only) was built in the session scratchpad at
`<scratchpad>/at/Bellhop/bellhop.exe`. The scratchpad survived one container restart but
is not durable. If it is gone, `tools/build_oracles.sh` rebuilds Bellhop and KRAKEN.
Without `GPUSONAR_BELLHOP` the Bellhop-dependent tests skip.

A container restart kills background jobs (it happened once, mid-suite). Committed and
pushed work is safe; anything only in the scratchpad or a running process is not.

## 4. GPU work goes through Hugging Face Jobs

The owner's instruction: "Use huggingface to test all gpu things" and "You can use hf to
build cuda". The `hf` CLI is installed and logged in. The pattern:

1. `git archive --format=tar.gz -o gpusonar_r6.tar.gz HEAD`
2. `hf upload mszarski/gpusonar-ci gpusonar_r6.tar.gz gpusonar_r6.tar.gz --type dataset`
3. Launch a job that downloads the archive and runs `tools/gpu_job.sh`.

The launcher script lived in the scratchpad (`hf/launch_r6.sh`); its whole content is:

```bash
#!/usr/bin/env bash
# launch_r6.sh <flavor> <timeout> [-e KEY=VAL ...]
flavor=$1; to=$2; shift 2
hf jobs run --detach --flavor "$flavor" --timeout "$to" --secrets HF_TOKEN "$@" \
  nvidia/cuda:12.6.3-devel-ubuntu24.04 bash -c "apt-get update -qq && apt-get install -y -qq python3-venv python3-pip > /dev/null && python3 -m venv /opt/b && /opt/b/bin/pip install -q huggingface_hub && export PATH=/opt/b/bin:\$PATH && hf download mszarski/gpusonar-ci gpusonar_r6.tar.gz --type dataset --local-dir /w > /dev/null && mkdir -p /w/s && tar -xzf /w/gpusonar_r6.tar.gz -C /w/s && cd /w/s && (bash tools/gpu_job.sh; echo JOB_EXIT \$?)" \
  2>&1 | grep -o "mszarski/[0-9a-f]\{24\}" | head -1
```

Job modes of `tools/gpu_job.sh` (its header lists every switch and variable; uploads need `-e GPUSONAR_UPLOAD_REPO=<dataset>`, and `GPUSONAR_UPLOAD_PATH` defaults to the mode's name, except for the gallery). `benchall` also reads `GPUSONAR_SCALE` (default 1), `GPUSONAR_BENCHALL_ARGS` and `GPUSONAR_BENCHALL_PYTEST`; `elevbench` reads `GPUSONAR_ELEVBENCH_ARGS`; the gallery `GPUSONAR_GALLERY_ARGS`:

| Mode | How | Typical time on `l4x1` | Output |
| --- | --- | --- | --- |
| whole test suite (default) | `-e GPUSONAR_ORACLES=1 -e GPUSONAR_CERES=1 -e GPUSONAR_HOLOSCAN_CPP=1` | about 80 min, timeout 150m | log only; grep for `passed` and `JOB_EXIT` |
| gallery | add `-e GPUSONAR_JOB=gallery -e GPUSONAR_UPLOAD_REPO=mszarski/gpusonar-ci` | about 40 min, timeout 4h | 57 files uploaded to the dataset folder **`gallery/`** (fixed name; `GPUSONAR_UPLOAD_PATH` is ignored in this mode) |
| every-stage benchmark | `-e GPUSONAR_JOB=benchall -e GPUSONAR_UPLOAD_PATH=<folder>` | 150m | json and md in that folder |
| full-rate elevation bench | `-e GPUSONAR_JOB=elevbench -e GPUSONAR_UPLOAD_PATH=<folder>` | 60m | `popoto_stage2.json` |
| live-chain ablation (issue `sonar-y7a`) | `-e GPUSONAR_JOB=liveablate -e GPUSONAR_UPLOAD_PATH=<folder>` (default folder `liveablate`; `GPUSONAR_LIVEABLATE_ARGS` appended to `tools/live_chain_ablation.py --backend cuda`) | not recorded | `live_chain_ablation.json` in that folder (committed as `docs/benchmarks/live_chain_ablation_l4_e8d30de.json`) |
| imaging benchmark | `-e GPUSONAR_JOB=benchmark` (`GPUSONAR_SCALE` 2, `GPUSONAR_PIXEL` 0.05 m, `GPUSONAR_MODES` `sas,fls,flank` by default; runs `tools/benchmark.py --backend cuda`) | not recorded | log only; nothing uploaded |

Then `hf jobs wait <id>`, `hf jobs logs <id>`, and
`hf download mszarski/gpusonar-ci --type dataset --include "gallery/*" --local-dir <dir>`.
Copy gallery files into `docs/figures/`. `tests/python/test_realistic_scenes.py::test_documented_full_size_classes`
reads `docs/figures/gallery.json`, so a new gallery can break it; update its expectations
honestly, with a comment saying what got worse.

## 5. Where things are

| Need | Read |
| --- | --- |
| What works, what is wrong, questions for the owner | `docs/STATUS.md` (open items 1 to 43, owner questions 1 to 27) |
| Why anything is the way it is | `docs/DECISIONS.md` (D-1 to D-140) |
| Every measured number and its source | `docs/RESULTS.md` |
| Scenes, with figures and per-scene results | `docs/SCENARIOS.md`, `docs/figures/` |
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
- `tools/popoto_12km_study.py`, `tools/fjord_rehearsal.py`, `tools/design_report.py`: the
  studies behind the design documents.

Working pattern that held up: delegate a self-contained change to a subagent in an
isolated worktree with a written brief ("lead the report with what failed; do not edit
STATUS, RESULTS, DECISIONS or README; list stale generated files"), merge its branch,
then write the decision, status and results text yourself and rerun the suites. Agents'
reports contained errors more than once; check their numbers against the files before
quoting them.

## 6. State of verification at `2160fe8`

- CPU: 138 of 138 ctest entries pass (whole pytest run included), 2026-11-19.
- GPU (NVIDIA L4, tree `ebdb22a`): 1493 passed, 24 skipped, none failed. The skips are
  data or tools not in the job. One test's expectation was changed after that job and has
  run only on CPU.
- Not in either suite: the generator's self-tests. Since D-142 and D-143 all six builds
  pass every one on both engines (42 of 42).
- Nothing has run on real hardware or real data. Nothing has run on DGX Spark or Jetson.

## 7. What was just done, and what it broke

The owner said "apply the defaults you recommended". Four changes, D-137 to D-140:

1. **Arrivals kept by energy** in the trial, 12 km and `popoto_env_*` scenarios instead of
   a typed count.
2. **`operation.site_kind: enclosed`** in a build spec writes `chain.min_snr_db: 15` (for
   fjord and harbour water). No shipped build uses it.
3. **`elevation.ray_method: arcs`** is the default in both engines; `steps` selectable.
4. **Scenario sound speed follows temperature** in 44 of 47 scenarios (1496.8, 1489.8,
   1482.4 m/s). Three keep a typed speed on purpose.

Consequences that are open (STATUS items 38 to 43):

- `popoto_static_2x32` failed its `shallow` and `rehearsal` self-tests at eight bounces
  (`sonar-jny`). Resolved by the owner's answers (D-142, D-143): the trials keep the whole
  series (cap 16) and the judge folds ambiguous candidates into depths. Two rows in 100 m
  at 2.6 to 3 km still cannot measure depth; they now say so with an honest, wide sigma.
- **The regenerated gallery is worse in several scenes** (`sonar-ic5`), none examined:
  `popoto_12km` ship classed `multipath`; `fls_170k_traffic` crossing boat classed `line`
  at 15.4 m/s (true about 4); `fls_120k_cluttered_seabed` 9 movers, 8 not the boat;
  `popoto_env_winter_sand_200m` one false mover; `popoto_env_gravel_30m` peak down 3.3 dB.
  **Since measured by seed (STATUS item 44, RESULTS "The gallery's changes, seed by
  seed")**: the 12 km, crossing-boat, cluttered-seabed and winter-sand changes are the luck
  of one seed; the classes flip from seed to seed under old and new settings alike. The
  gravel change is real and comes from the energy rule (D-137). What remains open is the
  fragility itself: the crossing boat is right on only 2 of 7 seeds, the 12 km ship on 0
  to 2 of 6.
- The self-test's range criterion now scores the first arrival; `range_m` sits 6 to 32 m
  behind the target in shallow water.
- The energy rule still stops at its caps (8 bounces, 64 arrivals) with a warning.
- Stale generated files (`sonar-zxk`, STATUS item 42): regenerated at `e10b17f` on
  2026-11-19 and 20, each compared with the file it replaced (RESULTS). Three tools reuse
  stored results (`ssp_range_study`, `fjord_rehearsal`, `decay_time_sensitivity`): rerun
  them into an empty `--json` file, with the settings their tables name. Why
  `sst_validation` moved since `5956b10` is bisected and settled (D-144).

## 8. Other open work (bd issues)

| Issue | What |
| --- | --- |
| `sonar-0ra` P1 | Mode that feeds Holoscan Sensor Bridge's receiver, simulator behind its emulator |
| `sonar-o7m` P1 | Vehicle adapters for the live chain: board samples and navigation |
| `sonar-sul` P1 | Run simulator calibration and the live chain on real data |
| `sonar-01u` P4 | Run the multiscale detector on the device: it is the 6 to 7 ms the live chain on CUDA gained since `15f1121` (`sonar-y7a`, measured and closed) |
| `sonar-ydi` P2 | False-alarm rate unmeasured at the default budget; three-row cell-mean constant drifts |
| `sonar-2j3` P2 | Regenerate depth planner and fjord depth plan after the detector recount (D-134) |
| `sonar-ofa` P2 | 12 km gallery ship misclassified (now overlaps `sonar-ic5`) |
| `sonar-9z8` P2 | Thinned cluttered-seabed test scene classes neither boat at 1496.8 m/s |
| `sonar-d91` P3 | Two-row depth sigma too wide for shallow targets at 1 km |
| `sonar-mib` P3 | FLS detection merge (`gallery.FLS_MERGE`, off): cures one scene, harms two (D-135) |

Also known: the design report still reads the older `gpu_benchmark_all_l4.json`, not
`gpu_benchmark_all_l4_d9cb5b7.json`.

## 9. Waiting on the owner

The three questions asked after the defaults were answered and applied (D-142, D-143):
the shallow trials keep their whole bounce series (cap 16), `popoto_static_trial_3row`
keeps bounces by energy, and the self-test judges ambiguous elevations by depth. All six
builds pass their self-tests again.

Standing questions are in `docs/STATUS.md` under "Open questions for the project owner":
hardware (converter, projector, frame, navigation unit), target classes, operating water,
recording, wire format, and two defaults not applied (16 and 18). Already decided by the
owner: two rows recommended on the face; the FLS merge stays off; the deployment defaults
were accepted; the tracker takes the most probable pairs first (D-141).

## 10. Design numbers to keep straight

- 12 km link-budget margins for one, two and three rows at 197 dB: −1.1 / +1.4 / +2.9 dB;
  single-ping detection probability 0.22 / 0.87 / 0.99. One row of 32 hydrophones needs
  198 dB. These are for a steady point echo in noise.
- In simulation the ship at 12 km is detected on 6 of 6 pings and tracked from ping 3 in
  100 m over sand with one, two and three rows; not detected in the deep-silt and
  summer-mud waters.
- Full-rate GPU ping (0.5 to 13 km at 102.4 kHz, C++ chain, L4): 0.48 / 1.02 / 1.51 s for
  one, two and three rows against a 17.9 s ping interval; timed on a loaded machine, so
  upper figures.
