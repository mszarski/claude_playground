# Task: run simulator calibration and the live chain on real data

Issue `sonar-sul` (P1). Written 2026-11-23 at commit `6600df8` on branch `docs-handoff` for an
agent with no history of this project. Read `docs/HANDOFF.md` first, then this file.

## 0. What is not known, and blocks the work

**No real data is in hand.** `tools/calibrate_sim.py` (`python/gpusonar/simcal.py`) and the
live chain (`tools/sonar_live.py`, `python/gpusonar/live.py`, `libs/live`) have only seen
simulated input; the only measured data the suite has met are other people's channel
soundings (UACR, Watermark, Smart Ocean; `docs/RESULTS.md`, "Time spreading against
measured channels"). Which sonar recorded what, in which format, with which gains and
geometry, is not known. The owner has to supply the data and the facts in section 3 first.
Do not guess a file format, a gain or a geometry and fit to it: a fit on wrong metadata
produces confident wrong offsets.

What can be done before the data arrives: read sections 1 and 2, run the existing tests
named there, and prepare the questions. Nothing else.

This task depends in part on `sonar-o7m` (`docs/tasks/VEHICLE_ADAPTERS.md`): a live Popoto
capture needs the board adapter. A Popoto **recording** can be processed without it, as
WAV with a hand-written sidecar (section 2.2).

## 1. Goal and done-criteria

Goal: the first real data through the two paths, and an honest record of how far the
models are from it.

1. **FLS or SAS through `tools/calibrate_sim.py`**: at least one real forward-looking sonar
   log (`fls --log` or `fls --frames`) or one real SAS image (`sas --image`) calibrated,
   with the report, the overrides file and, for FLS, the misfit plot (`--plot`) kept
   outside the repository, and the numbers in section 5 recorded.
2. **Popoto through `tools/sonar_live.py`**: at least one real Popoto recording processed
   by the live chain on **both engines** (`--engine python` and `--engine cpp`), the two
   report files compared, and the numbers in section 5 recorded.
3. **What the models miss**: for each data set, a list of the discrepancies between data
   and simulation, each with the model it points at (section 6) and whether a config
   change explains it.
4. **Tests**: any code written for this task (a reader for a vendor log, a converter, a
   comparison tool) has tests on synthetic fixtures that mimic the real format, written
   before or with it, all passing; tests that read the real data skip when its path is not
   set (the pattern of `GPUSONAR_UACR_DIR` in `tests/python/test_channel_replay.py`). Never
   mark the issue done with failing tests.
5. **Docs and proposals**: `docs/SIM_CALIBRATION.md` gains a "Real data" section (what was
   run, the misfit, the limits found) if the owner agrees its numbers may be published in
   the repository; proposed text for `docs/RESULTS.md`, `docs/STATUS.md` and, where a
   decision was taken, `docs/DECISIONS.md` goes in your report (section 7).

## 2. Current state (checked in the code at `6600df8`; brought to D-201 on 2026-11-26)

### 2.1 Simulator calibration (`simcal.py`, `tools/calibrate_sim.py`, `docs/SIM_CALIBRATION.md`)

- **FLS log, raw element data**: `tools/calibrate_sim.py fls --log FILE.h5 --scenario NAME
  [--levels absolute|relative] [--ranges FIRST LAST STEP] [--bearings ...] [--max-pings 20]`.
  The file is a container of ping records (`docs/RECORD_FORMAT.md`), read with
  `make_source({"type": "container", ...})`. Each ping is imaged with `recon.fan_image`
  and reduced to a range profile, the median over bearings (`simcal.observe_records`,
  `_profile`). `--levels` defaults to absolute for a log.
- **FLS frames from another beamformer**: `fls --frames FILE.npz` with `intensity
  [frame, range, bearing]` (power, not dB), `ranges` (m), `bearings_deg`
  (`simcal.observe_frames`). Always relative: the other system's processing gain is
  unknown.
- **The fit** (`simcal.calibrate_fls`): each boundary in the scenario's `reverberation`
  list and the ambient noise are simulated alone (`_simulate_part`, through
  `run_scenario`), imaged the same way, and the observed profile is fitted as a
  non-negative sum of them (`scipy.optimize.nnls`, relative error so every range counts
  the same). Out: per boundary `offset_db`, `share` (its largest fraction of the fitted
  profile) and `constrained` (share >= 0.1); the noise offset and whether it is
  constrained; for a relative log the held term (`--anchor noise` or `boundary:0`);
  `fit_rms_db`, the RMS of 10 log10(fitted / observed) over the range bins. `report()`
  prints them; `overrides()` gives scenario overrides (`reverberation[*].offset_db`,
  `receiver.ambient_noise.level_db`) for `load_scenario_config(name, overrides)`; `--plot`
  draws observed, fitted and each part against range.
- **SAS image**: `tools/calibrate_sim.py sas --image FILE --pixel M [M] [--display-db MIN
  MAX] [--block 48]` (`simcal.calibrate_sas_image`). Out (`SasCalibration`): scintillation
  index, speckle correlation area (resolution cell), scatterers per cell, density per m²,
  `shadow_below_seabed_db`, `bare_fraction`, `shadow_clipped`. `overrides()` asks for at
  most 5000 scatterers per m². No absolute level from a SAS image, by design.
- **Checked only against itself**: `tests/python/test_simcal.py`
  (`test_absolute_log_gives_back_the_hidden_strengths_and_noise`,
  `test_relative_log_gives_differences_only`, `test_a_term_the_log_cannot_see_is_reported_not_guessed`,
  `test_frames_from_another_beamformer_are_relative`,
  `test_speckle_statistics_tell_sparse_from_developed`,
  `test_an_overall_gain_changes_nothing_for_an_image`, `test_display_scale_can_be_undone`).
  `docs/SIM_CALIBRATION.md` reports the self-check: hidden +6.0 / -4.0 dB / 38.0 dB found
  as +5.9 / -4.0 / 38.6 dB with a 0.16 dB RMS misfit (at the tree of D-181; +6.1 / -4.0 / 38.0
  and 0.17 dB when first written, the move not traced: D-182); SAS densities 400 and 6000 per m²
  found as 620 and 8000. It says model error is not in those numbers because the "real"
  log came from the same simulator.
- **Not checked**: whether `fls --log` works on records of a flank or Popoto sonar
  (`observe_records` calls `recon.fan_image`, built for FLS). Do not use it on Popoto data
  without first testing it on a simulated Popoto log.
- The shipped FLS and SAS sonars (`configs/sonars/fls_*.yaml`, `sas_*.yaml`) are design
  values; their `unconfirmed` lists cover array, transmit and receive. A real log needs a
  sonar file for the sonar that recorded it, and a scenario (copy the nearest in
  `configs/scenarios/`, e.g. `fls_120k_transit.yaml`, and set the sonar, depth, attitude,
  water depth and `reverberation` list first; the calibration fits strengths, not
  geometry).
- Related tools for other parts of a real recording: `tools/fit_noise.py RECORDING --out
  model.h5 [--check]` fits a noise model (spectrum, coherence, tail) to a passive recording
  (`.wav`, a sidecar `.yaml` or a container `.h5`), used in a scenario as
  `receiver.ambient_noise: {model: fitted, file: ...}`; `tools/fit_sediment.py` fits the
  bottom scattering model to measured backscattering strength (CSV, calibrated levels);
  `python/gpusonar/calibrate.py` (`estimate_channel_errors`, `store_calibration`)
  estimates a real array's channel gains, phases and positions from a known source.

### 2.2 The live chain on a Popoto recording

- Inputs `tools/sonar_live.py` can take without new code: `--replay FILE.h5` (a container,
  e.g. a recording the chain wrote), a deployment file whose `source` is `{type: wav,
  sidecar: FILE.yaml}` (`WavSource`) or `{type: wav_watch, directory: DIR}`
  (`WavWatchSource`), or `flank_udp` blocks (`docs/tasks/VEHICLE_ADAPTERS.md`).
- **A WAV sidecar carries the array itself** (`elements`, `transmit`, `pings.times`,
  `pings.window`, `boards` with `reference_channel`, `recording_start_ns`, a static `pose`;
  `docs/RECORD_FORMAT.md`, "WAV sidecar"). `WavSource` does not read the sonar config. It
  refuses contradictions (element count against channels, sample rates, file lengths) and
  drops pings whose window runs off the file (`dropped_pings`).
- `--engine cpp` runs the C++ chain behind the Python program on the same records.
  `build/bin/sonar_live --replay FILE.h5` runs it standalone on a container.
- **Navigation for a recording**: `tools/sonar_live.py` ignores the `nav` section with
  `--replay`, `--scenario` and `--emulate`. A WAV sidecar has one static `pose`. A moving
  vehicle's recording with a separate navigation log has **no offline path** that I found:
  `NavListener` keeps fixes `keep_s` (120 s) behind the newest one and takes them live from
  UDP, so a navigation log replayed faster than the pings (`tools/nav_adapter.py --replay`)
  would be gone before the pings that need it. A static or bench recording is not affected.
  Writing an offline stamping tool is code beyond this task's tools: ask the owner first.
- Rehearsal of the same geometry in simulation: `tools/sonar_live.py CONFIG --scenario NAME
  --out DIR` on a scenario copied and set to the trial (`popoto_trial_100m.yaml`,
  `popoto_static_trial_2row.yaml` and the `fjord_*` and `popoto_env_*` scenarios are the
  nearest shipped ones).
- What `docs/DEPLOYMENT.md` "First time on the real hardware" asks of the first runs:
  bench recording with a known signal on every input, checked with `tools/inspect_tap.py`
  (sample rate, channel order, clipping, the reference bursts); `wav_watch` with
  `nav: {static: ...}` and `sync.ok`, offsets and drifts against the bench's prediction; a
  known target at known range and bearing (a bearing mirrored about broadside means
  `chain.look` is wrong); channel calibration from that target at three bearings
  (`calibrate.py`); under way, keep every recording.
- Engine parity on recordings is held today by `tests/python/test_live_cpp.py`
  (`test_program_replays_a_recording_like_the_python_chain`,
  `test_cpp_chain_gives_the_python_chains_report`, `test_report_fields_match_the_python_chain`,
  `test_one_deployment_file_runs_either_engine`) and `tests/python/test_wav_watch.py`.
- Hardware facts in config, all unconfirmed: `configs/sonars/popoto_8.yaml` (102.4 kHz,
  24 bits, 2.61 V full scale, 43 dB gain, -208 dB re 1 V/µPa), `popoto_array.yaml`,
  `popoto_4x7.yaml`, `popoto_long_range.yaml`; the site in
  `configs/deploy/popoto_vehicle.yaml` `chain.coverage` is the illustrative
  `shelf_100m_sand`, not a real one (STATUS question 12).

## 3. Questions for the owner (ask before any code)

**The data**

1. Which data sets exist or will exist: a forward-looking sonar log, a SAS image or
   mosaic, a Popoto recording (bench, static in water, under way), a passive (no transmit)
   Popoto recording, a sound-speed cast, a pinger sounding? Where are they, and may an
   agent read them (path on this machine, or a private location on the approved Hugging
   Face dataset)?
2. For each: the instrument (make and model or the configs file it corresponds to), the
   file format and any interface document or reader, and whether it is raw element data
   or already beamformed (FLS), or a display picture, linear power or dB (SAS; pixel size
   along and across track; was it resampled, clipped or compressed?).
3. Levels: are the front-end gains, the source level and the receive sensitivity known and
   true for the recording (absolute levels), or not (relative)? Gain changes during the
   recording?
4. Geometry: sonar depth and attitude, water depth, bottom type, sea state or wind, sound
   speed (a cast?), tilt of the sonar. For Popoto: the hydrophone positions and channel
   order, the board-to-channel map, which inputs carry the synchronisation reference.
5. Timing: the transmit times (a trigger channel? a log?), and the clock of the recording
   against the navigation's.
6. Navigation for the recording: static (where?) or under way (which log, which format;
   see `docs/tasks/VEHICLE_ADAPTERS.md` questions 10 to 14)?
7. Ground truth: known targets (range, bearing, depth, size, target strength or type),
   other traffic, or none?

**Use and publication**

8. May derived numbers (offsets, misfit, detection counts, SNR) and figures from the data
   be committed to `docs/`, or must they stay private? Is attribution needed?
9. May a confirmed instrument description (a sonar file) be committed, or must it stay on a
   private `GPUSONAR_CONFIG_PATH` (`python/gpusonar/config.py`)?
10. If the data contradicts a model enough to change a default (section 6): change it, add
    an opt-in alternative, or only record it?
11. For an under-way Popoto recording: may an offline navigation stamping tool be written
    (section 2.2), or will the owner supply recordings with the pose already in them?

## 4. Plan

1. **Baseline.** Build (section 9) and run `tests/python/test_simcal.py`,
   `tests/python/test_wav_watch.py` and `tests/python/test_live_cpp.py` (one file at a
   time). Note anything already failing.
2. **Data outside the repository.** Put the data where the owner says, never under the
   worktree. Reference it by an environment variable (propose a name such as
   `GPUSONAR_REAL_DATA_DIR` in your report; do not rely on it existing elsewhere) or by an
   absolute path in a private config. Check `git status` before every commit: no data file,
   no private config, no path that reveals the owner's identity or site.
3. **Describe the instrument and the setting in config.** A sonar file and a scenario (FLS),
   or a WAV sidecar and a deployment file (Popoto), with each fact's source in a comment
   and every fact not confirmed listed `unconfirmed`. Hardware facts never go into code.
4. **Readers, only if needed.** If a vendor format has to be converted (to a container of
   ping records, to the `.npz` frames layout, or to WAV with a sidecar), write the converter
   with tests on a synthetic fixture that mimics the format exactly, generated in the test
   (fixed seed, a few pings of `gpusonar.sim` output written in that format). Converting
   is preferred to a new source type in the live chain; a new source used by the live chain
   must exist in both engines (VEHICLE_ADAPTERS.md, path C) and needs the owner's go-ahead.
5. **Check the reader on the real data** before any fit: `tools/inspect_tap.py` on the
   records (per-channel levels, spectrum, clipping, channel order, the reference bursts),
   the first ping's `chain settings {...}` line, and for FLS a look at one beamformed
   frame. A wrong channel order or sample rate shows here, not in the fit.
6. **FLS / SAS calibration.**
   - First a dry run on simulated data of the same sonar and scenario (the `test_simcal.py`
     pattern: a log simulated with hidden offsets must give them back). If the dry run
     fails, the fit cannot be trusted on the real log; stop and report.
   - Then the real data: `tools/calibrate_sim.py fls --log ... --scenario ... --levels
     absolute|relative --out cal.yaml --plot fit.png` (or `--frames`, or `sas --image`),
     with `--levels absolute` only if question 3 says the levels are true. Keep `cal.yaml`
     and `fit.png` outside the repository until the owner agrees to publish.
   - Apply the overrides (`load_scenario_config(name, overrides)`), simulate again and
     compare: profile, and for a log with targets, their level above the background.
7. **Popoto through the live chain.**
   - Bench or static first: `tools/sonar_live.py CONFIG --out DIR --engine python`, then the
     same with `--engine cpp` into another directory; compare the two `report.jsonl` files
     line by line (detections, tracks, `sync`). Any difference is an engine-parity bug: make
     a minimal fixture that reproduces it, fix it in the engine that is wrong, and add the
     fixture to the parity tests.
   - Compare with the rehearsal of the same geometry in simulation (`--scenario` on the
     copied scenario): background level per range, detection of the known target (range,
     bearing, `snr_db` against the link budget less the losses in DEPLOYMENT.md, "What
     `snr_db` is"), false detections per ping against `chain.cfar.false_alarms_per_ping`
     (1e-3 by default; the false-alarm rate at the default budget is itself unmeasured,
     issue `sonar-ydi`).
   - A passive stretch, if there is one: `tools/fit_noise.py RECORDING --out model.h5
     --check`, and the simulated noise against it.
8. **Record** (section 5) and **interpret** (section 6). Write the report.

GPU: none of this needs a GPU. If a real-data run on the CUDA backend is wanted, the data
cannot go into the job tarball (it is not committed); ask the owner whether and where it
may be uploaded on the approved Hugging Face dataset before doing so (section 8).

## 5. What to measure and record

Every number names its tool, the commit, the data set (by a neutral label the owner agrees,
not a private name) and the settings.

**FLS log**: `fit_rms_db` and the range span it covers; per boundary `offset_db`, `share`,
`constrained`; the noise level and whether it was constrained; absolute or relative and the
held term; the residual against range (where in range the fit is worst: near field, the
first bottom return, the far end); the dry-run result on simulated data of the same setting.
**SAS image**: scintillation index, cell area against the sonar's design cell, scatterers
per cell, density, shadow depth, bare fraction, whether clipped.
**Popoto recording**: sample rate and channel checks from `inspect_tap.py`; per ping
`sync` (`ok`, `worst_residual_us`, `offsets_us`, `drift_ppm`); the known target's range and
bearing errors and `snr_db` against prediction; false detections per ping; tracks and
classes; `processing_seconds` against the ping interval; the counters `tools/sonar_live.py`
prints at the end (`dropped_blocks`, `overrun_pings`, `bad_blocks`, `reports_split`,
`sbl_fallbacks`), the report fields `pings_without_nav` and `pings_without_velocity`, and
`WavSource.dropped_pings` (not printed: read it from the source); whether Python and C++
reports were identical; the noise fit (`fit_noise.py` report and `--check` round trip).

**A seafloor backscatter fit** (`tools/fit_sediment.py`; estimators in `python/gpusonar/measure.py`;
G-9 of `docs/tasks/SST_GAPS_JACKSON.md`). How it is judged:
- Each value is 10 log10 of a mean INTENSITY: S_b from the calibrated level by (G.13) with Psi from
  (G.28)-(G.29) (`measure.backscattering_strength_db`, `psi_gaussian`; main lobe only; c tau / 2 well
  below the range; horizontal beam about 20 degrees or less). Averaging dB instead puts Rayleigh
  speckle 2.507 dB low whatever the number of samples (Jackson and Richardson 2007 (16.15); Dahl and
  Plant 1997 eq. (6)); for the sea surface's slow modulation 3.4-4.5 dB
  (`docs/references/surface_papers_mcdaniel_dahl.md` Sec. 3.4). Say which average was taken.
- The error of a value follows from its number N of independent samples (App. G.2.3:
  10 log10(1 + 1/sqrt(N))): 15 samples for 1.0 dB, 16 for 0.97 dB, 68 for 0.5 dB, 285 for 0.25 dB,
  1844 for 0.1 dB (`measure.samples_for_error_db`). N is the window's duration over the resolution
  (pulse length for CW, about 1/bandwidth for FM; p. 429) times the beams that do not overlap times
  the pings that see different seafloor (a still sonar's pings do not: pp. 422-424). Example, 10 m
  above a flat bottom, one beam, one ping: a 5-degree grazing band holds 126 / 24 / 7 samples at
  10-15 / 25-30 / 45-50 degrees with a 0.2 ms CW pulse, 25 / 5 / 1.5 with 1 ms CW, 505 / 98 / 29 with
  a 20 kHz FM sweep: a steep band needs many pings or beams for 0.5 dB.
- Fit with `--independent-samples N` (or a per-value `n_samples` column). The book's rule is an
  upper 1-sigma bar; on synthetic N = 16 data the fitted parameters fall within their reported 1 sigma
  in 55 % of cases (nominal 68 %) with it, 61 % with `--sample-error gamma` (the test
  `test_recovered_parameters_lie_within_their_reported_errors_at_the_expected_rate`, made and fitted with
  the perturbation model, the default when it was measured): read the
  reported errors as somewhat optimistic.
- A fit passes when the rms residual is within about the stated error (the tool notes it when it is
  more than twice), every parameter used downstream is constrained (not in `unconfirmed_fields`), no
  parameter ends on a `--prior-bounds` limit, and the result is recorded with the roughness model used
  (`--roughness-model`; the tool's default is `small_slope` since A10 of `docs/references/CONFORMANCE.md`,
  `perturbation` before it; with one frequency and volume-dominated data a small-slope fit without
  `--prior-bounds` can trade sigma2 for a much rougher relief, `test_sediment_fit.py`) and the extended
  report (covariance, predicted-strength error). Where the roughness models differ by less than the
  error over the data (the report says so), the data cannot choose among them; at D-193's Gallaudet
  site they differ by under 0.05 dB and two band means fix neither w2 nor gamma2.
- Calibration bias is not in these errors: an x dB error in the absolute level moves w2 and sigma2 by
  the factor 10^(x / 10) unseen by the fit. State the calibration's own uncertainty beside the fit.

Where results go: the report to the owner or orchestrator, with **proposed** text for a new
`docs/RESULTS.md` section (each row naming what it is compared with, as that file does) and
for `docs/STATUS.md` ("Where the code has run", the open items it closes or opens, "What is
next" item 3). Do not edit `STATUS.md`, `RESULTS.md` or `DECISIONS.md` yourself unless told
you own them. Figures go to `docs/figures/` only with the owner's permission (question 8).

## 6. When the real data contradicts a model

1. **Check the inputs first.** Most misfits on a first real data set are metadata: wrong
   channel order, sample rate, gain, sensitivity, levels taken as absolute that are not,
   sonar depth or tilt, water depth, a boundary missing from the scenario's
   `reverberation` list, a beam pattern that is not the config's. Fix the config, rerun,
   and say what changed.
2. **Then name the model the misfit points at**, and record it as a finding, not a fix:
   bottom or surface scattering law (a residual that grows or falls with grazing angle),
   noise spectrum or coherence, propagation (a range dependence no strength offset can
   follow), the array model (sidelobes higher than the `sidelobe_floor_db` assumption,
   channel errors before `calibrate.py`), the receiver (overload, AC coupling,
   `receive.overload`), the detector's background statistics (more false alarms than the
   design rate), the tracker or classifier (classes that flip). Say what the data cannot
   tell (a term reported `not constrained`).
3. **Do not tune to hide it.** Do not change a default, refit a constant until the plot
   looks right, or loosen a test. New behaviour is opt-in; a default result does not move
   without the owner's decision (question 10). Propose a `DECISIONS.md` entry in the report.
4. **Stop and ask the owner** when the fix would change the record schema, a default
   result, a published model's equation, or when a paper and a cited reference disagree in
   a way that changes results. Never invent a number a paper reports; a new source is
   proposed as a `docs/REFERENCES.md` row with its status (confirmed, via secondary, ours,
   unconfirmed).
5. **Keep the contradiction testable without the data**: where possible, a synthetic test
   that reproduces the effect seen (e.g. a profile shape the current model cannot fit), and
   a test on the real data that skips without its environment variable.

## 7. Rules you must follow

From the owner's brief (`docs/HANDOFF.md` section 2) and the project's working rules:

- Tests before or alongside every change. Never mark a milestone or issue done with failing
  tests. Never weaken a test to make it pass without saying so plainly.
- **Stop and ask the owner before**: adding a large or copyleft dependency; changing the
  record schema (`docs/RECORD_FORMAT.md`, `libs/core/include/sonar/core/record.hpp`); changing
  any FPGA code; acting when a paper and a cited reference disagree in a way that changes
  results; changing code beyond this task.
- GPL and LGPL code only as a separate test oracle or optional tool (as Bellhop and KRAKEN
  are), never linked into the core. No code copied from a closed or GPL source. Every
  open-source decision in `docs/OPEN_SOURCE.md`; every data set's terms recorded there too
  (it already lists the measured channel sets and PICMUS that way).
- Unconfirmed hardware facts live in config, not code. Do not invent numbers a paper or
  datasheet reports.
- The two live-chain engines stay equal: a behaviour change in one is made in both and held
  by engine-parity tests.
- **Real data is never committed**: owner-supplied vehicle and board recordings, FLS logs,
  SAS images, navigation logs, and licensed data (UACR, Watermark, Smart Ocean, PICMUS).
  Keep it outside the repository; reference it by path in a config or an environment
  variable; tests that need it skip without it. `.gitignore` covers `*.h5` but not `.wav`,
  `.npz`, `.npy` or pictures: check `git status` before each commit.
- Privacy: the owner's identity, the vehicle, the site and the project never go into a web
  query or any external request; generic search terms only. No accounts, forms or emails.
  Paywalled, login-gated or bot-challenged pages: move on, never circumvent. APL-UW TR 9407
  is distribution-limited: never fetch or read it. This task needs no web access.
- GPU work only on the owner's approved Hugging Face Jobs setup (section 8). Never print,
  paste or commit the token.
- Issue tracking is `bd` (beads), prefix `sonar-` (`export PATH=$PATH:$HOME/.local/bin:$HOME/go/bin`;
  `bd show sonar-sul`, `bd update sonar-sul --claim`, `bd create` for each model
  contradiction found, `bd close` when done). Every `bd` write re-exports
  `.beads/issues.jsonl`, which needs its own commit. If an orchestrator tells you not to
  run `bd`, list follow-ups in your report instead.
- Commits small and topical, with the trailer lines the harness gives you. Push only where
  the owner says.
- Writing style: plain statements; what failed, got worse or is uncertain first; every
  number names the tool and commit it came from.

## 8. GPU jobs (only if needed)

`docs/HANDOFF.md` section 4: `git archive` the committed tree to a tarball, upload it to the
owner's private Hugging Face dataset named there, and launch a job on an NVIDIA CUDA devel
image that downloads it and runs `tools/gpu_job.sh` (`bash tools/gpu_job.sh --help` lists
the modes and variables; the default builds with CUDA and runs ctest and pytest; uploads
happen only with `GPUSONAR_UPLOAD_REPO`, into `GPUSONAR_UPLOAD_PATH`). The job gets the
Hugging Face token as a job secret. One job at a time, flavor `l4x1`, timeout at most 3 h.
Real data is not in the tarball; uploading it anywhere needs the owner's permission.

## 9. Build and test on this machine

15 GB RAM, 4 cores, no GPU, shared with other work (`docs/HANDOFF.md` section 3):

```bash
export OMP_NUM_THREADS=1 PYTHONPATH=$PWD/python
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSONAR_BUILD_TESTS=ON -DSONAR_BUILD_PYTHON=ON \
      -DSONAR_ENABLE_CUDA=OFF -DSONAR_BUILD_HOLOSCAN=OFF \
      -DSONAR_CERES_DIR=/opt/ceres-nosuitesparse -DSONAR_ALLOW_SUITESPARSE=OFF
cmake --build build -j1
python3 -m pytest tests/python/test_simcal.py -q -p no:cacheprovider     # one file at a time
```

Never run the whole suite at full width here; it takes about two hours.
