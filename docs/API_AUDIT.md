# API audit: clarity, documentation, references, tests

Read-only review of the tree at commit `2c5bb8a` (2026-11-20), in six parts: the C++ core
libraries (ids C, I, D, T, R, S, M below, prefixed `cpp-`), the live chain in both engines
(`L`), the Python package (`P`), configuration keys against `CONFIGURATION.md` (`K`), the
tools and programs (`tools-T`), and physical models against their sources (`R`). Every
finding has file:line evidence; the ones marked **checked** were re-read against the code
before this list was written. Nothing has been changed yet: each group needs the owner's
decision (the owner's rule for this audit: no code change without asking).

Duplicates found by more than one part are merged (the ids of each part are kept in
brackets). Line numbers are at `2c5bb8a`.

## Overall

The suite is in good shape as an API. Most C++ headers and Python docstrings state units,
frames and error behaviour; nearly every physical model cites a source and says how far it
was confirmed (`REFERENCES.md` keeps a status per source); most models are tested against a
closed form, a printed value or an oracle; the two live-chain engines agree on every default
and are compared stage by stage in tests; environments and build specs reject unknown keys.

The problems are of five kinds:

1. **A few settings are silently lost or misread on the way to the C++ program**
   (`sonar_live`'s flat `key=value` interface reads every value with `atof`), and a few
   configuration values are silently accepted when wrong.
2. **A few models or defaults behave differently from what their documentation says.**
3. **Missing citations for the processing algorithms** (CFAR, MVDR, the tracker, the
   detection statistics); `REFERENCES.md` covers the simulator only.
4. **Stale or missing documentation**: some keys, the report's base fields, a few
   docstrings, a few statements made stale by later decisions.
5. **Untested public pieces**: half the tools, some library functions; the Bellhop/KRAKEN
   oracle tests skip silently when the programs are not installed.

## A. Behaviour: settings lost, values misread, documentation contradicted (CODE)

| id | where | what | consequence | proposed fix |
| --- | --- | --- | --- | --- |
| A1 (L1) **checked** | `tools/build_from_spec.py:133-166`; `libs/live/src/options.cpp:122` | The generated `chain.options` writes nested settings as Python text: `decimate={}`; `options.cpp` reads `atof("{}") = 0`. | The one-row builds' decimate-first choice is off when deployed with `sonar_live` (`popoto_testauv_1x32`, `popoto_face_1x32`): full rate, raw recording. Results are equivalent (test_decimate); compute and disk are not. | One shared flattener for both tools; `options.cpp` refuses non-numeric values where it expects a number; a test feeding `chain.options` through `set_pair`. |
| A2 (L2) | `tools/sonar_live.py:78-82`; `options.cpp:159-160,179-190` | The exported command writes profile booleans and lists as Python text (`True`, `[0.0, 0.017]`). | The profile ensemble is off; the first gradient of the list reads 0. | As A1. |
| A3 (L3) | `environment.hpp:92`; `options.cpp`; `tools/sonar_live.py:114-116` | `nav.surface_z` has no `sonar_live` key. | Lost in the C++ program (water depth from altitude, the built profile). | Add `nav_surface_z` and export it. |
| A4 (L4) | `live.py:42`; `chain.hpp:111`; `options.cpp:198` | Python's `backend: auto` uses CUDA when present; the program defaults to host and reads `auto` as host; `backend` is not exported. | A build with `compute.backend: cuda` runs on the host under `sonar_live`. | Export `backend`; accept `auto`; document the default. |
| A5 (L5) | `live.py:1337`; `chain.cpp:771-857` | `pings_without_nav` is on every Python report, never on the program's; two C++ comments say the fields are the same. | A reader of the program's lines misses the field. | Add it to `Report`; test the program's raw field set. |
| A6 (K15) **checked** | `live.py:374,1211`; `pipeline.py:216` | `look`, `window`, `backend` values are not checked. | `look: Starboard` (capital S) silently looks to port; a misspelt window is rectangular. | Check against the allowed values, as `beamformer` is. |
| A7 (K2) **checked** | `buildspec.py:728`; `sim/receiver.py:159,226` | The generator writes `receive.highpass_corner_hz`; the simulator reads it only under `receive.overload`. | The 100 Hz AC coupling of every generated sonar is not simulated. | Write it under `overload`, or drop it and its `unconfirmed` entry. |
| A8 (K4) | `sim/bellhop.py:70-72`; `environment.py:609-645` | The eigenray tier with no bottom silently uses 1600 m/s, 1.8, 0.5 dB/λ; `CONFIGURATION.md:134` says an undescribed bottom is an error (only checked for `image_source`). | A silent placeholder seabed. | Raise as for `image_source` (D-104). |
| A9 (cpp-S1) **checked** | `libs/sim/include/sonar/sim/reverb.hpp:40` | `ReverbParams::sound_speed = 1500.0`; every sibling struct defaults to NaN (HARDCODED_AUDIT K-2 lists this header). | A caller who forgets it gets 1500 m/s silently. | Default NaN; the existing check then refuses it. |
| A10 (P5) **checked** | `python/gpusonar/lloyd.py:270-293` | `bellhop_transfer` normalises by the horizontal range (`hypot(range, 0.0)`); its docstring and `one_way` use the slant range. | Differs at short range or steep geometry; used by the depth planner (`depthplan.py:112`). | Use the slant range (needs the depths) or correct the docstring; a test against `one_way` in isovelocity water (Bellhop). |
| A11 (R2) **checked** | `python/gpusonar/acoustics.py:27-30` | Below 5 PSU: Marczak plus Mackenzie's salinity and depth terms, a mix of our own; it jumps at 5 PSU (+0.8 / +1.3 / +1.7 m/s at 4 / 10 / 20 °C). | A discontinuity in sound speed with salinity. | Mark it ours; blend or state the jump; test Marczak's values. |
| A12 (tools-T1 to T3) **checked** | `tools/ssp_range_study.py`, `fjord_rehearsal.py`, `decay_time_sensitivity.py` | They resume from the committed JSON: a rerun recomputes nothing already stored, and their help does not say so; `decay_time_sensitivity` keys ignore `--pings`, `--taps`, `--max-arrivals`. | A "regeneration" that relabels old results (this happened on 2026-11-19). | `--fresh`, or the run options in the key; say it in the help. |
| A13 (tools-T9) | `tools/gallery.py:935,966`; `README.md:120` | `--quick` writes to `docs/figures` by default. | README's command overwrites the committed full-size gallery, which a test reads. | `--quick` to a scratch directory by default. |
| A14 (tools-T10, T11) **checked** | `tools/fetch_picmus.py:21-32`; `parts_list.py`, `architecture_diagrams.py`, `rough_surface_bands.py` | No argparse: `--help` does real work. | `fetch_picmus.py --help` creates a directory named `--help` and downloads into it; the others overwrite committed outputs. | argparse with `description=__doc__`. |
| A15 (K6) | `sim/receiver.py:83` | `receive.reference.mode` other than `sync_chirp` silently adds no reference. | A misspelt mode removes the sync reference. | Raise on an unknown mode. |
| A16 (K1) | `configs/sonars/flank_32.yaml:33` | `receive.bandpass_hz` is read by nothing. | Looks modelled, is not. | Remove, or implement. |

## B. Tests missing or weak (CODE: tests only)

| id | what | proposed test |
| --- | --- | --- |
| B1 (R1) | Sound speed has no reference test; `REFERENCES.md:149` names arlpy as its oracle, no test calls it. | Against arlpy `uwa.soundspeed` or printed check values. |
| B2 (R3) | Absorption is checked against a published table only at 0.1 to 10 kHz and about 1 °C; the sonars run at 100 to 750 kHz. | A published high-frequency or warm-water table, or arlpy over 10 kHz to 1 MHz. |
| B3 (cpp-S2) | The Rayleigh coefficient's lossy branch is untested (only the lossless limits). | A lossy case against a textbook value. |
| B4 (L13) | The `key=value` path is tested only with a minimal chain. | Round-trip every `cpp_chain_options` key through the binding and through `set_pair`; compare `ChainOptions`. |
| B5 (R11) | 13 oracle test files skip when Bellhop/KRAKEN are absent; a plain `pytest` passes without any oracle. | A switch that fails when oracles are expected; record the last oracle run in STATUS. |
| B6 (tools-T15) | 25 of 52 tools are run by no test (among them the three resumable studies, `inspect_tap`, `calibrate_sim`). | `--help` and quick-mode smoke tests. |
| B7 (cpp-C6) | Directivity's −3 dB-at-half-FWHM and `freq_ratio` are not asserted (the test re-implements the formula). | C++ test at sin = bw/2, behind boresight, freq_ratio 2. |
| B8 (P11) | `recon/multigpu.py` `backproject_tiles` claims "the same as a single call", never tested (hard-coded CUDA). | A `backend=` parameter and a host test against `sas_image`. |
| B9 (P12) | `sources/wav_watch.py` `WavWatchSource` (the file route for capture) is untested. | A temp-directory test: late WAV picked up once; a bad file to `failed`. |
| B10 (P13) | `scatterer_doppler_variance`, `iso_delay_angle`, `coherence_time` only tested end to end. | Direct closed-form and histogram tests. |
| B11 (cpp-R2) | `recon` `cull_mask` is public, unbound, tested only end to end. | Move to a detail namespace, or bind and test. |

## C. Citations missing (DOC-ONLY, but each source must be read before it is cited)

The project's rule: do not invent what a paper reports. Each citation below is to be added
with the status `REFERENCES.md` uses (confirmed / via secondary / ours / unconfirmed), and
only "confirmed" where the source was actually opened.

| id | what has no source | where |
| --- | --- | --- |
| C1 (cpp-T2, L12, R10) | Cell-averaging CFAR factor N(Pfa^(-1/N) − 1) | `libs/detect` `cfar.cpp:16-18`, `cfar.hpp:47` |
| C2 (cpp-T2) | Lugannani–Rice saddlepoint tail of the detection statistics | `statistic.hpp:115` |
| C3 (L12, R10) | MVDR / Capon beamformer, and the diagonal loading 3e-2 (to be marked as our choice) | `libs/live` `beamform.hpp:63-78` |
| C4 (L12, R10) | Constant-velocity Kalman tracker and its association (D-141) | `detect.py:288`, `tracker.cpp` |
| C5 (L12) | Noncentral chi-square / Swerling detection probability | `live.py:253-319` |
| C6 (cpp-S2, R10) | Rayleigh fluid–fluid reflection, dB/λ to loss tangent (PHYSICS_NOTES 6.4 says "from memory, verify before use") | `paths.hpp:134-137`, `propagation.py:364-380` |
| C7 (P6) | Munk profile (Munk 1974; the textbook constants) | `sim/bellhop.py:168` |
| C8 (R9) | ASASIN, the SBL papers, Sæbø 2010, Cook 2007, the Acoustics Toolbox: cited in PAPER_NOTES / headers / OPEN_SOURCE, absent from `REFERENCES.md`, which declares itself simulator-only | `REFERENCES.md:1-8` |
| C9 (R4, R5) | Chapman–Harris and the Cox–Munk total slope read as published in their docstrings; `REFERENCES.md` marks both unconfirmed | `acoustics.py:194-206, 216-219` (docstrings: code files) |

## D. Documentation stale or contradicting the code (DOC-ONLY, Markdown files)

| id | where | correction |
| --- | --- | --- |
| D1 (P4, K3) **checked** | `CONFIGURATION.md:439` | The Python provider's `max_order` is not 2: bounces by energy, cap 8 (D-105). |
| D2 (R6) **checked** | `PHYSICS_NOTES.md:1580` | "the default stays the rule": `facet` is the default since D-85. |
| D3 (R7) | `PHYSICS_NOTES.md:2255` | "rule_of_thumb … the suite's defaults": the defaults before D-85. |
| D4 (R8) | `PHYSICS_NOTES.md:846` | −27 dB Lambert "classic default": the code uses −29 dB (D-89). |
| D5 (cpp-R1) **checked** | `DECISIONS.md:136-138` (D-09) | `culling.beam_factor` does not exist: the parameter is `fov_factor`, half-angle `fov_factor·FWHM/2`, default 0 (`sas_image` 1.13). |
| D6 (cpp-C2) | `DECISIONS.md:129` (D-08) | `reserve(bytes, alignment)` is `reserve(bytes)`. |
| D7 (tools-T6) **checked** | `RESULTS.md:1695` | Calls `gpu_benchmark_all_l4.md` the first run; that is `_15f1121.md`. |
| D8 (tools-T14) | `README.md:126-127` | Not every study tool writes a committed file (tonal_range_study prints; coverage_map, gpu_benchmark_all write only with an option). |
| D9 (tools-T7) | `tools/gpu_job.sh:6-21`; `HANDOFF.md:112-120` | The header and HANDOFF's mode table miss `benchmark`, `liveablate` and several variables. |
| D10 (cpp-M1) | `ARCHITECTURE.md:181` | The layout table has no row for `libs/motion`. |
| D11 (K13) | `CONFIGURATION.md:239` | Shows the deprecated `source_level_db`; use `source_level_db_rms`. |
| D12 (L8) | `CONFIGURATION.md:464` (and `chain.hpp:64`, `live.py:41`) | `farfield` and `mvdr` do run on the device (`chain.cpp:107-109`). |
| D13 (P1) | `ORG_ARL_SURVEY.md:252` | Lists self-noise and tonals as "ours better"; `self_noise_spectrum_db` and `add_tones` are not reachable from any scenario. |

## E. Documentation missing (DOC-ONLY, Markdown files)

| id | what to document | source of truth |
| --- | --- | --- |
| E1 (L6) | The report's base schema, with units and frames: `ping`, `sequence`, `time_ns`, `ownship_xyz`, `sync`, detections' `range_m` and `bearing_deg` (reference and sign), tracks' `id`, `kind`, `parent`, `x`/`y`, `vx`/`vy`, `speed_m_s`, `seen`, `mean_snr_db`, `processing_seconds` | `chain.cpp:771-857`, `live.py` |
| E2 (L10) | One table: deployment-file path → `sonar_live` key → unit → default (tracker, classify, sbl, sync, multipath groups; `sync.clip_level`; that `classify_min_speed` also sets `static_speed`) | `options.cpp` |
| E3 (L9) | `mvdr_subbands`, `mvdr_loading`, `gate_fft`, `sbl_kernels`: C++ only, in no doc and not in the Python engine | `options.cpp:60-62,199` |
| E4 (L11, L7) | Units of the tracker's `process_noise`, `gate`, `initial_speed_sigma`; which binding keys are radians without a suffix | `detections.hpp:72-80`, `bind_live.cpp` |
| E5 (K5) | Bellhop grid keys `range_step`, `depth_step`, `min_range`, `n_beams`, `angles` (degrees) with defaults | `sim/bellhop.py:332-335` |
| E6 (K6, K7) | `receive.reference.*` and `receive.boards[].*` key names and units (`clock_offset` s, `jitter_rms` s, `level` V) | `sim/receiver.py:76-100`, `simulator.py:462-468` |
| E7 (K8 to K12) | `platform.rates`, `attitude_rate_hz`, `pings.start_ns`; `polar_grid` `forward`/`across`; `rocks.density`; reverberation `n_azimuth`, `max_range`; `time_spread.frequency`, `seed`; top-level `compression_level` | `simulator.py`, `scene.py`, `timespread.py` |
| E8 (K14) | `chain.coverage.budget.source_level_db` is an RMS level | `coverage.py:48` |
| E9 (cpp-C5) | Asking for CUDA in a build without it throws, in every library | `platform.hpp` |
| E10 (tools-T8) | The command that made each `docs/benchmarks/bench_*`, `session_*`, `conditioning_*` file | `RESULTS.md` |
| E11 (K16) | `truth` blocks in scenarios are for tests and tools only (`first_x_m` read by nothing) | `configs/scenarios/popoto_12km_*row.yaml` |

## F. Comments and docstrings in code files (no behaviour change, but in code files)

| id | where | what |
| --- | --- | --- |
| F1 (cpp-T1) **checked** | `statistic.hpp:17` | Cites `tests/cpp/test_statistic.cpp`, which does not exist (the tests are in `test_detector.cpp`). |
| F2 (cpp-C1) | `allocator.hpp:3-6` | "a single device allocation": only backprojection uses the slab. |
| F3 (cpp-D1) | `baseband.hpp:19-21`, `pulse_compression.hpp:23-25` | "two-way range rate" is the radial (one-way) rate in the formulas. |
| F4 (cpp-C4, I1, S3) | `platform.hpp:30`, `record_file.hpp:52-63`, `reverb.hpp:1-10` | Lifetime of `cuda_device_name`'s pointer; the reader's exceptions; reverb's source (SST Sec. 11.3). |
| F5 (cpp-C3) | `waveform.hpp:14` | Type 3 (Arbitrary) not listed. (`waveform_eval` is used by pulse compression, not dead: see G2.) |
| F6 (P7 to P10) | `simulator.py:20` `make_pose`; `propagation.py:93,383`; `target_classes.py:53`; `buildspec.py` helpers | Missing docstrings: frame and units of `make_pose`, the provider tiers, `cylinder_ts_db`, `adc_full_scale` (V), `window_range`. |
| F7 (L8) | `chain.hpp:64`, `live.py:41` | As D12. |
| F8 (R4, R5) | `acoustics.py:194-206, 216-219` | As C9: mark unconfirmed in the docstrings. |
| F9 (tools-T12) | `tools/scenario_diagrams.py:269` | `ArgumentParser()` without a description. |

## G. Dead or duplicate public API (CODE: removal)

| id | what |
| --- | --- |
| G1 (cpp-C2) | `core::Stage`: no library subclass. |
| ~~G2 (cpp-C3)~~ | ~~`core::waveform_eval`: no caller.~~ Wrong: `libs/dsp/src/pulse_compression.cpp:142` calls it (found while fixing F5). It stays. |
| G3 (P2, P3) | `recon/gain.py` `has_water`, `taps.py` `tap_files`: no caller. |
| G4 (P1) | `sim/noise.py` `self_noise_spectrum_db`, `add_tones`: not reachable from a scenario (wire in, or mark library-only). |
| G5 (tools-T13) | `tools/popoto_truth_check.py`: a script with the ship track hard-coded twice; to a test, or remove. |

## In good shape (from the six parts)

- Units, frames, peak-versus-RMS, the `exp(+iωt)` sign and NaN-for-no-default are stated once
  and consistently in `record.hpp`; `RECORD_FORMAT.md` matches it field for field.
- Spot-checked formulas all hold: `rotation_ned`, `gaussian_sigma`, the TVG absorption
  example, Thorp's unit conversion, Chapman–Harris as typed, Pierson–Moskowitz's second
  moment, Munk's constants, Cron–Sherman, array gain against isotropic noise, the ray arcs,
  Poisson speckle.
- Oracle and closed-form tests: backprojection against an independent Python reference;
  detection statistics against Erlang and hypoexponential forms; Lloyd's mirror; KRAKEN
  against Bellhop; sediment models to 1e-10 of their published forms; TR 0702.
- The two live-chain engines: equal defaults, one derivation of sizes, engine-against-engine
  tests for clustering, tracker (both rules), classifier, the full chain, sync, the
  multiscale detector, elevation, depth, decimation, blanking, multipath depth.
- Environments and build specs are strict about unknown keys; every build-spec field is
  documented; documented defaults checked against the code matched.
- Every tool has a docstring; no option named in the docs is missing from its tool.
