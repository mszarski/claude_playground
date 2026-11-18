# Hard-coded assumptions: audit

Read-only audit of everything except the live chain (`libs/live/**` and
`python/gpusonar/{live,flank,detect,classify,elevation,sync}.py`, which have their own audit and owner).
Nothing was changed. Branch `claude/gpu-sonar-suite-u289p5` at `3113327`.

The goal it is measured against: a different sonar, platform or environment is a configuration change,
never a code change; hardware facts the owner has not confirmed live in configuration, flagged.

**Classes.** A: a literal that encodes one sonar, platform, board, band, environment or site.
B: a default that suits one sonar class and is silently wrong for another, where the right value can be
derived. C: a capacity limit. D: a unit or convention not stated at the interface. E: a hardware fact not
confirmed by the owner that is in code, or in a config without an `unconfirmed` flag. F: a constant whose
source I could not find (section 5; not counted as findings).

**How it was done, and what was not read.** Every file in scope was searched for numeric literals,
`default=`, `.get(..., literal)`, `constexpr`, `#define`, fixed-size arrays, sonar names, `1500`, sample
rates and port numbers, and the code around each hit was read. Read in full or nearly: all C++ headers
in scope, `apps/`, `sonar.py`, `sonar_equation.py`, `acoustics.py`, `sources/wav.py`, the sonar, build
and deploy configs. Read only around the hits: `sim/noise_fit.py`, `sim/channel_replay.py`,
`sim/sediment_fit.py`, `sim/reverb_bistatic.py`, `sim/reverb_relief.py`, `sim/reverb_specular.py`,
`sim/targets.py`, `sim/waveform.py`, `sim/bottom.py`, `sim/surface.py`, `tonal.py`,
`tools/sst_validation.py`, `tools/gpu_benchmark_all.py`, `tools/channel_*.py`. A finding in those files
may have been missed. "What breaks" is from reading the code; nothing was run.

## 1. Summary

| Area | A | B | C | D | E | Rows |
| --- | --- | --- | --- | --- | --- | --- |
| C++ core, io, dsp, detect (`K`) | 3 | 3 | 0 | 1 | 0 | 7 |
| C++ sim and bindings (`S`) | 3 | 2 | 2 | 0 | 0 | 7 |
| Imaging and motion, C++ and Python (`R`) | 1 | 3 | 0 | 1 | 0 | 5 |
| Apps and Holoscan operators (`A`) | 3 | 1 | 2 | 0 | 0 | 6 |
| Python package outside `sim/` (`P`) | 6 | 4 | 0 | 3 | 0 | 13 |
| Simulator, `sim/` (`M`) | 3 | 7 | 1 | 0 | 0 | 11 |
| Sources (`SR`) | 4 | 1 | 1 | 1 | 1 | 8 |
| Build spec and generator (`B`) | 6 | 2 | 1 | 0 | 1 | 10 |
| Tools (`T`) | 8 | 0 | 0 | 0 | 0 | 8 |
| Configs (`G`) | 4 | 0 | 0 | 0 | 7 | 11 |
| Tests (`X`) | 1 | 0 | 0 | 0 | 0 | 1 |
| **Total** | **42** | **23** | **7** | **6** | **9** | **87** |

A row is counted once, under its first class; several rows stand for a literal repeated in many places
(P-1 alone is about 35 places).

**Areas that are clean.** `libs/io` (record file, validation: every limit has a stated reason and raises
or warns). `libs/core` allocator, vectors, CUDA helpers. `python/gpusonar/output.py`, `metrics.py`,
`calibrate.py`, `taps.py`, `config.py` apart from P-13. `python/gpusonar/hardware.py` (one chip's limits,
named for the chip, source commit cited). `sources/picmus.py`, `sources/container.py`, `sources/base.py`.
`configs/sediments.yaml` was not audited row by row (STATUS open question 10 covers its two "confirmed"
rows).

### The five most consequential

1. **One site is typed in up to eight places in a scenario, and the ray tracer's frequency is typed
   beside the sonar (G-1, M-1).** Water depth is in `environment.water_depth`,
   `propagation.bottom_z` or `propagation.environment.depth`, and each `reverberation[].boundary_z`;
   wind is in `reverberation[].wind_speed`, `receiver.ambient_noise.wind_speed` and
   `time_spread.surface.wind_speed`; the bottom is in `propagation.bottom` and again as a Lambert
   `mu_db`; `propagation.frequency: 5000.0` does not follow `sonar:` or `sonar_overrides`. Change the
   sonar or the site and miss one: a silent wrong answer.
2. **Ambient noise in a baseband record is flat across the band and independent between hydrophones
   (P-8, `sim/simulator.py:316-327`).** The comment says this is true at half-wavelength spacing. Every
   baseband scenario uses it, the 12 km check included, so array gain against sea noise is the full
   10 log N whatever the pitch. Silent, optimistic.
3. **Detection and tracking thresholds are one array's, in metres, and exist in three copies (B-2).**
   `buildspec.DEFAULT_CHAIN` (2 m cell, 100 m guard, 300 m training, 15 m tracker sigma, 800 m excess
   range), the hand-written deploy files, and the chain's own defaults. A build spec for another band
   or pulse gets them unchanged. Silent, degraded detection.
4. **Source level is used as a peak pressure, and nothing says so at the interface (P-5).**
   `sonar.py:255-266` turns `source_level_db` into a peak amplitude; the sonar-equation tool and the
   link budgets use the same number in an energy equation; noise levels are rms. A datasheet figure is
   normally rms: 3 dB. `buildspec.py:450` admits it in a warning string; no config key states it.
5. **Hardware facts the plan does not confirm are unflagged, or have a fallback in code (G-4 to G-8,
   B-4, SR-1).** The 4 to 6 kHz band is below the 10 to 100 kHz the board is specified for and no key
   carries a flag for it; `boards.confirmed` lists `input_full_scale_vpp`, which removes the flag from
   `receive.adc_full_scale` although where the 37 mVpp is referred to is our reading; clock tolerance
   25 ppm, offset 0.3 ms and jitter 20 ns are fallbacks in `buildspec.py`; the board's stream format is
   ours, not Popoto's.

Close behind: the live app exits after 30 s without a block and its block queue has no bound (A-1,
A-2); the backprojection cull allows 2 cm of trajectory residual (R-1); multipath is cut at two bounces
or eight arrivals by default, silently (S-3, M-2).

## 2. Findings

Columns: where; the literal; class; what it assumes; what breaks and how (error, silent wrong answer,
degraded); proposed fix; effort (S under an hour, M half a day, L more); risk to existing outputs
(none / default: needs a default that reproduces today's value / changes); files the fix touches.

### C++ core, io, dsp, detect

| ID | Where | Literal | Cl. | Assumes | Breaks, how | Fix | Eff. | Risk | Touches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K-1 | `libs/core/include/sonar/core/record.hpp:33`; `python/gpusonar/sonar.py:17`; `sources/wav.py:27-31`; `python/bindings/module.cpp:93-95` | `enum Mode {Unknown, Fls, Sas, Flank}` | A | every sonar is one of three classes | a config with another `mode:` raises `KeyError` in `SonarModel.make_record`. `configs/sonars/subbottom.yaml` calls a sub-bottom profiler `sas` for want of a value. No C++ code branches on it. Error | keep the enum as a label, add `Other` and a free-text `metadata.mode_name`; unknown strings map to `Other` with a warning | S | none | `record.hpp`, `module.cpp`, `sonar.py`, `sources/wav.py`, `docs/RECORD_FORMAT.md` |
| K-2 | `record.hpp:169`; `libs/sim/.../paths.hpp:21,39,69,110`; `reverb.hpp:38`; `synth.hpp:70`; `libs/recon/.../backproject.hpp:49`; `libs/motion/.../dpc_solver.hpp:49` | `sound_speed = 1500.0` as a struct default | A | sea water near 13 C | a caller that forgets to set it gets 1500 in a 1480 m/s pool: 1.3 % in every range and wavelength. Silent | default NaN; `validate.cpp:208` already rejects a non-finite value, and the kernels' entry points should too | S | none (every shipped path sets it) | the six headers, `libs/io/src/validate.cpp` |
| K-3 | `record.hpp:110,119` | `boresight{0, 1, 0}`, `direction{0, 1, 0}` | D | an element looks to starboard | a record built in C++ without a boresight is a side-looking array. Silent | default NaN; `validate.cpp:35` already demands a unit vector | S | none | `record.hpp` |
| K-4 | `libs/dsp/.../conditioning.hpp:21-24`; `python/bindings/bind_recon.cpp:161-163` | time-varying gain: `40` dB per decade, absorption `0.0`, clamp `120` dB | B | two-way spherical spreading, no absorption | at 330 kHz absorption is 0.083 dB/m (12 C): the gain is 50 dB short at 300 m; a one-way (passive) record needs 20. Degraded display and background level | `absorption_db_per_m < 0` means "from the record": centre frequency and `environment`; spreading from a chain option | S | default | `conditioning.hpp`, `conditioning.cpp`, `bind_recon.cpp` |
| K-5 | `libs/detect/.../cfar.hpp:13-15`; `bind_detect.cpp:35`; `python/gpusonar/holoscan_ops.py:118` | `pfa 1e-4` (C++), `1e-6` (Holoscan operator); `guard 4`, `train 32` cells | B | the image's range step is about one resolution cell | on an image sampled at 6 cm with a 2 m compressed pulse the guard is inside the target: it masks itself. Two defaults for one quantity. Degraded | guard and training in metres, or in resolution cells from `c / 2B` and the image's step; one default table | M | default | `cfar.hpp`, `cfar.cpp`, `bind_detect.cpp`, `holoscan_ops.py` |
| K-6 | `libs/dsp/src/baseband.cpp:30` | band of a CW pulse `4.0 / duration` | B | no Doppler | a 1 s tone at 5 kHz keeps 4 Hz; 1 m/s of closing speed shifts the echo 6.7 Hz: filtered out. Silent | add a Doppler allowance `2 v_max f / c` from an option | S | default | `baseband.hpp`, `baseband.cpp`, `dsp_cuda.cu` |
| K-7 | `libs/dsp/src/pulse_compression.cpp:75-76,102`; `python/gpusonar/sim/waveform.py:76-80` | resampler `16` taps, Kaiser `12.0`, typed in C++ (twice) and in Python | A | the two engines keep the same numbers by hand | change one and the C++ and NumPy engines stop agreeing. Test failure, not a wrong answer | export the two constants through the bindings, as `ABSORPTION_TABLE_SIZE` is | S | none | `pulse_compression.cpp`, `bind_recon.cpp`, `sim/waveform.py` |

### C++ sim and bindings

| ID | Where | Literal | Cl. | Assumes | Breaks, how | Fix | Eff. | Risk | Touches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S-1 | `libs/sim/.../paths.hpp:57,75`; `python/gpusonar/sim/propagation.py:131,184,203`; `python/bindings/bind_simcpp.cpp:130-131` | wall coefficient `0.8`, bottom coefficient `0.5` | A | a made-up reflection, the same at every angle | an image-source scenario with `bottom_z` and no `bottom:` loses 6 dB a bounce at any grazing angle. Silent | no default: `bottom_z` without `bottom`, `sediment` or a coefficient is an error; a wall without `coefficient` likewise | S | default (list the scenarios that rely on it first) | `paths.hpp`, `propagation.py`, `bind_simcpp.cpp` |
| S-2 | `paths.hpp:62-63,111`; `backproject.hpp:87`; `sim/bellhop.py:53-55`; `buildspec.py:180`; `tools/channel_validation.py:69`; `tools/popoto_12km_study.py:48-49`; 14 scenarios | "sand" as `1600/1.8/0`, `1600/1.8/0.5`, `1650/1.9/0.8`, `1700/1.9/0.5`, sediment speed `1700` | A | a sandy bottom, four different ones | a scenario that leaves the bottom out gets a different sand at each tier. Silent, inconsistent | one named entry in `configs/sediments.yaml` referenced by name; no numeric default in code | M | default | the files listed, `configs/scenarios/*.yaml` |
| S-3 | `paths.hpp:76`; `propagation.py:131`; `bind_simcpp.cpp:131` | `max_order = 2` | B | two bounces carry the field | at 12 km in 100 m of water many more do; the rest are dropped. Silent under-estimate | no default in C++; in Python derive a suggestion from range / depth and log the order used | S | default | `paths.hpp`, `propagation.py` |
| S-4 | `python/bindings/bind_simcpp.cpp:324,345,401` | `1500.0` passed as the sound speed | A | these three entry points never use it | true today as far as I read; the day one does, a silent 1500 | drop the argument from these calls, or pass the caller's value | S | none | `bind_simcpp.cpp` |
| S-5 | `libs/sim/.../synth.hpp:44`; `synth.cpp:46`; `bind_sim.cpp:99` | `kMaxEnvelopeBands = 33` | C | a sampled pulse's gain needs at most 33 knots | more: `invalid_argument`. Error, as it should be | none needed; say the limit in `CONFIGURATION.md` | S | none | docs |
| S-6 | `synth.hpp:41`; `synth_kernel.hpp:37-41,122`; `sim/simulator.py:36` | absorption table of `16` points over the transmit band; `16` amplitude knots along a sweep | C | loss and beam pattern are smooth across the band | outside the band the table is clamped; a pattern null crossing a wide band is under-sampled. Degraded, small | leave; state it; size from relative bandwidth if a wide-band sonar appears | S | none | docs |
| S-7 | `synth.hpp:89-90,96` | arrival-time passes `(4, 4)`, `8` for sampled pulses | B | 12 m/s at 400 m and 8 kHz (the comment's case) | residual is `(R/c)(v/c)^n 2 pi f`: 1e-3 rad at 12 km and 5 kHz, still fine; not checked for another case | compute the passes from the record's window, top frequency and speed, or assert the bound | S | none | `synth.hpp`, `synth.cpp` |

### Imaging and motion

| ID | Where | Literal | Cl. | Assumes | Breaks, how | Fix | Eff. | Risk | Touches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R-1 | `libs/recon/.../backproject_kernel.hpp:361,364` | cull margin `+ 0.02F` m "trajectory residual allowance", `+ 1e-3F` rad | A | the platform strays under 2 cm from its first-order pose within one ping | a long ping (18 s at 12 km) or a turning platform with a navigation table strays by metres: pairs the exact test would keep are culled. Only with `fov_factor > 0`. Silent loss at the beam edges | the host already builds the residual table (`backproject.cpp`): store its largest norm in `BpGroup`, as `aperture` is | S | none | `backproject.hpp`, `backproject.cpp`, `backproject_kernel.hpp` |
| R-2 | `libs/motion/.../dpc_solver.hpp:53-55`; `python/gpusonar/motion.py:36-38` | scales `1e-6` s, `1e-3` m/s, `1e-2` m in C++; a sixteenth of a period, `2e-3`, `5e-3` in Python | D | C++: a carrier of some hundreds of kHz | a C++ caller at 5 kHz gets a Huber threshold of 1/200 of a period. Two defaults for the same three numbers. Degraded convergence | no defaults in C++ (required), Python's derivation is the one place | S | none through Python | `dpc_solver.hpp`, `bind_motion.cpp` |
| R-3 | `python/gpusonar/motion.py:28-31` | `window_samples 256`, `max_lag_samples 12`, `n_windows 8` | B | one sample rate and bandwidth | the largest sway between pings it can measure is 12 samples of range; more is rejected as incoherent. Degraded | `max_lag_m` and `window_m`, turned into samples from the record | M | default | `motion.py` |
| R-4 | `python/gpusonar/recon/sas.py:29`; `motion.py:39`; `pipeline.py:133` | `plane_z = 0.0`, `seafloor_z = 0.0` | B | the scenario's origin is on the seabed | a survey whose origin is at the surface images the wrong plane: defocus. Silent | default from `environment.water_depth` or the DVL altitude in the record; error when neither is there | M | changes where it applies | `recon/sas.py`, `motion.py`, `pipeline.py` |
| R-5 | `recon/sas.py:70-71`; `pipeline.py:138-139` | `fov_factor 1.13`, `cull_block 16`, each typed twice | B | the paper's element (`1 / 0.886`) | the two copies can drift | `pipeline.py` passes only what the config gives | S | none | `pipeline.py` |

### Apps and Holoscan operators

| ID | Where | Literal | Cl. | Assumes | Breaks, how | Fix | Eff. | Risk | Touches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A-1 | `apps/sonar_live.cpp:199`; `libs/holoscan_ops/src/operators.cpp:50`; `buildspec.py:748` (`timeout: 30.0`); `sources/hsb.py:476` (`15.0`) | "the stream has ended" after `30` s | A | blocks never pause for 30 s | a board that sends only inside the receive window with a longer ping interval, or a board that restarts: the app returns 0 and systemd restarts it. Silent stop | `idle_seconds` option (0: never), written by the generator from the ping interval | S | none | `apps/sonar_live.cpp`, `buildspec.py` |
| A-2 | `apps/sonar_live.cpp:171` | `std::deque` of blocks with no bound | C | processing keeps up | when it does not, memory grows until the process is killed; nothing is counted. Crash | bound it and count dropped blocks in the closing summary, as the Python source does (`queue_pings`) | S | none | `apps/sonar_live.cpp` |
| A-3 | `operators.cpp:45`; `buildspec.py:748,753,757`; `tools/sonar_live.py:74`; `configs/deploy/*.yaml`; `configs/deploy/sonar_live.service` | ports `5700`, `5702`, `5701` | A | one network plan | defaults in three code files and typed in every config; change one and the others disagree. Error at run time (nothing arrives) | no default in code: the deploy file is the one source | S | none | `operators.cpp`, `buildspec.py`, `tools/sonar_live.py` |
| A-4 | `operators.cpp:48` | `chunk_samples = 8192` | B | a frame length unrelated to the passive options | the app computes `n_fft + (n_average - 1) hop` and passes it; another host of the operator gets 8192. Degraded | no default: required when `listen` is true | S | none | `operators.cpp` |
| A-5 | `apps/sonar_live.cpp:134,163,165` | bind address `"0.0.0.0"` | A | every interface | cannot be limited to the sonar's network | `--bind` option | S | none | `apps/sonar_live.cpp` |
| A-6 | `apps/sonar_live.cpp:65-66` | `char out[1024]` for the stamped path | C | a path under 1 kB | longer: `strftime` returns 0 and the unstamped path is used, so a restart overwrites the last recording. Silent | fail instead of falling back | S | none | `apps/sonar_live.cpp` |

### Python package outside `sim/`

| ID | Where | Literal | Cl. | Assumes | Breaks, how | Fix | Eff. | Risk | Touches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P-1 | `sonar.py:144,166,395`; `sources/flank_stream.py:102`; `sources/wav.py:227`; `buildspec.py:313,565,583,803,1104,1115,1129`; `sim/scene.py:156`; `sim/passive.py:22`; `sim/noise.py:42,51,177`; `sim/surface.py` (nine signatures); `sim/bottom.py:485,495`; `sim/reverb_specular.py:84`; `sim/sediment_fit.py:145`; `sim/noise_fit.py:667`; `sim/simulator.py:184,186,209`; `acoustics.py:130,225,233`; `tools/*` (`C = 1500.0` in two studies, `--sound-speed` defaults) | `1500.0` as the default sound speed, about 35 times | A | sea water | any call that omits it in fresh or cold water. Silent | one `acoustics.NOMINAL_SOUND_SPEED`, used only by `sonar.resolve_environment`; every model function takes the speed as a required argument | M | none if callers pass it (check each) | the files listed |
| P-2 | `sim/simulator.py:38-39,156-157`; `sim/sources.py:534`; `sonar_equation.py:32-34`; `acoustics.py` signatures; `buildspec.py:813,1169`; `tools/popoto_range_study.py:47` | temperature `10.0` (or `12.0`), salinity `35.0`, depth `0.0` (or `50.0`), pH `8.0` | A | one ocean, two versions of it | a scenario without `environment.temperature` gets sea-water absorption at 10 C; the design tools use 12 C and 50 m. At 330 kHz absorption is 0.074 dB/m at 5 C and 0.108 at 20 C: 20 dB two-way at 300 m. Silent | `resolve_environment` fills these once and says in the processing log which were defaulted; `SonarEquation.from_sonar` takes the environment | M | none | `sonar.py`, `simulator.py`, `sim/sources.py`, `sonar_equation.py`, `buildspec.py` |
| P-3 | `sonar.py:393-398`; 23 scenarios | `sound_speed: 1500.0` beside `temperature: 12.0, salinity: 35.0` | D | the typed speed and the typed water need not agree | Mackenzie gives 1496.8 m/s for that water; geometry uses 1500 and absorption uses 12 C. Small, silent | warn when both are given and differ by more than 1 m/s | S | none | `sonar.py` |
| P-4 | `sonar.py:67,117,215`; `sources/wav.py:206` | default `array.boresight`: `[1, 0, 0]` in `apply_fov`, `[0, 1, 0]` elsewhere | D | forward for a sonar with a `fov` block, starboard without | the same missing key means two directions. Silent | make `array.boresight` required in a base file (17 inheriting files rely on the base's) | S | none | `sonar.py` |
| P-5 | `sonar.py:255-266`; `record.hpp:91`; `sonar_equation.py:1-9`; `buildspec.py:450` | amplitude `10^((SL + sensitivity + gain) / 20)` is a peak | D | `source_level_db` is a peak level | a datasheet rms level is simulated 3 dB low in power against rms noise, and the overload study is 3 dB off the other way. Silent | `transmit.waveform.source_level_reference: peak` or `rms` (default `peak`), stated in `CONFIGURATION.md` and applied in one function | S | default | `sonar.py`, `sonar_equation.py`, `buildspec.py`, docs |
| P-6 | `sonar_equation.py:36,52`; `tonal.py:853` | array gain `10 log10(N)` | B | hydrophones at half a wavelength or more | the test AUV's short row is closer: `reference_plan.py:143` says so in prose while the tool overstates. Optimistic | `SonarEquation.from_sonar` uses `buildspec.directivity_db` (it exists) | S | changes design numbers for close-pitch rows | `sonar_equation.py`, `tonal.py` |
| P-7 | `sonar.py:272-282` | receive window `2 r / c` | B | monostatic, two-way | a passive or bistatic sonar cannot state its window in range; `start` and `samples` is the way out | say so in `CONFIGURATION.md`; reject `range` when `transmit` is absent | S | none | `sonar.py`, docs |
| P-8 | `sim/simulator.py:316-327` | baseband ambient noise: the centre frequency's level, independent channels | B | half-wavelength pitch, a narrow band | pitch under half a wavelength: no shared noise, so full `10 log N`; 4 to 6 kHz: the spectrum falls about 3 dB across the band and is drawn flat. Every baseband scenario. Silent, optimistic | draw it as the real-passband path does (`sim/noise.py`: spectrum in bands, isotropic coherence) and mix down | M | changes | `simulator.py`, `sim/noise.py` |
| P-9 | `sim/simulator.py:383-384`; `buildspec.py:1148,1159`; `tools/parts_list.py:72,76`; `tools/popoto_range_study.py:44,49`; `tools/tonal_range_study.py:46-47` | wind `5.0` m/s, shipping `0.5` | A | one sea | five copies of the default noise environment | part of the named environment of G-2 | S | none | the files listed |
| P-10 | `tonal.py:44-82` | bearings `-80..80` by 1 deg, `row_tolerance_m 0.02`, envelope rate `100` Hz, 8 s segments, 1 s blocks | B | a kHz array listening to shipping; rows found again from positions with a 2 cm tolerance | rows closer than 2 cm merge (the chain has the same number). All are keys; the defaults are this array's | carry the row of each channel in the record's metadata (the `stacked_lines` layout knows it) instead of inferring it | M | none | `sonar.py`, `tonal.py` (and `elevation.py`, other owner) |
| P-11 | `simcal.py:148,64,76,225,267` | noise reference `50.0` dB, `120` bins, blocks of `48` pixels, density cap `5000` | A | one log's scale | all are arguments; fine as long as callers pass them | none beyond documenting | S | none | docs |
| P-12 | `processing.py:37`; `pipeline.py:11-15` (docstring example) | scan conversion `512 x 512` | A | one display | coarse for a wide fan | size from the fan's own sampling | S | none | `processing.py` |
| P-13 | `config.py:36` | config root `parents[2] / "configs"` | A | the package runs from its source tree | an installed package finds no shipped config. Error | ship the configs as package data; `GPUSONAR_CONFIG_PATH` already exists | S | none | `config.py`, packaging |

### Simulator, `sim/`

| ID | Where | Literal | Cl. | Assumes | Breaks, how | Fix | Eff. | Risk | Touches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M-1 | `sim/propagation.py:337-341`; scenarios (`frequency: 5000.0` 7 times, `8000.0` 5, `27500.0` 3) | the eigenray tier takes `frequency`, `max_range`, `environment.depth`, `ssp_speed` from `propagation` only | B | whoever edits `sonar:` also edits these | swap the sonar or override its band: Bellhop still traces at 5 kHz (bottom loss, attenuation). Silent. `time_spread` already takes the sonar's frequency (`simulator.py:536`) | `make_provider` receives the sonar and the environment: frequency defaults to the centre, `max_range` to the receive window, depth to `environment.water_depth`; explicit keys override; a contradiction is an error | M | none while they agree | `propagation.py`, `simulator.py`, `bellhop.py` |
| M-2 | `sim/bellhop.py:241,311` | `max_arrivals = 8`; the strongest are kept | C | eight arrivals hold the energy | seven scenarios set 40 because eight is not enough in shallow water; the default drops the rest. Silent | log the dropped energy fraction and warn above 1 %; default from range / depth | S | none | `bellhop.py` |
| M-3 | `sim/bellhop.py:241,251,253` | launch angles `-80..80` deg, range grid `max_range / 200`, depth grid `depth / 50` | B | arrivals vary slowly over 60 m of range at 12 km | amplitude is taken from the nearest grid node (delay is corrected to first order). Degraded near caustics and shadow edges | state it; make the steps follow the wavelength and the receive aperture | M | changes | `bellhop.py` |
| M-4 | `sim/scene.py:494`; `:515,588,599,604,613` | wire spacing `0.01` m; hull `4000`, boat `800`, wake `600` points, keel `40` per m2 | B | scatterers closer than a resolution cell at kHz to 100 kHz | at 750 kHz a wavelength is 2 mm: a wire at 1 cm is a row of separate points. Silent artefact | spacing and counts from the sonar's wavelength and range resolution (the scene builder is given the sonar), explicit keys override | M | default | `scene.py`, `simulator.py` |
| M-5 | `sim/scene.py:425-658`; `sim/simulator.py:212,253` | default strengths `mu_db` `-8, -10, -12, -15, -20, -27, -30`, wake `-25` dB/m, propeller `-5` dB | A | one frequency-independent strength per kind of object | fine as scenario defaults, but they live in code and are the same at 5 kHz and 750 kHz | a `scene_defaults` table in config, cited or marked illustrative | S | none | `scene.py`, new config |
| M-6 | `sim/timespread.py:265` | `coherence_length = 10.0` m, `taps = 6`, `doppler_lines = 8` | B | one realisation cell size for every frequency and sea | a 120 kHz sonar and a 5 kHz array share a 10 m cell (see also STATUS open question 13). Silent | derive from the surface's correlation length (wind) or state why 10 m | M | changes realisations | `timespread.py` |
| M-7 | `sim/bathymetry.py:21,86,123` | gradient step `0.25` m | B | relief wider than half a metre | ripples of 0.1 to 0.5 m (the SAS scenes) have their slope smoothed. Silent | step from the description's smallest scale, capped by the wavelength | S | changes for fine relief | `bathymetry.py` |
| M-8 | `sim/simulator.py:553,600,609,638,649` | epoch `1_700_000_000_000_000_000` ns, `turn_time 30` s, `correlation_pings 8`, INS at `10` Hz, compression `1` | A | one survey | all are keys with these defaults; the INS rate matters for `attitude_time: arrival` | list them in `CONFIGURATION.md`; no code change | S | none | docs |
| M-9 | `sim/sources.py:80-81` | retarded-time knots every `4e-3` s, oversampling `2.0` | B | a source whose range rate changes little in 4 ms | fast manoeuvres at high frequency. Degraded, small | knot spacing from the source's acceleration and the top frequency | S | none | `sim/sources.py` |
| M-10 | `sim/surface.py:104`; `sim/scene.py:364`; `acoustics.py:207` | water density `1025.0`, gravity `9.81` three times | A | sea water; `acoustics.water_density()` exists and is not used here; the sediment table assumes 1000 (STATUS question 10) | fresh water: bubble and sediment models use different water. Small, silent | one `GRAVITY`; density from `acoustics.water_density(environment)` | S | changes at the 2 % level | `surface.py`, `scene.py`, `acoustics.py` |
| M-11 | `sim/reverb.py:230-232,526`; `sim/noise.py:52,178` | at least `72` azimuth sectors, `9` elevation bins, `96` noise bands | B | resolution choices | derived from the array above the floor; fine | none | S | none | none |

### Sources

| ID | Where | Literal | Cl. | Assumes | Breaks, how | Fix | Eff. | Risk | Touches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SR-1 | `sources/flank_stream.py:10-27,50-55,68-69,89-90` | block format `GSFB`, 128-byte header, 3-byte samples | E | the boards stream this over UDP | the plan documents `.wav` recording, not streaming; the format is ours. Another word length cannot be carried (`BlockError`; `buildspec.py:393` warns) | a bytes-per-sample field (the header has reserved bytes from offset 44); flag `source` as unconfirmed in the deploy files | M | none | `flank_stream.py`, `libs/live/src/udp_source.cpp` (other owner), `configs/deploy/*` |
| SR-2 | `sources/flank_stream.py:207`; `sources/hsb.py:526`; `sources/nvus_scanner.py:95`; `tools/tonal_range_study.py:40`; `tools/live_benchmark.py:24` | default sonar `"flank_32"`, `"l12_5d"`, `"popoto_long_range"`, `popoto_vehicle.yaml` | A | the caller meant that sonar | a caller that omits it gets another array's template: channel-count error at best, wrong geometry at worst | required argument | S | none | the files listed |
| SR-3 | `sources/flank_stream.py:282`; `:213,234` | emulator block of `128` samples; receive buffer `8 << 20`; `recv(65535)` | C | 32 channels | over about 170 channels a block passes the datagram limit: send error. `tools/build_from_spec.py:388` already sizes it from the channel count | size the block in the emulator the same way | S | none | `flank_stream.py` |
| SR-4 | `sources/hsb.py:41` | `FRAME_SAMPLES = 400` ("6.25 ms of the 64 kSPS stream ... 32 channels") | A | `flank_32` | 96 channels at 102.4 kHz: a frame of 115 kB, three times the burst the comment measured losing frames. Degraded: lost frames | frame samples from a byte budget, the channel count and the rate | S | none for `flank_32` | `hsb.py`, `tools/sonar_live_hsb.py` |
| SR-5 | `sources/wav.py:46-48` | 24-bit PCM reported as `bits = 32` | D | the word length does not matter | the plan's documented path (`.wav` from the board) writes `adc_bits: 32` unless the sidecar corrects it. Wrong metadata, silent | read the bits per sample from the file's header | S | none | `wav.py` |
| SR-6 | `sources/wav.py:59-76,196-226` | the sidecar describes the array again: `elements`, `boresight`, `beamwidth`, `transmit`, `boards` | A | one flat line of elements | a real recording needs a hand-written copy of the sonar file; stacked rows cannot use `{count, pitch}`; the copy can disagree with `configs/sonars` | accept `sonar: <name>` with overrides and build the geometry through `SonarModel` | M | none | `wav.py`, `docs/RECORD_FORMAT.md` |
| SR-7 | `sources/nvus.py:69`; `sources/nvus_scanner.py:70,96` | `sound_speed = 1480.0` | A | the pool | the two sonar files carry `nominal_sound_speed: 1480.0`; the adapter does not read it | take it from the sonar model | S | none | `nvus.py`, `nvus_scanner.py` |
| SR-8 | `sources/nvus.py:27-31` | 64 channels, 48-byte records, 12 bits | B | one capture board | named for that board, limits checked with errors. Listed for completeness | none | S | none | none |

### Build spec and generator

| ID | Where | Literal | Cl. | Assumes | Breaks, how | Fix | Eff. | Risk | Touches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-1 | `buildspec.py:678,748` | `"mode": "flank"`, `"type": "flank_udp"`, `"host": "0.0.0.0"` | A | every build is a flank array on UDP | a forward-looking build cannot be generated | spec keys with these as the stated default | S | none | `buildspec.py` |
| B-2 | `buildspec.py:173-178`; `configs/deploy/popoto_vehicle.yaml`, `popoto_elevation_rehearsal.yaml` (typed by hand); the chain's own defaults | `range_cell_m 2.0`, guard `100` m, training `300` m, `min_snr_db 9`, tracker `15` m and `1` deg, `max_excess_range 800`, bearings `-60..60` by `0.5` | B | a 4 to 6 kHz array, 0.1 to 1 s pulses, kilometres of range | another band or pulse gets the same windows. Degraded detection, silent | derive in the generator: cell from `c / 2B`, guard from the compressed pulse plus the multipath spread at the water depth, training as a multiple, tracker sigma from cell and beamwidth; `operation.chain` still overrides; a test holds the six builds to today's values | M | default | `buildspec.py`, regenerated `configs/deploy/*` |
| B-3 | `buildspec.py:179` | reference burst `38-48 kHz`, 4 ms, 2 mV, lead 5 ms, tail 10 ms | A | 102.4 kHz sampling and a band below 32 kHz | another rate or band: `check()` raises (`:419-422`), so an error, not a wrong answer | default the burst to the top fifth of the Nyquist band, as the schema's own text says | S | default | `buildspec.py` |
| B-4 | `buildspec.py:550,686` | `clock_offset_max_s 3.0e-4`, `clock_tolerance_ppm 25.0`, `jitter_rms_s 2.0e-8` | E | Popoto boards' clocks | none is in the plan; a spec that leaves them out gets them silently | no fallback: absent means unknown, and the clocks file is not generated without them | S | none (all six specs give them) | `buildspec.py` |
| B-5 | `buildspec.py:639,723-736` | elevation sector `-30..30` by `0.25`, roll sigma `1.0` or `0.2`, rate sigma `0.5`, depth sigma `5.0`, gradient `0.017` and sigma `0.05`, elevation error `0.05` | A | this array's geometry and this owner's words | typed in code and again in every build spec; the deploy files flag them (good) | one `configs/builds/defaults.yaml` the specs extend | M | none | `buildspec.py`, `configs/builds/*` |
| B-6 | `buildspec.py:803-870` | self-test trials: 12 C, `1.5` m/s, roll `2` or `3` deg, targets `45` m apart, windows `-150 / +250` m, bearings `-15, 5, 25`, `4` pings at `4` s | A | ranges of hundreds of metres to kilometres | a short-range build: targets fall outside the window and the self-test fails. Error | scale by the receive window and the range resolution | M | default | `buildspec.py`, `tools/build_from_spec.py` |
| B-7 | `buildspec.py:1148-1171`; `tools/reference_plan.py:58,63,71` | performance: wind `5`, ranges `500..12000`, targets `+25` and `+5` dB, threshold `13` dB, 12 C and 50 m | A | the 12 km requirement | another requirement is a code change | a `requirements:` block in the build spec | M | default | `buildspec.py`, `tools/reference_plan.py`, `tools/parts_list.py` |
| B-8 | `buildspec.py:498-503` | `64` (the sparse Bayesian solver's channel limit) | C | the limit in `libs/live` stays 64 | the warning and the limit can part company | read the constant from the bindings | S | none | `buildspec.py`, `bind_live.cpp` (other owner) |
| B-9 | `buildspec.py:755-758,781-783` | `/data/sonar/...`, taps of `2` pings; passive frames of `0.04 fs` samples, `12` averages | A | one vehicle computer | paths typed in code and in each spec | part of `defaults.yaml` of B-5 | S | none | `buildspec.py` |
| B-10 | `buildspec.py:393` | `24` bits | B | the stream's word length | another length warns and then cannot be streamed (SR-1) | follows SR-1 | S | none | `buildspec.py` |

### Tools

| ID | Where | Literal | Cl. | Assumes | Breaks, how | Fix | Eff. | Risk | Touches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-1 | `tools/gallery.py:120-125,562-567,579,647-651` | imaging and detection parameters by scenario name: range grids, averaging, bearings, cell, guard `6` m, training `30` m | A | ten named scenarios | a new scenario or sonar is a `KeyError` until the tool is edited | a `processing:` block in the scenario, or a file in `configs/chains/` (there is one); range step from `c / 2B`, bearings from the `fov` block | M | none | `gallery.py`, `configs/chains/*` |
| T-2 | `tools/design_report.py:49-62,152-153,353-355,524-529` | the design sonars, their scenarios, target strengths, and each requirement's frequency, range and field of view | A | six sonars | requirements are in code, not beside the designs they test | `configs/design/requirements.yaml` | M | none (`test_design` checks the document) | `design_report.py`, new config |
| T-3 | `tools/popoto_elevation_study.py:61-66`; `tools/popoto_range_study.py:41,47`; `tools/reference_plan.py:125`; `tools/popoto_elevation_bench.py:47-48`; `tools/popoto_12km_study.py:37-39,47-49` | `C = 1500.0`, `F_LOW, F_HIGH = 4000, 6000`, `N_ROW = 32`, `1500.0 / 5000.0`, sonar names per row count | A | the studies and the configs describe the same array | change the band in the build spec and the studies still use 4 to 6 kHz. Silent divergence | read band, count and speed from the sonar or build spec named on the command line | S each | none today (they agree) | the five tools, their JSON outputs |
| T-4 | `tools/parts_list.py:34-60,85,163,171` | the plan's quotes and prices as Python strings; "standard" is 32 a row and 1 to 3 rows; `+ 50.0` m; `170.8` | A | one plan, one module | another plan is a code edit | quotes in `configs/builds/plan_quotes.yaml` | S | none | `parts_list.py` |
| T-5 | `tools/build_diagrams.py:43-48`; `tools/architecture_diagrams.py:248` | the six stages as a table of file names | A | six builds | a seventh is not drawn | list `configs/builds/*.yaml` | S | none | the two tools |
| T-6 | `tools/benchmark.py:52-110`; `tools/live_benchmark.py:24` | `sas_auv`, `flank_32`, `1500.0`, `2 * 10000.0 / 1500.0` | A | benchmark cases | fixtures of a benchmark; the collection time should come from the record's window | low priority | S | none | `benchmark.py` |
| T-7 | `tools/build_from_spec.py:380-388` | `1500.0`, pose at 10 m, block `128 * 32 // n_ch` | A | a 16 kB datagram budget | the budget is typed as its factors | name the budget once, share it with SR-3 | S | none | `build_from_spec.py`, `flank_stream.py` |
| T-8 | `tools/sonar_live.py:74` | `"build/bin/sonar_live"`, port default `5700` | A | run from the source tree | elsewhere: file not found. Error | an option and the deploy file | S | none | `tools/sonar_live.py` |

### Configs

| ID | Where | Literal | Cl. | Assumes | Breaks, how | Fix | Eff. | Risk | Touches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G-1 | `configs/scenarios/*.yaml` (for example `popoto_12km.yaml`, `fls_120k_transit.yaml`) | water depth up to 3 times, wind up to 3, sound speed 2, the bottom 2, per file | A | the editor keeps them equal | change the site and miss one. Silent | one `environment:` block (`water_depth`, `wind_speed` or `sea_state`, `sediment`, sound speed or profile) that propagation, reverberation, noise and time spreading default from; explicit keys override; a test loads every scenario both ways | L | none while they agree | `simulator.py`, `propagation.py`, the scenarios |
| G-2 | 23 scenarios; 22; 14 | `environment: {sound_speed: 1500.0, temperature: 12.0, salinity: 35.0}`; `ambient_noise: {wind_speed: 5.0, shipping: 0.5}`; `bottom: {speed: 1650.0, density: 1.9, attenuation: 0.8}` | A | one coastal site, copied | a second site is 59 edits | named environments (`configs/environments/<name>.yaml`), or `base:` for scenarios as sonars have | M | none | `config.py`, the scenarios |
| G-3 | `configs/sonars/fls_120k.yaml`, `fls_170k.yaml`, `fls_330k.yaml` | identical but for name, band, pitch, sample rate and two `fov` numbers | A | three hand-kept copies | a correction to the shared front end is made three times | `base: fls_128` | S | none | the three files, a new base |
| G-4 | `configs/sonars/popoto_8.yaml` | `adc_full_scale: 2.61`, `transmit.lever_arm`, `array.layouts.*.pitch` not under `unconfirmed` | E | 37 mVpp is at the connector (plan note 2); a projector position; a pitch | the flag list says less than the notes do | add `receive.adc_full_scale`, `transmit.lever_arm`, `array.layouts` | S | none | `popoto_8.yaml` |
| G-5 | `configs/sonars/popoto_long_range.yaml`; all six `configs/builds/*.yaml` | band `4.0e3-6.0e3` on a board "specified for 10-100 kHz" (plan note 1) | E | the inputs work an octave below their specification | no key carries a flag; `tools/parts_list.py:251` says it in prose only | `boards.band_hz` with a status in the build spec, and `receive.band` in `unconfirmed` | S | none | `buildspec.py`, the builds, regenerated sonars |
| G-6 | `configs/builds/*.yaml`: `boards.confirmed: [model, inputs, gain_db, input_full_scale_vpp]`; `buildspec.py:664` | "confirmed" full scale | E | where 37 mVpp is referred to | it removes the flag from `receive.adc_full_scale` in every generated sonar file | take `input_full_scale_vpp` out of `confirmed`, or add `input_full_scale_reference: connector` as an unconfirmed key | S | none (flags only) | the builds, regenerated sonars |
| G-7 | generated `configs/sonars/popoto_*row.yaml`: `hardware.boards.clock_tolerance_ppm: 25.0` | 25 ppm | E | the boards' oscillators | written into the sonar file with no flag (the `_clocks` overlay is flagged) | flag `hardware.boards.clock_tolerance_ppm` | S | none | `buildspec.py` |
| G-8 | `configs/builds/*.yaml`: `inputs: 8`, wiring sums the reference into a hydrophone input | 8 hydrophones a board | E | a summing circuit the plan does not describe (plan note 4: 7 hydrophones and the reference) | flagged only in the `_clocks` overlay, not in the sonar file the trials use | flag `hardware.sync` in the base generated file | S | none | `buildspec.py` |
| G-9 | the six `configs/builds/*.yaml` and `configs/sonars/popoto_8.yaml` | the board block (about 20 lines: model, inputs, gain, full scale, high-pass, rate, bits, clocks) six times, and its rate, bits, `2.61`, `43.0`, `-208.0`, `100.0` again in the base sonar | A | the copies agree | "only what differs from the base is written": a build whose board differs from `popoto_8` in a key the generator does not compare is silently the base's | a board module, `configs/builds/modules/board_pmm6081.yaml`, as rows have; the base sonar's values checked against it | M | none | `buildspec.py`, the builds, a new module |
| G-10 | `configs/sonars/flank_32.yaml` | `count: 32`, `pitch: 0.094`, `6.0e3-10.0e3`, `source_level_db: 200.0`, `bandpass_hz` | E | a flank array nobody has designed | `unconfirmed` lists only `receive` and `transmit.lever_arm`; ten scenarios and 27 test files use this sonar | add `array` and `transmit` | S | none | `flank_32.yaml` |
| G-11 | `configs/deploy/popoto_vehicle.yaml`, `popoto_elevation_rehearsal.yaml` | no `unconfirmed` list at all | E | hand-written before the generator | the generated deploy files flag `chain.sync`, `nav.static` and the elevation priors; these two flag nothing | generate them, or add the list | S | none | the two files |

### Tests

| ID | Where | Literal | Cl. | Assumes | Breaks, how | Fix | Eff. | Risk | Touches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| X-1 | 27 files under `tests/python` load `flank_32`; 28 name a `popoto_*` config | a shipped sonar file used as a fixture | A | shipped design values never change | correcting an unconfirmed number in `flank_32.yaml` (G-10) breaks unrelated tests | a sonar file under `tests/fixtures/` for tests that only need "an array"; tests of a design keep the design's file | M | none | `tests/python/*`, `tests/fixtures/` |

## 3. Order of work

Batches are sets of files that do not overlap, so they can be given to different people at once, except
where a dependency is named.

| Batch | Findings | Files | Notes |
| --- | --- | --- | --- |
| 1. Flags | G-4, G-10, G-11 | `configs/sonars/popoto_8.yaml`, `flank_32.yaml`, two hand-written deploy files | config only; first, it is an hour |
| 2. Environment in one place | P-1, P-2, P-3, P-9, M-1, M-2, M-3, S-2 (Python side), G-1, G-2 | `sonar.py`, `acoustics.py`, `sim/simulator.py`, `sim/propagation.py`, `sim/bellhop.py`, `sim/surface.py`, `sim/bottom.py`, `sim/noise.py`, `configs/scenarios/*`, new `configs/environments/` | the largest; outputs must not change: a test loads every scenario before and after |
| 3. Levels and noise | P-5, P-6, P-8 | `sonar.py`, `sonar_equation.py`, `sim/simulator.py`, `sim/noise.py`, `tonal.py` | after batch 2 (same files). P-8 changes results: owner's decision first |
| 4. C++ outside the chain | K-1 to K-7, S-1, S-3 to S-7, S-2 (C++ side), R-1, R-2 | `libs/core`, `libs/dsp`, `libs/detect`, `libs/sim`, `libs/recon`, `libs/motion`, `python/bindings` except `bind_live.cpp` | independent of 2 and 3 except `propagation.py` defaults in S-1 and S-3: give those two lines to batch 2 |
| 5. App and operators | A-1 to A-6 | `apps/`, `libs/holoscan_ops/` | independent |
| 6. Build spec | B-1 to B-10, G-5 to G-9 | `buildspec.py`, `tools/build_from_spec.py`, `tools/reference_plan.py`, `tools/parts_list.py`, `configs/builds/**`, regenerated `configs/sonars/popoto_*row*.yaml` and `configs/deploy/*` | B-2 after the chain's owner settles the chain's keys; B-8 and B-10 need the chain's constants exported |
| 7. Sources | SR-1 to SR-8 | `python/gpusonar/sources/*`, `tools/sonar_live_hsb.py` | SR-1 with the chain's owner (`udp_source.cpp`) |
| 8. Imaging, motion, scene | R-3, R-4, R-5, M-4 to M-7, M-9, M-10 | `motion.py`, `recon/*`, `pipeline.py`, `sim/scene.py`, `sim/timespread.py`, `sim/bathymetry.py`, `sim/sources.py` | M-4, M-6, M-7 change results unless today's values stay the default: decide per item |
| 9. Tools | T-1 to T-8 | `tools/*.py` except those in batch 6, `configs/chains/`, new `configs/design/` | after 2 for the tools that read environments |
| 10. Tests | X-1 | `tests/python/*`, `tests/fixtures/` | last; makes batch 1's `flank_32` corrections safe |

Decisions the owner has to make before the work: P-5 (is a source level a peak or an rms figure),
P-8 and P-6 (accept lower predicted SNR for close-pitch rows), M-4, M-6, M-7 (accept changed
simulated records), B-2 (derive the thresholds or keep them typed).

## 4. Conventions to write down once

One page (`docs/CONVENTIONS.md`, linked from `RECORD_FORMAT.md` and `CONFIGURATION.md`) and an assertion
where a module can check it.

| Convention | Where it stands today |
| --- | --- |
| **Frame.** North, east, down; body x ahead, y to starboard, z down; roll, pitch, yaw | stated in `RECORD_FORMAT.md:19` and `CONFIGURATION.md:12`; used throughout the suite. The root `CLAUDE.md` describes another package (`trajectory_classifier`, east-north-up): say so in one line so nobody carries it over |
| **Default boresight** | `[0, 1, 0]` in `record.hpp:110`, `sonar.py:67,215`, `wav.py:206`; `[1, 0, 0]` in `sonar.py:117` (P-4) |
| **Source level: peak or rms** | peak in `sonar.py:255-266` and `record.hpp:91`; unstated in `sonar_equation.py`; rms for tonal sources (`sim/sources.py:18`), peak for harmonic tables (`:21`); rms for every noise level; `buildspec.py:450` notes the 3 dB (P-5) |
| **Receive sensitivity and gain** | dB re 1 V/uPa, a negative number (`CONFIGURATION.md:102`); `adc_full_scale` is volts peak after the gain (`buildspec.py:81`), while the board's figure is volts peak-to-peak before it (`:80`) |
| **Noise** | spectrum level in dB re 1 uPa2/Hz; `noise_rms` in sample units, per real component for complex data (`sim/receiver.py:12`); a baseband record carries twice the real signal's power (`simulator.py:321`) |
| **Angles at a boundary** | grazing, in radians, in `acoustics.py`, `sim/bottom.py`, `sim/surface.py`, `paths.hpp:98-101`; degrees in config keys ending `_deg`. No function takes an incidence angle. Not asserted anywhere |
| **Density** | a ratio to water in `paths.hpp:63` and the sediment table (`rho`); g/cm3 in `bellhop.py:54`; kg/m3 in `acoustics.water_density` and `surface.py:104`; the ratio's water is 1000 kg/m3 by STATUS question 10, the bubbles' is 1025 (M-10) |
| **Attenuation** | dB per wavelength for sediments (`paths.hpp:64`, `bellhop.py:55`); dB/m for water (`acoustics.absorption`); dB/km in `channel_replay.py:617`; dB/kyd inside the Thorp formula |
| **Time** | sample times are seconds after transmit (`record.hpp:57`); `transmit.time_ns` is ns since the Unix epoch, UTC; board time is "this board's minus the common timeline" (`record.hpp:179`); a stream block counts samples since capture start (`flank_stream.py:16`) |
| **Signal sign** | `exp(+i w t)` (`sim/targets.py:84`); the baseband carrier removed is `center_frequency` |
| **Receive window** | two-way: `2 r / c` (`sonar.py:279`); ranges in the chain's reports are half the two-way path, not corrected for propagation (STATUS, "Found along the way") |
| **Range of a CFAR window** | cells in `libs/detect` and `holoscan_ops.py`; metres in the chain options and `tools/gallery.py` (K-5) |
| **Wind** | m/s at 10 m, used as the 19.5 m wind (Pierson-Moskowitz) and the 12.5 m wind (Cox-Munk) without correction (`acoustics.py:206,212`); knots inside Chapman-Harris |
| **Depth** | positive down from the surface at z = 0; `boundary_z`, `bottom_z`, `surface_z` are z in that frame; `plane_z` defaults to 0, which is the surface, not the seabed (R-4) |

## 5. Constants whose source I could not find

Not findings. Each is used as a constant of nature or of a model; the comment or the three documents
(`OPEN_SOURCE.md`, `PHYSICS_NOTES.md`, `DECISIONS.md`; there is no `REFERENCES.md`) should say where it
is from, or that it is ours.

| Where | Constant | What was found |
| --- | --- | --- |
| `tools/popoto_range_study.py:54`; `tools/parts_list.py:171` | `170.8` in `SL = 170.8 + 10 log10(P) + DI` | used in `DESIGN.md`, no source given |
| `acoustics.py:207,221` | Pierson-Moskowitz `0.21 U^2 / g`, `a = 0.0081`, `b = 0.74` | named, no paper cited in the comment |
| `sim/simulator.py:212,253`; scenarios | Lambert `mu_db = -27` | discussed in `PHYSICS_NOTES.md`; no citation at the default |
| `buildspec.py:180`; 14 scenarios | sand `1650 m/s, 1.9, 0.8 dB/wavelength` | `sim/bottom.py:116-119` calls it illustrative; no source |
| `paths.hpp:57,75` | wall `0.8`, bottom `0.5` | none; ours (S-1) |
| `buildspec.py:735` | sound-speed gradient `0.017` per second | `PHYSICS_NOTES.md` has the number; the build specs call it "isothermal water: an assumption" |
| `buildspec.py:550,686` | `25` ppm, `0.3` ms, `20` ns | none; ours (B-4) |
| `sim/noise.py:22-26` | the four-term ambient noise fit | cited to Coates (1990) through two secondary copies; the original was not opened (the file says so) |
| `acoustics.py:152` | sea-state wind speeds | from another package's table that cites TR 9407, which was not read (the file says so) |
| `sim/bellhop.py:100` | Munk profile `epsilon = 0.00737`, axis `1300` m, `1500` m/s | named "Munk"; no citation in the comment |
| `python/bindings/bind_beamform.cpp:83` | adaptive beamformer loading `3e-2`, `32` sub-bands, blocks of `256` | `PHYSICS_NOTES.md` discusses loading; the default's origin is not stated (chain's owner) |
| `libs/dsp/.../conditioning.hpp:34,46,57-58` | `gamma 0.1`, `alpha 0.01`, median `0.2`, percentile `99.99` | cited to the paper's equations 9 to 12; whether the values are the paper's or ours is not said |
| `sim/timespread.py:265` | coherence length `10` m | none; ours (M-6) |
| `sim/bottom.py:115` | `PLACEHOLDER_SAND` roughness and volume numbers | stated to be illustrative, chosen for -30 dB at 20 degrees and 100 kHz |
