# Task: Doppler in the simulator and in the live chain

Issue `sonar-u1fu`. Written 2026-11-26 on branch `a-doppler` (from the integration branch at `f5054b4`) for an
agent with no history of this project. Read `docs/HANDOFF.md` first, then this file. The owner's instruction, that
day: "start a workstream to use doppler in simulation and real".

This package is the scope, a measured baseline and the plan. It changed no default, no scenario file, no part of
the live chain and no record field. The only code is a study tool, `tools/doppler_study.py` (it runs scenarios with
in-memory edits and processes the records in NumPy), which wrote every number in section 3.

## 0. What is not known, wrong or risky (read first)

1. **The gain cannot be measured honestly for surface-limited cases yet.** The simulated reverberation has a
   physical Doppler spectrum for a bottom seen from a moving platform (measured, section 2.1). It has **no Doppler
   spread from the moving sea surface** at the low grazing angles of long range: no Bragg phase speed, no orbital
   motion, no wind drift (`docs/STATUS.md`, "Simulator limits so far"; measured here: an 8 m/s wind leaves the
   surface's line as narrow as the pulse allows). Every gain below where the surface sets the background is an
   upper bound. Package D1 fixes this, opt-in, and needs a source the repository does not have.
2. **Today's chain cannot see a CW echo from a mover at all.** Both engines convert a record to baseband with the
   transmitted band only (`libs/live/src/chain.cpp:146` `dsp::to_baseband(record, {}, ...)`; Python through
   `recon.fls.prepare`, `python/gpusonar/live.py:1391`). For a 1 s CW at 8 kHz that band is 1/T: the record comes
   out at 8.7 Hz of sample rate and an echo at 2 m/s (21 Hz off) is filtered away, 92 dB down (measured with
   `core.to_baseband` on a free-field `flank_32` ping, this package). The option to keep room exists
   (`max_range_rate`, `doppler_allowance_hz`, `libs/dsp/include/sonar/dsp/baseband.hpp:18-28`); the chain does not
   pass it. Package D3 starts there.
3. **A CW at one frequency loses the band's multipath diversity.** In 100 m of water the echo of a 1 s CW at
   8 kHz on an image-source path set carried 18.9 dB less energy than the 6-10 kHz LFM of the same energy (the
   paths' coherent sum at one frequency sits in a fade; free field the two are equal within 0.01 dB). The fade
   depends on frequency, range and depths, so it changes from ping to ping and from target to target. A
   Doppler-resolving waveform for shallow water needs several frequencies (section 4).
4. **What the transmitter can send is unknown.** No projector is chosen (`docs/design/projector_selection.md`;
   `configs/sonars/popoto_long_range.yaml` lists `transmit` as unconfirmed). Whether it can send a CW, a CW and a
   sweep in one ping, or two bands, and at what peak power, are owner questions (section 5). A Hann-tapered CW of
   the same energy as the sweep needs 4.1 dB more peak power (measured: the simulator scales amplitude by the
   source level before the window).
5. **A CW gives up range resolution**: c T / 2 = 750 m for 1 s, against c / 2B = 0.19 m for the 6-10 kHz sweep and
   0.38 m for 4-6 kHz. The chain's tracker, multipath grouping, depth from multipath timing and the range-Doppler
   correction (D-181) all assume the sweep's resolution. A CW path is an addition to the sweep path, not a
   replacement.
6. **Nothing in this document has met real data.** The record carries what Doppler processing needs without a
   schema change (section 2.4), but whether a real log has velocities and clock rates good enough is unchecked.
7. The margins here are one target-only ping against a few background pings, with a clairvoyant background mean
   (section 3.1): a real detector estimates the background from its neighbours and loses a little. The chain's
   CFAR, clustering, tracking and several-ping integration are not in them.

## 1. Goal and done-criteria

Goal: a moving target below the boundary reverberation in level but separated from it in Doppler is detected by
the live chain, in both engines, opt-in; the simulator's reverberation has a Doppler spectrum good enough to
measure that; the link budget and coverage picture can credit it; and a real log can carry what it needs.

Done means:

1. `flank_patrol`'s small vehicle (D-198: below the reverberation by design) is detected and tracked by the live
   chain with the Doppler path on, in both engines, on a scene variant with a Doppler-resolving waveform, with no
   loss of the submarine and the wreck (the stationary wreck must still be detected: by the sweep).
2. With the Doppler path off (the default) every report line, record and figure is bit-identical to before.
3. Detections from the Doppler path carry a measured range rate; the tracker and classifier may use it (opt-in).
4. The simulator's surface reverberation has a Doppler spread from a cited model (opt-in), and the bottom's
   own-motion spectrum is held by a test.
5. The budget and coverage picture have a Doppler credit computed from the simulator's own Doppler model (opt-in),
   held against `tools/doppler_study.py` within a stated tolerance.
6. Tests for each, written first; the engine-parity file `tests/python/test_live_cpp.py` extended; the full GPU
   suite run before merge.

## 2. What exists (checked in the code at `f5054b4`)

### 2.1 The simulator

| Stage | What it does with Doppler | Where | Engines |
| --- | --- | --- | --- |
| Echo synthesis, analytic pulses (CW, LFM, Gaussian) | exact: arrival times of start, middle and end of the pulse per (path, channel) from moving transmitter, scatterer and receiver; the echo is time-scaled (envelope and carrier) with range curvature | `libs/sim/include/sonar/sim/synth_kernel.hpp:77`; `python/gpusonar/sim/waveform.py:40-50` | C++ host and CUDA (same header); NumPy reference |
| Echo synthesis, sampled pulses (LFM, HFM, stepped FM, Barker, m-sequence, code, samples, file) | the same, the envelope interpolated (D-75, D-81) | `python/gpusonar/sim/waveform.py` `synthesize_into`; `libs/sim` `pair_sampled` | C++ host and CUDA; NumPy reference. A CW cannot be a sampled pulse (`f_start` must differ from `f_stop`, `waveform.py:546`) |
| Moving targets, platform | scene items carry `velocity`; the platform has `velocity` and attitude rates; per ping and within a ping (first order) | `python/gpusonar/sim/simulator.py` `run_scenario`; `docs/RECORD_FORMAT.md` "pose" | all |
| Per-ray range rates (rough-surface bounces) | `time_spread.doppler`: each tap sinusoid is a ray with a path-length rate; Gaussian spectrum of rms width 2 f v sin g / c (D-76, D-84; the sin g reading is ours, `docs/references/CONFORMANCE.md` 2.8) | `python/gpusonar/sim/timespread.py:105-118, 760-770`; `propagation.py:63-64, 1035-1093`; `libs/sim/src/paths.cpp:670-751` | pairing C++ host (no CUDA pairing); synthesis C++ host and CUDA. Target echoes only, not reverberation |
| Reverberation, default (`geometry: monostatic`) | each azimuth sector's sequence and pulse shifted by 2 (v . u) f / c of the receiver's horizontal velocity, as if every path were horizontal (D-43); sectors = 2 per beamwidth, at least 72 (`default_azimuth_count`) | `python/gpusonar/sim/reverb.py:470-494` (NumPy), `libs/sim/src/reverb.cpp:221-229, 294-310` (C++), `libs/sim/include/sonar/sim/reverb_kernel.hpp:120-170` and `reverb_batched.hpp:130-149` (CUDA) | all three |
| Reverberation, `geometry: bistatic` | components keyed by (sector, elevation bin, bin of total Doppler 0.5/T wide), exact factor of TR 0702 eqs. (71)/(72) per eigenray pair (D-69, D-80) | `python/gpusonar/sim/reverb_bistatic.py:78-120`; `reverb.py:662-721` | component generation Python; the series C++ host and CUDA |
| Moving scatterers | `scatterer_motion: {velocity, rms_speed, rms_vertical_speed, wind_speed}` (bistatic only): a mean shift and a Gaussian spread realised as Doppler lines; `wind_speed` sets the rms **vertical** speed only (D-72) | `reverb.py:536, 574-604, 689-700`; `reverb_bistatic.py:258-300` | as bistatic |
| Relief | `bathymetry`: per-cell Doppler from the platforms' rays (D-73) | `python/gpusonar/sim/reverb_relief.py:163-262` | as bistatic |
| One-way sources | exact time scaling for moving sources and receiver | `python/gpusonar/sim/sources.py:37, 196, 239` | NumPy streamer, both engines |
| Basin late tail, direct sound, overload | **no Doppler** | `python/gpusonar/sim/basin.py:72` | NumPy |

**Is the simulated reverberation's Doppler spectrum physical? Measured** (`tools/doppler_study.py
reverb-spectrum`; `popoto_long_range`, a 1 s CW at 5 kHz with a Tukey 0.5 window, platform at 10 m depth moving
3 m/s along the array in 100 m of water, Lambert bottom −27 dB, image sources, one ping; spectrum of the beam
output over 2 s from 1.5 to 3.5 km, Hann-weighted; expected shift 2 v f sin(bearing) / c, full extent ±20.0 Hz):

| Bearing from broadside | −60° | −40° | −20° | 0° | +20° | +40° | +60° |
| --- | --- | --- | --- | --- | --- | --- | --- |
| expected, Hz | −17.4 | −12.9 | −6.9 | 0.0 | +6.9 | +12.9 | +17.4 |
| peak, monostatic (default), Hz | −18.4 | −13.3 | −6.8 | +0.9 | +7.6 | +11.3 | +17.6 |
| −10 dB extent, monostatic, Hz | −19.5 to −15.4 | −14.7 to −11.9 | −9.1 to −5.8 | −1.3 to +2.0 | +6.9 to +8.4 | +10.7 to +15.2 | +15.8 to +18.9 |
| peak, bistatic, Hz | | −12.8 | | −1.0 | | +12.9 | |
| static platform, surface, wind 8 m/s (bistatic, `scatterer_motion.wind_speed`), −10 dB extent, Hz | | −1.6 to +1.8 | | −2.5 to +1.9 | | −2.4 to +0.3 | |

So:

- **Bottom, own motion: physical.** Each beam's reverberation sits at the expected own-motion shift within the
  pulse's resolution (1 Hz; peaks within 0.1 to 1.6 Hz on one ping), with a width set by the beam and the pulse;
  the beams near endfire (±70°, ±80°) also hold the other end's sidelobe energy. The default mode treats
  every path as horizontal (the cosine of a 3° path elevation is 0.9986, ours); the bistatic mode is exact.
  Sector quantisation (2 per beamwidth) makes the within-beam spectrum a few lines; this was not checked
  separately (a 1 s pulse does not resolve them at this array's beamwidth).
- **Surface: not physical at long range.** With `wind_speed` the scatterers heave vertically only, and at a
  grazing angle of a few degrees the vertical component of the path is small: the −10 dB extent is within about
  1 Hz of that of a still bottom (static platform, monostatic: about −1.2 to +1.2 Hz), i.e. set by the pulse. A real wind-driven surface backscatters
  with the Bragg waves' phase speed and the orbital and drift motion of longer waves (general physics; no source
  read in this repository gives a level, a spectrum or a validated model: section 7). In `flank_patrol` and the
  Popoto scenes the surface item is Chapman-Harris in the default (monostatic) mode: it carries the platform's
  shift and nothing else.
- `flank_patrol`'s platform does not move, so its whole background is one line at zero Doppler, as narrow as the
  pulse: the most favourable case there is for Doppler (section 3).

### 2.2 The live chain (both engines)

| Stage | Today | Where |
| --- | --- | --- |
| Baseband | the transmitted band only; no Doppler allowance passed (fine for an LFM, fatal for a CW mover: section 0 item 2) | `libs/live/src/chain.cpp:145-150` `prepare`; `python/gpusonar/live.py:1391` → `recon/fls.py:104-118`; the option: `libs/dsp/include/sonar/dsp/baseband.hpp:18-28`, `libs/dsp/src/baseband.cpp:25-36` |
| Pulse compression | one hypothesis, zero Doppler (`doppler_scale` 1) | `libs/dsp/include/sonar/dsp/pulse_compression.hpp:24-29`; C++ `pulse_compression.cpp:102-177` |
| Doppler bank | exists outside the chain: `processing.doppler_bank` compresses once per range rate and keeps **one hypothesis per record** (D-25; "per-range or per-beam selection is not done", `docs/STATUS.md` "Conditioning limits"). Python only; used by `tools/sst_validation.py:940-957`, `tools/gpu_benchmark_all.py:913`, tests (`test_sst_validation.py::test_doppler_bank_finds_a_moving_highlight_through_a_sampled_pulse`, `test_sampled_waveform.py`, `test_batched.py::test_doppler_bank_takes_a_backend`) | `python/gpusonar/processing.py:10-36` |
| Beamforming | conventional (backprojection kernel default, far-field, MVDR), Hann aperture, ±40° at 0.5° | `live.py` `DEFAULTS` (`window`, `bearings_deg`), `flank.py:78-106` |
| Detection | range × bearing map, multiscale CFAR on 2 m cells; no Doppler axis | `python/gpusonar/detect.py:209`; `libs/live` `detector.hpp` |
| Range-Doppler coupling | opt-in `range_doppler: off/ownship/full` corrects the LFM's range for closing speed (factor f0 T / B; D-181) | `live.py:70-76, 1689, 1764`; `chain.cpp:220-231, 701, 799`; `detect.py:94-105`; `classify.py:367` |
| Range rate of a detection | none measured; `Detection` has range, bearing, SNR, cells, depth, extent | `libs/live/include/sonar/live/detections.hpp:20-35`; `detect.py:66` |
| Tracker | constant-velocity Kalman in the horizontal plane, range and bearing measurements only | `detect.py:349-425`; `detections.hpp:67-110` |
| Classifier | speed from a line fitted through the track's positions over pings (`fit_velocity`); `mover` if speed ≥ 0.5 m/s and well above its sigma | `classify.py:221, 255-330`; `libs/live/src/classify.cpp` |
| Passive | Goertzel tonal lines with a frequency track (Doppler of a radiating source) and DEMON | `python/gpusonar/tonal.py:1-15`; `libs/live/include/sonar/live/tonal.hpp:6` |
| Link budget, coverage | reverberation term since `6b5eb79` (`sonar-w2w3`), **no Doppler credit** (its "not in it" lists say so) | `python/gpusonar/reverb_budget.py:28-32`; `libs/live/include/sonar/live/coverage.hpp:38-40`; `docs/PHYSICS_NOTES.md` Sec. 32 |

Tests that already touch Doppler: `tests/python/test_lfm_range_doppler.py` (three tests: the LFM coupling and the
`ownship` correction), `tests/python/test_sst_validation.py::test_pursuit_target_is_below_the_reverberation_but_separated_in_doppler`
(single-channel spectrogram, CW 0.1 s, 20 m/s chase), `::test_reverberation_gives_each_eigenray_pair_its_own_doppler`,
`tests/python/test_sim.py` (Doppler and receiver motion against analytic values).

### 2.3 Waveforms and transmit configuration

- Analytic types in the record: CW (0), LFM (1), Gaussian (2), arbitrary (3), each with `f_start`, `f_stop`,
  `duration`, `window_alpha` (Tukey) (`libs/core/include/sonar/core/record.hpp:49-102`). A CW is windowed too
  (`libs/core/include/sonar/core/waveform.hpp:75-76`).
- Sampled sources (`transmit.waveform.type: sampled`): `lfm`, `hfm`, `stepped_fm`, `barker`, `mseq`, `code`,
  `samples`, `file` (`python/gpusonar/sim/waveform.py:12-27`). There is no pulse train or CW-plus-sweep source;
  one could be given as `samples` today (the band must be stated and differ from a point).
- `source_level_db_rms` scales the pulse's amplitude before the window (measured: a Hann CW at the same key
  carries 0.39 of the energy; free field).
- The Popoto long-range transmit: 4-6 kHz LFM, 1 s, 197 dB, unconfirmed (`configs/sonars/popoto_long_range.yaml`);
  `flank_32`: 6-10 kHz LFM, 1 s, 200 dB, unconfirmed. The design's own Doppler note
  (`docs/design/projector_selection.md` Sec. 1.8): a 1 s, 2 kHz sweep tolerates about ±0.38 m/s before a 1 dB loss
  (c / 2BT, ours), "the vehicle's own 1.5 m/s resolved along the beam already exceeds" it off broadside; the
  remedies named there are a replica bank or an HFM sweep. Nothing in `docs/design/` says whether the projector can
  send a tone, two pulse types in one ping, or two bands; the band needs a mechanical Q of 2.4 or less or
  equalisation (Sec. 1.2), and the 1 s pulse at rated drive is on no candidate's sheet (Sec. 1.4).

### 2.4 What the record and the configs already carry (no schema change needed)

`docs/RECORD_FORMAT.md` (schema 1.0.0, frozen):

- the transmit waveform: type, band, duration, Tukey fraction, and for an arbitrary pulse the replica itself as
  sent (lines 115-122): a CW-plus-sweep train fits as an arbitrary replica;
- `tx_pose` and `rx_pose`: position, attitude, **velocity** (NED) and attitude rates at the transmit (line 149);
- `navigation/ins`: velocity and attitude rate history through the ping; `navigation/dvl`: body velocity and
  altitude (lines 158-163);
- `environment/sound_speed` at the sonar (line 180);
- `boards`: each board's clock offset and **drift in ppm** (line 194).

What Doppler processing could need beyond it and how it would be carried without a schema change: the split of a
train into segments (CW from 0 to T1, sweep from T1 to T2) is configuration (`chain.doppler.segments`), or derived
from the replica; the transmitter's own clock error against the boards is not in the record (a calibration value,
configuration). None of the packages below needs a record change; if a reader finds one unavoidable it is an owner
question (section 5), not a package.

## 3. Baseline: what Doppler processing would buy on existing scenes

### 3.1 Method (ours; `tools/doppler_study.py`)

Per target: a target-only ping (that scene item alone, no reverberation, no noise), background-only pings (no
scene; reverberation and noise as the scene has them; each a new draw) and noise-only pings. Each record goes to
complex baseband with room for ±(v_max + 1) m/s, is beamformed on the target's bearing (Hann aperture, the chain's
window), and is correlated with the replica of each range-rate hypothesis: the transmitted pulse time-scaled by
(c − v)/(c + v), unit energy, on a grid at half of c / (2 f T). **Zero-Doppler margin** = the target's peak power
through the zero hypothesis over the mean background power in that beam and hypothesis within ±0.1 s (sweep) or
±0.5 T (CW) of the target's delay: the chain's view of one ping. **Bank margin** = the same in the hypothesis where
the target peaks, the background normalised in that (beam, hypothesis) cell: a per-cell normalised Doppler filter
bank. The noise-only margin of the same cells is the ceiling that perfect rejection of reverberation reaches.
"Gain" below is bank minus zero-Doppler margin of the same waveform, and "against the sweep" is the CW bank margin
minus the sweep's zero-Doppler margin (what changing the waveform and adding the bank buys over today).

Waveforms: the scene's own (an LFM sweep), and a CW at the band centre with the same duration and **the same
energy**: `cw` (Tukey 0.05, as the sweep) and `cw_hann` (Hann, +4.1 dB peak power). Speed resolution
c / (2 f T): 0.094 m/s at 8 kHz, 0.15 m/s at 5 kHz, 0.76 m/s for `sst_pursuit`'s 0.1 s at 9.9 kHz. Range
resolution: sweep c / 2B = 0.19 m (6-10 kHz) or 0.38 m (4-6 kHz); CW c T / 2 = 750 m (1 s).

The study's zero-Doppler margin of the small vehicle agrees with the figure the reverberation term was built to
explain: 7.0 dB here, "6.6 dB" in `docs/PHYSICS_NOTES.md` Sec. 32 (D-198's 3.8 to 5.1 dB is in the gallery's 4 m
cells, a different measure).

### 3.2 Results

All numbers: `docs/tasks/doppler_baseline.json` (the commands and every field), status **ours** (simulated records,
`tools/doppler_study.py`, seed of each scene, CPU, Bellhop for the three eigenray scenes). "Ceiling" = the same
cells against noise alone. "Penalty" = ceiling − margin: what the reverberation costs. Range rate is relative to the
platform, positive opening, with the platform's share in brackets.

**`flank_patrol`, scene as shipped (platform still; seed 17).** Sweep = 6-10 kHz LFM, 1 s (0.19 m range
resolution); CW = 1 s at 8.0 kHz, same energy (750 m; speed resolution 0.094 m/s).

| Target | Range rate, m/s | Sweep, zero Doppler (today): margin / ceiling, dB | Sweep with bank | CW Hann, bank: margin / ceiling | CW Tukey 0.05, bank | Penalty: sweep today → CW Hann bank | CW Hann bank − today |
| --- | --- | --- | --- | --- | --- | --- | --- |
| submarine (8 highlights, +16 dB in all, crossing 3 m/s) | −0.61 (0) | 26.6 / 46.3 | 28.6 / 48.2 | 63.4 / 62.2 | | 19.7 → 0 | **+36.8** |
| small vehicle (−3 dB, the hard target of D-198) | +2.00 (0) | 7.0 / 31.2 | 12.9 / 37.0 | 39.8 / 38.5 | 18.0 / 37.6 | 24.2 → 0 | **+32.8** (Tukey: +11.0) |
| wreck (+12 dB, on the bed) | 0.00 (0) | 29.8 / 46.2 | 29.8 / 46.2 | −16.6 / 34.6 | | 16.4 → 51.2 | **−46.4** |

**The small vehicle at other radial speeds, `flank_patrol`, CW Hann with the bank** (its velocity replaced by a
speed straight away from the platform):

| Platform | Range rate (= over ground), m/s | 0 | +0.10 | +0.20 | +0.30 | +0.50 | +1.00 | +2.00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| still | margin / ceiling, dB | −14.8 / 43.9 | −7.5 / 47.2 | 5.7 / 48.6 | 22.7 / 50.6 | 39.0 / 45.8 | 39.0 / 38.8 | 39.8 / 38.5 |

| Platform | Over ground, m/s | −2.0 | 0 | +0.25 | +0.50 | +1.0 | +2.0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.5 m/s along the array (ridge in this beam at −0.62 m/s, its −10 dB span −0.80 to −0.42) | range rate | −2.62 | −0.62 | −0.37 | −0.12 | +0.38 | +1.38 |
| | CW Hann bank: margin / ceiling, dB | 36.0 / 36.0 | −35.4 / 20.9 | −11.1 / 24.9 | 29.2 / 37.1 | 50.3 / 48.2 | 36.8 / 38.3 |
| | sweep, zero Doppler: margin / ceiling | 9.0 / 33.1 | | | 10.2 / 34.3 | | 8.8 / 32.8 |
| | CW Hann bank − today's sweep | +27.0 | | | +19.0 | | +28.0 |

**Long range and the SST example** (each scene's own platform motion):

| Scene, target | Range rate, m/s | Sweep today: margin / ceiling, dB | Sweep with bank | CW Hann bank: margin / ceiling | Penalty: sweep → CW | CW − today |
| --- | --- | --- | --- | --- | --- | --- |
| `popoto_far_mover`, ship at 6 km (6-10 kHz, 1 s; CW at 8 kHz; platform 1.5 m/s) | −0.99 (−0.20) | 27.2 / 27.8 | 27.8 / 28.4 | 49.4 / 51.5 | 0.6 → 0 | +22.2, **none of it from rejecting reverberation** |
| `popoto_12km`, ship at 12 km (4-6 kHz, 1 s; CW at 5 kHz, 0.15 m/s; platform 1.5 m/s) | −0.50 (−0.10) | 24.0 / 24.6 | 25.9 / 26.5 | 34.7 / 34.1 | 0.6 → 0 | +10.7, **none of it from rejecting reverberation** |
| `sst_pursuit` (its own CW: 0.1 s at 9.9 kHz, Tukey 0.1; 5 channels; 20 m/s chase; resolution 0.76 m/s) | −17.5 (−20.0) | zero hypothesis −15.6 / −15.3 | bank 12.7 / 32.8 | (the scene is a CW) | 20.1 remains | +28.3 over its own zero hypothesis |


### 3.3 What the numbers say

1. **Where the background is bottom reverberation and the target moves, the Doppler path removes the
   reverberation penalty entirely.** `flank_patrol`'s small vehicle: 24.2 dB below its noise ceiling today, at its
   ceiling with a Hann CW and the bank, 32.8 dB better than today's view (7.0 → 39.8 dB). The submarine gains 36.8 dB.
   Under way at 1.5 m/s the vehicle gains 27 to 28 dB at ±2 m/s over ground and 19 dB at +0.5 m/s.
2. **The gain is zero, or a large loss, inside the reverberation's own Doppler span.** A still platform: the CW is
   worse than today's sweep (7.0 dB, 12.9 with the sweep's bank) below about 0.25 m/s of range rate (−14.8 dB at 0,
   +5.7 at 0.2, +22.7 at 0.3); the reverberation's −10 dB span is ±0.14 m/s (1.5 resolution cells) and beyond it the
   CW's spectral sidelobes. Under way: the ridge
   sits at the ownship's share along the beam (−0.62 m/s at 24°) with a −10 dB span of ±0.19 m/s, and a target at
   0 or +0.25 m/s over ground is buried (−35.4, −11.1 dB). Stationary targets (the wreck: −16.6 dB against +29.8
   today) are lost to a CW: **the sweep must stay**.
3. **The CW's taper decides it.** At 2 m/s a Tukey 0.05 CW leaves 19.6 dB of reverberation penalty (18.0 dB margin)
   where a Hann CW of the same energy leaves none: the reverberation's energy at the target's shift is in the
   transmitted pulse's own spectral sidelobes, which no receive window removes. A Hann CW costs 4.1 dB of peak power
   for the same energy (question 1). A real projector's spectrum sets this: the replica as sent matters (D7).
4. **At long range in these scenes the background is noise, not reverberation**: the far mover's and the 12 km
   ship's sweep margins are 0.6 dB under their ceilings. Doppler has nothing to reject there. The CW's higher
   number (+22.2 and +10.7 dB) is its 750 m cell gathering all the ship's highlights and multipath arrivals, which a
   sweep resolves into separate cells; it is not a Doppler gain, and a sweep chain integrating over the target's
   extent (`cfar.integrate_cells`, D-97) recovers part of it. **For the 12 km requirement as these scenes state it,
   Doppler is not the lever**; it would be if the operating water is reverberation-limited at 12 km (question 4).
5. **Today's sweep already loses to Doppler mismatch**: a bank of sweep replicas adds 5.9 dB on the small vehicle
   (2 m/s), 2.0 on the submarine, 1.9 on the 12 km ship, against a 1 s, 4 kHz sweep's tolerance of about
   ±0.19 m/s (c / 2BT). This is the cheapest gain in the plan (D4), it needs no new waveform, and it does not
   separate reverberation (the sweep's reverberation is within 10 dB of its peak in every hypothesis across ±3 m/s).
   The rate the sweep's bank picks lies on its range-Doppler ridge and is not a measurement (12 km: +1.35 m/s for a
   true −0.50).
6. **`sst_pursuit`**: its own 0.1 s CW with a Tukey 0.1 taper leaves 20.1 dB of surface-reverberation penalty after
   the bank at 2.5 m/s (3.3 resolution cells) from the ridge: consistent with the existing single-channel figure
   (`docs/figures/sst_validation.json`: the spectrogram cell 8.3 dB over the reverberation, the band level
   6.4 dB under). The surface there is modelled in the bistatic mode with only the platform's Doppler.
7. **A CW at one frequency fades**: its noise ceiling for the small vehicle moves from 38.5 to 50.6 dB with the
   target's speed alone (the arrivals' phases at one frequency), and on an image-source path set its echo energy was
   18.9 dB below the sweep's (free field: equal). Plan several frequencies (option E) before trusting one tone.
8. **What range resolution costs**: the CW detection's range is the 750 m cell (the reported ranges here fall 33 to
   84 m from the sweep's); track association, multipath grouping and depth from timing need the sweep's 0.19 m.
   The CW path gives the range rate and the detection; the sweep gives the place.
9. **Not measured**: the surface's own Doppler spread (section 0 item 1: every result here where the surface is a
   large part of the background is an upper bound); a CFAR's estimation loss; several seeds; the chain itself.


## 4. Design options for the flank array

Hardware facts are unknown; every limit below is a question (section 5), kept in config when it arrives. The
families and trade-offs in this section are general sonar knowledge, labelled so; the only source read in this
repository on them is the GSFM dissertation (arXiv:1809.11009) as `docs/design/projector_selection.md` [A2]
records it (Doppler tolerance, HFM "optimally Doppler tolerant", continuous active sonar): via secondary for this
document (not reopened here).

| Option | What it is | Buys | Costs | Needs from the hardware (questions) |
| --- | --- | --- | --- | --- |
| A. Sweep only, Doppler bank on the sweep | today's LFM with a bank of time-scaled replicas | recovers the sweep's mismatch loss for fast movers (+2.0 dB for the submarine, +5.9 dB for the small vehicle at 2 m/s, section 3) and a coarse range rate; nothing against reverberation | the bank's compute | nothing new |
| B. HFM sweep | Doppler-tolerant sweep (`source: hfm` exists) | no mismatch loss at any speed without a bank (general knowledge; [A2] via secondary) | no Doppler separation from reverberation; no range rate | the same band and pulse |
| C. CW and sweep in sequence (one ping: CW of T1, then sweep of T2) | the CW path gives range rate and separation from the reverberation ridge; the sweep keeps range resolution and stationary targets | the Doppler gain of section 3 for movers outside the ridge | the energy is split (or the ping is longer: duty, blind range c (T1 + T2) / 2); the CW's multipath fading (section 0 item 3); a second processing path | can the projector and amplifier send a tone and a sweep back to back; peak power for a tapered CW; duty for 2 s |
| D. Split band (CW or narrow sweep in one sub-band, sweep in the rest, at the same time) | e.g. 4.0-4.2 kHz tone band and 4.4-6 kHz sweep | both paths each ping, no longer ping | two signals at once need linear drive (peak-to-average rises), each path has part of the energy and band; the sweep's resolution falls with its band | linearity of the amplifier, response across the band (Q ≤ 2.4 already asked) |
| E. Several CWs at different frequencies (a comb, sequential or simultaneous) | frequency diversity against the multipath fade of item 3 | Doppler separation with less fading; each tone a speed estimate | energy per tone falls with their number; combining needs care | as C or D |
| F. Doppler-sensitive coded pulses (GSFM, Costas, pseudo-random phase codes, sinusoidal FM) | a "thumbtack" ambiguity: range and Doppler resolution together (general knowledge; [A2] for GSFM, via secondary) | range rate and range from one pulse; separation from reverberation in range and Doppler | a large bank (hypotheses at c/(2fT) over the speeds of interest), ambiguity sidelobes spread reverberation over the plane (a pedestal below the peak by about the time-bandwidth product), more compute; phase codes need wide band at the projector | wide band at linear drive; phase coherence |
| G. Continuous active sonar | long sweep while listening | many looks per cycle, lower peak power ([A2] via secondary) | receiving while transmitting: not possible with the projector on the same face (`projector_selection.md` Sec. 1.8) | only with the projector apart |

Recommendation for the first measurement and the first chain work: **C with a Hann-tapered CW** in simulation,
because it is the simplest to synthesise (an arbitrary replica of a train, or two analytic pings back to back),
gives the measured gain where it exists, keeps the sweep for everything that needs resolution, and maps to the
least hardware: one more waveform in the projector's drive. E is the answer to the fade once C works. F is a
research option.

## 5. Questions for the owner (ask before packages D2, D6 and any default)

1. **The transmitter.** Can the projector and its amplifier send (a) a tone, (b) a tone followed by a sweep in one
   ping, (c) two signals in two sub-bands at once? At what peak power for a Hann-tapered tone of 1 s (4.1 dB over
   the sweep for the same energy)? With what duty if the ping grows to 2 s? Changes: which of options C, D, E is
   planned in D2; the energy split in the scenes.
2. **The slowest target speed of interest**, by class (the build's classes say ship 0-6 m/s, XL AUV 1-4,
   large submerged 2-8, all assumed: `configs/builds/targets/reference_classes.yaml`). Changes: the CW length (the
   speed resolution is c / (2 f T): 0.15 m/s for 1 s at 5 kHz) and whether the Doppler path can help at all for a
   target crossing at 12 km (the 12 km ship's radial speed is about 0.4 m/s).
3. **Must stationary and slow targets stay detectable?** (A wreck, a bottomed vehicle, a hovering vehicle.) The
   Doppler path cannot help them; with a split ping (option C or D) the sweep loses part of its energy. Changes:
   whether the sweep keeps all the energy and the CW is extra (a longer ping), or the energy is split.
4. **The 12 km requirement's target class and aspect**: is a crossing ship (little radial speed) the case that
   sizes the array, or an approaching one? Changes: whether Doppler is a design lever for 12 km at all (section 3).
5. **The navigation unit's velocity accuracy** along the beam and its latency (STATUS question 6 asks for the
   datasheet): the reverberation ridge's place per beam is 2 v f sin(bearing) / c, and a target's speed over ground
   is its measured range rate less the ownship's part. Changes: whether the chain centres its ridge exclusion from
   navigation or from the data.
6. **Is an optional `range_rate_m_s` field on detections (only when the Doppler path is on) acceptable** in the
   report line (`docs/DEPLOYMENT.md`, "The report line", a contract)? Changes: D3's output.
7. **Real data**: is there, or will there be, a recording with a CW transmit (any sonar) over a known bottom with
   the platform's velocity logged? Changes: whether D1's surface model can be checked against anything.

## 6. Plan (packages)

Each package is independently mergeable, opt-in, with tests first. Order and dependencies at the end.

Conventions for every package: work in your own worktree; tests first (a test that fails before the change and
passes after); new behaviour opt-in with the default bit-identical (prove it: the existing test files of what you
touch pass unchanged, and a test compares the default output with values from the tree before you); where the live
chain changes, the same change in `python/gpusonar/live.py` / `detect.py` / `classify.py` and `libs/live` in the same
package, held by `tests/python/test_live_cpp.py`; an unknown key refused with the nearest known one
(`tests/python/test_dict_keys.py`); one `docs/DECISIONS.md` entry per package, leading with what got worse.

### D1. Surface reverberation with a Doppler spread at low grazing (simulator, opt-in)

- **Goal.** A surface `reverberation:` item can carry the horizontal motion of the scatterers that dominates at low
  grazing: a mean shift (wind drift, the advancing Bragg waves) and a spread (orbital motion of longer waves), so
  that a surface-limited Doppler gain can be measured. Today `scatterer_motion` has `velocity` (a mean, by hand) and
  `rms_speed` (isotropic) besides the vertical `wind_speed` term (D-72); none is tied to wind or frequency.
- **First: a source.** Find and read an open source for the Doppler spectrum of sea-surface backscatter at low
  grazing against wind and frequency (generic web searches; arXiv and open journals only; TR 9407 is never fetched).
  Record what it gives in `docs/references/` with its equations, and its status. **If no source is found, stop and
  bring the owner the choice**: an `ours` model (Bragg phase speed from the deep-water dispersion relation, two
  advancing and receding Bragg lines weighted by the wind's direction, a Gaussian orbital spread from
  `acoustics.rms_surface_vertical_speed`, the horizontal orbital rms being equal to the vertical for deep-water
  waves, ours) marked `ours`, or none.
- **Files.** `python/gpusonar/sim/reverb.py` (`scatterer_motion` keys, e.g. `bragg: true`, `drift_m_s`,
  `orbital_rms_m_s`), `reverb_bistatic.py` (lines from the model; the existing `doppler_lines`), the monostatic path
  (a surface item in the default mode would need components: either refuse the new keys without
  `geometry: bistatic`, as `scatterer_motion` does now, or give the monostatic generator a per-sector line set; the
  first is smaller). C++ and CUDA are not touched: the generator already takes per-component shifts.
- **Tests first.** `tests/python/test_reverb_doppler.py` (new): (a) the existing own-motion spectrum held: the
  `reverb-spectrum` measurement of section 2.1 as a test (peak within 2/T of 2 v f sin(b) / c at ±20°, ±40°: 2 Hz for 1 s, the one-ping measurement's worst was 1.6 Hz;
  −10 dB extent inside the beam's span plus 2/T) on both engines; (b) with the new keys, the surface line's mean
  and rms width equal the model's closed form within 10 % (many pings, one beam); (c) without them, bit-identical
  to before (the 16-array comparison of D-69/D-80); (d) unknown keys refused.
- **Default that could move:** none (opt-in). If the owner later wants it on in `flank_patrol` or the Popoto
  scenes, bring before -> after of the scene's records (reverberation level per beam unchanged within 0.2 dB; its
  spectrum widened by the stated amount) and of the gallery.
- **Must not:** change the default generator's output; invent a coefficient a paper reports; fetch TR 9407.
- Size M. No dependency. Can start now.

### D2. A Doppler-resolving waveform in the transmit configuration and in scenes (simulator, opt-in)

- **Done 2026-11-26** (branch `d2-train`): see "D2 done" at the end of this file, which also says what option C
  as planned did NOT deliver (the sweep's reverberation fills the tone's Doppler cells).

- **Goal.** Scenes with option C (a tapered CW followed by the scene's sweep in one ping) and, for comparison, a CW
  alone, so that the chain work (D3) has records to run on and the owner can see the trade.
- **Files.** `python/gpusonar/sim/waveform.py`: a sampled source `train` (a list of segments, each `cw` with
  `frequency`, `duration`, `window` or `lfm`/`hfm` with its band, and `gap`), built on the existing envelope path, so
  synthesis (C++ host and CUDA, `pair_sampled`) and the record's arbitrary replica (type 3, no schema change) carry
  it unchanged. A segment table for processing goes in configuration, not in the record: the chain's
  `doppler.segments` (D3) or a sidecar of the scene. New scene files (new names only, existing ones untouched):
  `flank_patrol_doppler` (`flank_patrol` with a Hann CW of 1 s at 8 kHz then the 6-10 kHz sweep, each at the energy
  question 1 settles; first proposal: the same energy each, i.e. 3 dB less in the sweep than today, which must be
  brought to the owner as a change of the scene's sweep margin), `flank_patrol_doppler_underway` (the same at
  1.5 m/s along the array), `popoto_far_mover_doppler`, `popoto_12km_doppler`.
- **Tests first.** `tests/python/test_sampled_waveform.py` additions: the train's envelope equals the analytic CW
  and LFM pieces (to 1e-6); a moving point target's echo of the train equals the sum of the echoes of its pieces
  delayed (NumPy and C++ engines; the CUDA comparison in the GPU suite); `f_start`/`f_stop` of the record span the
  train's band. `tests/python/test_scenarios.py` (or the file that loads every scene): the new scenes load and run
  one quick ping.
- **Default that could move:** none (new scenes, new source). The new scenes' figures go in `docs/SCENARIOS.md`.
- **Must not:** change `flank_patrol` or any existing scene; add a waveform field to the record.
- Size M. No dependency (D3 can start with the analytic CW type meanwhile). Can start now.

### D3. The Doppler path in the live chain: baseband with room, a Doppler filter bank per beam, per-cell normalisation, a measured range rate (both engines, opt-in)

- **Goal.** With `chain.doppler: {mode: bank, ...}` the chain detects in a (beam, range, range-rate) volume and each
  detection carries its range rate; off (the default) nothing changes.
- **Keys (proposal).** `doppler.mode: off | bank`; `doppler.rates: [lo, hi]` m/s (default from the sonar: ±(the
  largest class speed + the ownship's speed)); `doppler.step` (default c / (4 f T): half the resolution);
  `doppler.segments` (which part of the replica the bank runs on: the CW piece of a train; default the whole pulse
  if it is a CW); `doppler.window: hann` (the receive weighting of each filter); `doppler.ridge: data | navigation |
  none` (where the reverberation ridge is excluded: from the data's own Doppler profile per beam, from the ownship
  velocity, or not at all); `doppler.ridge_halfwidth_m_s` (default: the beam's own-motion span plus two resolution
  cells). The baseband conversion passes `max_range_rate` = the bank's largest |rate| when the bank is on
  (section 0 item 2: without it a CW mover is filtered away).
- **Processing.** After the beamformer (any of the three), per beam: correlate with the replica time-scaled per
  hypothesis (for a CW this is a short-time Fourier transform of the beam with the pulse as window, hop T/4: cheaper
  than one correlation per hypothesis); power; per (beam, hypothesis) a CFAR over range (the existing multiscale or
  mean statistic along range, cells of c T / 2 for a CW); a detection's `range_rate` is the hypothesis (refined by a
  parabola over the neighbours, as bearings are, D-176); its range is the cell's (coarse: c T / 2 for a CW). The
  sweep path keeps running as today; detections of the two paths are reported together, each marked with its path,
  and a CW detection whose bearing and coarse range match a sweep detection lends it its range rate (opt-in merge
  rule; the tracker gets one detection, not two).
- **Files.** `python/gpusonar/live.py` (`DEFAULTS["doppler"]`, validation, the stage), a new
  `python/gpusonar/doppler.py` (the bank, the CFAR per hypothesis, the ridge); `libs/live/src/doppler.cpp`,
  `libs/live/include/sonar/live/doppler.hpp`, `chain.cpp` (the stage; `prepare` passes the allowance),
  `options.cpp` (keys), `detections.hpp` (`Detection::range_rate`, NaN when not measured); the report line's optional
  `range_rate_m_s` (owner question 6); `docs/CONFIGURATION.md`, `docs/DEPLOYMENT.md`. CUDA: the bank as batched FFTs
  (cuFFT) in `libs/dsp` (a `doppler_bank` next to `pulse_compression`, host and device), used by the C++ chain.
- **Tests first** (`tests/python/test_doppler_chain.py`, new; `tests/cpp/test_doppler.cpp`, new):
  (a) a free-field CW ping with one point target at 1 m/s: today's chain does not detect it (the baseband filter) and
  the bank does, with `range_rate` within half a step; (b) a Lambert bottom, static platform, a weak mover 30 dB below
  the reverberation in a sweep cell and 1 m/s away: detected by the bank, not by the sweep; a stationary one at the
  same level: detected by neither (the bank must not invent Doppler); (c) a platform at 1.5 m/s: no detection on the
  ridge of any beam over 20 pings of reverberation alone with `ridge: data` (false alarms at the configured rate);
  (d) default off: every report line of the shipped chain configs bit-identical on the test pings
  (`test_live_cpp.py` cases unchanged); (e) engine parity of (a) to (c) within the chain's existing tolerances
  (range rate within 0.01 m/s); (f) host against CUDA bank within single-precision rounding (GPU suite).
- **Acceptance on scenes** (with D2's `flank_patrol_doppler`, seeds 17 to 26): the small vehicle detected on at
  least 5 of 6 pings on 8 of 10 seeds by both engines, range rate within 0.1 m/s of its true 2.0 m/s; the submarine
  as well; the wreck still detected by the sweep path; false detections per ping not above today's on the same seeds
  plus one. Section 3 says the margin is there (39.8 dB against the bank's background for the Hann CW at a static
  platform; 36.8 dB under way at 1.5 m/s), so a failure points at the chain, not the physics.
- **Default that could move:** none. **Must not:** change `prepare`'s default band (only with the bank on); change
  the sweep path's numbers; add a record field.
- Size L. Depends on nothing for (a) to (c) (they use the analytic CW type); the scene acceptance needs D2.

### D4. The sweep's Doppler bank (both engines, opt-in): the cheapest gain

- **Goal.** Section 3 measured that today's zero-Doppler sweep loses 5.9 dB on the small vehicle at 2 m/s and
  2.0 dB on the submarine at 0.6 m/s to mismatch alone (a 4 kHz, 1 s sweep tolerates about ±0.19 m/s): a bank of
  time-scaled sweep replicas per beam (or per detection's beam and range only) recovers it, and gives a coarse
  range rate. It does not separate reverberation (the sweep is Doppler-tolerant in the range-Doppler sense: its
  reverberation is within 10 dB of its peak in every hypothesis across ±3 m/s, section 3, where the CW's falls by more than 10 dB within ±0.14 m/s).
- **Files.** As D3 (the same stage with `doppler.mode: sweep_bank`), or a narrower first step: the existing
  `processing.doppler_bank` made per beam and range block, in both engines.
- **Tests first.** A point target at 2 m/s on a 1 s, 4 kHz sweep: the bank's peak 5 to 6 dB above the zero
  hypothesis (the closed form of the mismatch, ours, derived in the test); engine parity; default bit-identical.
- **Default that could move:** if the owner wants it on by default, every chain result moves (bring before -> after
  on the gallery's ten seeds, worst first). Opt-in until then.
- Size M. Shares D3's stage: do it with or after D3, not in parallel on the same files.

### D5. Tracker and classifier use the measured range rate (both engines, opt-in)

- **Goal.** The tracker's measurement model takes range rate where a detection has one (an extended Kalman update
  with the radial component of the track's velocity relative to the ownship), and the classifier can call a track a
  mover from range rate over ground in one or two pings instead of a fitted line over many.
- **Files.** `python/gpusonar/detect.py` (`Tracker`), `classify.py` (a rule `mover_from_range_rate` with a
  significance test), `libs/live` `detections.hpp`, `tracker.cpp`, `classify.cpp`.
- **Tests first.** A crossing target and an approaching one with exact range rates: the track's velocity converges
  in fewer pings than without (stated number); a static clutter point with a range rate equal to the ownship's part:
  never a mover; NaN range rates change nothing (bit-identical to today); engine parity.
- **Needs from navigation:** the ownship's velocity along the beam to better than half the speed resolution
  (0.075 m/s at 5 kHz and 1 s, ours: c / (4 f T)) for the over-ground speed; question 5.
- Size M. Depends on D3 (detections with range rates).

### D6. The link budget's and coverage picture's Doppler credit (opt-in), built on `sonar-w2w3`'s term

- **Goal.** For a target class with a radial speed, the budget's reverberation term is the reverberation in the
  target's Doppler cell, not all of it: the same rings and beam integral (`reverb_budget.level_db`; C++
  `coverage_reverberation_db`), but each ring's power spread over Doppler by the ownship's shift per azimuth
  (2 v f cos(phi) / c, the beam pattern weighting each azimuth) and convolved with the pulse's spectrum (the CW's
  window sets the leakage: section 3 measured 22 dB between a Tukey 0.05 and a Hann CW at 2 m/s), then read at the
  target's shift. Surface terms take D1's model when it exists, else their line at the ownship shift with a warning
  that the surface's own spread is missing.
- **Files.** Whatever `sonar-w2w3` lands in `python/gpusonar/reverb_budget.py`, `coverage.py`,
  `libs/live/src/coverage.cpp` (owned by another agent today: start only after it merges; do not edit them before).
- **Tests first.** Against `tools/doppler_study.py margin` on `flank_patrol` (static: the small vehicle's bank
  background within 2 dB of the budget's Doppler-cell term at 0.3, 0.5, 1 and 2 m/s) and on the under-way variant;
  zero speed gives the existing term exactly; Python and C++ within 0.3 dB.
- **Default that could move:** none if opt-in (a `doppler` field of a class or a study option). If the owner wants
  coverage pictures to credit Doppler by default, every coverage figure moves: bring before -> after.
- Size M. Depends on `sonar-w2w3` merged; better after D1 for surface-limited cases.

### D7. Real data and navigation needs (documentation and a check tool; no schema change)

- **Goal.** Say what a real log must hold for the Doppler path and check a recording for it.
- **What a log must carry (all fields exist):** the transmitted pulse as sent (an arbitrary replica from a reference
  hydrophone where possible: a projector's real spectrum sets the CW's leakage, which section 3 shows is decisive);
  `tx_pose`/`rx_pose` velocity and `navigation/ins` velocity through the ping; `navigation/dvl`; each board's clock
  drift (`boards`, ppm); the sound speed at the sonar.
- **Accuracies (ours, from the resolution c / (2 f T)):** a clock error of e ppm shifts every echo by e f, the same as
  a range rate of e c / 2: 100 ppm is 0.075 m/s, half a cell at 5 kHz and 1 s, so a clock known to 10 ppm is
  plenty and the boards' configured or estimated rate (D-185) covers it; the transmitter's own clock against the
  boards' is not in the record and must be configuration (unconfirmed). Ownship velocity along the beam: see D5.
- **Files.** `docs/tasks/REAL_DATA.md` (a Doppler subsection: owner-approved edit), a tool
  `tools/doppler_readiness.py` reading a container and reporting per ping which of the above are present and
  finite, with a test on simulated records with fields removed.
- **Must not:** add a record field; ask for real data beyond what REAL_DATA.md's questions already ask without the
  owner.
- Size S. No dependency. Can start now.

### D8. GPU cost and kernels

- The bank on a CW is a short-time Fourier transform per beam: per ping B beams x H hops x an FFT of T fs points.
  For the Popoto 12 km window (1.4 km, 1.9 s at the scene's 8 kHz baseband), 161 beams, hop T/4: 161 x 8 x an FFT
  of 8192 points, about 7e8 flops a ping (5 N log2 N per FFT): milliseconds on an L4, about a second on one host
  core; after decimating a CW's beams to a few hundred hertz (the band it needs: section 0 item 2) it is tens of
  times less (estimates, ours; measure them). A sweep bank (D4)
  is a correlation per hypothesis: H hypotheses x the existing compression's cost (55 for ±10 m/s at 1 s and
  2 kHz, `projector_selection.md` Sec. 1.8): too costly per beam on the whole map at 32 to 96 channels; do it before
  beamforming per channel block (as `doppler_bank` does) or only around detections.
- Kernels: batched cuFFT for the STFT (`libs/dsp/src/dsp_cuda.cu` already holds the compression's plans, D-42); a
  per-(beam, hypothesis) CFAR reuses the detector's kernels with one more axis. Measured in the GPU job's benchmark
  mode (`GPUSONAR_JOB=benchmark`, a new case) before any default.

### D9. Validation scenes and acceptance numbers (with D2 and D3)

- `flank_patrol_doppler` (static) and `flank_patrol_doppler_underway` (1.5 m/s): the numbers of D3's acceptance;
  the wreck detected by the sweep on every ping where it is today; the small vehicle's measured range rate.
- `popoto_far_mover_doppler`, `popoto_12km_doppler`: the ship's margin by path, with the expectation set by section 3
  (the 12 km ship's radial speed is small: the Doppler path may not help; report it either way).
- `sst_pursuit`: the bank on its CW reproduces the existing single-channel result's direction (the echo separated
  from the ridge; `docs/figures/sst_validation.json`: 8.3 dB in the spectrogram cell where the band level is 6.4 dB
  below the reverberation).
- The ten-seed evaluation tool (`tools/classify_rules_eval.py`, the gallery) gains a `--doppler` switch.

### Order, dependencies and what can run in parallel

| Package | Depends on | Touches | Can start |
| --- | --- | --- | --- |
| D1 surface Doppler | a source (or the owner's choice of `ours`) | `sim/reverb.py`, `sim/reverb_bistatic.py`, new test | now |
| D2 waveform and scenes | owner question 1 for the energy split (a proposal can be built before) | `sim/waveform.py`, new scene files | now |
| D3 chain Doppler bank | none for its unit tests; D2 for the scene acceptance | `live.py`, new `doppler.py`, `libs/live` (chain, options, detections, new doppler), `libs/dsp` | now |
| D4 sweep bank | D3's stage (same files) | as D3 | after D3, or inside it |
| D5 tracker and classifier | D3 | `detect.py` `Tracker`, `classify.py`, `libs/live` tracker and classify | after D3 |
| D6 budget credit | `sonar-w2w3` merged; D1 for surfaces | `reverb_budget.py`, `coverage.py`, `coverage.cpp` | after w2w3 |
| D7 real-data needs | none | `tools/doppler_readiness.py`, a test, REAL_DATA.md (approved) | now |
| D8 GPU | D3 | `libs/dsp` CUDA | with or after D3 |
| D9 validation | D2, D3 | new scenes, eval tools | last |

D1, D2, D3 and D7 touch disjoint files and can run in parallel. D2 + D3 deliver the first measured end-to-end gain
on `flank_patrol` (the small vehicle detected through the chain); D1 makes the gain honest where the surface sets the
background.

## 7. Not verified, and sources wanted

- Surface reverberation Doppler at low grazing (Bragg phase speed, orbital motion, drift; level and shape of the
  spectrum against wind and frequency): no source in `docs/references/` or `docs/REFERENCES.md` gives one (the
  three surface papers read give none: `docs/references/surface_papers_mcdaniel_dahl.md` 3.7). Wanted for D1.
- The families of section 4 are general knowledge; only [A2] is recorded as read, by another agent.
- The study tool's margins are one ping of one seed per target; the background is a few pings. A ten-seed run was
  not made.
- Nothing was run on a GPU.

## 8. What was run for this document

- `tools/doppler_study.py`, the commands in `docs/tasks/doppler_baseline.json`, one at a time on the shared 4-core
  CPU with `OMP_NUM_THREADS=1`, 2026-11-26, Bellhop from `tools/build_oracles.sh`. The worktree used the main
  checkout's built module (2026-11-25) through a temporary symlink, removed afterwards; no scene here needs the
  image-plane or newer sediment bindings, and the C++ engine (the default) ran every case.
- Times: a background ping of `flank_patrol` 13 to 20 s; a target-only ping with the bearing search and the bank 50
  to 80 s (the submarine's 8 highlights 270 to 470 s); `popoto_12km` 32 to 50 s of background, 120 to 200 s a target;
  `sst_pursuit` 31 s a background ping. The whole batch about 50 min.
- Also checked by hand (reported in section 0, not in the JSON): `core.to_baseband` on a free-field 1 s CW ping of
  `flank_32` keeps 8.7 Hz of sample rate by default and 223 Hz with `max_range_rate=5`, and the 2 m/s echo's peak
  moves from 3.7e-11 to 1.5e-6; echo energies of a free-field and an image-source ping for the sweep, a Tukey CW and
  a Hann CW (equal energy free field for sweep and Tukey CW; Hann at the same `source_level_db_rms` 0.39 of it;
  image sources: the Tukey CW 18.9 dB under the sweep).

## Follow-ups (for the orchestrator to file)

- D1 to D9 of section 6, one issue each; D1, D2, D3 and D7 can start in parallel.
- The live chain's baseband conversion has no Doppler allowance (section 0 item 2): harmless with the sweep, fatal
  for any CW or narrowband transmit. Belongs to D3; worth a warning now if a chain config ever carries a CW
  waveform (both engines).
- `docs/STATUS.md` "Conditioning limits": add that `to_baseband`'s default band drops Doppler-shifted narrowband
  echoes; "Simulator limits": the measured statement that surface reverberation has no Doppler spread at low grazing
  even with `scatterer_motion.wind_speed`.
- `source_level_db_rms` scales the amplitude before the window, so a tapered pulse's true RMS level is below the
  key's (0.39 of the energy for Hann): a naming question for `docs/CONFIGURATION.md` and `docs/RECORD_FORMAT.md`'s
  wording (no schema change).
- `flank_patrol`'s header says "reverberation has no Doppler spread": true of the scene (a still platform), but the
  simulator does have own-motion spread; no edit made (scenario files untouched).

## D2 done (2026-11-26, branch `d2-train`): what was built and what it found

Appended; nothing above was rewritten. Status of every number here: **ours** (simulated records, CPU, Bellhop,
`tools/doppler_study.py`, one seed per scene; JSON in the run logs named below, not committed).

**What got worse or did not work (first).**

1. **Option C as planned (a tone at the band centre, then the sweep, in one ping) does not give the tone its Doppler
   gain where the bottom reverberates.** The sweep's reverberation, from ranges c T / 2 nearer, arrives with the
   tone's echo, is broadband (6-10 kHz) and fills every Doppler cell of the tone. On `flank_patrol_doppler` the
   small vehicle's tone with the bank is at **10.9 dB** over its background (ceiling 36.3 dB), where the tone alone
   of the baseline reached its ceiling (39.8 / 38.5 dB, section 3.2); under way **7.8 / 36.8 dB** (baseline 36.8 /
   38.3). It is still above the train's sweep at zero Doppler (6.4 and 7.3 dB), but by 4.5 and 0.5 dB, not 33.
   Controls on the static scene, small vehicle, two background pings: the sweep muted (`--mute 2`, the tone keeps
   its half of the energy): **33.9 / 36.3 dB** (the tone back at its ceiling); the order reversed (sweep, then
   tone): tone 17.7 / 34.8 dB and the sweep's zero-Doppler margin falls to 1.4 dB (the tone's reverberation now
   lies under the sweep); **the tone moved outside the sweep's band (10.5 kHz, above 6-10 kHz): 37.7 / 34.4 dB**,
   at its ceiling. So a tone and a sweep in one ping work only if the tone is outside the sweep's band (option D's
   band split, sequential) or the two are in separate pings. This is a question for the owner (section 5 question
   1: can the projector send a tone outside the sweep's band?), and changes what D3 can show on the shipped scenes.
2. **Option E's comb helps the fade only a little** (item 3 below), and inside the sweep's band it meets the same
   reverberation: a 4-tone comb (6.5, 7.5, 8.5, 9.5 kHz, Schroeder phases, the tone's half of the energy), each
   tone's bank summed incoherently, gave the small vehicle **3.9 / 30.9 dB** on `flank_patrol_doppler` (single tone
   10.9 / 36.3); with the sweep muted it reaches its ceiling, 31.1 / 30.9 dB, still 5.4 dB under the single tone's
   ceiling at this geometry (2.2 km, 40 m), where the fade table below has the comb 5.3 dB under the tone.
3. The tone's margins in the margin study scatter by about +-3 dB between runs: its background is a mean over a
   1 s window of a filter about 1 Hz wide, about two independent values a ping, and the study takes two pings (the
   baseline took six for the tone's reverberation, two for noise). The tests' tolerances say so.

**The waveform** (`sim/waveform.py` `build_train`; keys in `docs/CONFIGURATION.md`): `transmit.waveform: {type:
sampled, source: train, segments: [...]}` with `cw` (one `frequency` or several `frequencies` at once), `lfm`,
`hfm` and `gap` segments, each with its Tukey `window_alpha`, and opt-in `energy_share` or `source_energy_db` per
segment. Built as one sampled envelope, so the C++ host and CUDA kernel (`pair_sampled`) and the NumPy reference
synthesise it unchanged; Doppler is the exact time scaling of the whole train; the record holds it as its replica
(type Arbitrary; no field added). **Why a tone could not be a sampled pulse**: the envelope's sample rate is
`oversample` x the stated band and the gain knots span that band, so `f_start == f_stop` gave a zero rate
(`build_waveform` refuses it, as before for `source: lfm`); a train's tone takes f +- `tone_halfwidth_hz` (16 / T)
as its band, or the sweep's band when there is one. No C++ was changed.

**Energy accounting** (`sonar.py`): `source_level_db_rms` keeps its meaning (the peak of the envelope before the
taper); the energy at 1 m is SL + 10 log10(integral |b|^2 du), measured on a free-field echo within 1e-3 dB.
Opt-in `source_energy_db` states the energy and SL is derived. The four scenes keep their base scene's energy and
give each segment half; the Hann tone then sets the peak: +1.1 dB (flank, 201.1 dB rms) and +1.0 dB (Popoto).

**Measured** (tests in `tests/python/test_pulse_train.py`, `tests/python/test_doppler_scenes.py`):

- one-segment train = today's sampled sweep, bit for bit (samples and record, both engines); one-tone train against
  the analytic CW: 1.8e-6 relative rms (interpolation);
- the train's envelope against its closed-form pieces: 1.4e-12 on its grid, 4.8e-6 of the peak between samples
  (the default interpolator's own floor, 2.9e-6 for a single tone too);
- engines: C++ against NumPy 6.0e-8 rms (worst sample 1.5e-7 of the peak) on a train with beams, absorption and
  motion; on one thinned ping of each new scene 1.4e-8 to 1.1e-6 rms; CUDA: a gated test (not run here: no GPU);
- Doppler: a target closing at 6 m/s: the echo is the train time-scaled by (c + v) / (c - v) to 5.7e-8; the tone's
  peak at 8064.255 Hz against 8064.257 expected; the sweep segment's bank peaks at -6.0 m/s; the echo is the sum of
  the pieces' echoes to 1.6e-16;
- both live chains (unchanged) read train records and report the same detections.

**The scenes against the baseline** (small vehicle / ship, two background pings, the baseline's bearing; margin /
ceiling in dB; expected: ceilings -3.0 dB, noise-limited margins -3.0 dB, reverberation-limited margins unchanged):

| Scene | Tone with the bank (baseline tone alone) | Sweep at zero Doppler (baseline) | |
| --- | --- | --- | --- |
| `flank_patrol_doppler` | 10.9 / 36.3 (39.8 / 38.5) | 6.4 / 28.3 (7.0 / 31.2) | tone: the sweep's reverberation (item 1) |
| `flank_patrol_doppler_underway` (+2 m/s over ground: +1.38 m/s) | 7.8 / 36.8 (36.8 / 38.3) | 7.3 / 29.9 (8.8 / 32.8) | the same |
| `popoto_far_mover_doppler` | 47.9 / 51.7 (49.4 / 51.5) | 24.1 / 24.2 (27.2 / 27.8) | noise-limited: as expected within the tone's scatter |
| `popoto_12km_doppler` | 33.2 / 28.8 (34.7 / 34.1) | 21.2 / 21.3 (24.0 / 24.6) | the same |

**The fade** (`tools/doppler_study.py fade`: target-only pings, the target moved along its bearing, four
hydrophones, echo energy over transmitted energy per segment, the tone's against the sweep's):

| Water, target | Geometries | Single tone minus sweep, dB: mean / min / max / std | 4-tone comb minus sweep |
| --- | --- | --- | --- |
| `flank_patrol_doppler` (Bellhop), small vehicle | 15: 1.8, 2.0, 2.2, 2.6, 3.0 km x 20, 40, 60 m | -1.9 / **-11.0** / +7.6 / 5.4 | -3.2 / **-9.3** / +2.4 / 3.8 |
| `popoto_far_mover_doppler` (Bellhop), ship as one point at 4 m | 5: 5.8 to 6.6 km | -1.4 / **-8.6** / +6.2 / 5.3 | -3.1 / **-7.3** / +2.8 / 3.4 |
| the same water, free field (check) | 2 | 0.0 (within 0.014) | 0.0 |

The baseline's 18.9 dB was one image-source geometry; over these 20 Bellhop geometries the tone's worst is
11.0 dB under the sweep, and the spread 18.7 dB. Option E (`frequencies`: four tones across the band, energy
shared) narrows the spread to 11.7 dB and lifts the worst case by 1.3 to 1.7 dB, at a mean cost of 1.3 to 1.7 dB:
cheap to configure (the key exists), small in effect, and inside the sweep's band it meets item 1. Not a fix.

**What D3 can rely on**: a train record (type Arbitrary, the replica as sent, `f_start`/`f_stop` the train's
band, `duration` the whole train) and the segment table from the sonar's configuration
(`SonarModel.sampled_waveform.segments`: kind, start, duration, band, frequencies, taper, energy); a segment's
replica at any range rate (`TrainSegment.signal`, as `tools/doppler_study.py` `segment_replica` uses it); exact
Doppler on every segment on both engines. Not: a tone path free of the sweep's reverberation on these scenes (item 1).

Runs (scratch, not committed): `tools/doppler_study.py margin <scene> --target <label> --pings 2 --bearing <deg>
--vmax 3|4` per scene (the slow test, `GPUSONAR_DOPPLER_STUDY=1`, 42 min for the four); the controls with `--mute
2`, a reversed-order and a tone-above-band copy of `flank_patrol_doppler`; `fade flank_patrol_doppler --target
small_vehicle --ranges 1800 2000 2200 2600 3000 --depths 20 40 60 --comb 4 --channels 4`, the same with
`--fidelity free_field`, and `fade popoto_far_mover_doppler --target ship --ranges 5800 6000 6200 6400 6600 --comb 4
--channels 4`.
