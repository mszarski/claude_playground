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
- **Done 2026-11-26 (branch `d4-bank`), opt-in, and it loses on every scene measured**: section 9.

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

## 9. D4 done: the sweep's Doppler bank in the live chain (2026-11-26, branch `d4-bank`)

### 9.1 What got worse or did not work (read first)

1. **On every scene measured the bank lowers the margin of every target, through both engines.** The chain keeps per map
   sample the largest power over N hypotheses; that maximum's noise background (the detector's training cells are
   maxima too) rises 4.8 to 5.6 dB (N = 17 to 33, measured on noise, `doppler_bank_threshold.json`), and the mismatch
   it recovers on these targets is smaller than that: margins fall 3.4 to 5.3 dB (table 9.4). Stationary targets lose
   the whole rise (the wreck: 5.3 dB). It is off by default and must stay off for now.
2. **The thresholds are not moved (0 dB), so the false-alarm rate on the maximum is BELOW the configured one** (measured:
   at 3 false detections a ping 50 -> 0 on noise, 5 -> 0, 18 -> 0, 20 -> 0 on the three scenes' reverberation; at the
   default budget 0 -> 0 on 4 pings each). The shift that would restore the configured rate is negative and was measured
   only at loosened budgets (-0.6 to -3.0 dB at 3 a ping, -0.3 to -2.4 dB at 0.3 a ping, for 3 to 33 hypotheses); it
   shrinks toward the default budget (1e-3 a ping) by about 0.6 dB a decade, where a CPU cannot count it. Applying a
   value from a looser budget would raise the false-alarm rate above the configured one, so none is applied
   (`threshold_db` sets one). Even with the 0.3-a-ping shift every target still loses 1.0 to 2.9 dB (table 9.4).
3. **The multiscale detector is the wrong statistic for a maximum over hypotheses.** For the single-sample test the
   cost of searching N hypotheses at an equal false-alarm rate is small (the maximum's tail is about N times one map's,
   measured: 12 to 16 times for N = 17; ours: 10 log10((x + ln N) / x), 0.6 dB at x = 14), but a uniform shift of every
   test cannot get there: the detector-level shift that holds the rate (-1.7 dB for N = 17 at 0.3 a ping) is far from
   the single-sample test's own (about -4.2 dB, from the measured rise and tail), presumably because the cell-mean test
   of a maximum behaves differently (not isolated here). A per-test shift, or D3's CFAR per (beam, hypothesis) with
   the maximum taken over normalised statistics and its own thresholds, is the follow-up.
4. **The range rate is not a measurement on extended or multipath targets**: the submarine, the far mover's ship and the
   12 km ship read rates scattered over +-3 m/s from ping to ping (true -0.03 to -0.99 m/s; the baseline found the same,
   section 3.3 item 5). On a point (the tests) and on the wreck it is within one step; the small vehicle's one bank
   detection read 2.06 m/s (true 2.00).
5. **Cost: 22 to 51 times the chain (table 9.5; "22 to 32" before a correction of 2026-11-26 left out the far mover, 44 and 51 times) on the scenes' pings, 2.3 to 2.9 times on the Popoto trial pings** (one compression
   and one map per hypothesis; table 9.5; the trial's window is short and its other stages weigh more). Measured on this
   CPU only; nothing ran on a GPU.
6. **Sidelobes of very strong echoes**: on the test fixture (targets 80 to 90 dB over the noise) the bank adds detections
   60 dB and more under the strongest echo at far bearings, which the detector's sidelobe envelope (made for the matched
   pulse; the bank raises its range part by the mismatch pedestal only within per_beam x step x T) does not explain. Not
   seen on the scenes (their echoes are 20 to 30 dB over the background).
7. The plan's test expectation ("the bank's peak 5 to 6 dB above the zero hypothesis" for a point at 2 m/s on a 1 s,
   4 kHz sweep) was the scene's study number. The closed form for a point is 11.7 dB (ours, derived in
   `tests/python/test_doppler_bank.py`: the residual chirp's Fresnel integral with the delay free; centred it would say
   13.5 dB); an ideal sweep gives 11.7 dB, the simulator's 5 % Tukey pulse on the map's samples 11.1 dB. Through the
   detector, after the background's rise of 6.4 dB, the detection's snr_db gains 5.7 dB.

### 9.2 What exists

- Option `chain.doppler: {mode: sweep_bank, max_range_rate_m_s: 3.0, step_m_s: None, hypotheses: None,
  unknown_ownship_m_s: 2.0, threshold_db: None, report: false}` (`docs/CONFIGURATION.md`; flat `doppler_*` keys for
  `sonar_live`), both engines: `python/gpusonar/doppler.py`, `libs/live/include/sonar/live/doppler.hpp`,
  `libs/live/src/doppler.cpp`, the stage in `live.py` and `chain.cpp`. Off is the chain as before, byte for byte.
- Hypotheses r_k = k x step, step = c / (2 B T) (0.1875 m/s at 4 kHz, 0.375 m/s at 2 kHz, 1 s; the worst loss between two
  hypotheses 0.24 dB, ours: phi <= pi / 4). Each beam takes the 2K + 1 nearest its own-ship rate -v . u_b from the mean of
  the record's transmit and receive pose velocities (K = ceil(max_range_rate / step): 17 per beam at 2 kHz, 33 at 4 kHz
  for 3 m/s; the scenes' slightly faster sound speed made it 19 and 35). No finite velocity: every beam centred on 0 and
  the span widened by `unknown_ownship_m_s` (processed, not refused: a live chain must not stop on one bad fix).
- Each hypothesis is compressed from the same baseband record with the pulse time-scaled exactly ((c - r)/(c + r);
  the zero one is the chain's own compression, `doppler_scale` exactly 1) and **read 2 f0 T r / (B c) later** (its own
  LFM coupling): every hypothesis then shows an echo where the zero one shows it, so the maximum takes the matched peak
  at today's range instead of smearing it ±f0 T Δr / B along the range-Doppler ridge (without this, measured on the test
  fixture, detections moved up to 9 m and strong echoes grew 21 m extents and spurious detections). Consequences: the
  bank moves no detection; `range_doppler` with the bank takes each detection's closing speed from the bank
  (-range_rate) and `full` moves no track further (no double correction; tested).
- Baseband: `max_range_rate` is passed only when the default band's untapered room (0.1 B) does not hold 2 r f_top / c:
  never for the shipped sweeps (35 Hz against 400 Hz at 6-10 kHz), so their baseband is today's.
- Detector: `DetectorOptions::threshold_offset_db` (every test's threshold, 0 now) and `pedestal_m` (the range sidelobe
  envelope raised to the mismatch loss out to per_beam x step x T); both 0 without the bank, the plan bit for bit.
- Each detection carries `range_rate` and `range_rate_step` (`Detection`, both engines); on the report line only with
  `report: true` (owner question 6).
- Restrictions (refused): an LFM transmit only, `cell_statistic: multiscale`, `beamformer: backprojection`.
- CUDA: the existing kernels in a loop (each hypothesis' compression and map on the device); a CUDA-gated test
  (`test_doppler_bank.py::test_the_bank_on_a_gpu_gives_the_hosts_detections`) is for the GPU job; not run here.
- Tools: `tools/doppler_bank_threshold.py` (the false-alarm measurement), `tools/doppler_bank_eval.py` (the end-to-end
  table), `tools/live_chain_timing.py --doppler`.

### 9.3 For D3

D3 plugs in beside this stage, not in it: `doppler.plan` / `plan_doppler` already give per-beam own-ship centring, the
hypothesis grid and the baseband allowance rule; `doppler.hypothesis_maps` is the map maker (D3 adds a tone's
short-time Fourier transform per beam as a second one) and `doppler.fold` / `fold_hypothesis` the combiner (D3 adds a
CFAR per (beam, hypothesis) as a second combiner; item 3 above says that one is needed for the sweep as well). The
detection's `range_rate` fields, the report switch and the `range_doppler` interaction are shared.

### 9.4 End to end through the chain (`tools/doppler_bank_eval.py`, docs/tasks/doppler_bank_eval.json; ours)

Scenes as shipped, 6 pings, their seeds, Bellhop eigenrays; chain defaults (bearings -40 to 40 every 0.5,
backprojection, multiscale, host); bank `max_range_rate_m_s: 3`. Margin = snr_db over the single-sample test's
threshold, mean over the pings both detect. Gross: the chain as it is (thresholds unchanged). Net at 0.3 a ping: gross
minus the shift measured at 0.3 false detections a ping (-1.7 dB for 17-19 hypotheses, -2.4 dB for 33-35), the nearest
to an equal false-alarm rate measured (an estimate: the default budget's shift is smaller in size). Python and C++
gave the same numbers to the printed digit on every row. Worst first:

| Scene, target | Hypotheses a beam | Margin off -> on, dB | Gross | Net at 0.3 a ping | Detected off -> on (of 6) | Tracked off -> on | Baseline's sweep bank gain (section 3) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flank_patrol wreck (still) | 35 | 14.63 -> 9.33 | -5.30 | -2.9 | 6 -> 6 | static -> static | 0.0 |
| popoto_12km ship | 19 | 9.11 -> 5.57 | -3.55 | -1.9 | 6 -> 6 | mover -> mover | +1.9 |
| popoto_far_mover ship | 35 | 12.35 -> 8.84 | -3.50 | -1.1 | 6 -> 6 | mover -> mover | +0.6 |
| flank_patrol submarine | 35 | 11.69 -> 8.32 | -3.37 | -1.0 | 6 -> 6 | mover -> mover | +2.0 |
| flank_patrol small vehicle | 35 | detected on one ping each (off ping 6 at 12.5 dB snr; on ping 3 at 9.8 dB, rate 2.06 m/s) | | | 1 -> 1 | no -> no | +5.9 |

Why it differs from the baseline: the baseline normalised each (beam, hypothesis) cell by a clairvoyant background of
its own and took the target's best cell; the chain takes the maximum over hypotheses first and estimates one background
from it, which pays the rise of item 1. The mismatch itself is recovered (the point target: 11.1 of 11.7 dB on the map).
What does not improve at all, as expected: stationary targets (nothing to recover) and reverberation-limited margins
(the sweep's reverberation is in every hypothesis).

### 9.5 Cost a ping (CPU of this machine, 1 thread, host backend; median after the first ping)

| Case | Channels | Hypotheses (all beams) | Python off -> on, s | C++ off -> on, s |
| --- | --- | --- | --- | --- |
| flank_patrol (2.4 km window, 161 beams) | 32 | 35 | 4.79 -> 138.8 | 4.04 -> 129.9 |
| popoto_far_mover | 32 | 45 | 2.72 -> 119.1 | 2.45 -> 124.9 |
| popoto_12km (1.4 km) | 32 | 25 | 1.37 -> 30.8 | 1.08 -> 27.1 |
| Popoto trial pings of `tools/live_chain_timing.py` (one row, decimated first) | 32 | 17 a beam | 1.08 -> 3.10 | 0.91 -> 2.64 |
| the same, three rows (board synchronisation) | 96 | 17 a beam | 2.17 -> 5.08 | 1.71 -> 3.97 |

The 0.065 s a ping at 96 channels in `docs/STATUS.md` is the C++ chain on an L4 GPU; the bank multiplies the compression
and map part by the hypothesis count (25 to 45 here); its cost on the GPU was not measured (`tools/live_chain_timing.py
--doppler` in the GPU job).

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

## D3 done (2026-11-26, branch `d3-tone`): a CFAR combiner for the sweep bank, the tone's Doppler filter bank

Appended; nothing above was rewritten. Every number here is **ours** (simulated records, the shared 4-core CPU, one
thread a process, host backend, Bellhop eigenrays, the scenes' own seeds). Commands and every field:
`docs/tasks/doppler_d3_eval.json`.

### D3.1 What failed, got worse, is uncertain or was not run (read first)

1. **The train scene with the tone above the band (`flank_patrol_doppler_tone`, new) was not measured end to end.**
   Simulating its 6 pings (a 2 s sampled train through the eigenray reverberation, 32 channels) ran 4 h 40 min on this
   shared CPU and did not finish. It was stopped; the brief's 1 h bound for an evaluation run applies. The figures
   "small vehicle and submarine with the tone outside the band in one ping" are therefore NOT measured through the
   chain. What stands in for them: D2's study (the tone at 10.5 kHz at 37.7 dB over its background, the sweep 6.4 dB)
   and the alternating-pings run on `flank_patrol` (D3.3), where the tone at 10.5 kHz and the sweep travel in separate
   pings through the same water. To do: `tools/doppler_tone_eval.py flank_patrol_doppler_tone --cache DIR` in the GPU job
   (CUDA synthesis), or with fewer pings. The gated test `test_doppler_scenes.py::test_the_chain_with_the_tone_path_sees_the_small_vehicle_and_keeps_the_wreck`
   (`GPUSONAR_DOPPLER_CHAIN=1`) holds its acceptance and was not run.
2. **The first scene run of the tone path failed**: 43 to 50 false detections a ping on flank_patrol's tone pings at the
   notch's edges, and the wreck was given 0.23 m/s. The leakage rule (D3.3 (c)) was designed after seeing this, on the
   same records. It brought them to 1, 0 and 0, but that is not an independent test. One false detection
   (2983 m, 22.8 degrees, +0.24 m/s, 18.4 dB) is not explained.
3. **Today's chain on a train compresses with the whole train**, and on `popoto_12km_doppler` that loses the ship on every
   ping (0 of 6). With the tone path on, the sweep path now uses the sweep part of the replica alone (6 of 6). The
   default (tone off) is unchanged and keeps that loss: D2's train scenes are blind through the default chain on this
   scene. This is a finding about D2's scenes, not a change of a default.
4. **The tone gives a rate on the 12 km ship on one ping in six.** Its range rate (-0.50 m/s) lies 2.7 resolution cells
   from its beam's ridge, at the notch's edge, and the leakage rule removes the rest. The owner has set the 12 km
   requirement for the large submerged class first and the large ship second: the scenes that matter for it are those
   classes at 12 km (no such scene exists yet). How much the tone helps there depends on their radial speed against the
   ridge (expected small for a crossing target: section 3.3 item 4).
5. **The normalisation for the tone is along rate, not per (beam, hypothesis) along range** as the brief asked: a ping
   holds 13 lags on flank_patrol at a quarter of 1 s, too few for a background per (beam, hypothesis). Part A's sweep
   combiner is per (beam, hypothesis).
6. **The threshold correction of part A was measured only at loosened budgets** (3, 1 and 0.3 a ping, long-range array,
   60 noise pings); at the default 1e-3 a ping it is the planned value (0.62 to 0.70 dB), not a measured one. The flank
   array's noise count was stopped for time.
7. **Engines**: the two engines' tone paths are one C++ function (`tone_stage`), called by both, like `plan_doppler`.
   Only the merge and the reporting are written twice. Parity of the tone stage is by construction. A NumPy reference of
   the matched filter in the test holds the function itself. Without the tone path, the Python chain on a CW record of a
   short window makes a map of nonsense (snr of thousands of dB), where the C++ chain refuses it. This divergence predates
   D3 and is a follow-up.
8. **The small vehicle in part A is thin**: margins 0.48 to 1.37 dB on its three pings (it is below the reverberation by
   design); the sweep bank does not separate reverberation.
9. Nothing ran on a GPU; the GPU suite was not run.

### D3.2 Part A: `doppler: {mode: sweep_bank, combine: cfar}`

D4's bank took the largest power over each beam's hypotheses and let the detector estimate its background from that
maximum (9.1 item 1). `combine: cfar` normalises each (beam, hypothesis) map by its own background first: the detector's
own estimate (`detector_background`, both engines) on that hypothesis' cell means, over the cells the detector keeps
(blanking). The largest normalised value per sample is kept (`fold_normalised`; the hypothesis that holds it gives the
detection's coarse rate as before) and the detector tests it with the background it was normalised by: the map it is
handed is that maximum times the background of the beam's centre (own-ship) hypothesis, and that background is given to
it (`multiscale_cells` / `detect.multiscale` take an optional `level`; without it the detector is as before, bit for bit).

The threshold correction for N hypotheses a beam, **derived** (ours): each normalised map is a statistic the detector's
thresholds were planned for, and P(max over N > x) <= N P(one > x) (the union bound), so planning the thresholds for the
budget divided by N holds the configured rate, on the safe side because neighbouring hypotheses are correlated (one
step of the sweep loses 0.97 dB: their noise is nearly the same). In dB the correction is the single-sample threshold
of the divided budget over the undivided one; in an exponential tail about 10 log10((x + ln N) / x).

**Measured** (the live chain itself, C++ engine; `tools/doppler_bank_threshold.py --chain cfar`): noise-only pings of the
long-range array (60 pings, 800 m, bearings -60 to 60 every 0.5), budgets loosened so that a CPU counts:

| Budget a ping | Bank off: detections (of the budget) | cfar, 7 a beam (1 m/s): detections, ratio to off, correction | cfar, 17 a beam (3 m/s) |
| --- | --- | --- | --- |
| 3 | 128 (0.71) | 109, 0.85, +0.74 dB | 105, 0.82, +1.04 dB |
| 1 | 56 (0.93) | 37, 0.66, +0.67 dB | 43, 0.77, +0.94 dB |
| 0.3 | 14 (0.78) | 12, 0.86, +0.61 dB | 12, 0.86, +0.86 dB |

Every ratio is within three Poisson sigmas of 1 (0.26 to 2.8 at 0.3 a ping, 0.55 to 1.26 at 3) and under it: the union
bound holds the rate and is not loose by much here. On the scenes' reverberation-only pings (4 each, budget 3 a ping):
flank_patrol 5 -> 5, popoto_far_mover 18 -> 17, popoto_12km 20 -> 7; at the default budget 0 -> 0 on all three. **Not
measured**: the default budget (1e-3 a ping): a CPU cannot count it (a few hundred thousand pings); the correction there
is the planned one (0.62 to 0.70 dB on the scenes, below), not a measured one. The flank array's noise count (35 a beam)
was started and stopped for time (40 pings at budget 3: off 88; the cfar count not finished).

**End to end** (`tools/doppler_bank_eval.py SCENE --combine cfar`, 6 pings, both engines; the off and max rows are
D4's `docs/tasks/doppler_bank_eval.json`, the same records: the off margins here agree with it to the printed digit).
Margin = snr_db less the single-sample threshold of the plan in use, mean over the pings both detect. Python and C++
gave the same numbers to the printed digit on every row. Worst first:

| Scene, target | Hypotheses a beam | Threshold correction | Margin off, dB | D4 max | cfar | cfar - off | Detected off / max / cfar (of 6) | Tracked off -> cfar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flank_patrol wreck (still) | 35 | +0.69 dB | 14.63 | 9.33 | 14.15 | -0.48 | 6 / 6 / 6 | static -> static |
| popoto_12km ship | 19 | +0.62 dB | 9.11 | 5.57 | 9.63 | +0.52 | 6 / 6 / 6 | mover -> mover |
| popoto_far_mover ship | 35 | +0.70 dB | 12.35 | 8.84 | 13.65 | +1.30 | 6 / 6 / 6 | mover -> mover |
| flank_patrol submarine | 35 | +0.69 dB | 11.69 | 8.32 | 13.20 | +1.51 | 6 / 6 / 6 | mover -> mover |
| flank_patrol small vehicle (2.0 m/s) | 35 | +0.69 dB | -0.82 (ping 6 only) | (ping 3 only) | +0.48 to +1.37 (pings 3, 4, 5) | | 1 / 1 / 3 | no -> yes (static) |

So the acceptance of part A holds on these scenes: the still target loses less than the correction (0.48 against
0.69 dB: the maximum over normalised neighbours adds 0.21 dB of snr), every mover gains (the mismatch the bank recovers
less the correction), and the small vehicle, detected on one ping without the bank and with D4's max, is detected on
three with ranges rates 1.87 to 2.25 m/s (true 2.00) and tracked (called static: three pings). Its margins (0.5 to 1.4
dB) are thin: it is below the reverberation by design (D-198) and the sweep's bank does not separate reverberation.

Cost a ping (median after the first): the same as D4's max (the background of each hypothesis is a cell-mean pass):
flank_patrol Python 4.26 -> 140.1 s, C++ 4.02 -> 121.6 s; popoto_far_mover 2.61 -> 119.2, 2.24 -> 111.0;
popoto_12km 1.20 -> 30.0, 1.07 -> 26.0.

### D3.3 Part B: `doppler: {tone: bank}` (both engines; one C++ function, `tone_stage`)

What it is (`docs/CONFIGURATION.md` has every key; `libs/live/include/sonar/live/doppler.hpp` the reasons):

- **Where the tone is.** A CW record is a tone-only ping (its own frequency, length, taper). A train (an arbitrary
  record) names its tone in the chain's configuration (`tone_frequency_hz`, `tone_start_s`, `tone_duration_s`,
  `tone_window_alpha`; `doppler.tone_from_sonar(sonar)` reads them from the sonar's segment table): the record carries
  the train as sent, not its table (no schema change), so a train without them is refused by name. A linear-sweep record
  has no tone path (the other ping of an alternating pair). Any frequency, inside or outside the sweep's band.
- **Baseband.** The conversion gets the allowance that keeps the tone and its largest shift (2 r f / c + 8 / T) inside
  its untapered band (`plan_tone`); for the shipped sweeps without a tone nothing changes.
- **The bank.** Per beam of the far-field beamformer (the chain's bearings and aperture window) on the baseband record
  before compression: the beam mixed to the tone and cut to the band the hypotheses need by an exact FFT decimation; at
  lags a quarter of the tone apart, the matched filter of the tone's envelope at each range-rate hypothesis
  (r_k = k x c / (4 f T): half the resolution; the echo's envelope time scaling, under 2 ms at 3 m/s and 1 s, left out:
  ours). Held against a NumPy matched filter at the full rate within 0.1 dB on every cell within 25 dB of the peak
  (`test_doppler_tone.py`). Each beam's 2K + 1 hypotheses are centred on its own-ship rate, as the sweep bank's.
- **The ridge.** (a) A notch: hypotheses within the own-ship rate's spread over the beam's half-power width plus three
  resolution cells are not tested (still reverberation, still targets: the sweep's). (b) A CFAR along RATE per (beam,
  lag): the mean over the hypotheses 5 to 36 steps away either side, notch excluded; the threshold the cell-averaging
  factor for exponential power, training cells counted at step / (enbw x resolution) each, at the budget over every cell
  tested (union bound). Not along range, as the brief asked: a ping holds too few lags (13 on flank_patrol at a quarter of
  1 s) for a background per (beam, hypothesis); the per-(beam, hypothesis) normalisation of part A is the sweep's.
  (c) **Leakage** (added after the first scene run): a flagged cell no more than 10 dB above what a stronger cell of its
  beam and lag (the ridge, or a stronger cell) leaks into it through the tone's own spectrum (the window's power spectrum,
  its envelope) is dropped. Without (c), flank_patrol's tone pings gave **43 to 50 false detections a ping** at the notch's
  edges and gave the wreck a rate of 0.23 m/s; with it 1, 0 and 0 on those pings, the wreck none (designed on these records: D3.1 item 2).
- **Detections and merge.** Touching flagged cells are one detection, refined by parabolas (rate, bearing, lag); its
  range is the lag's (coarse: c T / 2). Each tone detection gives its rate to the strongest sweep detection within c T / 4
  and 2 degrees (`doppler_path: both`: the sweep's range and bearing, the tone's rate); one that finds none is its own
  detection (`tone`) with a range sigma of c T / 8 for the tracker (`Detection::range_sigma`, NaN for every other
  detection: the tracker as before); every other sweep detection is as it was (`sweep`). A stationary target seen only
  by the sweep and a mover seen only by the tone both survive (tests).
- **A train's sweep path compresses with the sweep part of the replica alone** (the tone's samples zeroed,
  `sweep_only_replica`; added after the popoto run): with the whole train as the replica the sweep path pays the tone's
  half of the energy in its noise and its c T / 2 pedestal in the map. On `popoto_12km_doppler` the whole-train replica
  gave the ship on **0 of 6** pings through the chain (the tone path on 3, tone-only); with the sweep replica 6 of 6.
- **Tone-only pings** make no sweep map and do not size the chain (sizes derive from the first sweep ping; a tone-only
  ping before it is reported, not tracked). Tracker change: only the per-detection range sigma.

Measured, end to end, both engines (`tools/doppler_tone_eval.py`; Python and C++ report lines agree within 0.05 m,
0.05 degrees, 0.05 dB and the path on every scene; tone margins are the Python engine's tone stage, the same function):

**`flank_patrol_doppler_tone`** (one ping: the tone at 10.5 kHz, then the sweep): **not measured** (D3.1 item 1).

**Alternating pings** (`--alternate 10.5e+3` on `flank_patrol`: a 1 s Hann CW at 10.5 kHz with the sweep's whole energy
(199.86 dB re 1 uPa^2 s) on pings 0, 2, 4, the scene's sweep on 1, 3, 5; 12 s apart):

| Target (true range rate) | Tone pings: detected, rate m/s, tone margin dB, range error m | Sweep pings: detected | Tracked at the last ping |
| --- | --- | --- | --- |
| small vehicle (+2.00) | 3 of 3: 1.972, 1.992, 1.988; 19.5, 22.2, 26.0; +132, +50, +8 | 1 of 3 (ping 5, 12.5 dB snr) | yes (22 m off; called static) |
| submarine (-0.61 to -0.43) | 2 of 3: -0.568, -0.420; 23.9, 26.6; -8, +36 (ping 2 missed) | 3 of 3 | yes (13 m; uncertain) |
| wreck (still) | 0 of 3 (the notch) | 3 of 3 (27.9 to 28.0 dB snr) | yes (18 m; uncertain) |

False detections: one tone detection on tone ping 0 (2983 m, 22.8 degrees, +0.24 m/s, 18.4 dB: just outside the notch;
not explained). Without the bank the chain cannot take the tone pings at all (the C++ chain refuses a CW of fewer
than three range cells; see D3.1); off here is the sweep pings alone: small vehicle 1 of 3, submarine and wreck 3 of 3.

**`popoto_12km_doppler`** (as shipped: the 5 kHz tone INSIDE the 4-6 kHz sweep band, the background noise; the ship's
range rate -0.50 m/s, its beam's own-ship rate near -0.10 m/s: the ship lies 2.7 resolution cells from the ridge, at
the notch's edge):

| Path | Ship detected (of 6) | snr dB | Rate | Tracked |
| --- | --- | --- | --- | --- |
| off (the whole train as the replica) | 0 | | | no |
| tone path on (sweep replica alone) | 6: ping 0 `both`, pings 1 to 5 `sweep` | 18.5 to 21.8 | -0.573 on ping 0 (true -0.499); tone margin 15.9 dB | yes (mover, 33 m) |
| tone path on, before the leakage rule and with the whole train | 3, `tone` alone | 29.5 to 32.4 | -0.573, -0.486, -0.486 | no |

So at 12 km the tone gives a rate on one ping in six: the ship is too slow against its ridge for the notch and the
leakage rule (expected, DOPPLER.md 3.3 item 4); the gain on this scene is the sweep replica's. Background (2 pings,
targets removed): default budget 1 -> 0, budget 3: 11 -> 8 (2 of them the tone's).

Cost a ping (median after the first; C++ / Python): popoto_12km_doppler 1.12 / 1.24 s off -> 2.65 / 2.95 s with the
tone path; flank_patrol's tone-only pings 0.50 to 0.57 s (C++; no sweep map), its sweep pings 3.8 s (the sweep path
alone, as off: 4.1 s). Nothing ran on a GPU (the tone stage's beamformer runs on the chain's backend; the bank
and the CFAR are host loops).

### D3.4 What is left

- The surface's own Doppler spread (D1) is still missing: every tone margin on a surface-limited background is an upper
  bound (section 0 item 1).
- The tone's matched filter is the configured Tukey envelope, not the record's replica as sent (a real projector's
  spectrum sets the leakage: section 3.3 item 3).
- A per-(beam, hypothesis) background for the tone along range needs longer windows or pings pooled (not done).
- The tone path is refused with `range_doppler` other than off and with `elevation` (not made for them).
- The tracker takes the tone's range rate nowhere (D5): only its range sigma.
- D8: the bank and both CFARs on the GPU; the GPU suite (`tools/gpu_job.sh`) was not run.

## D8: the Doppler stages on the device (2026-11-27, branch `cuda-doppler`)

Appended; nothing above was rewritten. Status of every number: ours.

### D8.1 What is not verified (read first)

1. **None of the device code has run on a device.** `libs/live/src/doppler_cuda.cu` was compiled here with `nvcc` 12.0
   for sm_89 (the L4) as a separate object, and its symbols match the host code's declarations; the project was never
   built with `SONAR_ENABLE_CUDA=ON` here and nothing was linked against cuFFT. The CUDA-gated tests of
   `tests/python/test_doppler_device.py` skip on this machine.
2. **The gain on the device is not measured.** The design below removes the transfers and host passes the loop makes;
   how many seconds that saves is for the GPU job (`tools/live_chain_timing.py`, per-stage seconds now in its rows).
3. What does not move to the device: each hypothesis' filter is still built on the host (`dsp::compression_filter`: the
   replica and one transform of n_fft points a hypothesis, the same code as `pulse_compress`); the tone path's grouping
   of flagged cells and its merge; the detector re-uploads the combined map (its own statistic on the device, as before);
   the elevation option's rows keep the loop.

### D8.2 Where the time went (the loop, from the code, `backend: cuda`)

Per hypothesis other than the zero one, C++ engine (`chain.cpp` before D8): a host copy of the baseband record;
`pulse_compress`: the filter built and transformed on the host, the record and the filter uploaded, the record
downloaded; `range_bearing_power`: `make_inputs` copies the samples on the host, `backproject_cuda` uploads them, the
points and a zero image, and downloads the image; on the host the power, the cells of the map and the fold (max), plus
the cell means and `detector_background` (cfar). That is 5 uploads, 2 downloads and 5 to 7 host passes over the record
or the map a hypothesis. The Python engine the same, its fold in NumPy. The tone path: the beams on the device and
downloaded (n_beams x n samples), then per beam on the host a transform of N points, the band cut, one of M points, the
matched filters of every (hypothesis, lag), the CFAR along rate and the leakage rule.

Measured on an L4 by the coordinator (Hugging Face job 6b08c9d954072e8b71685cd6, branch `gpujob` 28e84ae,
`tools/live_chain_timing.py`, median s a ping after the first, backend cuda), off -> max -> cfar: flank_patrol C++
0.112 -> 3.558 -> 3.935, Python 0.196 -> 4.684 -> 5.370; popoto_far_mover C++ 0.062 -> 2.410 -> 2.766, Python 0.114 ->
4.232 -> 4.824; popoto_12km C++ 0.019 -> 0.372 -> 0.532, Python 0.061 -> 1.063 -> 1.198. The tone path on
popoto_12km_doppler C++ 0.033 -> 0.207, Python 0.048 -> 0.241. So the bank costs 19 to 45 times the chain on the device.
The acceptance target of D8 is the bank's cost a ping on these three scenes after the change, by the same tool in the
GPU job (not yet run).

Transfers a ping of the loop, from the code and the scenes' sizes (a noise ping of each scene's sonar and window, a
moving platform, bearings -40 to 40 every 0.5, `max_range_rate_m_s: 3`; scratchpad script, not committed):

| Case | Channels x samples | n_fft | Hypotheses (all beams) | Map points | Loop: up / down a ping | Batched: up / down a ping |
| --- | --- | --- | --- | --- | --- | --- |
| flank_patrol | 32 x 50400 | 62500 | 43 | 2 895 585 | 5.12 / 1.55 GB, 301 calls | 34.6 / 23.9 MB, 1 call |
| popoto_far_mover | 32 x 37632 | 49896 | 43 | 1 929 585 | 3.50 / 1.08 GB | 26.9 / 16.0 MB |
| popoto_12km | 32 x 17248 | 23328 | 23 | 900 312 | 0.87 / 0.27 GB | 8.8 / 7.7 MB |

### D8.3 What moved to the device, and the batching (`libs/live/include/sonar/live/doppler_batched.hpp`)

- One formulation, written once as kernels over an executor (`dsp::HostExec` / `dsp::CudaExec`, as the batched
  beamformers): `doppler.batched: auto` (default) runs it on the cuda backend and keeps the loop on the host, `on` runs it
  on the host too (the same kernels on CPU threads: how it is tested here, bit for bit against the loop), `off` keeps the
  loop on cuda.
- Sweep bank: the baseband record uploaded once a ping; every channel's spectrum once per transform length (one length
  for every hypothesis of the shipped sweeps); then per batch of hypotheses: each filter times every spectrum, one
  batched inverse transform, the crop; **the backprojection kernel itself** (`recon::BpKernel`, unchanged: one kernel
  for all modes) launched once per hypothesis on the device-resident compressed samples, its group carrying the
  hypothesis' read shift in `start_time`; the fold (max) or the cell means, **the detector's own background kernels**
  (`detector_kernel.hpp`, one row per (hypothesis, beam)) and the normalised fold (cfar). At the end the combined map's
  cells (or `normalised_map`) and one download of the map, the hypotheses and the cells (and the background, cfar).
  Both engines call it (`sweep_bank_stage`; Python `doppler.batched_maps`).
- Tone path: the far-field beams are made and kept on the device (`farfield_batched_device`), then each beam's
  transform, the band cut, one batched inverse transform, the matched filters of every (beam, hypothesis, lag), the CFAR
  along rate and the leakage rule; power, ratio and flags come back (n_beams x hypotheses x lags, under 0.3 MB here) and
  the grouping stays on the host.
- Memory a ping (all buffers counted by `sweep_bank_bytes`; the transforms' work memory is cuFFT's): flank_patrol 123 MB
  + 52.6 MB a hypothesis (max; cfar 125 + 58.3), 2.4 GB (2.6) for all 43; popoto_far_mover 85 + 38.2, 1.7 GB (1.9);
  popoto_12km 40 + 17.8, 0.45 GB (0.53). The batch is as many hypotheses as half the device's free memory holds after
  the fixed part (`sweep_bank_batch`; `doppler.batch` sets it): on a 24 GB L4 every case is one batch. Batches never mix
  transform lengths and keep hypothesis order, so the fold is the same in batches of any size (tested on the host: one
  batch, batches of 1, 2, 3, 4).

### D8.4 Tests and tolerances

- Host (`tests/python/test_doppler_device.py`, GoogleTest `DopplerBatched`): the batched form on the host gives the
  loop's report lines byte for byte, flank and long-range records, max and cfar, unbatched and in batches; the maps and
  the hypothesis of every sample likewise; the tone path's power, ratio and detections likewise; the Python engine with
  it reports its loop's lines within the engines' tolerances. Default-off proof, D3's method (sha256 of the report lines
  without processing_seconds, 2 pings, both engines, host backend, a build of the base `02bce49` against this branch):
  flank_patrol and popoto_12km with the defaults, `doppler: {mode: off}` and `range_doppler: ownship` (12 of 12 equal,
  and equal to D3's own hashes), and popoto_12km with the bank's host loop, max and cfar (4 of 4 equal). The C++ engine
  with `batched: on` (one batch, batches of 3) gives the loop's hashes (4 of 4 equal). The Python engine with `batched:
  on` does not (4 of 4 differ, as expected: its loop maps with its own geometry code and folds in NumPy; the batched
  stage is the C++ one), within the engines' tolerances on the test records. Not hashed: the tone path on
  popoto_12km_doppler (its simulation did not finish in 50 min here); the tone loop's code was moved, not changed, and
  is held by `test_doppler_tone.py` and by the batched form's bit equality with it.
- Device (skipped here): maps within 1e-4 of each beam's strongest sample plus 1e-4 relative (the estimate: cuFFT
  against pocketfft and the device's sin and cos give 1e-6 to 1e-5 of the strongest sample); the winning hypothesis
  equal on all but 1e-3 of the samples; detections within 0.05 m, 0.05 degrees, 0.05 dB and the same range rate (the
  engines' tolerances), both engines, both combiners, one batch and batches of 1 or 2; the two engines equal on the device.

## Large submerged at 12 km (2026-11-27, branch `scene12`, issue `sonar-qbrg`): the sweep, the repaired bank and the tone

Appended; nothing above was rewritten. The owner set the 12 km requirement for `large_submerged` first and the ship
second (2026-11-26, D-235) and chose a tone outside the sweep's band in the same ping, alternating tone and sweep pings
as the fallback. Every number here is **ours** (simulated records, the shared 4-core CPU, one thread a process, host
backend, Bellhop eigenrays, one seed); the class's values are **assumed** (`configs/builds/targets/reference_classes.yaml`).
Commands and every field: `docs/tasks/doppler_large_submerged_eval.json`.

### LS.1 What failed, got worse, is uncertain or was not run (read first)

0. **Chain finding: the tone above the band makes false tone detections through the array's grating lobes.** At 6.5 kHz
   the 0.15 m pitch (half a wavelength at 5 kHz) is 0.65 of a wavelength, so a beam steered beyond +-32.4 degrees has a
   grating lobe near endfire (steered to -34.75 degrees: a lobe at +74.8; ours, sin = sin(-34.75) + lambda / d). The
   tone path's notch is centred on the steered beam's own-ship rate only, and the own-motion ridge of the water behind the grating lobe (-1.5 sin 74.8 =
   -1.45 m/s there) is tested as a target: **3 to 5 false tone detections on every tone ping** (alternating pings 0, 2, 4:
   5, 4, 3; the one train ping: 4), all at |bearing| 31.95 to 40.0 degrees, 15.8 to 28.7 dB, rates +-1.1 to +-1.6 m/s,
   through both engines (one C++ function). The configured budget is 1e-3 a ping. Not fixed: which remedy is a design
   choice for the owner (a notch about each grating lobe's own-ship rate as well, both engines through `tone_stage`; the
   tone path's bearings limited to the grating-lobe-free sector, +-32.4 degrees here; or the tone BELOW the band,
   e.g. 3.5 kHz, where the pitch is 0.35 of a wavelength and no lobe appears, at a coarser 0.21 m/s resolution and a
   +-0.70 m/s notch). The sweep (4-6 kHz) has no grating lobe within +-40 degrees (from 41.5 degrees at 6 kHz).
   The same applies to `flank_patrol_doppler_tone` (10.5 kHz on flank_32's pitch, not checked here).
1. **The train scene (tone and sweep in one ping, the owner's chosen route) was measured on ONE ping only**: its 2 s
   sampled train took 58 min a ping to simulate on this shared CPU (4 pings ran 69 min without finishing and were
   stopped, as was a 3-ping run at 69 min); no tracking, no background count. Six pings with a background are a GPU-job
   item (LS.5).
2. **The fallback (alternating pings) was measured on 3 pairs only** (6 pings, 3 of them tone pings; one seed), and its
   tone pings have no background count (the tool has none for `--alternate`): the tone path's false-alarm rate on this
   water is not measured here.
3. **The tone cannot help the crossing target, by construction**: on its beam's own-motion ridge it is inside the
   notch on every tone ping (0 of 3) and seen by the sweep alone. Crossing targets are where the requirement's class
   spends much of its time; only the sweep holds them (6 of 6 here).
4. **Tracks do not get the tone's rate** (D5 not done): with alternating pings the oblique target (2 m/s, 0.50 m/s
   closing) is tracked as `static` and the others `uncertain` (`uncertain` too from the 3 sweep pings alone, 39.8 s
   apart); the sweep-only chain called all three `mover` over 6 pings 19.9 s apart. With the repaired bank (`combine: cfar`) the crossing target is called `static` (2 m/s across the
   line of sight; `mover` without the bank). Both are classification regressions against the sweep-only chain on this
   scene.
5. **The sweep bank's range rate is not a measurement here either** (9.1 item 4): closing -7.1 to -4.1 m/s and once
   +2.2 (true -5.6), oblique -3.4 to +7.9 (true -0.50), crossing -1.5 to -0.75 (true -0.39). The tone's rate is
   (within 0.06 m/s, LS.3).
6. **The repaired bank is refused on a train record** (`mode: sweep_bank` takes an LFM transmit only): on the owner's
   chosen waveform it cannot be combined with the tone path today. Follow-up, both engines.
7. **The evaluation tools' truth was wrong for a fast mover at 12 km**: taken at the ping's time, it is 45 m off for
   the closing target by the time the pulse reaches it (8 s one way), and the 60 m gate about the hull's centre misses
   its nearest highlight (50 m forward of the centre, and 14 m more of the sweep's range-Doppler coupling). The first
   run reported it "not detected on any ping" through both engines with the bank off and on; it is detected on 6 of 6.
   Fixed in the tools only, opt-in (`--truth-at reflection`, `--gate-range-m`; defaults unchanged, so every committed
   run reproduces). Not a chain defect. The earlier scenes' movers have small radial speeds or are near (0.5 to 2 m/s,
   2 to 12 km): their truth's range is off by 1 to 4 m (ours: radial speed x R / c), inside the 60 m gate.
8. Simulation cost on this CPU: the sweep scene 6 pings in 27 min, the tone-only pings of the fallback 6 in 50 min (the
   machine was shared, load 4 to 5). One seed; no surface Doppler spread (section 0 item 1); nothing on a GPU; the GPU
   suite was not run.

### LS.2 The scenes (`configs/scenarios/`, opt-in, new files)

- `popoto_12km_large_submerged`: `popoto_12km` (100 m of isovelocity water over sand, `popoto_long_range` one row of 32,
  ownship 1.5 m/s, seed 51, window 11.3 to 12.7 km) with the ship replaced by three `large_submerged` targets at the
  class's LOW target strength: five highlights of +18.01 dB, 25 m apart along a 100 m hull (+25.00 dB in all, as the
  ship is built), 50 m deep (inside the class's 30 to 300 m), at 6, 2 and 2 m/s (inside 2 to 8):
  `closing` (-15 degrees, 12.2 km, bow-on: range rate -5.61 m/s, 6.0 m/s from its beam's ridge), `oblique` (0 degrees,
  2 m/s at 75.5 degrees to the line of sight: -0.50 m/s, 0.50 m/s from the ridge, like `popoto_12km_doppler`'s ship at
  0.40) and `crossing` (+15 degrees, 2 m/s across: -0.39 m/s, on the ridge). The shipped sweep only.
- `popoto_12km_large_submerged_tone`: the same with a 1 s Hann tone at **6.5 kHz** then the 4-6 kHz sweep in one 2 s ping,
  half the energy each (the total is the shipped sweep's, 196.72 dB re 1 uPa^2 s at 1 m). The rule, the one
  `flank_patrol_doppler_tone` used (10.5 kHz over a 6-10 kHz sweep): **0.5 kHz above the top of the sweep's band**. No
  number about the projector is needed (the simulator has no projector response); that the projector sends 6.5 kHz is
  stated in the file as unconfirmed, as are the 2 s ping (10 % duty against the 5.6 % the source is sized for) and the
  split. At 6.5 kHz the 0.15 m pitch is 0.65 of a wavelength: no grating lobe within +-32.4 degrees of broadside
  (beyond it, LS.1 item 0).
- The fallback, alternating pings: no file. `tools/doppler_tone_eval.py popoto_12km_large_submerged --alternate 6.5e+3`
  simulates the sweep scene and a copy whose transmit is a 1 s Hann CW at 6.5 kHz with the sweep's whole energy, and
  takes their pings in turn (tone first).
- Tests: `tests/python/test_large_submerged_scenes.py` (9: loading and YAML numbers, each file against its base, the
  class's values, the stated radial speeds, one thinned ping on both engines: 1.2e-8 and 1.4e-8 relative rms).

### LS.3 Measured (both live-chain engines; worst first)

The notch the chain computes (`libs/live/src/doppler.cpp`, the tone plan): own-motion spread over the beam's half-power
width, 1.5 m/s x 0.5 x 0.886 lambda / L = 0.033 m/s at 6.5 kHz (independent of bearing for motion along the array), plus
3 resolution cells of c / (2 f T) = 0.115 m/s: **+-0.38 m/s about each beam's own-ship rate** (+-0.49 m/s at 5 kHz:
0.043 + 3 x 0.150). Hidden: any target whose radial speed over ground is within +-0.38 m/s, whatever its speed: at 2 m/s
the headings within 11 degrees of square to the line of sight (12 % of headings), at 8 m/s within 2.7 degrees (3 %); the
leakage rule (D3.3 (c)) may widen it near strong echoes (not separated here). On these targets: `crossing` (0.00 from the
ridge) inside, `oblique` (0.50) and `closing` (6.0) outside.

Margin = snr_db less the single-sample threshold of the plan in use (13.06 dB off; 13.85 dB with the bank, +0.78 for 45
hypotheses a beam at 8 m/s, 51 in all), mean over the pings detected; the tone's margin is over its own CFAR's threshold
(15.09 dB; 15.22 dB on the train ping). On the alternating and train rows the bank-off column is the sweep pings' (snr_db
where the tool records no threshold).
Range rate: the detection's against truth at the pulse's reflection. Python and C++ gave the same numbers to the printed
digit (bank: every row; tone: the report lines agree within 0.05 m, 0.05 degrees, 0.05 dB and the path).

| Scene / ping plan, target (range rate; from the ridge) | Bank off: detected, margin dB | `combine: cfar` (8 m/s): detected, margin, rate m/s | Tone path: detected by the tone, its margin, rate m/s | Tracked at the last ping: off / cfar / tone path |
| --- | --- | --- | --- | --- |
| sweep only, crossing (-0.39; 0.00) | 6 of 6, 11.41 | 6 of 6, 12.00 (+0.59), -1.50 to -0.75 | (no tone) | mover / **static** / - |
| sweep only, closing (-5.61; 6.0) | 6 of 6, 7.98 | 6 of 6, 8.78 (+0.80), -7.1 to -4.1, once +2.2 | (no tone) | mover / mover / - |
| sweep only, oblique (-0.50; 0.50) | 6 of 6, 9.22 | 6 of 6, 10.08 (+0.87), -3.4 to +7.9 | (no tone) | mover / mover / - |
| alternating (3 tone + 3 sweep pings), crossing | sweep pings 3 of 3 | (not run) | **0 of 3** (in the notch) | uncertain / - / uncertain |
| alternating, oblique | sweep pings 3 of 3 | (not run) | 3 of 3, 14.2 / 23.2 / 29.3, -0.537 / -0.477 / -0.505 | uncertain / - / **static** |
| alternating, closing | sweep pings 3 of 3 | (not run) | 3 of 3, 25.8 / 24.0 / 17.5, -5.602 / -5.582 / -5.595 | uncertain / - / uncertain |
| one ping of tone + sweep (train, 6.5 kHz), crossing | sweep 1 of 1, snr 24.5 (whole train as replica) | (refused on a train) | **0 of 1** (in the notch); sweep 1 of 1, snr 25.0 | (one ping) |
| train, oblique | **0 of 1** (whole-train replica, D3.1 item 3) | (refused) | 1 of 1, `both`, tone margin 8.5, -0.551 | (one ping) |
| train, closing | 1 of 1, snr 17.8 | (refused) | 1 of 1, `both`, tone margin 24.5, -5.598; sweep snr 18.5 | (one ping) |

Background (the sweep scene without targets, 4 pings, seed + 1000): at the default budget 0 false detections off and 0
with `cfar`; at 3 a ping 20 off and 7 with `cfar`. Cost a ping (median after the first): bank off 1.32 s (Python) /
1.00 s (C++), `cfar` with 45 hypotheses a beam 61.2 / 53.5 s; the alternating run 1.6 / 1.4 s a ping (tone pings make
no sweep map). Tone range errors (the tone's lag is coarse, c T / 2 = 750 m): closing +68, +28, +39 m; oblique +214,
+76, +39 m; the sweep's on the closing target -46 to -66 m (its bow highlight and the range-Doppler coupling).

**Against the planner.** `tools/popoto_depth_planner.py --build popoto_face_1x32 --environment shelf_100m_sand
--own-depth 9 11 --ts 25 --target-depth 49 51` (scratch, not committed; the shipped plan's figures for the class at this
site are +15.4 dB for 3 x 32 at 65 m and +11.0 dB worst over 30 to 99 m for 1 x 32 at 9 m): at the scenes' own depth
(10 m) and the targets' 50 m, **+13.5 dB of one row's SNR over the 15.8 dB the detector needs for Pd 0.5**, i.e. one row's
SNR 29.3 dB for one +25 dB point (noise-limited: 0.6 dB from its noise-only figure). What the planner puts in its 15.8 dB
and the chain's snr_db pays: the aperture window (1.8 dB) and the map's sampling (0.5 dB); so the planner predicts a
chain snr_db of 27.0 dB and a margin over the 13.06 dB threshold of **13.9 dB for one point of +25 dB**. Measured
(bank off): crossing 11.4 dB (2.5 dB under: inside 3 dB), oblique 9.2 (4.7 under), closing 8.0 (5.9 under). The two
gaps over 3 dB are the targets' structure, not the budget: seen bow-on (closing) or obliquely (oblique: 6.3 m between
highlights in range against the sweep's 0.38 m) the five highlights are resolved, each +18.0 dB, 7.0 dB under the
whole; per highlight the planner predicts 6.9 dB, and the chain gives 1.1 (closing) and 2.3 dB (oblique) more,
presumably from the neighbouring highlights and multipath arrivals within the detector's cells (not separated). Across
the line of sight (crossing) the highlights share one range cell and add; their phases set the ping-to-ping spread
(snr 22.1 to 26.6 dB). So 12 km is held for this class at this site through the chain on every ping, but with 3 to 6 dB
less margin than the planner's point target for an extended hull off broadside.

### LS.4 What it says

1. At this site the sweep alone detects all three on every ping; the background is noise (0 false detections at the
   default budget). The tone adds no detection here: it adds a **range rate** (within 0.06 m/s on the 8 tone looks at
   targets outside the notch: 6 alternating, 2 in the train ping), nothing for a target on its ridge, and false
   detections beyond +-32 degrees through the grating lobes (item 0).
2. The repaired bank adds 0.6 to 0.9 dB net on all three (the mismatch it recovers less its +0.78 dB correction for 45
   hypotheses a beam), and its rate is not a measurement.
3. What the tone path needs to be useful for this class: D5 (the tracker taking the tone's rate: today a 2 m/s target is
   called `static` on it), the bank and the tone on one train record, and the train scene through the chain at six
   pings (GPU job, LS.5).

### LS.5 Not run on this CPU: the GPU job

What was too slow here, as command lines for the GPU job (`GPUSONAR_JOB=doppler`, since added to `tools/gpu_job.sh` (D-241; the sub-run `ls12`), then by
another package; `$OUT` its output folder; Bellhop needed; each line its own step):

    python3 tools/doppler_tone_eval.py popoto_12km_large_submerged_tone --max-rate 8 --truth-at reflection --gate-range-m 100 --background 2 --cache "$OUT/ls12" --json "$OUT/ls12_tone_train.json"
    python3 tools/doppler_tone_eval.py popoto_12km_large_submerged --alternate 6.5e+3 --pings 12 --max-rate 8 --truth-at reflection --gate-range-m 100 --cache "$OUT/ls12" --json "$OUT/ls12_tone_alternate12.json"
    python3 tools/doppler_bank_eval.py popoto_12km_large_submerged --combine cfar --max-rate 8 --truth-at reflection --gate-range-m 100 --cache "$OUT/ls12" --json "$OUT/ls12_bank_cfar.json"

The third repeats this section's sweep-only run (27 min to simulate here, about 15 min of chains) as the check that the
GPU's records give the same table. The tone path's false detections through the grating lobes (item 0) will be in the
first two.

Follow-ups: the grating-lobe false detections of the tone path (item 0; a design choice first); D5 (the tone's rate into
the tracker and classifier); the sweep bank on a train record (both engines); a
background count for `--alternate`; several seeds and the class's high target strength near broadside (+41.5 dB);
`docs/handoff/README.md`, `handoff/SIMULATION.md` and `STATUS.md` still say 95 scenario files (96 since
`flank_patrol_doppler_tone`, 98 now).

### D8.5 What the first GPU suite found, and the repair (2026-11-27, branch `cuda-fix`)

**With the Doppler options off, programs on cuda stopped agreeing.** Four report-line comparisons that had been exact
failed on the L4 (batch5; late digits of `snr_db` and of a track's `speed_m_s`): `sonar_live --replay` against the
Holoscan program, the merged per-board streams against the whole-array stream (twice), and the program's ping count.
Cause, found by reading and confirmed with `nm` on objects compiled here: `doppler_cuda.cu` instantiated the kernel
templates `parallel_for_kernel<recon::BpKernel>` and `<FarfieldSumKernel>` (and the detector's three) for the device a
second time. Without relocatable device code, each `.cu` file registers its own device body for the same weak host stub,
and the program's link and load order picks one. `doppler_cuda.cu` is compiled with `--fmad=false`,
`backproject_cuda.cu` and `beamform_cuda.cu` are not, so the default chain's backprojection ran one body in one program
and the other in another. The repair: the shared kernels run on the device only through `launch_on_device` in their own
files (`backproject_cuda.cu`, `detector_cuda.cu`; the tone's beams through `farfield_cuda_device` in `beamform_cuda.cu`),
so each kernel template has one device instantiation in the whole build and the default chain's device code is the
code before D8. Guard: `test_doppler_device.py::test_no_kernel_template_is_instantiated_for_the_device_in_two_cuda_files`
(`nm` on every built `*.cu.o`; it fails on batch5's objects and passes now) and a CUDA-gated test of the default chain in
fresh processes. The device timing of D8 above (the coordinator's L4 run: flank_patrol C++ off / max / cfar 0.116 /
0.424 / 0.453 s) was measured with the bank's backprojection compiled with `--fmad=false`. It now runs as
`backproject_cuda.cu` compiles it, and was measured again after the fix (D-242: job 6b090cd56c5d61fff1888217; flank_patrol 0.115 -> 0.393 -> 0.418 s, off -> max -> cfar).

## Grating-lobe notch for the tone path (2026-11-27, branch `notch`, issue `sonar-bc5r`)

Appended; nothing above was rewritten. The owner (2026-11-27): "i guess implement the notching with the tone path but the
default is still not the tone path? lets make sure we have test coverage and docs for both and understand tradeoffs."
So the tone path (opt-in) gets the notch; the default chain (the plain sweep, every Doppler option off) is unchanged,
byte for byte; both paths' tests are listed below; the trade-offs are `docs/design/doppler_tradeoffs.md`. Every number
here is **ours** (simulated records made in the tests, the shared CPU, host backend) unless named otherwise.

### GL.1 What failed, got worse, is uncertain or was not run (read first)

1. **The diagnosis of LS.1 item 0 was right but incomplete.** Notching each beam's visible grating lobes (sin g =
   sin b + m lambda / d, |sin g| <= 1) removed the false detections at 32 to 40 degrees, and uncovered weaker ones at 24
   to 31.5 degrees whose lobe centre lies just BEYOND endfire: the lobe's skirt still takes in the endfire water, where
   the water is densest in sine (dtheta = ds / cos theta). The remedy therefore also notches the endfire ridge of every
   beam whose response towards endfire (the aperture's array factor at the tone, over its main lobe's) is at least
   `tone_lobe_floor_db` (-40 dB, ours). Its cost is the larger part of the table in GL.3.
2. **What is left: the far sidelobes of strong reverberation.** On a field of 2000 still scatterers all round (the test
   record), false tone detections without the lobe notch -> with it: **15 -> 0** in 6 pings at -35 dB each, **15 -> 2**
   at -30 dB, **11 -> 5** in 4 pings at -25 dB; at 5 kHz (no lobe) 0, 0 and 2. A strong field reaches every beam
   through its far sidelobes (-40 to -60 dB, Hann) at every rate within +-|v| (the own speed), which no notch about one
   ridge removes; the tone path before this package has the same at 5 kHz. Not fixed (follow-up: a leakage rule across
   beams, as 6b is along rate).
3. **The scenes were not rerun here.** `popoto_12km_large_submerged_tone` and the alternating run take 58 min a ping and
   50 min on this CPU. GPU baseline (the coordinator's run of 2026-11-27, Hugging Face job 6b08f68754072e8b7168e9c6,
   tree `d54deb9`; ours, not yet in the repository): 4 target-free pings of the train scene, at the default budget
   **0 false detections with the tone path off, 22 with it on (all 22 the tone's)**; at 3 a ping 14 and 37. Expected
   after this package: near 0 at the default budget, because the 3 to 5 a ping of LS.1 all lay at 32 to 40 degrees
   on the lobes' ridge (rates +-1.1 to 1.6 m/s), which are now not tested; what could remain is item 2's mechanism
   (the reverberation of that scene is weaker than the test's -30 dB field: unmeasured). Acceptance: `GPUSONAR_JOB=doppler
   GPUSONAR_DOPPLER_RUNS=ls12` (`tools/doppler_gpu_runs.sh`) on this branch.
4. **`flank_patrol_doppler_tone` is not affected.** Its platform is still: every lobe's ridge is the beam's own (0 m/s),
   so the lobe notch adds no cell (proved bit for bit on a still flank_32 record at 10.5 kHz, test (d)). Its 12 tone
   false detections in 4 background pings at 3 a ping are its budget (12); the 1 at the default budget (and D3.1 item 2's
   one) is not a grating lobe and is not explained. A flank array under way at 10.5 kHz has the defect (GL.3).
5. **Device code: no change, never run.** The device form takes the notch mask as it is (`ToneBankBatchedInput::notched`,
   made on the host by `tone_notch_mask`); the batched form on the host gives the loop's power, ratio and detections bit
   for bit with the lobe notch (test). The CUDA-gated tests (2 new) skip here; `nvcc` is not on this machine.
6. Limits of the rule: only a UNIFORM line takes a lobe notch (rows sharing coordinates count once; any other array
   none: `pitch` 0 in the plan); lobe directions are taken in the beam's half-plane, so a velocity ACROSS the line
   (crab) leaves the mirror half's ridge, 2 v_across cos g away, unnotched (small near endfire); omnidirectional
   elements (none configured: the lobe at the main lobe's level; a real element pattern lowers it and is not known:
   unconfirmed hardware, it would live in config).

### GL.2 The remedy (`libs/live/include/sonar/live/doppler.hpp` step 4b; both engines, both backends)

- `plan_tone` lists, per beam, every visible grating lobe and the endfire directions that pass the floor, each with
  its ridge -v . u_g and a notch by the beam's own rule at g (own-motion spread over the half-power width plus
  `tone_notch_cells` resolution cells, or `tone_notch_m_s` when given). `tone_notch_mask` makes the (beam, hypothesis)
  mask from the beam's notch and its lobes'; `tone_stage` (the loop and the batched form, host and device) takes it.
  The Python engine calls the same C++ function; only the option words are written twice (`gpusonar/doppler.py`).
- `doppler.tone_lobe_notch: auto | off` (default `auto`: on whenever the tone path is on and a lobe adds a hypothesis;
  `off` for comparison) and `doppler.tone_lobe_floor_db` (-40). Written to the C++ options and their JSON only when not
  the default.
- Said, not reported: the plan carries `lobes`, `lobe_hidden` (m/s per beam), `lobe_beams` and, when any, in its JSON
  `lobe_beams`, `lobe_hidden_max_m_s`; `sonar_live` prints it once on stderr ("tone grating-lobe notch"). The report
  line is unchanged (a contract: section 5, question 6).
- Arithmetic (GoogleTest `DopplerToneLobes`): the lobes at -34.75 degrees (one, 75.57 degrees at 1500 m/s), its mirror;
  pitch exactly half a wavelength (none inside +-90; at endfire steering the other endfire); a lobe exactly at endfire
  (pitch one wavelength: -90 and +90); several (2.5 wavelengths: sines +-0.4, +-0.8); just beyond endfire with and
  without slack; the pitch of a line (any order, two rows, not uniform); the mask, its hidden interval, a still
  platform, an unknown velocity, a non-uniform line; the options.

### GL.3 What it costs: the extra blind interval per beam (from `plan_tone`; platform 1.5 m/s along the line)

The lobe's ridge lies |v| lambda / d from the beam's in radial speed over ground (2.31 m/s on popoto_long_range at
6.5 kHz, 2.28 on flank_32 at 10.5 kHz): CLOSING targets on port beams (negative bearings), OPENING on starboard (the
mirror). Hidden: m/s of the beam's hypotheses notched by a lobe and not by the beam's own notch (+-0.38 m/s on
popoto_long_range at 6.5 kHz, +-0.25 on flank_32 at 10.5 kHz). Headings hidden: the share of all headings of a target at
that speed whose radial speed over ground falls in the lobes' notches (the beam's own notch excluded; the own notch
alone hides 12 % at 2 m/s and 3 % at 8 m/s at 6.5 kHz, LS.3). Inside +-22 degrees nothing is added. Ours; scratch
script on `core.live_tone_stage`, c 1500 m/s, `max_range_rate_m_s` 3 (8 gives the same rows).

| Array, tone | Bearing (deg, either side) | Visible lobe: direction deg, ridge m/s from the beam's +- notch | Endfire response, ridge +- notch | Hidden m/s | Headings hidden at 2 m/s | at 8 m/s |
| --- | --- | --- | --- | --- | --- | --- |
| popoto_long_range 6.5 kHz | 40 | 63.6, 2.31 +- 0.38 | -33.9 dB, 2.46 +- 0.35 | 0.87 | 8.5 % | 3.7 % |
| | 36 | 71.9, 2.31 +- 0.38 | -6.4 dB, 2.38 +- 0.35 | 0.75 | 8.5 % | 3.3 % |
| | 34 | 78.3, 2.31 +- 0.38 | -1.1 dB, 2.34 +- 0.35 | 0.81 | 8.5 % | 3.2 % |
| | 32 | none | -0.2 dB, 2.29 +- 0.35 | 0.69 | 7.2 % | 2.9 % |
| | 28 | none | -13.8 dB, 2.20 +- 0.35 | 0.69 | 12.1 % | 2.9 % |
| | 24 | none | -37.7 dB, 2.11 +- 0.35 | 0.69 | 15.6 % | 2.9 % |
| | 22 and less | none | under -40 dB | 0 | 0 | 0 |
| flank_32 10.5 kHz | 40 | 61.3, 2.28 +- 0.25 | -33.7 dB, 2.46 +- 0.21 | 0.68 | 0.0 % | 2.7 % |
| | 36 | 68.7, 2.28 +- 0.25 | -13.7 dB, 2.38 +- 0.21 | 0.57 | 0.0 % | 2.3 % |
| | 32 | 81.8, 2.28 +- 0.25 | -0.3 dB, 2.29 +- 0.21 | 0.50 | 0.0 % | 2.0 % |
| | 28 | none | -6.8 dB, 2.20 +- 0.21 | 0.43 | 3.2 % | 1.8 % |
| | 24 | none | -31.5 dB, 2.11 +- 0.21 | 0.43 | 10.3 % | 1.8 % |
| | 22 and less | none | under -40 dB | 0 | 0 | 0 |

At 2 m/s the lobe band (2.11 to 2.31 m/s +- the notch) is the target coming nearly straight in (or out): 7 to 16 % of
headings on popoto_long_range; at 8 m/s 3 to 4 %. A 2 m/s `large_submerged` closing bow-on at -36 degrees is not
reported by the tone; the sweep path still detects it (test (c)). The interval scales with the own speed (|v| lambda / d)
and vanishes on a still platform.

### GL.4 Tests (both engines; new files and additions)

`tests/python/test_doppler_tone_lobes.py` (11): the lobes from the geometry and the endfire rule against an array factor
computed in the test; (a) the field record: false tone detections with the lobe notch off, none with it (the stage and
the chain, both engines; the code before this package gives 2 on this record, both engines: fails before, passes
after); (b) a mover at -36 degrees outside every notch: the same detection and rate with and without; (c) a mover on its
beam's lobe ridge: not reported by the tone with the notch, reported without, and the sweep ping of the pair reports it
either way; (d) no visible lobe (5 kHz under way) and a still platform (flank_32, 10.5 kHz): auto equals off bit for bit
(stage power, ratio, detections, tests, factor; the report lines of both engines); the option. `test_live_cpp.py` (2):
the two engines equal with the notch and without; `sonar_live` on the recording gives the Python chain's lines and says
the notch once on stderr. `test_doppler_device.py` (3): the batched form on the host equals the loop with the notch bit
for bit; on the device (CUDA, skipped here) power and detections against the host, both engines. GoogleTest
`DopplerToneLobes` (3) and `Options.ToneLobeNotchKeysAreReadAndEchoedOnlyWhenSet`. `tests/python/test_doppler_paths.py`
(7): the coverage gaps below.

### GL.5 Default-off and no-lobe proof

sha256 of the report lines without processing_seconds (D3's method), host backend, a build of the base `d54deb9`
against this branch: flank_patrol and popoto_12km, 2 pings, defaults, `doppler: {mode: off}` and `range_doppler:
ownship`, both engines (12 of 12 equal, and equal to D8's flank_patrol hashes); the tone path on records with no lobe
added (flank_32 CW 8 kHz under way and still, flank_32 10.5 kHz still, popoto_long_range 5 kHz under way), both
engines' report lines and the stage's power, ratio and detections (12 of 12 equal). 24 of 24.

### GL.6 Both paths: what each does and the test that holds it

| Path | Behaviour | Held by |
| --- | --- | --- |
| plain sweep (default) | off and `mode: off` are the chain as before, byte for byte | `test_live_cpp.py::test_the_doppler_bank_is_off_by_default_and_off_is_as_before`; hash proofs (D3, D8, GL.5; scratch, not tests) |
| plain sweep | no Doppler field on the report line (sweep ping and train ping), no rate for a mover | **new** `test_doppler_paths.py::test_the_default_chain_reports_no_doppler_field_and_gives_a_mover_no_rate` |
| plain sweep | a train compressed whole (D-237: 0 of 6 on `popoto_12km_doppler`); the replica untouched | **new** `test_doppler_paths.py::test_the_default_chain_compresses_a_train_whole_and_pays_for_it`; `test_doppler_tone.py` train test |
| plain sweep | a short CW: the C++ chain refuses it, the Python chain maps nonsense (a divergence, D3.1 item 7) | `test_doppler_tone.py::test_a_cw_mover_is_missed_today_and_found_by_the_tone_bank_with_its_range_rate` |
| plain sweep | still and moving targets detected, tracked, the engines equal | `test_live_cpp.py` (the parity tests), `test_live.py` |
| plain sweep | `range_doppler` ownship / full | `test_live_cpp.py::test_range_doppler_*` |
| sweep bank (max, cfar) | plan, fold, threshold, both engines, batched and device | `test_doppler_bank.py`, `test_doppler_cfar.py`, `test_live_cpp.py` bank tests, `test_doppler_device.py`, GoogleTest `Doppler*` |
| tone | the matched filter of the tone per rate (against NumPy, 0.1 dB) | `test_doppler_tone.py::test_the_bank_is_the_matched_filter_of_the_tone_at_each_rate` |
| tone | plan: step, notch, allowance; a sweep ping has no tone path | `test_doppler_tone.py::test_the_plan_spaces_...`, GoogleTest `DopplerTone.ThePlan...` |
| tone | a CW mover found with its rate; a still target not given a Doppler (the notch) | `test_doppler_tone.py::test_a_cw_mover_...`, `test_a_still_target_is_not_given_a_doppler`, GoogleTest `DopplerTone.AMoverIsFound...` |
| tone | the leakage rule | `test_doppler_tone.py::test_a_strong_still_echo_leaks_no_detection_out_of_the_notch` |
| tone | **the grating lobes' and endfire ridges notched; the cost; nothing changed without a lobe** | **new** `test_doppler_tone_lobes.py`, GoogleTest `DopplerToneLobes` |
| tone | train: the sweep's range, the tone's rate; sweep-only replica; a train without its tone refused | `test_doppler_tone.py` train tests |
| tone | alternating pings, one tracker, the engines equal | `test_doppler_tone.py::test_alternating_*` |
| tone | the merge; the tone's range sigma for the tracker | GoogleTest `DopplerTone.TheMergeLends...`, `test_doppler_tone.py::test_a_detections_own_range_sigma_...` |
| tone | the budget on noise (loosened: 3 and 30 a ping) | `test_doppler_tone.py::test_on_noise_the_tone_path_stays_inside_its_budget` |
| tone | off by default, said only when on, unknown words refused | `test_doppler_tone.py::test_the_option_is_off_...`, **new** `test_doppler_tone_lobes.py::test_the_option_is_auto_...` |
| tone | refused with `range_doppler` (both engines) and `elevation` (both, and the C++ chain itself) | `test_doppler_tone.py::test_the_option_is_off_...`; **new** `test_doppler_paths.py::test_the_tone_path_is_refused_with_elevation_and_range_doppler_in_both_engines` |
| tone | batched on the host = the loop, bit for bit; the device = the host (CUDA, skipped here) | `test_doppler_device.py` tone tests (**new** lobe variants) |
| tone | `sonar_live` runs it like the Python chain | **new** `test_live_cpp.py::test_program_runs_the_tone_paths_lobe_notch_like_the_python_chain_and_says_it` |
| tone | the scene acceptance (`flank_patrol_doppler_tone`) | `test_doppler_scenes.py::test_the_chain_with_the_tone_path_sees_the_small_vehicle_and_keeps_the_wreck` (gated `GPUSONAR_DOPPLER_CHAIN=1`, not run here) |

Gaps not closed: the false-alarm rate at the DEFAULT budget is counted only in GPU runs, for the plain sweep and for the
tone path (no test can count 1e-3 a ping on a CPU); the far-sidelobe false detections of strong reverberation (GL.1
item 2) have no remedy and no test that asserts them; the tone path with `mode: sweep_bank` on the same chain (a
train's bank is refused, LS.1 item 6; an alternating pair with both on) has no end-to-end test; the mirror half's ridge
for a velocity across the line (GL.1 item 6) is untested; a multi-row array's plan is tested only through
`uniform_pitch`; the Python engine's CW nonsense (D3.1 item 7) is stated by a test, not fixed; the tracker does not use
the tone's rate (D5).

### GL.7 The attribution of false detections (`tools/doppler_tone_eval.py`, added after GL.1 to GL.6)

Every false detection on the background pings and every detection on the target pings that matches no target is saved
in the JSON with its bearing, range, rate, snr and path, whether its beam has a visible lobe or an endfire notch, whether
its rate lies within a lobe's notch, and its distance from its beam's own ridge and from the nearest lobe's ridge in notch
widths; kinds `lobe_ridge`, `notch_edge` (within 2 own-notch widths), `other`, `no_rate`. The geometry is the plan's
with the lobe notch auto whatever the run uses, so `--tone-lobe-notch off` says how many the lobes would explain. The
`ls12` sub-run of `tools/doppler_gpu_runs.sh` now runs the train scene twice on the same cached pings, the notch off
(`ls12_tone_train_lobe_off.json`) and on (`ls12_tone_train.json`): before and after in one job. Tests:
`tests/python/test_doppler_gpu_job.py` (the switch and the two passes; the kinds on a made geometry; on the lobe record
every false detection with the notch off is `lobe_ridge`, none with it).

### GL.8 Why the notch did not remove the train scene's false detections on the GPU (2026-11-27): the simulator

The coordinator's GPU run (Hugging Face job 6b0911926c5d61fff1888f01, branch `notch` at `f1c5d2d`, L4, cuda; ours):
`popoto_12km_large_submerged_tone`, 4 target-free pings, default budget: 22 false tone detections with the lobe notch
off, **25 with it on**. The cause is **not the chain**. It is the simulator.
- **The simulator's monostatic boundary reverberation shifts each azimuth sector by its Doppler at the record's centre
  frequency** (`python/gpusonar/sim/reverb.py`, `_series_numpy`: `doppler = 2 (v . u) fc / c`; the C++ generator the
  same), a narrowband frequency shift for the whole pulse. On the train (4.0 to 6.6 kHz, record centre 5.3 kHz) the
  6.5 kHz tone's still water lies at 5.3 / 6.5 = 0.815 of its physical ridge. The chain's ridges are the physical ones.
- Evidence, GPU: the 22 detections with the notch off lie on the lobe ridge scaled by 0.815 (rate less that ridge,
  mean -0.010 m/s, sd 0.061), not on the physical ridge (mean -0.276); with the notch on, the 25 lie 0.32 to 0.40 m/s
  from the physical ridge, at the notch's edge (+-0.38), where the scaled ridge's far side is cut. The alternating run
  (CW pings: the record's centre IS the tone, so the simulated Doppler is right) on the same job reports **0 false tone
  detections in 6 tone pings with the notch on**, on the device. Before the notch the CPU run had 3 to 5 a tone ping (LS.1).
- Evidence, host (scratch reproduction: popoto_long_range, 1.5 m/s along the line, a flat bottom 50 m below, the
  simulator's reverberation): the strongest rate of the beam at +-60 degrees is -1.33 / +1.33 m/s on a CW record (own
  ridge -+1.30) and -0.98 / +1.04 on the train (0.815 x 1.30 = 1.06); with the notch on the train gives 17 tone
  detections at +-1.0 to 1.12 m/s at 31 to 39 degrees (10 off), the GPU's pattern.
- The mask reaches every form: every detection's strongest cell lies outside every applied notch, by 0.07 to 2.5 steps
  on that record; its parabola moves its rate at most half a step inward, which is why the attribution of `f1c5d2d`
  (inside if within 1.0 notch width) called them "on a lobe's ridge". Now the attribution says `lobe_edge` for those,
  `DEFECT` for a rate inside an applied notch by more than half a step (none on any record here), and counts those
  on the ridge scaled by the record's centre frequency over the tone's.
- So on a TRAIN the scene's reverberation is wrong for the tone, and the tone path's false detections there measure the
  simulator, not the chain. The sweep's own reverberation on the train has the same approximation (4 to 6 kHz shifted
  at 5.3 kHz: at most 25 % off at the band's edges). Not fixed here: it moves every simulated moving-platform scene with
  boundary reverberation (a default result), so it is for the owner. The fix: the shift per frequency
  (2 (v . u) f / c, i.e. a time scaling), in the frequency domain where each sector's delay is already applied, in the
  NumPy and C++ generators and the CUDA sum; opt-in first, with before -> after on the shipped scenes.
- Tests: `test_doppler_tone_lobes.py::test_no_tone_detection_lies_inside_a_notch_applied_to_its_beam` (the field record
  and this reverberation train, both engines, loop and batched); the same on the device (CUDA-gated) and the batched
  inputs carrying the mask in `test_doppler_device.py`; the attribution's kinds and DEFECT in `test_doppler_gpu_job.py`.

## HFM: the hyperbolic sweep with the plain matched filter on a mover (2026-11-27, branch `txstudy`, issue `sonar-vjq7`)

Appended; nothing above was rewritten. The owner's baseline (2026-11-27) is the plain sweep with every Doppler option off,
and the transmitter is a free design choice. The trade study's transmit table found that the shipped linear sweep's peak
loss on a mover grows with the pulse as fast as its energy (section 0 of `docs/design/popoto_trade_study.md`). This
package asks whether a hyperbolic sweep (`transmit.waveform: {type: sampled, source: hfm}`, already in the simulator) makes
"plain filter, long pulse" work with no bank. Every number is **ours** (simulation; closed forms derived here); commands
and every field: `docs/tasks/hfm_doppler_eval.json`. No default moves; no chain file changed.

### HFM.1 What failed, is uncertain or was not run (read first)

1. **Not run through the 12 km eigenray scene.** The chain measurement is in FREE FIELD at 12 km (no multipath, no
   reverberation, Wenz noise), from a STILL platform, with both targets 20 dB above the class's low end (at +25 dB the
   free-field 1 s margin is about 0 dB, too little to measure a 20 dB loss). The scene itself (eigenrays, bottom, the
   1.5 m/s platform, the class at +25 dB), LFM and HFM, 1 s and 4 s, plain and bank: the GPU-job sub-run `hfm12`
   (HFM.6), not run.
2. **The HFM's cost is a range bias the plain filter cannot remove**: an up-sweep reads a closing echo nearer by
   f1 T / B per m/s, 3 T m per m/s at 4-6 kHz (measured on the point: -17 m at 1 s and 5.6 m/s, -67 m at 4 s, -191 m at
   8 s and 8 m/s). The plain filter gives no range rate to correct it. The linear sweep is no better: its coupling is
   f_c T / B (5/6 of the HFM's: -20 m at 1 s and 8 m/s, measured -20; -168 m at 8 s) and its peak is smeared besides;
   only the sweep bank gives a (coarse) rate, and it refuses an HFM (it takes a linear sweep only, `plan_doppler`).
3. **The hull's range errors are not interpretable** (the strongest of its five highlights, 0 to 100 m behind the bow,
   varies from ping to ping); the point's are the bias. The hull's relative margins are.
4. One seed; 3 pings a case (2 with the bank at 1 s; 1 at 2 and 4 s); the HFM's relative margin scatters by up to
   0.8 dB (8 s, 2 m/s: -0.8 on the point) about the closed form's 0.0 to 0.2 dB.
5. Not measured: reverberation (the HFM separates nothing from it, as the LFM's bank does not: section 3.3 item 5),
   the surface's Doppler spread, opening targets (closed form: smaller loss, 0.05 / 0.19 dB at 2 / 8 m/s), several
   targets per beam, the false-alarm rate (unchanged by construction: same band, same detector), a GPU.
6. The projector: the HFM puts its energy density as 1 / f^2 across the band (2.6 dB more at 4.25 kHz than at
   5.75 kHz, measured on the 1 s pulse); whether a projector sends that at the same peak level is not known (no
   projector is chosen), and the planner has no waveform shape (the trade study's zero-speed margins are the LFM's).

### HFM.2 Theory (ours, derived; `gpusonar.doppler.hfm_doppler`, `hfm_range_factor`; `tests/python/test_hfm_doppler.py`)

An echo of range rate r (opening positive) is the pulse time-scaled by eta = (c - r) / (c + r) (the simulator's exact
model, tested above). The **linear sweep** keeps a quadratic phase error of pi B T |r| / c at its ends after the delay
takes the linear term (D4, `tests/python/test_doppler_bank.py`): it grows with B T, so a longer pulse loses more. The
**hyperbolic sweep** has phase phi(u) = ln(1 + k f0 u) / k, k = (1/f1 - 1/f0) / T, and

    phi(eta t) = ln(eta) / k + phi(t + tau),    tau = (1/eta - 1) f1 T / (f0 - f1)     (exact)

so the scaled echo IS the pulse advanced by tau, in phase, over the band both share. Hence (4-6 kHz, c = 1500 m/s):

| | 2 m/s closing | 5.6 m/s | 8 m/s | grows with T? |
| --- | --- | --- | --- | --- |
| HFM peak loss, untapered: 1 + (1/eta - 1) f1 / B (the band lost at the top) | 0.07 dB | 0.20 dB | 0.28 dB | no |
| HFM peak loss, Tukey 0.1 (the simulator's pulse) | 0.02 dB | 0.11 dB | 0.18 dB | no |
| HFM range bias, - c tau / 2 = r c / (c - r) f1 T / B (3 T m per m/s) | -6 T m | -16.7 T m | -23.9 T m | yes |
| LFM peak loss, -20 log10 M(pi B T v / c), at 1 / 2 / 4 / 8 s | 8.8 / 11.7 / 14.8 / 17.8 dB | 13.2 / 16.4 / 19.3 / 22.4 | 14.8 / 17.8 / 20.9 / 23.9 | yes |
| LFM range coupling, f_c T / B (2.5 T m per m/s) | -5 T m | -14 T m | -20 T m | yes |

Checked by direct correlation of the time-scaled pulse with the unscaled replica at continuous lags (1, 2, 4, 8 s;
taper 0 and 0.1; +-2, 5.6, 8 m/s): the peak within 0.02 dB of the closed form, its lag within 10 microseconds (7.5 mm).
Source: ours, derived (the identity and both closed forms); the general statement that the HFM is "optimally Doppler
tolerant" is the GSFM dissertation's (arXiv:1809.11009, Ch. 2) as `docs/design/projector_selection.md` [A2] records it:
via secondary (not reopened here). What the HFM costs at zero speed (1 s, Tukey 0.1, the chain's unwindowed filter;
same test): -3 dB main lobe 0.359 m against the LFM's 0.352 m (2 % wider), first sidelobe -12.7 dB against -13.3 dB,
the 1 / f^2 spectrum (item 6); and no range rate.

### HFM.3 The simulator and the chain on an HFM record (no defect found; nothing changed)

- Simulator: a plain HFM's echo of a point closing at 8 m/s is the pulse time-scaled by (c + v) / (c - v) to 7.7e-8
  relative rms in BOTH engines (C++ `pair_sampled` and NumPy), and the unscaled pulse advanced by tau to 1e-5 inside the
  shared band; C++ against NumPy on a 1 s HFM, 32 channels, moving platform, absorption: 4.3e-9. (The existing Doppler
  tests covered a sampled burst pair, codes and trains; this is the plain HFM.)
- Record: an HFM is type Arbitrary with its real replica and rate in the existing fields (frozen schema; no change).
- Chain: both engines compress through `core.pulse_compress` (`recon.prepare`), whose `build_arbitrary` builds the filter
  from the RECORD's replica, not an LFM from `f_start` / `f_stop`. Proof (test): at zero speed the HFM's detection is
  58.04 dB against the LFM's 58.00 dB of the same energy (a linear replica on an HFM echo would lose tens of dB); the two
  engines' reports agree to the C++ engine's printed digits (1e-6 relative). The only waveform branches in the chain are
  the bank and the tone path (Doppler options, refused or not taken for an Arbitrary record) and the `range_doppler`
  correction (0 for anything but an analytic LFM; off by default). So no fix, and no default-off proof is needed.

### HFM.4 Measured through the chain (`tools/hfm_doppler_eval.py --pulses 1 2 4 8 --pings 3 --ts-offset-db 20`; both engines equal on every case)

Free field, 12 km, still platform; `point` +45 dB at -10 degrees and `hull` 5 x 38.01 dB bow-on at +10 degrees, both
closing along their lines of sight; each waveform at 197 dB rms, so the energy rises with T and is the same for LFM and
HFM at each T. Each cell: margin change against the same waveform, pulse and energy at zero speed, point / hull (dB),
then the closed-form peak loss; worst first. Zero-speed margins (point / hull): 1 s 18.9 / 12.1, 2 s
22.1 / 15.4, 4 s 24.9 / 18.1, 8 s 28.0 / 21.0 dB (LFM; HFM within 0.2 dB of them).

| Pulse, waveform | 2 m/s | 5.6 m/s | 8 m/s | Point's range error at 2 / 5.6 / 8 m/s (m) |
| --- | --- | --- | --- | --- |
| 8 s LFM | -15.7 / -12.1 (17.8) | -19.7 / -13.0 (22.4) | **-22.0 / -14.0** (23.9) | -36 / -97 / -168 |
| 4 s LFM | -12.5 / -12.3 (14.8) | -16.5 / -11.7 (19.3) | -18.1 / -11.9 (20.9) | -19 / -51 / -73 |
| 2 s LFM | -9.9 / -9.0 (11.7) | -13.9 / -12.0 (16.4) | -14.9 / -10.9 (17.8) | -10 / -26 / -36 |
| 1 s LFM (shipped) | -7.1 / -6.5 (8.8) | -10.5 / -10.1 (13.2) | -12.3 / -10.8 (14.8) | -5 / -13 / -20 |
| 1 s LFM, bank `combine: cfar` to 8 m/s (45 a beam), against the plain zero-speed margin | -0.2 / -0.6 | -0.5 / -0.7 | -0.6 / -0.5 | -5 / -14 / -20 |
| 8 s HFM | -0.8 / +0.0 (0.0) | -0.1 / +0.5 (0.1) | +0.2 / +0.4 (0.2) | -48 / -134 / -191 |
| 4 s HFM | -0.2 / +0.1 (0.0) | -0.4 / +0.3 (0.1) | +0.2 / +0.4 (0.2) | -24 / -67 / -95 |
| 2 s HFM | +0.2 / +0.3 (0.0) | -0.0 / +0.6 (0.1) | +0.5 / +0.5 (0.2) | -12 / -33 / -47 |
| 1 s HFM | -0.0 / +0.2 (0.0) | -0.1 / +0.8 (0.1) | +0.0 / +0.7 (0.2) | -6 / -17 / -24 |

The bank at 2 and 4 s (`tools/hfm_doppler_eval.py --pulses 2 4 --waveforms lfm --pings 1 --ts-offset-db 20 --bank --engines
cpp --bank-engines cpp`: ONE ping a case, C++ only), against the plain zero-speed margin of the same ping draw (point /
hull, dB):

| Pulse, LFM with `combine: cfar` to 8 m/s | 0 m/s | 2 m/s | 5.6 m/s | 8 m/s | Point's range error at 2 / 5.6 / 8 m/s (m) | Cost a ping |
| --- | --- | --- | --- | --- | --- | --- |
| 2 s (87 hypotheses a beam) | -0.9 / -0.9 | -1.4 / -1.0 | -1.6 / -0.8 | -0.9 / -0.7 | -10 / -28 / -40 | 87 s |
| 4 s (173 a beam) | -1.0 / -1.0 | -0.8 / -0.8 | -1.3 / -0.7 | -0.8 / -0.9 | -20 / -56 / -80 | 188 s |

So the bank holds the LFM's margin on a mover at every length tried, at 0.2 to 1.6 dB under the plain zero-speed margin
(its threshold correction and the maximum's background; 0.2 to 0.7 at 1 s) and at 45 to 180 times the plain chain's cost
on this CPU; the
HFM holds it with the plain filter at the plain chain's cost, and within the scatter of the zero-speed margin. The bank's
range is the LFM's coupling uncorrected (`range_doppler` off): f_c T / B per m/s, as the plain HFM's is f1 T / B.

The chain's detector integrates over cells, so the linear sweep loses 1.7 to 9.9 dB less than its peak's closed form (most
on the hull, whose smeared highlights share cells), but the loss still grows with T: **on a mover the linear sweep's
margin stays where the 1 s pulse's is, whatever its energy** (point at 8 m/s: 6.6, 7.2, 6.8, 6.0 dB at 1, 2, 4, 8 s,
while the zero-speed margin rises 9.1 dB). The HFM keeps the zero-speed margin within the scatter at every T and speed,
so its energy is bought on movers too. Cost a ping (C++, host, this CPU): plain 0.9 to 1.2 s for every pulse and either
waveform; the 1 s bank 42.5 s (45 hypotheses a beam).

### HFM.5 The trade study (`tools/popoto_trade_study.py --report`, section 0a, derived; old sections byte-identical)

Each transmit option's planner margin less the closed-form peak loss closing at 2 and 8 m/s, LFM and HFM. Least energy
that holds both `large_submerged` and `ship` at 12 km on a mover with NO bank, deep water with the apl_1994 bubble law
(without it in brackets where different):

| Sea state | LFM at 2 m/s | LFM at 8 m/s | HFM at 2 m/s | HFM at 8 m/s |
| --- | --- | --- | --- | --- |
| 2 | 197 dB (1 s 197) | 203 dB (1 s 203 only) | 197 dB (1 s 197, 2 s 194) | 197 dB |
| 3 | none in the grid (203 dB, 1 s 203) | none | 197 dB | 197 dB |
| 4 | none | none | 203 dB (1 s 203, 2 s 200, 4 s 197) [200 dB] | 203 dB [200 dB] |
| 5 | none | none | 206 dB (2 s 203, 4 s 200, 8 s 197) [203 dB] | 206 dB [206 dB: the 0.28 dB takes `large_submerged` under at 203] |
| 6 | none | none | 212 dB (8 s 203) | 212 dB |

So with the plain filter the HFM needs the same energy as a still target (section 0's least energy) at every sea state
but one marginal cell; the LFM holds an 8 m/s mover only at sea state 2 and 203 dB, and at sea state 3 and above no
option in the grid holds a mover at 2 m/s with the bubble law. The planner's margins are for the PEAK; the chain lost
less with the LFM (HFM.4), so the LFM columns are pessimistic by 2 to 10 dB, the HFM columns are not.

### HFM.6 Not run on this CPU: the GPU job (`GPUSONAR_JOB=doppler`, `GPUSONAR_DOPPLER_RUNS=hfm12`)

`tools/doppler_gpu_runs.sh` sub-run `hfm12` (only when named; default limit 150 min, `GPUSONAR_DOPPLER_TIMEOUT_HFM12`):
`tools/doppler_bank_eval.py popoto_12km_large_submerged --max-rate 8 --background 2 --truth-at reflection --gate-range-m
250 --backend cuda --sim-backend cuda --cache $OUT/cache` with, in order: HFM 1 s `--off-only`; HFM 4 s `--off-only`;
HFM and LFM 4 s at 190.98 dB rms `--off-only` (the 1 s pulse's energy, 196.72 dB re 1 uPa^2 s at 1 m); LFM 1 s and 4 s
`--combine cfar` (plain and bank; 45 and 173 hypotheses a beam). The 4 s runs at 197 dB carry +6.02 dB of energy over 1 s.
The HFM is set by `--set sonar_overrides.transmit.waveform.type=sampled --set ...source=hfm`, the length by `--set
...duration=4.0` (new opt-in switches of `doppler_bank_eval.py`: `--set`, `--backend`, `--sim-backend`, `--off-only`).
Its quick host form (one free-field ping each) runs in `tests/python/test_doppler_gpu_job.py` (about a minute). Risk: a
sampled 4 s pulse through the eigenray reverberation is the slow case on a CPU (D3.1 item 1); on the device not timed.

### HFM.7 What it says

1. **Yes**: with the plain filter an HFM of 4-6 kHz holds its zero-speed margin on a target closing at 2 to 8 m/s, at
   1 s and at 4 s: within -0.4 to +0.8 dB through both engines (-0.8 to +0.6 at 2 and 8 s), 0.02 to 0.18 dB by the closed form,
   independent of the pulse. The shipped linear sweep loses 6.5 to 12.3 dB at 1 s and 11.7 to 18.1 dB at 4 s on the same
   targets; the bank recovers it at 45 times the cost and loses 0.2 to 0.7 dB to its threshold correction (2 and 4 s: 0.7
   to 1.6 dB, at 87 and 180 times).
2. What the HFM costs: the range bias (3 T m per m/s, uncorrectable without a rate; the LFM's coupling is 2.5 T and also
   uncorrected with no bank), its 1 / f^2 spectrum on an unknown projector, 2 % in resolution and 0.6 dB in the first
   sidelobe, no range rate and no bank. It needs no chain change and no schema change.
3. For the owner: "plain filter, long pulse" becomes a valid design with an HFM; with a linear sweep it is not (a longer
   pulse buys nothing on a mover). Not a default: the shipped sweep stays linear until the owner chooses.

Follow-ups: the `hfm12` GPU run (the scene, the class's own strength, eigenrays); the projector's response across the band
for an HFM (question for the projector selection); a range-rate source for the HFM's bias (the tone path, D5) or the bias
stated in the report; the planner's waveform-blind zero-speed margin (the HFM's spectrum and absorption); an opening and a
crossing target through the chain; several seeds.

### HFM.8 The `hfm12` run on the device (appended 2026-11-27; D-247; nothing above rewritten)

HFM.1 item 1 and HFM.6 say the 12 km scene was not run. It has been: Hugging Face job
6b091ba66c5d61fff188a75c, branch `txstudy` at `9bb1b62`, NVIDIA L4, simulator and chain on the
device, C++ engine, `popoto_12km_large_submerged`, 6 pings, 2 background pings; the JSON is in the
approved dataset under `doppler_hfm12` (not committed). Ours; one seed. These records carry the
simulator's centre-frequency reverberation Doppler (D-249).

What it does not show, first: a noise-limited scene. This site is limited by reverberation, so four
times the energy buys 1 to 2.4 dB and the hyperbolic sweep's advantage of HFM.4 (free field, a
point, +20 dB targets) shrinks to about 1 dB on the closing hull.

| Transmission (plain filter unless said) | closing, -5.6 m/s | oblique, -0.5 m/s | crossing, -0.39 m/s | false detections, 2 background pings, default budget |
| --- | --- | --- | --- | --- |
| linear 1 s, 197 dB (shipped) | 7.98 dB | 9.22 dB | 11.41 dB | 0 |
| hyperbolic 1 s, 197 dB | 8.83 | 8.79 | 11.63 | 0 |
| linear 4 s at the 1 s energy (190.98 dB) | 7.20 | 9.17 | 10.37 | 0 |
| hyperbolic 4 s at the 1 s energy | 8.59 | 8.95 | 11.49 | 2 |
| linear 4 s, 197 dB | 9.17 | 11.37 | 12.48 | 0 |
| hyperbolic 4 s, 197 dB | 10.40 | 11.20 | 13.35 | 0 |
| linear 1 s with the bank, `combine: cfar` | 8.78 | 10.08 | 12.00 | 0 |
| linear 4 s with the bank, `combine: cfar` | 10.83 | 12.33 | 13.60 | 0 |

Every target is detected on 6 of 6 pings in every row. The 2 false detections of the hyperbolic
4 s at the 1 s energy are not explained. The bank's range rates on these hulls scatter by several
m/s (as D-232 found). Next: the deep site at sea state 4 or 5 with both sweeps (issue sonar-rctz).

### GL.9 Counted again with the corrected simulator (appended 2026-11-27; D-251, D-252; nothing above rewritten)

GL.8's counts were made with the simulator's centre-frequency reverberation Doppler. With the
per-frequency law the default (D-251) and the notch, on the device (job 6b0947c354072e8b71697a5a,
tree `81364e0`, `popoto_12km_large_submerged_tone`, 4 target-free pings, default budget, both
engines): tone path off 1 false detection; on with the lobe notch off 17, all on a lobe's ridge; on
with the notch (auto) 0. At 3 a ping: off 16; on with the notch 25, of which 3 are the tone's. Not
shown: more pings, the far-sidelobe case on a real scene.

## D5: tracks use the measured range rate (2026-11-27, branch `d5rate`, issue `sonar-evd4`; opt-in, both engines)

### D5.1 What failed, is uncertain or was not run (read first)

1. **The defect is not shown removed on the scenes that showed it.** `flank_patrol_doppler_tone` (D-241) and
   `popoto_12km_large_submerged` with alternating pings (D-243) need eigenray simulation that does not finish on this
   CPU, and no cached records of them exist here. The GPU job's `tone` and `ls12` sub-runs now take a second pass with
   the option on (section D5.4); not run.
2. **Free-field forms do not reproduce it** (labelled: `--set fidelity=free_field --set reverberation=[]`, not a
   measurement of the scenes). `flank_patrol_doppler_tone` free field, 6 pings, both engines: with the tone path on the
   small vehicle is already a mover from ping 3 (0-based) without the rate (the line fit suffices at its free-field
   signal), and the submarine too; with `range_rate: update` every class, ping by ping, is the same (the fit's mover
   rule comes first, so the reported velocities are the fit's too); the wreck static in both. The 12 km scene free field
   with alternating pings detects its targets on 0 to 2 of 12 pings and tracks none: says nothing either way.
3. **Where it is shown (ours, synthetic):** the chain on alternating tone and sweep pings of `test_doppler_tone.py`
   (flank_32, a 1.56 m/s mover at 600 m, 2 s pings): without the rate the one track is static on pings 4 to 7, with it a
   mover on the same pings, both engines, still platform and under way at 1.5 m/s. The tracker alone (a 2 m/s target at
   3 km, -1.2 m/s along the line of sight, 12 s pings, range sigma 20 m, bearing 1 degree, rate sigma 0.04 m/s, 20
   seeds): first a mover on ping 9 to 14 (median 12) without the rate, on ping 4 (min_points) on 18 seeds and 5, 6 on
   the others with it.
4. **The 12 km oblique target may stay static even with the rate**: its radial part is 0.50 m/s, below `min_speed`
   (1.0 m/s), and the rule tests |v|, whose part across the line of sight comes from the positions (about 200 m of
   bearing error at 12 km). Expected from the design; to be seen on the GPU pass.
5. **Not measured:** what `update_and_gate` does on a scene (it is tested on a made case: a wrong rate's pair refused, a
   new track born); false movers from an overconfident covariance on extended targets; the navigation accuracy (the
   0.05 m/s default is assumed, unconfirmed). The design adds one classifier rule (D5.2: the existing rules cannot read
   the filter).
6. Off is the chain as before: 10 of 10 report hashes (processing_seconds removed) equal a build of the base
   `c77190e`, with the defaults and with `tracker.range_rate: off` said explicitly: `flank_patrol` and `popoto_12km`
   (3 pings each, defaults), the tone path on `popoto_12km_doppler` (3 pings) and on alternating synthetic pings (8,
   still and under way), each in both engines.

### D5.2 Design

- **Frame.** The tracker's state is (x, y, vx, vy) in its horizontal, world-fixed frame: x across = east (the record's y),
  y forward = north (the record's x), metres from the array's centre at the first ping, velocities over ground
  (`live.py` and `chain.cpp` build each tracker detection there; unchanged).
- **The measurement** (ours, from the geometry). A detection's `range_rate` is the bank's, relative to the platform,
  opening positive (a still scatterer reads -v_o . u: `doppler.hpp`). With the track's position p, velocity v, the
  platform's position o (the array's centre at the ping) and velocity v_o (the mean of the transmit and receive poses'
  velocities from navigation, the one the bank centres its hypotheses on), all in the tracker's frame:

      z = h(x) + e,   h(x) = u . (v - v_o),   u = (p - o) / |p - o|

  Its Jacobian (an extended Kalman update) is H = [w_t / R, u], R = |p - o|, w = v - v_o, w_t = w - u (u . w) (the
  part across the line of sight). The vertical is left out: the slant's elevation (cos 0.5 degree = 0.99996) and
  v_z u_z. The still target seen from a moving platform is the test of the sign: z = -u . v_o, h = -u . v_o at v = 0,
  no innovation; read as over ground (v_o withheld) the same rate makes the track a mover or breaks it
  (`test_tracker_range_rate.py`).
- **Its variance.** sigma^2 = step^2 + sigma_nav^2: `range_rate_step` (the tone bank's hypothesis step c / (4 f T),
  half the resolution; the parabola refines within it, and the errors measured on the scenes stayed within one step:
  1.964 to 1.997 against 1.996 m/s at a step of 0.036 m/s, D-241; -0.454 to -0.551 against -0.50 at 0.058, D-243)
  and `tracker.range_rate_nav_sigma_m_s`, the own ship's velocity error along the line of sight (default 0.05 m/s: an
  assumed navigation accuracy, unconfirmed; the hardware's goes there; D5's plan asked for better than 0.075). Left
  out: the track's bearing error times its speed across the line of sight in u (the EKF's position part of H carries
  it to first order), the bank's quantisation correlation from ping to ping, and the clock (DOPPLER.md D7: 10 ppm is
  0.0075 m/s at the scale of these numbers).
- **Which rates are trusted.** Per detection: `Detection.range_rate_sigma` (both engines), NaN = not used. The chain
  sets it only with `tracker.range_rate` on, for a finite rate and step, and with `range_rate_from: tone` (default)
  only for `doppler_path` tone or both: the tone bank's. The sweep bank's (`doppler_path` sweep with `mode:
  sweep_bank`) scatter by several m/s on extended hulls (D-232) and are not a measurement; `range_rate_from: any`
  takes them too (not recommended). A ping without a finite velocity gives no rate.
- **The update** (sequential: the noises of the position and the rate are independent). The position update as before,
  then a scalar EKF update linearised at the updated state: y = z - h(x), S = H P H^T + sigma^2, K = P H^T / S,
  x += K y, P = (I - K H) P (the position update's own form). A new track takes its first detection's rate too.
- **Gating** (`update_and_gate`, opt-in within the option, since a wrong gate loses tracks): a (track, detection)
  pair whose rate innovation at the predicted state has y^2 / S >= gate^2 (the tracker's gate, 4 sigma) is no
  candidate. S is the rate's marginal; its correlation with the position's innovation is left out (ours). It only
  vetoes: the association's cost is the position's, as before.
- **The classifier: why a rule.** Every existing rule reads the line fitted through the track's measured positions
  (D-157): its speed and sigma come from the positions alone, so a better filter velocity cannot reach them (the
  filter's state is used today only to move a track shorter than three points). Hence one rule, `mover_range_rate`,
  active only with the option on: a track that took a rate (`rate_updates` > 0), whose filter velocity v and covariance
  P (state[2:4], cov[2:4, 2:4]) give v^T P^-1 v >= significance^2 (25: 2 degrees of freedom, 3.7e-6 for a still target
  if P is right) at |v| >= min_speed and seen on min_seen of its pings (min_points and the fit limit as for any track),
  is a mover where the fit does not already make it one; its reported velocity, speed and speed sigma are then the
  filter's (sqrt(u^T P u), u = v / |v|). The static rules are untouched: a still target with the rate keeps the fit's
  class (tested). Risks: an overconfident P (highlights jumping on a hull, process noise 0.05) gives false movers;
  `min_speed` reads |v|, whose part across the line of sight comes from positions, so a slow radial component (the 12 km
  oblique target: -0.50 m/s) alone does not reach min_speed 1.0 m/s.
- **Taps** (opt-in, both engines; with the option off they are as before): the tracker tap gains `range_rate` and
  `rate` [{track, detection, z, predicted, innovation, s, sigma}], and with `update_and_gate` each pair `rate_d2`; the
  classifier's evidence `rate_updates`, `filter_speed`, `filter_chi2` and the rule name `mover_range_rate`.
- **Unchanged:** no record field, no report-line field (a track's kind, velocity and speed change only with the option
  on), every default. The tracker and the classifier run on the host in both engines (`detect.py`, `tracker.cpp` with
  Eigen, called after the detector in `chain.cpp`); no `.cu` file changed.
- **Keys:** `chain.tracker.range_rate: off | update | update_and_gate`, `range_rate_from: tone | any`,
  `range_rate_nav_sigma_m_s` (`docs/CONFIGURATION.md`; `sonar_live` `tracker_range_rate*`).

### D5.3 Tests (both engines; each fails on the branch's base `c77190e`: 29 of 29 in the new file)

`tests/python/test_tracker_range_rate.py`: the measurement and its Jacobian (numerical); the closed-form covariance
(a still target ahead at 3 km, noise-free, no process noise, six pings 12 s apart, range sigma 2 m, rate sigma
0.06 m/s: the velocity sigma along the line of sight 0.0398 -> 0.0209 m/s, across 1.034 m/s either way, to 1e-9, both
engines; GoogleTest `Tracker.RangeRateGivesTheClosedFormVelocityVarianceAlongTheLineOfSight` the same); the velocity
error with noise (40 seeds: more than three times smaller along the line of sight, within 1.5 times the covariance's);
pings to a mover (D5.1 item 3); a still target stays static (never a mover; the fit's class on every ping); a still
target from a platform at 1.5 m/s stays static, and a naive form (the rate read as over ground) is wrong on at least
18 of 20 seeds (measured 20: a mover on 5, uncertain or broken on 6, a static track that lost detections on 9); an
untrusted rate (NaN sigma, the bank's scatter) changes nothing, bit for bit; `update_and_gate` refuses a wrong rate's
pair; the taps (the rate's members only with the option on); the two trackers and classifiers equal (1e-8, three modes,
both associations); off bit for bit as a tracker given no rate; the chain (both engines; alternating pings, still and
moving platform) static -> mover; the two chains' report lines and taps equal with `update` and `update_and_gate`;
which rates the chain trusts; the keys (said only when on, refused when wrong, in the binding and in `sonar_live`'s
options). `test_doppler_gpu_job.py`: the eval tool's switch and the second passes.

### D5.4 The GPU job (not run)

`GPUSONAR_JOB=doppler GPUSONAR_DOPPLER_RUNS=tone,ls12` (tools/gpu_job.sh with GPUSONAR_ORACLES=1 for Bellhop): the `tone`
sub-run runs `flank_patrol_doppler_tone` as before (-> `doppler_tone_eval.json`) and then again on the same cached
pings with `--tracker-range-rate update --background 0` (-> `doppler_tone_eval_rate.json`); `ls12` adds the same second
pass to `popoto_12km_large_submerged_tone` and to the alternating 12 pings of `popoto_12km_large_submerged`
(`ls12_tone_train_rate.json`, `ls12_tone_alternate12_rate.json`; its default limit 80 -> 110 min). Each JSON gives per
target the class ping by ping (`kinds`), `first_mover_ping`, and the last ping's speed, true speed and velocity error:
the before -> after of D-241's small vehicle and D-243's oblique target in one job.

## Deep site at 12 km: the transmit table's noise-limited claims through the simulator and the chain (2026-11-27, branch `deepsea`, issue `sonar-rctz`; prepared, the device run not yet made)

Appended; nothing above was rewritten. The trade study's transmit table (`docs/design/popoto_trade_study.md` sections 0
and 0a; D-247) rests on the deep, noise-limited site, but the only real-scene run (HFM.8) was the reverberation-limited
shelf. The owner has since set the requirement's sea state (D-255): the **baseline** is the shipped face (1 s, 197 dB)
up to sea state 3; an **alternate**, `configs/builds/popoto_face_3x32_ss5.yaml` (the same face with 4 s at 200 dB, the
least energy with a single listed projector that holds both classes at sea state 5 with the bubble law: +2.0 / +2.9 dB
by the planner), at sea state 5. This section is the acceptance check of both. Every number is **ours**.

### DS.1 Where the scene and the planner differ, and what is not known (read first)

1. **The ambient noise is the planner's at the band centre and not across the band.** Both are Wenz at the site's wind
   and shipping 0.5 (`tests/python/test_deep12_scenes.py`, to 1e-9 dB at 5 kHz, sea states 3, 4, 5 and 5 m/s). The
   simulator's spectrum is +1.62 dB at 4 kHz and -1.37 dB at 6 kHz against the centre; its band mean is +0.13 dB, the
   planner's flat: the scene is 0.13 dB noisier.
2. Absorption is the planner's at 5 kHz (Ainslie-McColm at 12 C, 50 m: equal to 1e-12); across the band two-way at
   12 km 5.96 / 7.79 / 9.96 dB at 4 / 5 / 6 kHz; on the compressed peak that is +0.02 dB against the centre's figure.
   Sound speed: 1497.6 m/s in the scene (Mackenzie at 12 C, 35 PSU, 50 m) against the planner's 1500.
3. The surface: the rough surface's height and the bubble loss at a bounce are the planner's (`lloyd.Surface`; the
   bubble loss from the simulator's table within 2 %). The planner's peak keeps only the coherent part; the simulator
   adds the scattered rest as later taps (`time_spread.surface`), which the chain's detector may collect.
4. **The site's `bubble_model` does not reach the simulator by itself**: the scene states it again in
   `time_spread.surface.bubble_loss_db` (the test holds the two equal). Changing that would change the simulator's
   reading of every environment with a bubble law (none shipped); not done.
5. The detector: the chain's single-sample threshold in this 11.3-12.7 km window is 9.45 dB (three rows added in power,
   `--chain-set elevation={}`), the planner's 9.89 dB for the build's 0.75-12.7 km window: `tools/deep12_eval.py`
   predicts the chain's margin against the threshold the chain used.
6. Geometry: one own depth, 225 m (the planner's for `ship` at both deep sites; for `large_submerged` alone it
   recommends 245 m at 5 m/s, 395 m at sea state 5, 445 m at sea state 3); the hulls at 50 m, the ship at 4 m. The
   planner's figure at the scene's geometry is not the table's (worst over the class's depths): at sea state 3 the point
   +2.74 against the table's +2.36 dB, the ship +9.01 against +3.32 dB (a near-surface target in a constructive lobe).
7. Not in the planner: the hulls' five highlights resolve in range off broadside (DOPPLER.md LS.3: 3 to 6 dB under a
   point); the chain's multiscale detector sums cells, so the linear sweep loses less on a mover than the closed form.
8. The bank with rows runs the per-hypothesis path, not D8's batched one (which takes one row): its cost on the device
   with 3 rows and 173 hypotheses a beam at 4 s is not measured.

### DS.2 The scenes (`configs/scenarios/`, opt-in, new files)

`popoto_12km_deep` (the planning site `deep_no_bottom_return` by name) and `popoto_12km_deep_sea_state_5`
(`deep_no_bottom_return_sea_state_5` by name, the apl_1994 law); sea states 3 and 4 by `--set environment.sea_state=N`.
Image-source paths, direct and surface image only: the cheapest tier that carries the surface path, its roughness, the
bubble loss and the site's noise, and the planner's own paths ("direct path and surface image (no bottom return),
isovelocity"); no Bellhop is needed (the water has no profile). The three-row face at 225 m under way at 1.5 m/s;
`popoto_12km_large_submerged`'s three hulls unchanged, a ship at +29.2 degrees (range rate -3.65 m/s) and a +25 dB point
moving with the platform (range rate 0: the planner's zero-speed prediction for every waveform). Cost on this CPU:
17 s a ping (1 s linear, 96 channels), 567 s a ping (4 s hyperbolic, sampled).

### DS.3 The run (`tools/doppler_gpu_runs.sh` sub-run `deep12`; `tools/deep12_eval.py` sets each case against the planner)

Cases `ss{3,4,5}_{lfm1, lfm4p200, hfm4p200, hfm1}` with the plain filter and `_cfar` (the bank, 8 m/s) for the linear ones;
6 pings and 4 target-free pings each; C++ chain, and the Python chain on `ss3_lfm1` (the baseline's acceptance). The 4 s
cases are the alternate's energy (200 dB); the 4 s at 197 dB of the first brief is dropped for it. CPU trials of one to
three pings (not results; plumbing): at sea state 3 the baseline's point +3.29 dB against +2.79 predicted (3 of 3), the
crossing hull +2.53 against +1.76 (3 of 3), and the closing hull, the oblique hull and the ship not detected on any of
3 pings with the plain filter (predicted -11.5, +1.1, -2.1 dB with the closed form's loss).

### DS.4 The stale outputs of D-251 (issue `sonar-rar0`; `tools/regen_stale.py --list`)

Stale (a moving platform and boundary reverberation), rerun by the `regen` sub-run: `doppler_bank_eval.json`'s
`popoto_far_mover` and `popoto_12km`; `doppler_d3_eval.json`'s two `cfar_scenes` and `tone_p12`;
`doppler_large_submerged_eval.json`'s `bank`, `alternate` and `train_one_ping`; `doppler_baseline.json`'s runs with a
moving platform (`fp_v15`, `fp_v15_lfm`, `fm_*`, `k12_*`, `sp_scene`, `rs_mono`, `rs_bi`); `docs/design/popoto_12km.json`
whole. Not stale: every part from `flank_patrol` (platform still; one ping's record SHA-256 equal under `doppler:
per_frequency` and `doppler: centre` here), the noise-only and planner parts. Kept as they were and not reproducible at
this tree: `tone_p12_wholetrain` (before 51d2b12 and 80df69d) and the first run's truth summary (LS.1 item 7).
Regenerated here: `docs/handoff/examples/gallery_quick_fls_120k_transit.json` (confirmed tracks 7 -> 10, the three more
`uncertain` and `static` clutter; detections a ping [10, 11, 10, 13] -> [11, 9, 9, 15]; with `doppler: centre` the old
numbers come back exactly); `examples/classify_rules_eval_quick.md` rerun: unchanged.

## Deep site at 12 km: measured (appended 2026-11-27; jobs 6b0a02346c5d61fff18a11a1, 6b0a02356c5d61fff18a11a4; nothing above rewritten)

NVIDIA L4, tree `deepsea` at `1166ca6`, simulator and chain on cuda, `popoto_12km_deep_sea_state_5` at sea states 3, 4,
5 (`--set environment.sea_state`), 6 pings and 4 target-free pings a case, the C++ chain (the Python chain on `ss3_lfm1`:
equal to 6e-7 dB and the same detections). Ours; one seed; site illustrative; class values assumed. Every figure:
`docs/tasks/doppler_deep12_eval.json` (`tools/deep12_eval.py`). Held: detected on half the pings or more (the design's
detection probability, 0.5 on one ping).

### DM.1 What failed (read first)

1. **The baseline fails its acceptance at sea state 3.** The shipped 1 s, 197 dB linear sweep with the plain filter:
   the closing hull 0 of 6, the oblique hull 0 of 6, the ship 0 of 6. It holds the crossing hull (6 of 6, +2.95 dB) and
   the planner's own +25 dB point (6 of 6, +3.41 dB against +2.79 predicted). With the bank (`combine: cfar`, not the
   baseline: every Doppler option is off in it) the ship comes back (6 of 6, +2.62 dB); the closing and oblique hulls
   stay at 0 of 6.
2. **The alternate fails its acceptance at sea state 5**, with every waveform tried at its energy (4 s, 200 dB):
   linear plain: closing 0, oblique 0, ship 0 of 6 (crossing 4, point 5 of 6); linear with the bank: closing 0, oblique 0
   (ship 5, crossing 5, point 6); hyperbolic plain: closing 2, oblique 0 (ship 6, crossing 5, point 6 of 6).
3. So **the transmit table's margins hold for an unresolved, non-moving target of the stated strength, and only for
   that**: the point is reproduced within 0.62 dB wherever it is detected on every ping (11 cases); a hull of the class's
   +25 dB whose five highlights resolve in range falls 2.4 to 6.3 dB under its own prediction with a filter that takes
   no Doppler loss, and the linear sweep with the plain filter loses more on a mover.
4. No false detection on any of the 72 target-free pings at the default budget (18 cases); at 3 a ping 0 to 9.
5. Not shown: more than one seed; the hull's high target strength near broadside (+41.5 dB); targets at other depths
   (the scene's hulls at 50 m, the ship at 4 m, the array at 225 m); sea state 4 and 5 without the bubble law.

### DM.2 Why, separated (evidence)

**(a) The hull's highlights resolve, and the detector tests no scale long enough to add them.** In this window the
multiscale detector tests two scales only: the single sample (0.375 m, threshold 9.45 dB) and a 2 m cell (5.5 looks,
4.88 dB on its mean) (`detector_scales`; `live.detector_plan`). The hulls' extent along the line of sight: crossing
0.1 m, oblique 25 m (highlights 6.3 m apart), ship 58.5 m (14.6 m), closing 100 m (25 m): every hull but the crossing
one puts each of its five +18.0 dB highlights in its own cell, 6.99 dB under the whole. Measured, hull less the point
in the SAME case, with a filter that takes no Doppler loss (bank or hyperbolic): crossing -0.2 to +1.0 dB; oblique
-6.0 / -5.9 / -4.8 / -4.2 dB (sea state 3 hyperbolic, 3 bank, 4 hyperbolic, 4 bank); closing -5.4 / -5.1 / -3.9 / -3.8.
The ship's own prediction is 6 dB over the point's (a near-surface target in a constructive surface-image lobe at this
geometry: factor +8.9 against +2.8 dB at 50 m); measured it is level with the point (-0.5 to +1.2 dB), i.e. 4.9 to 6.3 dB
under its own prediction, the same loss. Against the one-highlight prediction (the prediction less 6.99 dB) the resolved
hulls come out +0.6 to +4.6 dB above it (+0.6 to +2.9 where the margin is high; the larger figures at sea state 4, where
the mean is over the stronger pings only): the chain sees about one highlight and some of the rest (not separated: the
2 m cell, the highlights' surface images, the strongest-of-five draw). At sea state 3 with the baseline the oblique
hull's one-highlight prediction is -4.2 dB and the closing hull's -5.6: below the threshold, as measured (0 of 6).

**(b) Doppler mismatch, the linear sweep with the plain filter** (same case, linear plain / hyperbolic plain / bank):

| Sea state, pulse | `ship` (-3.44 m/s) | `closing` (-5.60) | `oblique` (-0.50) | `crossing` (-0.39) | `point` (0) |
| --- | --- | --- | --- | --- | --- |
| 3, 1 s 197 dB | 0 of 6 / +4.05 / +2.62 | 0 / 0 / 0 of 6 | 0 / 0 / 0 of 6 | +2.95 / +3.82 / +2.75 | +3.41 / +2.84 / +2.77 |
| 3, 4 s 200 dB | +2.38 / +11.67 / +10.89 | +0.25 (2 of 6) / +6.26 / +5.71 | +2.29 / +5.62 / +4.91 | +9.24 / +11.42 / +10.65 | +11.69 / +11.63 / +10.84 |
| 4, 4 s 200 dB | 0 of 6 / +6.21 / +5.28 | 0 of 6 / +1.54 (5) / +0.88 | +0.33 (1) / +0.58 (5) / +0.56 | +4.04 / +5.64 / +4.97 | +5.39 / +5.41 / +4.72 |
| 5, 4 s 200 dB | 0 / +1.64 / +0.85 (5) | 0 / +0.17 (2) / 0 | 0 / 0 / 0 | +0.76 (4) / +2.38 (5) / +2.26 (5) | +2.12 (5) / +2.17 / +1.29 |

At 4 s the linear sweep's plain filter loses 9.3 dB on the ship (closed form 17.2), 6.0 on the closing hull (19.3),
3.3 on the oblique (8.8), 2.2 on the crossing (7.6): less than the closed form (the detector's cells), growing with the
speed. At 1 s the ship is lost (0 of 6) where the hyperbolic sweep and the bank hold it. The bank costs its threshold
shift (9.45 -> 10.10 dB at 1 s, 10.31 at 4 s: the point 0.6 to 0.9 dB under the plain filter) and, with three rows (the
per-hypothesis path, not D8's batched one), 3.4 s a ping at 1 s and 25.6 to 25.8 s at 4 s on the L4 against 0.05 to
0.14 s plain. The hyperbolic sweep equals the plain filter on the point at 4 s (+11.63 against +11.69; -0.57 dB at 1 s)
and leads the bank by 0.1 to 0.8 dB on every mover.

**(c) Anything else.** The chain's threshold in the 1.4 km window (9.45 dB, against the planner's 9.89 for its 0.75-12.7 km
window) is in the prediction, and the point agrees: not a cause. Own depth: the scene's 225 m is the planner's for the
ship; for this class it recommends 245 to 445 m, but the planner's margin at the scene's geometry is no worse than the
table's (sea state 3: +2.74 against +2.36 dB): not a cause. The surface image at 50 m: in the prediction (+2.8 dB at sea
state 3), and the point agrees: not a cause. The scene's noise 0.13 dB over the planner's flat figure (DS.1): within the
point's agreement.

### DM.3 What it means for the two designs, and what would accept the resolved hull

- Accepted, measured: the baseline at sea state 3 for a point (unresolved) of +25 dB with +3.4 dB to spare, so down to
  about +21.6 dB; the hull of the class's +25 dB is accepted only across the line of sight. The alternate at sea state 5:
  the point with +2.2 dB (so about +22.8 dB); the hull across the line of sight; the ship with the hyperbolic sweep or the
  bank.
- **What the resolved hull needs (measured bracket):** with a filter that takes no Doppler loss the closing and oblique
  hulls are held (5 or 6 of 6) where the point's margin is +5.4 dB (sea state 4, 4 s at 200 dB) and not held where it is
  +2.2 to +3.4 dB. So the hull needs about **+2 to +3 dB more than the table gives at sea state 3 with the baseline**, and
  **about +3 dB more than the alternate at sea state 5**:
  - baseline, sea state 3: about 200 dB re 1 uPa^2 s at 1 m a ping (1 s at 200 dB, or 2 s at 197 dB; the table's cells
    +5.4 / +6.3 dB), with the hyperbolic sweep or the bank for the ship (model and the bracket; not run at that energy);
  - alternate, sea state 5: about 209 dB (4 s at 203 dB or 8 s at 200 dB; no single listed projector gives either,
    the table's projector column); the closest measured stand-in is sea state 4 at 206 dB, whose noise is 3.15 dB
    lower (point +5.4 dB): there the hulls are held with the hyperbolic sweep or the bank.
  - In target strength: the resolved hull behaves as a point 5 to 6 dB weaker; the class's low end would have to be about
    +27 dB (baseline, sea state 3) or +28 dB (alternate, sea state 5) for these designs as they are.
- Processing instead of energy (proposals; nothing changed): (i) the hyperbolic sweep or the bank removes the ship's
  loss (measured, above); (ii) a detector scale over the hull: a plain cell as long as the hull gains little (model: a
  25 m cell's mean of about 67 looks needs about as much total energy as the strongest highlight alone), a test that sums
  only five highlight cells would gain about 4.5 dB (model; needs the spacing, i.e. the aspect: not general); (iii)
  several pings combined: 1.2 / 2.6 / 4.4 dB for 2 / 4 / 8 pings by the trade study's own figure (not implemented).
- **The alternate's sweep: hyperbolic** (`source: hfm`), from the sea-state-5 cases: the ship 6 of 6 (bank 5, linear 0),
  the crossing hull 5 of 6 (bank 5, linear 4), the point +2.17 dB (bank +1.29); at the plain chain's cost. What it costs:
  no bank; a range bias of 3 T = 12 m per m/s at 4 s (-67 m at 5.6 m/s closing); its 1 / f^2 spectrum on a projector
  whose response across 4-6 kHz is not known. It does not accept the resolved hull at sea state 5 either. The config
  (`configs/builds/popoto_face_3x32_ss5.yaml`) still carries the linear sweep: the owner's choice.

### DM.4 Every case and target, worst first (measured less predicted)

Predicted: `predict_point` at the target's geometry less the plain filter's closed-form loss (zero for the bank); the
planner's own figure at the scene / the table's (worst over the class's depths at its recommended own depth).

| Case | Design | Filter | Target (range rate m/s) | Verdict | Detected | Measured margin dB | Predicted, zero speed | Closed-form loss | Predicted | Measured - predicted | One highlight, predicted | Planner at the scene / table's worst depth | Tracked (class) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ss3_hfm1 | hyperbolic 1 s, 197 dB | plain | closing (-5.60) | FAIL | 0 of 6 | - | +1.63 | 0.20 | +1.44 | - | -5.55 | +1.58 / +2.36 | no (-) |
| ss3_hfm1 | hyperbolic 1 s, 197 dB | plain | oblique (-0.50) | FAIL | 0 of 6 | - | +2.79 | 0.02 | +2.77 | - | -4.22 | +2.74 / +2.36 | no (-) |
| ss3_lfm1 | baseline (shipped: 1 s, 197 dB) | plain | closing (-5.60) | FAIL | 0 of 6 | - | +1.63 | 13.16 | -11.53 | - | -18.52 | +1.58 / +2.36 | no (-) |
| ss3_lfm1 | baseline (shipped: 1 s, 197 dB) | plain | oblique (-0.50) | FAIL | 0 of 6 | - | +2.79 | 1.73 | +1.05 | - | -5.94 | +2.74 / +2.36 | no (-) |
| ss3_lfm1 | baseline (shipped: 1 s, 197 dB) | plain | ship (-3.44) | FAIL | 0 of 6 | - | +9.06 | 11.18 | -2.12 | - | -9.11 | +9.01 / +3.32 | no (-) |
| ss3_lfm1_cfar | baseline (shipped: 1 s, 197 dB) | bank cfar | closing (-5.60) | FAIL | 0 of 6 | - | +0.99 | 0.00 | +0.99 | - | -6.00 | +1.58 / +2.36 | no (-) |
| ss3_lfm1_cfar | baseline (shipped: 1 s, 197 dB) | bank cfar | oblique (-0.50) | FAIL | 0 of 6 | - | +2.14 | 0.00 | +2.14 | - | -4.85 | +2.74 / +2.36 | no (-) |
| ss4_hfm1 | hyperbolic 1 s, 197 dB | plain | closing (-5.60) | FAIL | 0 of 6 | - | -4.85 | 0.20 | -5.05 | - | -12.04 | -4.90 / -3.95 | no (-) |
| ss4_hfm1 | hyperbolic 1 s, 197 dB | plain | oblique (-0.50) | FAIL | 0 of 6 | - | -3.74 | 0.02 | -3.75 | - | -10.74 | -3.79 / -3.95 | no (-) |
| ss4_hfm1 | hyperbolic 1 s, 197 dB | plain | point (+0.00) | FAIL | 0 of 6 | - | -3.74 | -0.00 | -3.74 | - | - | -3.79 / -3.95 | no (-) |
| ss4_lfm1 | baseline (shipped: 1 s, 197 dB) | plain | closing (-5.60) | FAIL | 0 of 6 | - | -4.85 | 13.16 | -18.02 | - | -25.01 | -4.90 / -3.95 | no (-) |
| ss4_lfm1 | baseline (shipped: 1 s, 197 dB) | plain | oblique (-0.50) | FAIL | 0 of 6 | - | -3.74 | 1.73 | -5.47 | - | -12.46 | -3.79 / -3.95 | no (-) |
| ss4_lfm1 | baseline (shipped: 1 s, 197 dB) | plain | ship (-3.44) | FAIL | 0 of 6 | - | +3.29 | 11.18 | -7.90 | - | -14.89 | +3.24 / -1.90 | no (-) |
| ss4_lfm1_cfar | baseline (shipped: 1 s, 197 dB) | bank cfar | closing (-5.60) | FAIL | 0 of 6 | - | -5.50 | 0.00 | -5.50 | - | -12.49 | -4.90 / -3.95 | no (-) |
| ss4_lfm1_cfar | baseline (shipped: 1 s, 197 dB) | bank cfar | oblique (-0.50) | FAIL | 0 of 6 | - | -4.38 | 0.00 | -4.38 | - | -11.37 | -3.79 / -3.95 | no (-) |
| ss4_lfm1_cfar | baseline (shipped: 1 s, 197 dB) | bank cfar | ship (-3.44) | FAIL | 0 of 6 | - | +2.64 | 0.00 | +2.64 | - | -4.35 | +3.24 / -1.90 | no (-) |
| ss4_lfm4p200 | sea-state-5 alternate (4 s, 200 dB), linear | plain | closing (-5.60) | FAIL | 0 of 6 | - | +4.17 | 19.30 | -15.14 | - | -22.13 | +4.15 / +5.10 | no (-) |
| ss4_lfm4p200 | sea-state-5 alternate (4 s, 200 dB), linear | plain | ship (-3.44) | FAIL | 0 of 6 | - | +12.31 | 17.25 | -4.94 | - | -11.93 | +12.29 / +7.16 | no (-) |
| ss5_hfm1 | hyperbolic 1 s, 197 dB | plain | closing (-5.60) | FAIL | 0 of 6 | - | -7.75 | 0.20 | -7.95 | - | -14.94 | -7.80 / -7.05 | no (-) |
| ss5_hfm1 | hyperbolic 1 s, 197 dB | plain | oblique (-0.50) | FAIL | 0 of 6 | - | -6.93 | 0.02 | -6.95 | - | -13.94 | -6.98 / -7.05 | no (-) |
| ss5_hfm1 | hyperbolic 1 s, 197 dB | plain | ship (-3.44) | FAIL | 0 of 6 | - | -2.01 | 0.12 | -2.13 | - | -9.12 | -2.06 / -6.15 | no (-) |
| ss5_hfm1 | hyperbolic 1 s, 197 dB | plain | point (+0.00) | FAIL | 0 of 6 | - | -6.93 | -0.00 | -6.93 | - | - | -6.98 / -7.05 | no (-) |
| ss5_hfm4p200 | sea-state-5 alternate (4 s, 200 dB), hyperbolic | plain | oblique (-0.50) | FAIL | 0 of 6 | - | +2.08 | 0.02 | +2.07 | - | -4.92 | +2.07 / +1.99 | no (-) |
| ss5_lfm1 | baseline (shipped: 1 s, 197 dB) | plain | closing (-5.60) | FAIL | 0 of 6 | - | -7.75 | 13.16 | -20.92 | - | -27.90 | -7.80 / -7.05 | no (-) |
| ss5_lfm1 | baseline (shipped: 1 s, 197 dB) | plain | oblique (-0.50) | FAIL | 0 of 6 | - | -6.93 | 1.73 | -8.67 | - | -15.66 | -6.98 / -7.05 | no (-) |
| ss5_lfm1 | baseline (shipped: 1 s, 197 dB) | plain | crossing (-0.39) | FAIL | 0 of 6 | - | -6.93 | 1.03 | -7.96 | - | -14.95 | -6.98 / -7.05 | no (-) |
| ss5_lfm1 | baseline (shipped: 1 s, 197 dB) | plain | ship (-3.44) | FAIL | 0 of 6 | - | -2.01 | 11.18 | -13.20 | - | -20.19 | -2.06 / -6.15 | no (-) |
| ss5_lfm1 | baseline (shipped: 1 s, 197 dB) | plain | point (+0.00) | FAIL | 0 of 6 | - | -6.93 | -0.00 | -6.93 | - | - | -6.98 / -7.05 | no (-) |
| ss5_lfm1_cfar | baseline (shipped: 1 s, 197 dB) | bank cfar | closing (-5.60) | FAIL | 0 of 6 | - | -8.40 | 0.00 | -8.40 | - | -15.39 | -7.80 / -7.05 | no (-) |
| ss5_lfm1_cfar | baseline (shipped: 1 s, 197 dB) | bank cfar | oblique (-0.50) | FAIL | 0 of 6 | - | -7.58 | 0.00 | -7.58 | - | -14.57 | -6.98 / -7.05 | no (-) |
| ss5_lfm1_cfar | baseline (shipped: 1 s, 197 dB) | bank cfar | crossing (-0.39) | FAIL | 0 of 6 | - | -7.58 | 0.00 | -7.58 | - | -14.57 | -6.98 / -7.05 | no (-) |
| ss5_lfm1_cfar | baseline (shipped: 1 s, 197 dB) | bank cfar | ship (-3.44) | FAIL | 0 of 6 | - | -2.66 | 0.00 | -2.66 | - | -9.65 | -2.06 / -6.15 | no (-) |
| ss5_lfm1_cfar | baseline (shipped: 1 s, 197 dB) | bank cfar | point (+0.00) | FAIL | 0 of 6 | - | -7.58 | 0.00 | -7.58 | - | - | -6.98 / -7.05 | no (-) |
| ss5_lfm4p200 | sea-state-5 alternate (4 s, 200 dB), linear | plain | closing (-5.60) | FAIL | 0 of 6 | - | +1.26 | 19.30 | -18.04 | - | -25.03 | +1.25 / +1.99 | no (-) |
| ss5_lfm4p200 | sea-state-5 alternate (4 s, 200 dB), linear | plain | oblique (-0.50) | FAIL | 0 of 6 | - | +2.08 | 8.80 | -6.72 | - | -13.71 | +2.07 / +1.99 | no (-) |
| ss5_lfm4p200 | sea-state-5 alternate (4 s, 200 dB), linear | plain | ship (-3.44) | FAIL | 0 of 6 | - | +7.00 | 17.25 | -10.24 | - | -17.23 | +6.99 / +2.91 | no (-) |
| ss5_lfm4p200_cfar | sea-state-5 alternate (4 s, 200 dB), linear | bank cfar | closing (-5.60) | FAIL | 0 of 6 | - | +0.41 | 0.00 | +0.41 | - | -6.58 | +1.25 / +1.99 | no (-) |
| ss5_lfm4p200_cfar | sea-state-5 alternate (4 s, 200 dB), linear | bank cfar | oblique (-0.50) | FAIL | 0 of 6 | - | +1.23 | 0.00 | +1.23 | - | -5.76 | +2.07 / +1.99 | no (-) |
| ss3_lfm4p200_cfar | sea-state-5 alternate (4 s, 200 dB), linear | bank cfar | ship (-3.44) | PASS | 6 of 6 | +10.89 | +17.23 | 0.00 | +17.23 | -6.33 | +10.24 | +18.07 / +12.38 | yes (mover) |
| ss3_hfm4p200 | sea-state-5 alternate (4 s, 200 dB), hyperbolic | plain | ship (-3.44) | PASS | 6 of 6 | +11.67 | +18.08 | 0.12 | +17.96 | -6.29 | +10.97 | +18.07 / +12.38 | yes (mover) |
| ss4_lfm4p200_cfar | sea-state-5 alternate (4 s, 200 dB), linear | bank cfar | ship (-3.44) | PASS | 6 of 6 | +5.28 | +11.45 | 0.00 | +11.45 | -6.17 | +4.47 | +12.29 / +7.16 | yes (mover) |
| ss3_hfm4p200 | sea-state-5 alternate (4 s, 200 dB), hyperbolic | plain | oblique (-0.50) | PASS | 6 of 6 | +5.62 | +11.81 | 0.02 | +11.79 | -6.17 | +4.80 | +11.79 / +11.42 | yes (mover) |
| ss3_lfm4p200_cfar | sea-state-5 alternate (4 s, 200 dB), linear | bank cfar | oblique (-0.50) | PASS | 6 of 6 | +4.91 | +10.96 | 0.00 | +10.96 | -6.04 | +3.97 | +11.79 / +11.42 | yes (static) |
| ss4_hfm4p200 | sea-state-5 alternate (4 s, 200 dB), hyperbolic | plain | ship (-3.44) | PASS | 6 of 6 | +6.21 | +12.31 | 0.12 | +12.19 | -5.98 | +5.20 | +12.29 / +7.16 | yes (mover) |
| ss3_lfm1_cfar | baseline (shipped: 1 s, 197 dB) | bank cfar | ship (-3.44) | PASS | 6 of 6 | +2.62 | +8.41 | 0.00 | +8.41 | -5.80 | +1.43 | +9.01 / +3.32 | yes (mover) |
| ss5_lfm4p200_cfar | sea-state-5 alternate (4 s, 200 dB), linear | bank cfar | ship (-3.44) | PASS | 5 of 6 | +0.85 | +6.15 | 0.00 | +6.15 | -5.31 | -0.84 | +6.99 / +2.91 | yes (uncertain) |
| ss5_hfm4p200 | sea-state-5 alternate (4 s, 200 dB), hyperbolic | plain | ship (-3.44) | PASS | 6 of 6 | +1.64 | +7.00 | 0.12 | +6.88 | -5.24 | -0.11 | +6.99 / +2.91 | yes (mover) |
| ss3_hfm1 | hyperbolic 1 s, 197 dB | plain | ship (-3.44) | PASS | 6 of 6 | +4.05 | +9.06 | 0.12 | +8.94 | -4.89 | +1.95 | +9.01 / +3.32 | yes (mover) |
| ss4_hfm4p200 | sea-state-5 alternate (4 s, 200 dB), hyperbolic | plain | oblique (-0.50) | PASS | 5 of 6 | +0.58 | +5.28 | 0.02 | +5.26 | -4.68 | -1.72 | +5.27 / +5.10 | yes (static) |
| ss3_hfm4p200 | sea-state-5 alternate (4 s, 200 dB), hyperbolic | plain | closing (-5.60) | PASS | 6 of 6 | +6.26 | +10.65 | 0.20 | +10.46 | -4.19 | +3.47 | +10.64 / +11.42 | yes (multipath) |
| ss3_lfm4p200_cfar | sea-state-5 alternate (4 s, 200 dB), linear | bank cfar | closing (-5.60) | PASS | 6 of 6 | +5.71 | +9.80 | 0.00 | +9.80 | -4.09 | +2.81 | +10.64 / +11.42 | yes (multipath) |
| ss4_lfm4p200_cfar | sea-state-5 alternate (4 s, 200 dB), linear | bank cfar | oblique (-0.50) | PASS | 6 of 6 | +0.56 | +4.43 | 0.00 | +4.43 | -3.87 | -2.56 | +5.27 / +5.10 | yes (uncertain) |
| ss4_hfm1 | hyperbolic 1 s, 197 dB | plain | ship (-3.44) | FAIL | 1 of 6 | +0.23 | +3.29 | 0.12 | +3.17 | -2.94 | -3.82 | +3.24 / -1.90 | no (-) |
| ss4_lfm4p200_cfar | sea-state-5 alternate (4 s, 200 dB), linear | bank cfar | closing (-5.60) | PASS | 6 of 6 | +0.88 | +3.31 | 0.00 | +3.31 | -2.43 | -3.68 | +4.15 / +5.10 | yes (uncertain) |
| ss4_hfm4p200 | sea-state-5 alternate (4 s, 200 dB), hyperbolic | plain | closing (-5.60) | PASS | 5 of 6 | +1.54 | +4.17 | 0.20 | +3.97 | -2.42 | -3.02 | +4.15 / +5.10 | yes (uncertain) |
| ss5_hfm4p200 | sea-state-5 alternate (4 s, 200 dB), hyperbolic | plain | closing (-5.60) | FAIL | 2 of 6 | +0.17 | +1.26 | 0.20 | +1.07 | -0.89 | -5.92 | +1.25 / +1.99 | no (-) |
| ss3_lfm4p200 | sea-state-5 alternate (4 s, 200 dB), linear | plain | oblique (-0.50) | PASS | 6 of 6 | +2.29 | +11.81 | 8.80 | +3.01 | -0.71 | -3.98 | +11.79 / +11.42 | yes (mover) |
| ss3_hfm4p200 | sea-state-5 alternate (4 s, 200 dB), hyperbolic | plain | crossing (-0.39) | PASS | 6 of 6 | +11.42 | +11.81 | 0.01 | +11.79 | -0.37 | +4.80 | +11.79 / +11.42 | yes (static) |
| ss3_lfm4p200_cfar | sea-state-5 alternate (4 s, 200 dB), linear | bank cfar | crossing (-0.39) | PASS | 6 of 6 | +10.65 | +10.96 | 0.00 | +10.96 | -0.31 | +3.97 | +11.79 / +11.42 | yes (mover) |
| ss3_hfm4p200 | sea-state-5 alternate (4 s, 200 dB), hyperbolic | plain | point (+0.00) | PASS | 6 of 6 | +11.63 | +11.81 | -0.00 | +11.81 | -0.18 | - | +11.79 / +11.42 | yes (static) |
| ss3_lfm4p200_cfar | sea-state-5 alternate (4 s, 200 dB), linear | bank cfar | point (+0.00) | PASS | 6 of 6 | +10.84 | +10.96 | 0.00 | +10.96 | -0.12 | - | +11.79 / +11.42 | yes (mover) |
| ss3_lfm4p200 | sea-state-5 alternate (4 s, 200 dB), linear | plain | point (+0.00) | PASS | 6 of 6 | +11.69 | +11.81 | -0.00 | +11.81 | -0.12 | - | +11.79 / +11.42 | yes (mover) |
| ss5_lfm4p200 | sea-state-5 alternate (4 s, 200 dB), linear | plain | point (+0.00) | PASS | 5 of 6 | +2.12 | +2.08 | -0.00 | +2.08 | +0.04 | - | +2.07 / +1.99 | no (-) |
| ss3_hfm1 | hyperbolic 1 s, 197 dB | plain | point (+0.00) | PASS | 6 of 6 | +2.84 | +2.79 | -0.00 | +2.79 | +0.05 | - | +2.74 / +2.36 | yes (static) |
| ss5_lfm4p200_cfar | sea-state-5 alternate (4 s, 200 dB), linear | bank cfar | point (+0.00) | PASS | 6 of 6 | +1.29 | +1.23 | 0.00 | +1.23 | +0.05 | - | +2.07 / +1.99 | yes (static) |
| ss5_hfm4p200 | sea-state-5 alternate (4 s, 200 dB), hyperbolic | plain | point (+0.00) | PASS | 6 of 6 | +2.17 | +2.08 | -0.00 | +2.08 | +0.09 | - | +2.07 / +1.99 | yes (static) |
| ss4_lfm4p200 | sea-state-5 alternate (4 s, 200 dB), linear | plain | point (+0.00) | PASS | 6 of 6 | +5.39 | +5.28 | -0.00 | +5.28 | +0.11 | - | +5.27 / +5.10 | yes (mover) |
| ss4_hfm4p200 | sea-state-5 alternate (4 s, 200 dB), hyperbolic | plain | point (+0.00) | PASS | 6 of 6 | +5.41 | +5.28 | -0.00 | +5.28 | +0.13 | - | +5.27 / +5.10 | yes (static) |
| ss4_lfm4p200_cfar | sea-state-5 alternate (4 s, 200 dB), linear | bank cfar | point (+0.00) | PASS | 6 of 6 | +4.72 | +4.43 | 0.00 | +4.43 | +0.29 | - | +5.27 / +5.10 | yes (mover) |
| ss5_hfm4p200 | sea-state-5 alternate (4 s, 200 dB), hyperbolic | plain | crossing (-0.39) | PASS | 5 of 6 | +2.38 | +2.08 | 0.01 | +2.07 | +0.31 | -4.92 | +2.07 / +1.99 | no (-) |
| ss4_hfm4p200 | sea-state-5 alternate (4 s, 200 dB), hyperbolic | plain | crossing (-0.39) | PASS | 6 of 6 | +5.64 | +5.28 | 0.01 | +5.27 | +0.37 | -1.72 | +5.27 / +5.10 | yes (static) |
| ss4_lfm4p200_cfar | sea-state-5 alternate (4 s, 200 dB), linear | bank cfar | crossing (-0.39) | PASS | 6 of 6 | +4.97 | +4.43 | 0.00 | +4.43 | +0.54 | -2.56 | +5.27 / +5.10 | yes (mover) |
| ss3_lfm1_cfar | baseline (shipped: 1 s, 197 dB) | bank cfar | crossing (-0.39) | PASS | 6 of 6 | +2.75 | +2.14 | 0.00 | +2.14 | +0.61 | -4.85 | +2.74 / +2.36 | yes (static) |
| ss3_lfm1 | baseline (shipped: 1 s, 197 dB) | plain | point (+0.00) | PASS | 6 of 6 | +3.41 | +2.79 | -0.00 | +2.79 | +0.62 | - | +2.74 / +2.36 | yes (static) |
| ss3_lfm1_cfar | baseline (shipped: 1 s, 197 dB) | bank cfar | point (+0.00) | PASS | 6 of 6 | +2.77 | +2.14 | 0.00 | +2.14 | +0.62 | - | +2.74 / +2.36 | yes (static) |
| ss5_lfm4p200_cfar | sea-state-5 alternate (4 s, 200 dB), linear | bank cfar | crossing (-0.39) | PASS | 5 of 6 | +2.26 | +1.23 | 0.00 | +1.23 | +1.02 | -5.76 | +2.07 / +1.99 | yes (uncertain) |
| ss3_hfm1 | hyperbolic 1 s, 197 dB | plain | crossing (-0.39) | PASS | 5 of 6 | +3.82 | +2.79 | 0.01 | +2.77 | +1.05 | -4.22 | +2.74 / +2.36 | yes (static) |
| ss3_lfm1 | baseline (shipped: 1 s, 197 dB) | plain | crossing (-0.39) | PASS | 6 of 6 | +2.95 | +2.79 | 1.03 | +1.76 | +1.19 | -5.23 | +2.74 / +2.36 | yes (uncertain) |
| ss3_lfm4p200 | sea-state-5 alternate (4 s, 200 dB), linear | plain | ship (-3.44) | PASS | 6 of 6 | +2.38 | +18.08 | 17.25 | +0.83 | +1.55 | -6.16 | +18.07 / +12.38 | yes (mover) |
| ss4_hfm1 | hyperbolic 1 s, 197 dB | plain | crossing (-0.39) | PASS | 3 of 6 | +0.05 | -3.74 | 0.01 | -3.75 | +3.80 | -10.74 | -3.79 / -3.95 | no (-) |
| ss4_lfm4p200 | sea-state-5 alternate (4 s, 200 dB), linear | plain | oblique (-0.50) | FAIL | 1 of 6 | +0.33 | +5.28 | 8.80 | -3.52 | +3.85 | -10.51 | +5.27 / +5.10 | no (-) |
| ss4_lfm1 | baseline (shipped: 1 s, 197 dB) | plain | point (+0.00) | FAIL | 1 of 6 | +0.94 | -3.74 | -0.00 | -3.74 | +4.67 | - | -3.79 / -3.95 | no (-) |
| ss4_lfm1_cfar | baseline (shipped: 1 s, 197 dB) | bank cfar | point (+0.00) | FAIL | 1 of 6 | +0.29 | -4.38 | 0.00 | -4.38 | +4.67 | - | -3.79 / -3.95 | no (-) |
| ss4_lfm1_cfar | baseline (shipped: 1 s, 197 dB) | bank cfar | crossing (-0.39) | FAIL | 2 of 6 | +0.47 | -4.38 | 0.00 | -4.38 | +4.86 | -11.37 | -3.79 / -3.95 | no (-) |
| ss3_lfm4p200 | sea-state-5 alternate (4 s, 200 dB), linear | plain | crossing (-0.39) | PASS | 6 of 6 | +9.24 | +11.81 | 7.55 | +4.26 | +4.99 | -2.73 | +11.79 / +11.42 | yes (mover) |
| ss4_lfm1 | baseline (shipped: 1 s, 197 dB) | plain | crossing (-0.39) | FAIL | 2 of 6 | +0.87 | -3.74 | 1.03 | -4.77 | +5.63 | -11.76 | -3.79 / -3.95 | no (-) |
| ss5_lfm4p200 | sea-state-5 alternate (4 s, 200 dB), linear | plain | crossing (-0.39) | PASS | 4 of 6 | +0.76 | +2.08 | 7.55 | -5.47 | +6.23 | -12.46 | +2.07 / +1.99 | no (-) |
| ss4_lfm4p200 | sea-state-5 alternate (4 s, 200 dB), linear | plain | crossing (-0.39) | PASS | 6 of 6 | +4.04 | +5.28 | 7.55 | -2.27 | +6.31 | -9.26 | +5.27 / +5.10 | yes (static) |
| ss5_hfm1 | hyperbolic 1 s, 197 dB | plain | crossing (-0.39) | FAIL | 1 of 6 | -0.41 | -6.93 | 0.01 | -6.95 | +6.54 | -13.94 | -6.98 / -7.05 | no (-) |
| ss3_lfm4p200 | sea-state-5 alternate (4 s, 200 dB), linear | plain | closing (-5.60) | FAIL | 2 of 6 | +0.25 | +10.65 | 19.30 | -8.65 | +8.90 | -15.64 | +10.64 / +11.42 | no (-) |

| Case | Target-free pings | False detections, default budget | At 3 a ping |
| --- | --- | --- | --- |
| ss3_hfm1 | 4 | 0 | 0 |
| ss3_hfm4p200 | 4 | 0 | 0 |
| ss3_lfm1 | 4 | 0 | 0 |
| ss3_lfm1_cfar | 4 | 0 | 2 |
| ss3_lfm4p200 | 4 | 0 | 0 |
| ss3_lfm4p200_cfar | 4 | 0 | 9 |
| ss4_hfm1 | 4 | 0 | 0 |
| ss4_hfm4p200 | 4 | 0 | 0 |
| ss4_lfm1 | 4 | 0 | 0 |
| ss4_lfm1_cfar | 4 | 0 | 2 |
| ss4_lfm4p200 | 4 | 0 | 0 |
| ss4_lfm4p200_cfar | 4 | 0 | 9 |
| ss5_hfm1 | 4 | 0 | 0 |
| ss5_hfm4p200 | 4 | 0 | 0 |
| ss5_lfm1 | 4 | 0 | 0 |
| ss5_lfm1_cfar | 4 | 0 | 2 |
| ss5_lfm4p200 | 4 | 0 | 0 |
| ss5_lfm4p200_cfar | 4 | 0 | 8 |


## The outputs D-251 made stale, regenerated (appended 2026-11-27; issue `sonar-rar0`; jobs 6b0a02356c5d61fff18a11a6, 6b0a02366c5d61fff18a11aa; nothing above rewritten)

`tools/doppler_gpu_runs.sh` sub-run `regen` on an NVIDIA L4, tree `deepsea` at `1166ca6`, simulator and chain on cuda
(the old runs: this CPU, host backend). Only the stale parts replaced (`tools/regen_stale.py --list`); each file carries
a `regenerated` block (what was replaced, what was kept and why). Old -> new of every number the sections above and
`docs/RESULTS.md` quote, worst first. The change mixes D-251 with the device (cuda against host) and the decisions since
the old runs; they are not separated.

1. **Worse.** `doppler_large_submerged_eval.json` `alternate` (alternating tone and sweep pings, LS.3): the closing hull is
   **no longer tracked at the last ping** (was `uncertain`, 17.5 m from the truth; both engines), with the tone path off
   and on; its detections are as before (3 of 3 sweep pings, 3 of 3 tone pings, rates -5.602 / -5.582 / -5.595 m/s). Not
   investigated.
2. `doppler_d3_eval.json` `tone_p12` (`popoto_12km_doppler`, D3): the background is now 4 pings (the committed one had 2,
   the recorded command says 4): tone path off 1 -> 4 false detections at the default budget (0.5 -> 1.0 a ping), 11 -> 30
   at 3 a ping (5.5 -> 7.5 a ping); tone path on 0 -> 0 at the default budget, 8 -> 24 at 3 a ping (24 attributed: 21 the
   sweep's, 3 the tone's, none on a lobe's ridge). The ship as before: 0 of 6 with the default chain, 6 of 6 with the
   tone path, the rate on 1 ping (-0.573 m/s), snr within 0.1 dB.
3. `combine: cfar` background at 3 a ping (D3.2): `popoto_12km` off 20 -> 16, cfar 7 -> 12; `popoto_far_mover` off 18 -> 16,
   cfar 17 -> 11; `popoto_12km_large_submerged` off 20 -> 16, cfar 7 -> 8. At the default budget 0 -> 0 everywhere.
4. Margins (9.4, D3.2, LS.3), both engines equal as before: `popoto_12km` ship off 9.11 -> 9.11, max 5.57 -> 5.57, cfar
   9.63 -> 9.63 dB; `popoto_far_mover` ship 12.35 -> 12.36, max 8.84 -> 8.83, cfar 13.65 -> 13.63; large submerged off
   7.98 -> 7.99 (closing), 9.22 -> 9.22 (oblique), 11.41 -> 11.38 (crossing), cfar 8.78 -> 8.81, 10.08 -> 10.09,
   12.00 -> 11.99; classes as before (the crossing hull `static` with cfar). The train ping (LS.3): closing snr 17.8 -> 17.9
   (bank off), 18.5 -> 18.3 (tone path); crossing 24.5 -> 24.3; tone margins and rates as before.
5. The bank's cost (9.5): the regenerated scenes' times are now the L4's, not this CPU's: `popoto_12km` C++ 0.020 s plain,
   0.098 s with the bank (5.0 times; the CPU's 25 times, table 9.5, is no longer in the file), cfar 0.108 s (5.8 times);
   `popoto_far_mover` 0.060 / 0.348 s (5.8 times; CPU 51), cfar 0.376 s (6.3 times).
6. `doppler_baseline.json` (section 3.2): within 0.1 dB everywhere but `sst_pursuit`'s bank 12.7 -> 12.5 dB (gain over its
   zero hypothesis 28.3 -> 28.1); `popoto_12km` 24.0 / 25.9 (the trade study's +1.9 dB bank credit, unchanged),
   `popoto_far_mover` 27.2 / 27.8, the flank under way (`fp_v15*`) as before; the reverberation spectra (2.1) the same
   peaks, two -10 dB edges moved by 0.4 and 0.06 Hz.
7. `docs/design/popoto_12km.json` (RESULTS "The 12 km study, rerun"): every case's pings detected, median SNR and track
   ping as before (winter sand two rows 31.4 -> 31.5 dB; Bellhop two rows "several possible elevations" on 5 -> 4 pings);
   class cases: the same detections and classes; speed errors moved by up to 0.5 m/s (the cylinder -1.95 -> -1.17 m/s);
   the Bellhop submerged mover's reported depth 27 -> 9 m (sigma 2.5 km, truth 60 m inside two sigma, as before).
   `docs/DESIGN.md` regenerated from it (the class table's speed and heading errors).
8. Not stale and kept (`regenerated.kept` in each file): every `flank_patrol` part (platform still: one ping's record
   SHA-256 equal under both laws), the noise-only and planner parts; not reproducible, kept as they were:
   `tone_p12_wholetrain`, the first run's truth summary.

### DM.5 The energy candidates, predicted before they are run (appended 2026-11-27; issue `sonar-rctz`; not run yet)

DM.3's "what the resolved hull needs" is the model and a bracket. These cases measure it; each prediction below was
written before the run (`tools/deep12_eval.py --predict-cases --json docs/tasks/deep12_round3_predictions.json`; `tools/deep12_eval.py`
puts it beside the measured verdict when the results are summarised). Point: the planner's predicted chain margin of the +25 dB
point (`predict_point` at its geometry, the chain's 9.45 dB threshold; for the bank less 0.75 dB, its threshold shift measured
0.65 / 0.86 dB at 1 / 4 s). Bracket (DM.3): the resolved hulls held where the point's margin is 5.4 dB or more, not at 3.4 dB
or less. One highlight: the hull's prediction less 10 log10(5). Duty: the pulse over the two-way time to 12.7 km plus the
pulse. Projector: the trade study's column (via secondary: the sheets in docs/design/projector_selection.md). Ours.

| Case | Energy (dB re 1 uPa^2 s at 1 m) | Point, predicted | One highlight: closing / oblique / ship | Bracket's verdict for the resolved hulls | Blind range | Duty | Range bias (m per m/s) | A single listed projector |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ss3_hfm2` (hyperbolic 2 s, 197 dB) | 200.0 | +5.80 dB | -2.5 / -1.2 / +5.0 | held | 1.50 km | 10.6 % | 6 | yes |
| `ss3_lfm2_cfar` (linear 2 s, 197 dB, bank) | 200.0 | +5.05 dB | -3.1 / -1.9 / +4.3 | inside the bracket (between 3.4 and 5.4 dB) | 1.50 km | 10.6 % | — | yes |
| `ss3_hfm1p200` (hyperbolic 1 s, 200 dB) | 200.0 | +5.79 dB | -2.6 / -1.2 / +5.0 | held | 0.75 km | 5.6 % | 3 | yes |
| `ss4_hfm2` (hyperbolic 2 s, 197 dB) | 200.0 | -0.73 dB | -9.0 / -7.7 / -0.8 | not held | 1.50 km | 10.6 % | 6 | yes |
| `ss5_hfm8p200` (hyperbolic 8 s, 200 dB) | 209.0 | +5.09 dB | -2.9 / -1.9 / +2.9 | inside the bracket (between 3.4 and 5.4 dB) | 6.00 km | 32.1 % | 24 | no |
| `ss5_hfm4p203` (hyperbolic 4 s, 203 dB) | 209.0 | +5.08 dB | -2.9 / -1.9 / +2.9 | inside the bracket (between 3.4 and 5.4 dB) | 3.00 km | 19.1 % | 12 | no |
| `ss5_hfm8p203` (hyperbolic 8 s, 203 dB) | 212.0 | +8.08 dB | +0.1 / +1.1 / +5.9 | held | 6.00 km | 32.1 % | 24 | no |

The one-highlight figures are under the threshold in every case but the ship's: on DM.4's runs the resolved hulls came out
0.6 to 4.6 dB above their one-highlight prediction, so the bracket, not that figure, is the prediction to test. Neither way
to 209 dB has a single listed projector (4 s at 203 dB: two TR-1075D in phase; 8 s at 200 dB: 3.0 dB short at 32 % duty).
The run: the `deep12` sub-run with these cases, the sea-state-3 candidates first (GPUSONAR_DOPPLER_DEEP12_CASES); each
case limited to 45 min; on the L4 the 4 s hyperbolic cases simulated 6 pings in 27 s (DM), the 8 s ones are untimed.

### DM.6 The energy candidates, measured (appended 2026-11-28; job 6b0a19496c5d61fff18a650c, NVIDIA L4, tree `7aa84e1`; nothing above rewritten)

The seven cases of DM.5, 6 pings and 4 target-free pings each, C++ chain on cuda; merged into `docs/tasks/doppler_deep12_eval.json`
(`acceptance.candidate_*`, each with the prediction written before the run). DM.3's "what the resolved hull needs" was the model
and a bracket; this is the measurement. Ours; one seed.

**What it does not show, first.** One seed. The resolved hulls are held with little to spare: at sea state 3 the oblique hull by
+0.2 to +0.7 dB and the closing hull by +0.8 to +1.3 dB (mean over the pings detected): a second seed may move a verdict. **The
oblique hull is detected on 6 of 6 pings and NOT tracked** in `ss3_hfm2` and in all three sea-state-5 cases (tracked, `static`, in
`ss3_hfm1p200` and `ss3_lfm2_cfar`); not investigated: `tools/doppler_bank_eval.py` keeps per ping only the strongest detection in
the gate and the tracks within 100 m of the hull's centre at the last ping, so whether a track exists elsewhere on the 100 m hull,
or none confirms, is not in the output (a follow-up: write the tracks' positions; a CPU rerun of one case takes about 30 min).

| Case | Predicted before the run (point; bracket) | closing | oblique | crossing | ship | point | Resolved hulls held? | False detections, 4 target-free pings (default / 3 a ping) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ss3_hfm2` (hyperbolic 2 s, 197 dB (200 dB a ping), plain filter, at sea state 3) | +5.80 dB; held | 6 of 6, +1.28 | 6 of 6, +0.73, not tracked | 6 of 6, +5.13 | 6 of 6, +6.26 | 6 of 6, +6.12 | **yes** | 0 / 1 |
| `ss3_hfm1p200` (hyperbolic 1 s, 200 dB (200 dB a ping), plain filter, at sea state 3) | +5.79 dB; held | 5 of 6, +1.23 | 5 of 6, +0.66 | 6 of 6, +5.71 | 6 of 6, +6.58 | 6 of 6, +5.71 | **yes** | 0 / 0 |
| `ss3_lfm2_cfar` (linear 2 s, 197 dB (200 dB a ping) with the bank, at sea state 3) | +5.05 dB; inside the bracket (between 3.4 and 5.4 dB) | 5 of 6, +0.83 | 5 of 6, +0.22 | 6 of 6, +4.81 | 6 of 6, +5.73 | 6 of 6, +5.05 | **yes** | 0 / 3 |
| `ss4_hfm2` (the hyperbolic 2 s, 197 dB candidate at sea state 4) | -0.73 dB; not held | 0 of 6, —, not tracked | 0 of 6, —, not tracked | 3 of 6, +1.46 | 6 of 6, +1.27 | 5 of 6, +0.90, not tracked | **NO** | 0 / 1 |
| `ss5_hfm4p203` (hyperbolic 4 s, 203 dB (209 dB a ping), plain filter, at sea state 5) | +5.08 dB; inside the bracket (between 3.4 and 5.4 dB) | 5 of 6, +1.22 | 4 of 6, +0.56, not tracked | 6 of 6, +4.71 | 6 of 6, +4.18 | 6 of 6, +5.07 | **yes** | 0 / 0 |
| `ss5_hfm8p200` (hyperbolic 8 s, 200 dB (209 dB a ping), plain filter, at sea state 5) | +5.09 dB; inside the bracket (between 3.4 and 5.4 dB) | 5 of 6, +0.33 | 2 of 6, +1.20, not tracked | 5 of 6, +5.12 | 6 of 6, +4.01 | 6 of 6, +4.90 | **NO** | 0 / 0 |
| `ss5_hfm8p203` (hyperbolic 8 s, 203 dB (212 dB a ping), plain filter, at sea state 5) | +8.08 dB; held | 6 of 6, +2.64 | 6 of 6, +2.40, not tracked | 6 of 6, +7.21 | 6 of 6, +6.67 | 6 of 6, +7.91 | **yes** | 0 / 0 |

Against the predictions: the three "held" predictions held (`ss3_hfm2`, `ss3_hfm1p200`, `ss5_hfm8p203`), the "not held" one did not
(`ss4_hfm2`: closing and oblique 0 of 6); of the three inside the bracket, two held (`ss3_lfm2_cfar`, `ss5_hfm4p203`, the oblique hull
on 4 of 6) and one did not (`ss5_hfm8p200`, the oblique hull on 2 of 6). The point came out within -0.19 to +0.33 dB of its prediction
where detected on every ping (`ss3_hfm2` +6.12 against +5.80; `ss5_hfm4p203` +5.07 against +5.08).

### DM.7 What the owner chooses, measured (replaces DM.3's model figures; nothing configured)

- **Sea state 3 (the baseline's).** +3 dB of energy, 200 dB re 1 uPa^2 s at 1 m a ping, with a filter that takes no Doppler loss,
  holds every target on 5 or 6 of 6 pings: hyperbolic 2 s at 197 dB (the shipped level: a TR-1075D or TR-1075A class source by
  the sheets, via secondary; blind range 1.5 km; duty 10.6 %; range bias 6 m per m/s), hyperbolic 1 s at 200 dB (TR-1075D only,
  free of cavitation below 29 m; blind 0.75 km; duty 5.6 %; 3 m per m/s), or linear 2 s at 197 dB with the bank (87 hypotheses
  a beam: 7.3 s a ping with three rows on the L4, against 0.05 s plain). The shipped 1 s, 197 dB linear sweep holds the
  crossing hull and the point only (DM.1). The candidate does not reach sea state 4 (`ss4_hfm2`: closing and oblique 0 of 6).
- **Sea state 5 (the alternate's).** 209 dB a ping holds the resolved hulls only as 4 s at 203 dB (oblique 4 of 6); 8 s at
  200 dB does not (oblique 2 of 6). 212 dB (8 s at 203 dB) holds every target on 6 of 6. **No single listed projector gives
  any of the three** (4 s at 203 dB: two TR-1075D in phase, unconfirmed; 8 s: 32 % duty, a 6 km blind range).
- **What a decision would change** (the owner's; none made here):
  - a hyperbolic sweep in a build: `python/gpusonar/buildspec.py` first (its `projector.waveform.type` takes `lfm` or `cw`
    only: an `hfm` type writing `{type: sampled, source: hfm}` into the generated sonar, with tests); then the build
    (`configs/builds/popoto_face_3x32.yaml` for the baseline, `popoto_face_3x32_ss5.yaml` for the alternate) and
    `tools/build_from_spec.py --write-shipped` (`configs/sonars/popoto_long_range_3row*.yaml` with the window opening at the
    new blind range, `configs/deploy/popoto_vehicle_elevation*.yaml`); then the generated documents (`docs/DESIGN.md`,
    `docs/handoff/hardware_builds.json`, `docs/design/projector_requirements.json`, the depth plan) and every result of the
    three-row face. The chain needs no change for the hyperbolic sweep (HFM.3); the bank refuses it;
  - the bank in the baseline instead: the deployment file's `chain.doppler: {mode: sweep_bank, combine: cfar}`, which reverses
    D-244's "every Doppler option off" baseline, at 87 hypotheses a beam for 2 s;
  - either way a default result moves: the owner's approval with before -> after first.

## The baseline becomes the 2 s hyperbolic sweep (D-256; 2026-11-28, branch `hfmbase`, issue `sonar-7h6q`)

Appended; nothing above was rewritten. The owner (D-256): the 12 km long-range baseline transmits a hyperbolic sweep of 2 s at
197 dB rms, 4-6 kHz, plain matched filter, every Doppler option off. Configured in `configs/sonars/popoto_long_range.yaml`
(POP-3's file) and the face builds `configs/builds/popoto_face_{1,2,3}x32.yaml` (`projector.waveform.type: hfm`, a new type of
the build spec that writes `{type: sampled, source: hfm}`). Every number below is **ours**; the CPU runs are quick forms.

### HB.1 What got worse, is uncertain or was not run (read first)

1. **Nothing inside 1.5 km.** The window opens at c T / 2 = 1500 m (750 m before). The three row and elevation scenes that
   showed the stacked rows at about 1 km (`popoto_elevation_free`, `popoto_elevation_shallow`, `popoto_elevation_rehearsal`)
   now state the former 1 s linear sweep by override, targets kept; the face builds' own simulated trials moved their near
   target from 1000 m to 1700 m. The stacked rows' elevation at 12 km with the new baseline is shown only by
   `tests/python/test_buildspec.py` (`popoto_12km_3row`, image-source and Bellhop: detected and tracked on both engines);
   no depth figure at 12 km was measured here. Whether a short-range mode is wanted (a short pulse for the first
   kilometres) is a question for the owner.
2. **A range bias on movers the plain filter cannot remove**: f_stop T / B = 6 m per m/s of range rate (closing reads
   nearer; `gpusonar.doppler.hfm_range_factor`), against the linear sweep's uncorrected f_c T / B = 2.5 m per m/s at 1 s. The
   range-Doppler correction (`chain.range_doppler: ownship | full`) is the linear sweep's: on a sampled record it gave a
   factor of 0, silently. It is now **refused** in both engines (and in `tools/gallery.py --range-doppler`) with the same words.
   The face builds' self-test checks the range against the truth displaced by the predicted bias (`build_from_spec.py`).
3. The deep-site acceptance of the baseline is the DM.6 run of `ss3_hfm2` (the same transmission, one seed, 6 pings): every
   target 6 of 6, the resolved hulls by **+1.28 (closing) and +0.73 dB (oblique): thin**; sea state 4 not held (`ss4_hfm2`).
   `ss3_baseline` (the shipped sonar with no waveform set) is not run yet: GPU job below.
4. **The face's own simulated rehearsal fails its depth check at the new near range.** `tools/build_from_spec.py
   popoto_face_3x32 --self-test`: free field and the board stream pass; the rehearsal (100 m of water, a target at 60 m,
   now 1.7 km out) gives depths 28, 35 m and none on its three pings where 60 is true (C++ and Python alike): not honest.
   With the 1 s linear sweep at 1.7 km it gives no depth on any ping; at the former 1000 m with that sweep it is honest
   (59, 60, 63 m). So it is the geometry the blind range forces, not the sweep. Not fixed; the multipath-sigma tests keep
   the former 1 km trial by statement (`tests/python/test_multipath_sigma.py`).
5. **Found on the way and fixed (both engines equal again):** on the 2 s hyperbolic record at 6 km the C++ and Python
   multipath-depth stages differed at 2 of 244 points of the depth likelihood's envelope (by up to 8, the depth and its
   sigma equal): the Python window's times were dt (k - n0) where the C++ engine's are -dt n0 + dt k, and the anchor's
   t0, on a half step when the echo peaks on an odd sample, rounded apart. `multipath_depth.window_times` and `likelihood`
   now use the C++ engine's arithmetic (the two agree to 1e-13 on the cases tested, 1e-9 before on the linear sweep).
6. The simulator's cost on this CPU: a sampled 2 s pulse simulates 5 to 17 times slower than the analytic 1 s sweep (the
   quick forms below; other jobs were running); the chain's cost a ping is unchanged (0.9 to 2.8 s, C++).
7. Projector (via secondary, the sheets as read in `docs/design/projector_selection.md`): the TR-1075D gives 2 s at 197 dB
   (201.7 dB at its 30 % rating, 10.6 % duty); the TR-1075A only with 0.1 dB to spare at 6 kHz; the Neptune T335 no longer
   qualifies (its 10 % duty rating); no sheet states the longest pulse at that drive; one amplifier rating names a 1 s burst.

8. **Tests that now state the 1 s linear sweep** (their levels, windows at about 1 km or stored figures were made with it;
   each says so in place; no tolerance changed): `test_detector.py` (its long-range sonar at 0.9 to 1.7 km, and the D-158
   golden lines' generator), `test_doppler_bank.py`, `test_doppler_cfar.py`, `test_doppler_device.py` (the bank), `test_elevation.py`
   (rows at 1 km), `test_multipath_sigma.py` (the build's former 1 km trial), `test_multipath_depth.py` (three tests tuned to the
   1 s echo), `test_hardcoded.py` (the ambiguity rule: with the 2 s sweep a second arrival 165 m behind the ship crosses the
   threshold), `test_reverb_budget.py` (the deep-silt prediction; with the 2 s sweep reverberation over noise is 3.2 dB higher,
   held), `test_rows_reverberation_gain.py` (the trade study's grid figures), `test_doppler_scenes.py` (the train's base sweep).
   The 1 s / 2 s difference that shows in `test_elevation.py` when it is not stated: two rows' repeats come 11.7 to 12.8 deg
   apart on the hyperbolic sweep against 11.5 at 5 kHz (its energy weighs the low end of the band, 1 / f^2).

### HB.2 Before -> after, the long-range scenes (quick forms on this CPU; both engines equal to 1e-6 dB on every case)

`tools/doppler_bank_eval.py SCENE --pings 3 --background 1 --off-only --engines python cpp --truth-at reflection --gate-range-m 250
--set propagation.max_arrivals=16` (`--chain-set elevation={}` for the stacked rows; no max_arrivals for the image-source scene).
Margin: the mean over the pings detected (C++). Range error: the detection's range less the truth's at reflection; the
predicted change is (6 - 2.5) m per m/s x the range rate (the HFM's bias less the linear sweep's coupling).

| Scene (target, range rate) | Detected | Margin (dB) | Range error (m), change predicted | Tracked (class) | False, 1 target-free ping |
| --- | --- | --- | --- | --- | --- |
| `popoto_face_2row_shallow` (target, -0.20 m/s) | 3 -> 3 of 3 | 19.45 -> 21.00 | -0.5 -> -1.0 (-0.7) | yes -> yes (uncertain) | 0 -> 0 |
| `popoto_12km_two_ships` (ship_b, -0.37) | 3 -> 3 | 5.56 -> 7.40 | -0.4 -> -1.6 (-1.3) | yes -> yes | 0 -> 0 |
| `popoto_12km_two_ships` (ship_a, -0.37) | 3 -> 3 | 8.21 -> 11.28 | -2.1 -> -3.1 (-1.3) | yes -> yes | |
| `popoto_12km_2row` (ship, -0.37) | 3 -> 3 | 10.51 -> 13.54 | -2.2 -> -3.3 (-1.3) | yes -> yes | 0 -> 0 |
| `popoto_12km_3row` (ship, -0.37) | 3 -> 3 | 12.33 -> 15.49 | -1.2 -> +0.1 (-1.3) | yes -> yes | 0 -> 0 |
| `popoto_12km` (ship, -0.37) | 3 -> 3 | 8.08 -> 11.57 | -2.2 -> -3.1 (-1.3) | yes -> yes | 0 -> 0 |

The gallery's long-range scenes, `tools/gallery.py --quick --backend host --no-animation` (4 pings, 16 arrivals), worst first:

| Scene | Detections a ping | Ship tracked; position error (m) | Peak over median, 32 hydrophones (dB) | Classes | Seconds (simulation and chain) |
| --- | --- | --- | --- | --- | --- |
| `popoto_env_deep_silt_1000m` | 2,2,3,3 -> 1,1,1,2 | no -> no (a track 301 m off, before and after) | 11.9 -> 11.9 | mover 1, multipath 1 -> mover 1 | 31 -> 205 |
| `popoto_env_summer_mud_60m` | 0 -> 0 (the summer shadow) | no -> no | 12.4 -> 12.5 | none -> none | 29 -> 248 |
| `popoto_12km` | 1,2,2,1 -> 1,1,2,3 | yes, 1.3 -> 2.9 | 23.2 -> 28.3 | mover -> mover | 62 -> 362 |
| `popoto_env_gravel_30m` | 1,1,1,1 -> 1,1,1,1 | yes, 16.9 -> 18.4 | 34.6 -> 35.2 | mover -> mover | 46 -> 355 |
| `popoto_env_winter_sand_200m` | 4,5,5,5 -> 4,5,5,5 | yes, 8.3 -> 8.6 | 31.2 -> 33.6 | mover 1, multipath 4 -> the same | 41 -> 346 |

Unchanged by construction (their transmit hashed equal before and after: type, band, length, levels, replica, window): the
scenes that state their own pulse (`popoto_12km_large_submerged` and the three elevation scenes, now by override; the trains
`popoto_12km_doppler`, `popoto_12km_large_submerged_tone`, `popoto_far_mover_doppler`), every 0.1 s static and test-AUV scene, the
27.5 kHz Popoto scenes, `popoto_far_mover` (6-10 kHz). FLS, SAS and flank_32: no resolved config changed; one simulated ping each
of `point_grid_flank_32`, `point_grid_fls_300k`, `point_grid_sas_auv`, `point_grid_popoto_8` hashed equal before and after.

### HB.3 The planner, the trade study, POP-3

- Free field, the ship-sized +25 dB at 12 km (`buildspec.performance`): one row's SNR 14.6 -> 17.6 dB; margin 1 / 2 / 3 rows
  -1.1 / +1.4 / +2.9 -> **+1.9 / +4.4 / +5.9 dB** (detection probability 0.22 / 0.87 / 0.99 -> 0.94 / 1.00 / 1.00): +3.01 dB of
  energy, the thresholds 0.019 / 0.014 / 0.012 dB lower for the window from 1.5 km.
- The depth plan (`docs/design/popoto_depth_planner.json`, three rows, recommended own depth): +3.0 dB at the noise-limited
  sites, +1.6 to +2.9 at the reverberation-limited ones; held at the same 4 of 6 sites by both classes; at
  `deep_no_bottom_return_sea_state_5` -4.0 / -3.1 dB (`large_submerged` / `ship`; -7.1 / -6.1 before); `summer_60m_mud_profile`
  -1.7 / -93.1 (-4.7 / -96.1). Every site equals the trade study's `2 s 197 dB` column within 0.001 dB.
- The trade study's section 0 marks `2 s 197 dB` as shipped (hyperbolic) and `1 s 197 dB` as shipped until D-256; the grid
  sections 1 to 6 stay of the 1 s sweep they were run at, now said so; the requirement sentence follows the shipped option
  (deep site up to sea state 3 with the bubble law, 4 without it).
- POP-3: `met` true and the status "partly met" as before (the summer mud's shadow is held by neither design); the cell now
  reads the baseline as measured holding every target at sea state 3 (thin), sea state 4 not held, and the alternate a
  placeholder needing a larger transmitter not yet identified.

### HB.4 The GPU job (not run here)

- The full gallery: `GPUSONAR_JOB=gallery` (the long-range scenes with the 2 s sweep; `docs/handoff/examples/live_popoto_12km_*`
  and `docs/figures`).
- The deep-site acceptance of the baseline from its own config: `GPUSONAR_JOB=doppler GPUSONAR_DOPPLER_RUNS=deep12
  GPUSONAR_DOPPLER_DEEP12_CASES=ss3_baseline,ss4_baseline,ss5_baseline,ss3_lfm1`, then `tools/deep12_eval.py DIR --json
  docs/tasks/doppler_deep12_eval.json` (keep the committed rows: merge), which makes `acceptance.baseline` the measured
  `ss3_baseline` instead of its stand-in `ss3_hfm2`.
- The main target on the shelf with the baseline: `GPUSONAR_JOB=doppler GPUSONAR_DOPPLER_RUNS=hfm12 GPUSONAR_DOPPLER_HFM12_BASELINE=only`
  (`popoto_12km_large_submerged`, which keeps its 1 s linear sweep as the bank's scene, with the 2 s hyperbolic sweep set).

## The short-range mode (2026-11-28, branch `shortmode`, issue `sonar-9ed5`; opt-in)

Appended; nothing above was rewritten. HB.1 item 1 (nothing inside 1.5 km with the 2 s baseline) has an opt-in answer:
a sonar's `short_range` block sends the same sweep cut to 0.25 s in its own slot after every long ping. The account, with
the options not chosen and their numbers, is `docs/design/short_range_mode.md`. Worst first: it costs the long mode 18 %
of its update rate (cycle 23.2 s instead of 18.93 s). Its short pulse has 9.0 dB less energy. The simulator does not model
one ping's echoes arriving in the next ping's window. `xl_auv` at +5 dB barely meets at the hand-over. On
`popoto_short_range_mode` (free field, both engines equal): the object at 800 m is seen on 5 of 5 short pings and on no
baseline ping; a mover crossing 1.5 km is one track; the 12 km ship's detections equal the baseline's at the same transmit
times. The rows' elevation at 1 km works on the short pings (`popoto_elevation_short_range`).
