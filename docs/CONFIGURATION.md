# Configuration reference

Everything about a sonar and about a simulated scenario is in a YAML file. Nothing in the
code is specific to one sonar. This page lists the keys; `docs/DESIGN.md` shows what the
owner's designs set them to and what follows.

- Sonars: `configs/sonars/<name>.yaml`, loaded with `SonarModel.load(name, overrides)`.
- Scenarios: `configs/scenarios/<name>.yaml`, run with `run_scenario(name)` or
  `tools/gallery.py --only <name>`.
- Any file may start from another with `base: <name>`; nested keys merge.
- A scenario changes its sonar with `sonar_overrides:`, using the same keys.
- Frame: x ahead (north), y to starboard (east), z down. Body frame the same on the vehicle.

## Changing a design without editing files

```python
from gpusonar.sonar import SonarModel
from gpusonar.config import load_scenario_config
from gpusonar.sim import run_scenario

# A forward-looking array with twice the elements and a 10 degree beam tilted 6 degrees up.
fls = SonarModel.load("fls_330k", {"array": {"count": 256},
                                   "fov": {"vertical_deg": 10.0, "tilt_up_deg": 6.0}})

# A longer SAS receive array (1.81 m, the MINSAS 180 class): 80 elements instead of 48.
sas = SonarModel.load("sas_minsas120", {"array": {"count": 80}})

# A Popoto array of 64 hydrophones: eight boards follow automatically.
popoto = SonarModel.load("popoto_array", {"array": {"count": 64}})

# The same scenario with another array size, speed or target.
cfg = load_scenario_config("popoto_far_mover", {"sonar_overrides": {"array": {"count": 16}},
                                                "platform": {"velocity": [2.5, 0, 0]}})
result = run_scenario(cfg)
```

## Sonar files

| Key | Meaning |
| --- | --- |
| `name`, `mode` | `mode` is `fls`, `sas` or `flank` (which processing the record is meant for) |
| `nominal_sound_speed` | m/s, used for wavelengths and default beamwidths |
| `unconfirmed` | list of keys whose values are placeholders until hardware confirms them |

### `fov` (optional): field of view in one place

| Key | Meaning |
| --- | --- |
| `fov.horizontal_deg` | full width at half maximum of each element and of the transmit beam in azimuth |
| `fov.vertical_deg` | the same in elevation |
| `fov.tilt_up_deg` | both boresights tilted up from the horizontal `array.boresight` (negative looks down) |

An explicit `beamwidth_az` / `beamwidth_el` under `array` or `transmit` wins over the block.

### `array`: the receive elements

| Key | Meaning |
| --- | --- |
| `array.layout` | `uniform_line` (default), `line` (positions along one axis), `stacked_lines` (parallel rows at different heights) or `explicit` (x, y, z each) |
| `array.rows` | for `stacked_lines`: one entry per row, `{up, out, along}` in metres (towards the vehicle's top; along the boresight, for a curved hull; along the axis), optionally its own `count`, `pitch` or `positions`. Channels run row by row |
| `platform: {name, span_m, height_m, mounting}` | what there is to mount on (the test AUV's hull, the static drop-in frame, the 2 m × 9 m face): read by the studies, the diagrams and the tests, not by the processing. On the face also `usable_span_m` and `row_baseline_m` (the owner's working values: rows within 7 m, outer rows 1.5 m apart); on the static array `static: true` and `depth_m` |
| `array.interleave` | for `stacked_lines`: take this many channels from each row in turn instead (4 with two rows and boards of eight: a board then holds the same stations of both rows and they share its clock) |
| `array.count`, `array.pitch` | number of elements and spacing (m) of a uniform line |
| `array.positions` | for `line`: metres along the axis; for `explicit`: `[x, y, z]` per element |
| `array.axis` | direction the elements run along, body frame |
| `array.boresight` | direction the elements face, body frame |
| `array.origin` | offset of the array centre from the vehicle reference point |
| `array.element_width`, `array.element_height` | element size (m); beamwidth is 0.886 λ / size when not given |
| `array.beamwidth_az`, `array.beamwidth_el` | element beamwidths (rad); absent and no size means omnidirectional |
| `array.layouts`, `array.layout_choice` | several named layouts for one board (for example dense and sparse); keys beside `layouts` override the chosen one |

### `transmit`

| Key | Meaning |
| --- | --- |
| `transmit.kind` | `point` (a projector) or `plane_wave` |
| `transmit.lever_arm` | projector position, body frame |
| `transmit.boresight`, `transmit.direction` | where the projector faces; plane-wave direction |
| `transmit.aperture_width`, `transmit.aperture_height` | projector size; or `beamwidth_az`, `beamwidth_el` directly |
| `transmit.waveform.type` | `lfm`, `cw` or `gaussian` |
| `transmit.waveform.f_start`, `f_stop`, `duration`, `window_alpha` | band (Hz), pulse length (s), edge taper |
| `transmit.waveform.source_level_db` | dB re 1 µPa at 1 m; with `receive.sensitivity_db` it makes samples volts |
| `transmit.waveform.amplitude` | echo amplitude of a 0 dB target at 1 m when the chain is uncalibrated |
| `transmit.waveform.type: sampled` | a pulse given by samples (`gpusonar/sim/waveform.py`); `f_start`, `f_stop` are then the occupied band and are required. Echoes of a sampled pulse are synthesised by the C++ kernel (host or CUDA), or by the NumPy reference when that engine is selected; `table_factor: 0` or more than 32 `gain_knots` runs in NumPy, with a warning |
| `transmit.waveform.source` | for `sampled`: `lfm`, `hfm`, `stepped_fm` (with `steps`), `barker` (with `code`: its length), `mseq` (with `degree`, optional `taps`), `code` (with `code`: chip values, or `[re, im]` pairs), `samples` (inline list), `file` |
| `transmit.waveform.file`, `dataset` | `.wav` (real passband), `.npy` or HDF5 dataset (real passband, or complex envelope) |
| `transmit.waveform.samples`, `sample_rate`, `carrier` | inline samples (real passband, or `[re, im]` pairs of the envelope), their rate (Hz; a WAV file has its own), and the carrier removed from complex samples (default: the band centre) |
| `transmit.waveform.rolloff` | phase codes: fraction of the band over which the chips' spectrum is rolled off to zero (0.25) |
| `transmit.waveform.normalise` | `peak` (default: unit peak envelope, so `amplitude` keeps its meaning) or `none` |
| `transmit.waveform.oversample` | envelope samples per cycle of bandwidth for built-in sources and the least for files (8) |
| `transmit.waveform.interp_taps`, `interp_window`, `interp_beta` | interpolator: taps (16), `kaiser` (default, shape 12) or `hann` (TR 0702 eq. 17) |
| `transmit.waveform.gain_knots` | frequency intervals across the band for frequency-dependent absorption and beam patterns (8; 0: at the band centre) |
| `transmit.waveform.replica_sample_rate` | rate of the real replica stored in the record (default 4 × the top of the band) |
| `transmit.direct_path` | absent or `false` (default): as before. `true`: the simulator adds the pulse as the hydrophones hear it straight from the projector at `transmit.lever_arm` (`sim/sources.py` `direct_arrival`): each element's own distance for the level (source level, 1/r), delay and phase (the array's near field), the element patterns, the projector's pattern per element when it has one, and the surface and bottom images when the scenario's provider has boundaries. `{boundaries: false}`: the straight arrival only. Analytic pulses: the C++ kernel on both engines; sampled pulses: C++ or NumPy by engine. Matters when the receive window includes the pulse, or with `receive.overload` |

### `receive`

| Key | Meaning |
| --- | --- |
| `receive.kind` | `real` (passband samples) or `baseband` (complex, already down-converted) |
| `receive.sample_rate`, `adc_bits`, `adc_full_scale` | sampling |
| `receive.sensitivity_db`, `gain_db` | hydrophone sensitivity (dB re 1 V/µPa) and front-end gain |
| `receive.window.range` | `[near, far]` in metres: the receive window |
| `receive.tvg` | analogue gain ramp: `time` and `gain_db` knots |
| `receive.gain_ramp` | a named hardware ramp instead (`gpusonar/hardware.py`) |
| `receive.mux.groups` | which elements each transmit captures, when fewer channels than elements |
| `receive.board_size` | channels per capture board; boards are created to fit `array.count` |
| `receive.boards` | boards spelled out, each with clock offset, drift, jitter and reference channel |
| `receive.reference` | the shared synchronisation signal wired to every board |
| `receive.overload` | absent (default): the receiver as before. Present (even empty): a behavioural front end that clips and recovers (`sim/receiver.py` `apply_front_end`; `PHYSICS_NOTES.md` Sec. 21), NumPy on both engines, real (passband) records only. **No key below defaults to a hardware value**; example values in the scenarios and tests are illustrative. With `transmit.direct_path` the time from the transmit to the opening of the window is simulated too (direct arrival only), so the window starts in the state the transmit left. The truth (per-channel overload, intervals) is in `ScenarioResult.front_end`, `simulate_ping(front_end_truth=...)` and the simulator taps, never in the record |
| `receive.overload.preamp: {limit_v, gain_db, dead_time, dead_output, recovery_tau, scope}` | the input stage, modelled only when given. `limit_v`: its output swing in volts at its output, required (one number, or `[negative, positive]` magnitudes for unequal rails); `gain_db`: the part of `receive.gain_db` up to that output (default: all of it); `dead_time` (s, 0): after its last clipped sample the output stays on that rail (`dead_output: rail`, default) or at zero (`zero`); `recovery_tau` (s, 0): the gain then returns as 1 − exp(−t/τ); `scope`: `channel` (default) or `board` (the dead time and recovery are shared by all channels of a capture board) |
| `receive.overload.highpass_corner_hz` | the AC coupling between the input stage and the ADC, modelled only when given: a one-pole high-pass. A clipped burst whose output has a mean (unequal rails, a rail held through a dead time) leaves a charge that decays with 1/(2π f) and can hold the ADC on its rail |
| `receive.overload.adc: {limit_v, clip, dead_time, dead_output, recovery_tau, scope}` | the ADC stage: clips at `receive.adc_full_scale` (or `limit_v`; `clip: false` removes the stage); the other keys as for `preamp` |
| `receive.overload.gain_switch: {gain_db, until, ramp, report}` | transmit-time gain switching, the mitigation to compare: the input gain is changed by `gain_db` (required, negative) until `until` seconds after the transmit (default: the pulse length) and returns linearly in dB over `ramp` seconds. `report: true` (default) writes it into the record's existing gain-ramp fields, which `apply_tvg` undoes (the live chain does not); not together with `receive.tvg` unless `report: false` |

## Diagrams and checks that follow the files

| Tool | What it writes |
| --- | --- |
| `tools/design_report.py` | `docs/DESIGN.md`, `docs/design/*.png`: derived numbers, link budgets and the requirement checks |
| `tools/scenario_diagrams.py` | `docs/scenarios/<name>.png` for every scenario: plan and side view, and for Bellhop scenarios the sound-speed profile and transmission loss |
| `tools/popoto_range_study.py` | `docs/design/popoto_range_study.png`: source level against frequency for a given range (`--range`, `--ts`, `--pulse`, `--depth`) |
| `tools/direct_sound_study.py` | `docs/design/direct_sound_study.json`: the projector's direct level and the overload in dB at every hydrophone of a sonar file, and for 0.1 s and 1 s pulses the time after the pulse at which each channel is back within tolerance and the minimum range that leaves, for illustrative recovery parameters (`--pulses`, `--sets`, `--gain-tolerance-db`, `--offset-tolerance-db`, `--overrides`) |
| `tools/architecture_diagrams.py` | `docs/architecture/*.png`: data flow and taps, stage by backend, deployment, simulator calibration |
| `tools/rough_surface_bands.py` | `docs/figures/rough_surface_bands.png`: what makes the bands under a rough sea and which frame they are fixed in |
| `tools/gallery.py` | `docs/figures/<name>.png` and `<name>.gif`: the simulated image, detections and tracks, and an animation, with numbers in `gallery.json` (`--quick` thins the scene, `--no-animation` skips the GIF) |

## Scenario files

| Key | Meaning |
| --- | --- |
| `sonar`, `sonar_overrides` | which sonar, and changes to it |
| `fidelity` | `free_field`, `image_source` (flat surface and bottom, walls), `sediment` (refraction into the bed), `eigenray` (Bellhop, external) |
| `propagation` | options of that tier (boundary depths, bottom properties, bounce order, Bellhop grid, arrivals kept, wind). For `eigenray`: `environment: {depth, ssp_depth, ssp_speed, bottom_speed, bottom_density, bottom_attenuation, bathymetry}` is the ocean (sound-speed profile, bottom type, depth) |
| `time_spread` | time spreading of bounce paths (`sim/timespread.py`, D-62): `surface: {wind_speed}` or `{rms_height, mean_square_slope, rms_vertical_speed}`, `bottom: {rms_height, mean_square_slope}`, `taps` (6), `coherence_length` (10 m), `min_spacing`, `incoherent: false` for coherent loss only. Absent: bounce paths are fully coherent |
| `reverberation` item `{model: volume, layer: [top, bottom], sv_db}` | a volume-scattering layer (fish, plankton, bubbles) as SST's sheet at the layer's centre: strength per unit area = volume strength × thickness |
| `reverberation` item `{model: fluid_sediment, boundary_z, rho, nu, delta, w2, gamma2, sigma2}` | bottom scattering from interface roughness and sediment volume (`sim/bottom.py`, D-63) in place of Lambert's law. All six are needed; none has a default |
| `reverberation` item `geometry: bistatic` | take the transmitter's and the receiver's own positions and velocities (`sim/reverb_bistatic.py`): iso-time ellipses, scattering strength at the true grazing angles and bistatic azimuth (`fluid_sediment` uses the azimuth; Lambert and `volume` do not depend on it; `chapman_harris` is a backscatter fit used at the mean grazing angle), Doppler of each platform along its own ray, arrival elevation in the steering to the hydrophones. Default `monostatic`: the transmitter is taken to be at the array centre, moving with it. With `bistatic`: `n_elevation` (9) bins of arrival elevation, `max_doppler_bins` (128; the arrival directions are split by total Doppler shift in bins half the pulse's resolution 1/T wide, at most this many, so that eigenray pairs and times with different shifts keep their own), `azimuth_subsamples` (1) azimuths evaluated per sector |
| `reverberation` item `strength_map: {range: [...], offset_db: [...], azimuth_deg: [...], origin: [north, east]}` | a boundary whose strength varies over it: dB added to the scattering strength against horizontal range (and azimuth, degrees from north towards east) about a point fixed in the world; `offset_db` is `[ranges]` or `[azimuths][ranges]`, interpolated linearly, held beyond the last range. Needs `geometry: bistatic`. The boundary stays flat |
| `reverberation` item `statistics: {model: k, shape, cell_time, cell_azimuth_deg}` | K-distributed reverberation: a gamma texture of unit mean and the given `shape` per resolution cell on the Gaussian speckle (mean power unchanged; `shape` has no default; large `shape` is Rayleigh). A cell is `cell_time` long (default 1/bandwidth of the pulse) and `cell_azimuth_deg` wide (default wavelength / array extent). Default `{model: rayleigh}` |
| `reverberation` item `scatterer_motion: {velocity: [n, e, d], rms_speed, rms_vertical_speed, wind_speed}` | scatterers that move (needs `geometry: bistatic`). `velocity` (m/s, world): their mean velocity (surface drift, a current carrying a volume layer), shifting each cell by `-f/c V·(a_in + a_back)` (TR 0702 eq. 72). `rms_speed`: random motion on every axis (volume scatterers); `rms_vertical_speed`: vertical only (a heaving surface); `wind_speed` sets the latter to the rms vertical speed of a fully developed wind sea (the `time_spread.surface` parameter of the same name describes the same sea for forward bounces). They spread each direction's Doppler into a Gaussian of width `(f/c) sqrt(rms_speed² |k|² + rms_vertical_speed² k_z²)`, `k = a_in + a_back` (2 for backscatter). No drift is derived from the wind: give `velocity`. A spread below 1/20 of the pulse's own resolution 1/T changes nothing |
| `reverberation` item `bathymetry: ...` | a seabed with relief in place of `boundary_z` (needs `geometry: bistatic`; `sim/bathymetry.py`): `{plane: {depth, slope: [dD/dx, dD/dy], point: [x, y]}}`, a grid `{x: [...], y: [...], depth: [[...]]}` (`depth[x][y]`), or `{range: [...], depth, azimuth_deg: [...], origin: [x, y]}` about a fixed point; depths positive down, held beyond the tables' edges. Each cell is at its own depth with its facet's true area, grazing angles and bistatic azimuth in the facet's frame; cells hidden from either platform by nearer relief, or turned away, return nothing (`shadowing: false` turns the hidden test off). Legs with a bottom bounce before the cell are left out by default (`bottom_bounces: flat` keeps them as flat-bottom image paths, an approximation); with the `eigenray` tier Bellhop is run once per azimuth sector over that bearing's depth profile instead (transmitter and receiver over the same point). A level bathymetry is the flat calculation exactly |
| scene item `seafloor` with `bathymetry: ...` | the same description for a point-scatterer seabed: points at the bed's depth with its facet normals, amplitudes scaled for the facet's true area (`depth`, `slope`, `ripples` are then not used) |
| `reverberation` item `band_taper: true`, `guard: seconds` | against leakage of the strong start of a long record into its weak end in the reverberation generator (matters when the reverberation falls by more than about 50 dB across the record): roll the pulse spectrum off above half the generation rate's Nyquist frequency and generate two pulse lengths (or `guard` seconds) of silence after the record. Off by default so that earlier results do not change |
| `time_spread.surface.bubble_loss_db` | energy absorbed by near-surface bubbles at each surface bounce: dB, or `{grazing_deg: [...], loss_db: [...]}`. No wind law is built in |
| `time_spread.surface.bubble_loss_db: {model: apl_1994, wind_speed}` | RECOMMENDED wind law: `1.26e-3 / sin(grazing) U^1.57 f_kHz^0.85` dB per bounce (U >= 6 m/s; below, the 6 m/s value times `exp(1.2 (U - 6))`), as implemented in arlpy (BSD) and UnderwaterAcoustics.jl (MIT), which attribute it to the APL-UW 1994 model (TR 9407 II-21; not read). No range or cap is stated by either; the grazing angle is floored at 2 degrees and a warning is raised above 20 m/s (our guards) |
| `time_spread.surface.bubble_loss_db: {model: hall_novarini_layer | hall_novarini_screened, wind_speed}` | our physical models from the Hall-Novarini population: every radius with its damped response and refraction in the layer (`_layer`), plus the sound an opaque layer scatters back (`_screened`, an upper bound that saturates near 13 to 15 dB at 10 degrees). 2 to 3 times the dB of `apl_1994` at 10 to 30 kHz, 10 m/s: for study, not the default choice |
| `time_spread.surface.bubble_loss_db: {model: hall_novarini, wind_speed}` | the same loss from a wind-speed law (`sim/surface.py`): Hall-Novarini bubble population with resonant extinction, two slant crossings of the layer. `wind_speed` defaults to the section's. Warns outside 3.3-330 kHz (zero there), above 20 m/s, and above 10 kHz, where it is probably too high. No better-founded published law could be read (none is printed on arXiv; journal sites refused): a measured table remains the recommended input, and this law an upper estimate |
| `reverberation` item `fluid_sediment`: `sediment` | a name in `configs/sediments.yaml` (or in `sediment_file`) supplying the six parameters; keys in the item override it. Shipped: the 20 sediment classes from `sandy_gravel` (-1 phi) to `clay` (9 phi), each complete, and `placeholder_sand` (unconfirmed, warns). `medium_sand` and `very_fine_silt` are read from arXiv:1809.08459 Table 1 (water 1000 kg/m3 and 1500 m/s assumed for the ratios); the other 18 take density, speed and loss from UnderwaterAcoustics.jl and `w2`, `gamma2`, `sigma2` from the published grain-size relations (the file's header and each row's `provenance:`; `docs/REFERENCES.md`). These are default values for a sediment class, not a survey of a site. An entry may give a `physical:` block in source units, or `reflection:` plus `scattering: {w2_cm4, gamma2, sigma2}`, instead of the six parameters |
| `propagation.bottom: {sediment: <name>}` (`image_source`) | the fluid half-space of a named sediment from `configs/sediments.yaml` (or `sediment_file`) in place of `{speed, density, attenuation}`: speed = the table's speed ratio x the scenario's sound speed, density = its density ratio, attenuation = its loss parameter x 40 pi / ln 10 dB per wavelength. Keys given beside the name win. Works in both engines (the name is resolved before the engine is called) |
| `propagation.environment: {sediment: <name>}` (`eigenray`) | the same for Bellhop's bottom half-space, in place of `bottom_speed`, `bottom_density`, `bottom_attenuation` (giving both is an error); the speed ratio is applied to the water sound speed at the bottom of the profile |
| `configs/sediments.yaml`: entries by kind | Every row has density, sound speed and loss (`reflection: {density, sound_speed, delta, water_density, water_sound_speed}`, read from UnderwaterAcoustics.jl, MIT, `src/stdlib.jl:113-136`, which attributes the table to APL-UW TR 9407 IV-6 Table 2) and is usable by name for bottom reflection (the two rows above). The 20 sediment rows also have `mean_grain_size_phi` and the scattering parameters. `rock` and `pebbles` are not fluid sediments: they stay `reflection_only: true`, `fluid_sediment` reverberation and seafloors refuse them (naming `w2`, `gamma2`, `sigma2`) unless the item gives those itself, and they carry `empirical_backscatter: {model: lambert, mu_db, mu_db_range, basis}` instead, a Lambert parameter from published measurements (`bottom.empirical_backscatter`; use `mu_db` with `model: lambert`). `pebbles`' value rests on one statement at one angle and warns. `sandy_gravel`'s sound speed is the corrected 1.337 x water (the source table's 1.357 is a typing error; the entry's note) |
| `bottom.sediment_from_grain_size(phi)` (Python) | the six parameters for a site's mean grain size in phi units (-1 to 9): `w2`, `gamma2`, `sigma2` from the published relations, density, speed and loss interpolated between the tabulated classes (our interpolation). Pass the result as the item's keys |
| `seafloor` with `model: fluid_sediment`: `sediment` | a name in `configs/sediments.yaml` (or `sediment_file`), as for the reverberation item; the item's own values win |
| `sea_state` (0 to 9) in place of `wind_speed`, anywhere in a scenario | the median wind speed of that sea state: 0.8, 2.6, 4.4, 6.9, 9.8, 12.6, 19.3, 26.5, 30.6, 32.9 m/s (`acoustics.SEA_STATE_WIND_SPEED`; printed in UnderwaterAcoustics.jl `src/stdlib.jl:411-422`, which attributes it to APL-UW TR 9407 II-4 Table 2). `run_scenario` rewrites every mapping that has `sea_state` to have `wind_speed` before anything reads it (`time_spread.surface`, its `bubble_loss_db`, `reverberation` items and their `scatterer_motion`, `receiver.ambient_noise`, `propagation` for the Bellhop tier, a `sea_surface` spectrum). Whole states only; giving both keys in one mapping is an error unless they agree. Functions called directly take `wind_speed`: use `acoustics.sea_state_wind_speed` |
| `reverberation` item `fluid_sediment`, and `seafloor` with `model: fluid_sediment`: `roughness_model` | `small_slope` (default since D-86: first-order perturbation theory where that is valid, the Kirchhoff result in the specular direction; what current literature uses), `perturbation` (first order alone; the default before D-86: name it to reproduce earlier records; it diverges at normal incidence), `kirchhoff` (near specular), `composite` (large-scale slope average, small grazing angles), `combined` (small-slope averaged over large-scale slopes: all three regimes). `small_scale_kh` (0.5) sets the composite split |
| `reverberation`: which surface model in which band (D-87) | **Below 12 kHz**: `chapman_harris` (fitted at 0.4-6.4 kHz, below about 40 degrees; above that an extrapolation; the formula is not confirmed against the print). **12 to 70 kHz, 3 to 15 m/s**: `bubble_layer_surface`, the one model here compared with data in a stated range (+-5 dB; a lower bound between about 30 and 70 degrees). **Above 70 kHz**: nothing we could open is validated; `bubble_layer_surface` there is an extrapolation and warns, `kraus_surface` states no range, and `chapman_harris` lies above the level a saturated bubble layer returns once the frequency passes (10 degrees grazing; wind 3 / 5 / 6 / 8 / 10 / 15 m/s) 416 / 127 / 84 / 43 / 26 / 10 kHz (`kraus_surface`: never / 511 / 223 / 74 / 36 / 12 kHz; `surface.saturation_crossing_frequency`). There is no default surface model: an item without `model` is a Lambert boundary, so every surface item names its law, and no shipped scenario's surface item lies in 12-70 kHz |
| `reverberation` item `chapman_harris` / `kraus_surface`: `cap: bubble_saturation` | opt-in: the strength is the smaller of the law and the bubble layer's saturation level, delta_r sin(grazing) / (8 pi delta) (-27.9 / -29.3 / -30.4 dB at 10 degrees for 12 / 30 / 68 kHz; `surface.bubble_saturation_db`). Without it the law is unchanged, and a warning is given once per law when it is used at 12 kHz or above where it exceeds that level at 10 degrees. The level is the thick-layer limit of the bubble term, not its largest value (the term peaks 3.7 dB higher at intermediate optical depth) and not a bound near normal incidence (wave facets) |
| `reverberation` item `{model: kraus_surface, boundary_z, wind_speed}` | empirical high-frequency surface backscattering strength (`sim/surface.py`) in place of Chapman-Harris. No validity range is stated in its source; warns below 10 kHz, held constant above 60 degrees |
| `reverberation` item `{model: bubble_layer_surface, boundary_z, wind_speed, extinction}` | surface backscattering strength from the near-surface bubble layer plus wave facets (`surface.bubble_layer_surface_db`): the two terms of the published high-frequency surface model whose formulas could be read (Gallaudet 2001, eqs. (8), (10)). The bubble term saturates (about -28 to -30 dB at 10 degrees for 12 to 70 kHz), where the empirical laws keep growing with wind and frequency. `extinction`: which bubble loss law gives the layer's optical depth, `apl_1994` (default) or `hall_novarini`. The small-scale roughness term of the published model is not included (its empirical function is not printed), so between about 30 and 70 degrees this is a lower bound. Stated range 12-70 kHz, 3-15 m/s; warns outside |
| `reverberation` item `{model: searay_bottom, boundary_z, bottom_type}` | empirical bottom backscattering strength by type: `mud`, `sand`, `gravel`, `rock` (or 1-4). `rock` lies above the largest Lambert law energy conservation allows (mu = 1/pi, -5 dB) by up to 7.5 dB between 10 and 45 degrees (-7.2 dB at 20 degrees against -14.3 dB) at every frequency, and warns once (D-88); for rock use the next row |
| `reverberation` item `{boundary_z, sediment: rock}` (or `pebbles`) | Lambert's law with the measured parameter of a `configs/sediments.yaml` entry that is not a fluid sediment: `rock` mu = -16 dB (published fits -23 to -10 dB), `pebbles` -9 dB (one statement; warns as unconfirmed). An explicit `mu_db` wins. A fluid sediment's name is refused here (use `model: fluid_sediment`). Against `searay_bottom: rock` the level is 14.3 / 18.1 / 15.8 dB lower at 10 / 20 / 45 degrees |
| `time_spread.surface.rms_height` beside `wind_speed` | the surface's rms height (m) in place of the wind law's (Pierson-Moskowitz); slope and vertical speed stay the wind's unless given too. Use it whenever a height is known: measured seas were 1.4 times and a third of the wind law's height (`docs/PHYSICS_NOTES.md` Sec. 21.4) |
| `time_spread.surface.height_scale` | multiplies the rms height in force (the wind law's, or `rms_height`); default 1 |
| `time_spread.tap_shape` | K-type level fluctuation of the scattered part of a bounce path: a gamma texture of this shape on each ray's taps (scintillation index of a tap 1 + 2 / shape). A number, or `{grazing_deg: [...], shape: [...]}`. Not given or infinite: complex Gaussian taps, the output unchanged. `time_spread.texture_rate` (default 0.25): how fast the texture changes, as a fraction of the tail's Doppler width. A free parameter fitted to data, not a derived one |
| `time_spread.angle_spread` | `true`: each tap of the tail leaves and arrives in the direction of the boundary facet that gives its delay, so an array sees the tail with reduced spatial coherence (`false`). Works on both engines |
| `time_spread.doppler` | `true`: the tail varies during a ping (frequency spreading); each tap becomes `doppler_lines` rays with their own frequency shift (`false`: the tail changes only from ping to ping). Works on both engines (the C++ path pairing carries a range rate per ray); paths per scatterer grow by about `doppler_lines` squared |
| `time_spread.doppler_spectrum` | `gaussian` (default: rms width 2 f v sin(grazing) / c / sqrt(1 - coherent fraction), derived in `sim/timespread.py`; the shape measured channels show) or `lorentzian` (SST's AR(1) process, half-width 2 f v sin(grazing) / c). Also sets the ping-to-ping fading |
| `time_spread.doppler_lines` | sinusoids the process of each tap is made of (8). Changing it changes the realisation |
| `time_spread.decay_time` | `facet` (default since D-85: mss (1 + sin² grazing) t1 t2 / (t1 + t2) with the shape that Gaussian facet slopes give; measured-over-model median 0.99 at the one site with measured channels, scatter a factor 5 per arrival) or `rule_of_thumb` (TR 0702: 2 mss t1 t2 / (t1 + t2), exponential; the default before D-85, about twice too long at that site; name it to reproduce TR 0702 or earlier records) |
| `time_spread.multi_bounce` | `convolve` (default: the convolution of the bounces' tails; a bounce delays the tail only when it scatters) or `sum` (the earlier rule: one exponential whose mean is the sum of the bounces' means; wrong when the bounces differ) |
| `environment` | `sound_speed`, `temperature`, `salinity`, `water_depth` |
| `absorption` | `francois_garrison` (default), `ainslie_mccolm`, `thorp` (the two-term expression SST uses, TR 0702 eq. (42); no temperature, salinity or depth dependence; for like-for-like comparisons at low frequency: it warns above 100 kHz, where it lacks the pure-water term) or `none` |
| `seed` | random seed of the scene, noise and reverberation |

### `platform` and `pings`

| Key | Meaning |
| --- | --- |
| `platform.position`, `rpy`, `velocity` | start position, attitude (rad) and velocity; the platform moves between and during pings |
| `platform.stop_and_go` | a rail: still during each ping |
| `platform.motion` | sway and heave that navigation does not report (`sway_rms`, `heave_rms`, `correlation_pings`, `nav_knows_disturbance`) |
| `platform.raster` | a stop-and-go grid of positions: `x: [start, stop, step]`, `y: [...]` |
| `platform.legs`, `platform.speed` | survey legs: each `{start, heading_deg, pings or length}`; yaw follows the heading. For a multi-element SAS, set `speed × pings.interval` to half the array length so phase centres are evenly spaced |
| `pings.count`, `pings.interval` | number of transmits and time between them |

### `scene`: a list of objects

Every object may have `label`, and `velocity: [vx, vy, vz]` to move it.

| `type` | What it is | Main keys |
| --- | --- | --- |
| `points` | point targets | `positions`, `target_strength_db` |
| `polar_grid` | point targets on a range × bearing grid | `ranges`, `bearings_deg` |
| `wire` | a thin line of points | `start`, `end`, `spacing` |
| `seafloor` | random scatterers on the bed | `x_range`, `y_range`, `density`, `lambert_mu_db`, `depth`, `slope`, `ripples` |
| `plate` | a flat surface | `origin`, `edge_u`, `edge_v`, `density`, `mu_db` |
| `box` | six faces; casts a shadow | `min`, `max`, `density`, `mu_db` |
| `sphere` | casts a shadow | `centre`, `radius`, `density`, `mu_db` |
| `cylinder` | casts a shadow | `start`, `end`, `radius`, `density`, `mu_db`, `caps`, `occlude` |
| `cable` | cylinders along waypoints | `waypoints`, `radius`, `density`, `mu_db` |
| `hull` | a ship hull, afloat or wrecked | `origin`, `length`, `beam`, `depth`, `heading_deg`, `count`, `mu_db`, `keel_down`, `occlude` |
| `rocks` | random spheres on the bed | `x_range`, `y_range`, `count`, `radius_range`, `depth` |
| `debris` | random boxes on the bed (crates, blocks); each casts a shadow | `x_range`, `y_range`, `count`, `size_range`, `depth`, `density`, `mu_db` |
| `boat` | a boat afloat: hull, fin keel, propeller and rudder cluster, and optionally the bubble wake left astern (labelled `<label>_wake`, fixed in the water) | `origin` (amidships, waterline), `length`, `beam`, `draft`, `keel_depth`, `heading_deg`, `count`, `propeller_ts_db`, `wake: {length, count, strength_db_per_m, spread, depth}` |
| `sea_surface` | a patch of the surface with waves, seen from below: facets whose tilt follows the waves, normals into the water. The wave shape travels through the facets at each train's deep-water speed; the speckle is new every ping | `x_range`, `y_range`, `density`, `mu_db`; `spectrum: {wind_speed, significant_height, direction_deg, spread_deg, components}` for an irregular wind sea and/or `waves: [{amplitude, wavelength, direction_deg, phase}]` for regular trains; `isotropic_db` (scattering that does not depend on tilt); `velocity` (surface drift); `frozen: true` (waves do not travel); `coherent: true` (keep one speckle) |
| `pool` | walls and floor facing inwards | `x_range`, `y_range`, `z_range`, `density` |
| `volume` | scatterers through a box | `min`, `max`, `count`, `rms_amplitude` |
| `seafloor` with `model: fluid_sediment` | the same random scatterers, but each one's strength follows the bottom scattering model (`sim/bottom.py`: interface roughness plus sediment volume) at every ping's own grazing angles, instead of Lambert's law. Same mean level as a `reverberation` boundary of that model | `rho`, `nu`, `delta`, `w2`, `gamma2`, `sigma2` (all required; the keys of the `fluid_sediment` reverberation boundary), `offset_db`; not `lambert_mu_db` |
| `target` | a rigid set of highlights and smooth shapes on a trajectory (`sim/targets.py`); amplitudes and glint positions are worked out at each ping from where the projector and receiver are | `origin`, `velocity`, `heading_deg`, `pitch_deg`, `roll_deg`, `turn_rate_deg`; `highlights`, `shapes`, `aspect_table`, `common_center`, `random_position`, `occlude` (below) |

A `target`'s frame has x ahead, y to starboard, z down.

| `target` key | Meaning |
| --- | --- |
| `highlights` | a list of `{position, target_strength_db` (or `amplitude`)`, delay, response, aspect_table}`: `delay` is seconds between reception and re-emission; `response: {frequency, magnitude_db, phase_deg}` is a complex frequency response, used as its value at the centre frequency plus its group delay there; `aspect_table` (as below) is added to that highlight's strength |
| `aspect_table` | `{aspect_deg, ts_db}` and optionally `elevation_deg` (then `ts_db` is `[elevations][aspects]`): target strength in dB re 1 m² against the sonar's bearing in the target frame (0 = from ahead, 90 = starboard beam; a table ending at 180 is symmetric, one reaching 360 is periodic), interpolated linearly in dB. On the target it is the strength of the whole highlight set (their own strengths then only share it out) |
| `shapes` | smooth rigid bodies that return specular glints (Kirchhoff approximation, `k a >> 1`): `{shape: sphere, centre, radius}`; `{shape: cylinder, start, end, radius, caps}`; `{shape: plate, origin, edge_u, edge_v, two_sided}`; `{shape: disc, centre, normal, radius}`. Each takes `reflection` (1 rigid, -1 pressure release) and, except the sphere, `resolved: true` with `spacing` (m, at most half a wavelength): a row or grid of coherent points instead of one highlight carrying the far-field pattern. Needed inside `L²/λ`, i.e. for imaging; a warning says so |
| `common_center` | SST's `commonCenter`: every highlight takes the direction and spreading loss of the target's origin and keeps only its own arrival time |
| `random_position` | RMS (m, per axis) of a Gaussian jitter of the highlight positions, drawn once |
| `occlude` | the target's spheres and cylinders also cast shadows (targets that do not move) |

### Background

| Key | Meaning |
| --- | --- |
| `reverberation` | a list of boundaries: `{boundary_z, mu_db}` (Lambert) or `{boundary_z, model: chapman_harris, wind_speed}`. Without `mu_db`, a model or a sediment the Lambert parameter is -29 dB (D-89; -27 dB before): the deep-water value at 530 and 1030 Hz (Mackenzie 1961 as quoted by Muggleworth, NPS 1994, hdl 10945/28529, p. 34), not a sonar-frequency one, so name a sediment (`{model: fluid_sediment, sediment: ...}`, `{sediment: rock}`) or give the site's `mu_db`; the bare default warns once at 10 kHz and above. Every shipped scenario gives `mu_db` and is unaffected. Fixed in the world: a moving sonar sees it Doppler shifted by bearing, through its own beam patterns. `offset_db` moves a boundary's strength whatever its model (what `tools/calibrate_sim.py` writes). A surface modelled by a `sea_surface` scene item must not also be listed here |
| `receiver.ambient_noise` | `{level_db}` flat, or `{wind_speed, shipping}` for the four-term ambient fit |
| `receiver.ambient_noise.statistics` | non-Gaussian ambient noise (`sim/noise.py`). `{model: k, shape, correlation_time, common}`: K-distributed; without `correlation_time` TR 0702's KNoise (a K variate per sample and channel, then the spectrum filter: kurtosis 3/shape for a white spectrum only); with it a unit-mean gamma texture held over cells that long on the Gaussian noise (spectrum and array coherence kept), shared by all hydrophones unless `common: false`. `{model: alpha_stable, alpha}`: impulsive symmetric alpha-stable samples, white in band, independent channels; for alpha < 2 there is no variance and the configured level sets the scale (that of the Gaussian noise of the same spectrum; identical to it at alpha = 2). Default (no key, `model: gaussian`, infinite `shape`): unchanged |
| `receiver.ambient_noise: {model: fitted, file, gain_db, channels, include_tones, allow_uncovered}` | noise generated from a model fitted to a multichannel recording by `tools/fit_noise.py` (`sim/noise_fit.py`): its spectrum, the coherence between hydrophones and its tail. Levels are in the recording's sample units; `gain_db` (0) rescales. The model's channel count must match the sonar's, or `channels` picks a subset; frequencies the recording did not cover are refused unless `allow_uncovered` (zero noise there); `include_tones` (false) adds back the lines the fit removed |
| `sources` | radiating sources heard one way (`sim/sources.py`): a list of `{label, origin (or position), velocity, heading_deg, turn_rate_deg, signal}`; `signal` is one mapping or a list. `{type: noise, level_db, slope_db_per_octave, reference_frequency, band}` or `{type: noise, spectrum: {frequency, level_db}}` (dB re 1 uPa^2/Hz at 1 m); `{type: tonals, lines: [{frequency, level_db, phase_deg}], family: {fundamental, harmonics: [[n, 20 log A_n, phase_deg]]}}` (rms source level for a line; TR 0702 eq. 29 for a family); `{type: waveform, waveform: {...}, source_level_db, times | repeat: {first, interval, count} | ping_offset}` (any transmit waveform; default the sonar's, emitted with each ping); `{type: file, path, dataset, sample_rate, level_db | scale, start, loop}`. Levels use `receive.sensitivity_db + gain_db` as ambient noise does. Pulses run in the C++/CUDA kernel, the rest in a NumPy streaming synthesiser |
| `sources[].stream: {knot_interval, oversample}` | the streaming synthesiser of continuous one-way signals (`noise`, `tonals`, `file`): the retarded time is solved exactly every `knot_interval` seconds (0.004) and the delay read linearly between, which is off by h² v_t² / (8 r c) seconds for a source passing at range r with speed v_t across the line of sight; `knot_interval: auto` halves the spacing until that is under 0.01 rad at the top of the band, and a fixed spacing that leaves more than 0.1 rad warns. `oversample` (2): the emission-time series is held at this multiple of the record's rate |
| `pings.transmit: false` | a passive capture: no echoes, reverberation or transmit reference; the record keeps its form and length (the receive window) |
| `reverberation` item `exclude_specular: true` (or `{mean_square_slope, rms_height, reflection}`) | `geometry: bistatic` items only: take out of the scattering strength the near-specular part that `time_spread` already carries as the tail of each bounce (`sim/reverb_specular.py`). Use it when the transmitter's pulse also arrives one way over that boundary (a `sources:` entry for a separate transmitter). `true` reads the slopes and height from `time_spread` and the reflection coefficient from the provider. Refused with a warning on a monostatic item |
| `receiver.noise_rms`, `receiver.adc` | white noise at the ADC; whether to quantise |
| `synthesis.beam_floor_db` | skip paths this far down both beam patterns (speeds up narrow-beam scenes) |

## Processing chains and deployment files

| File | Read by | Keys |
| --- | --- | --- |
| `configs/chains/<name>.yaml` | `pipeline.run` | `source: {type, ...}`, `steps: [{stage, ...}]`, `backend`, `output: {geotiff, log, debug_dir, taps: {dir, stages}}` |
| `configs/deploy/<name>.yaml` | `tools/sonar_live.py` | `source`, `nav: {port, host}` or `nav: {static: {...}}`, `chain`, `output: {recording, report, report_udp, taps: {dir, pings}}` |
| `configs/nav/<name>.yaml` | `tools/nav_adapter.py` | a mapping from a vendor's navigation stream to the chain's datagrams: `format`, `csv`, `time`, `position`, `attitude`, `velocity`, `sound_speed`, `altitude`, `water_depth`, `sigma`, `output` (`python/gpusonar/navadapter.py`; `generic_csv.yaml` is a worked example, `sprintnav_template.yaml` a template with every vendor-side entry unconfirmed) |

### Steps of a chain file: keys derived from the record, and where the defaults live

A step passes on only the keys its config gives; every other parameter takes the default of the function it
calls (`recon.sas_image`, `recon.fan_image`, `recon.stripmap_grid`, `motion.MotionOptions`), stated there once.
Keys added by the audit of hard-coded assumptions (`HARDCODED_AUDIT.md`); each default reproduces the earlier behaviour:

| Key | Meaning |
| --- | --- |
| `{stage: tvg}`: `spreading` | `two_way` (40 dB per decade: an echo; the default), `one_way` (20: a passive or transponder arrival), `auto` (two-way when the record has a transmit pulse) or a number. The older `spreading_db_per_decade` still works |
| `{stage: tvg}`: `absorption` | `none` (default: 0 dB/m, right at a few kHz), `record` (Francois-Garrison at the record's centre frequency for the temperature and salinity the record states; an error if it states none) or a number of dB/m **one way** (the law doubles it). At 330 kHz and 12 °C it is 0.083 dB/m: the default law is 50 dB short at 300 m. With `record`: `temperature`, `salinity`, `depth`, `ph` (8.0), `model`, `frequency` override what the record gives. The older `absorption_db_per_m` still works |
| `{stage: tvg}`: `max_gain_db`, `reference_range`, `undo_hardware_ramp` | the clamp (120 dB), the range of 0 dB (1 m), whether the recorded analogue ramp is divided out first (true). The step's log entry gives `clamped_samples`, `clamp_range_m` and `shortfall_db`: where the law was cut at the clamp and by how much at the last sample |
| `{stage: sas_image}`: `plane_z` | z (m, down) of the imaging plane, or `seafloor`: from the records (each ping's z plus its DVL altitude; without a DVL, `environment.water_depth` with the surface at z = 0; an error with neither). Default 0.0, which is the seabed only for a scenario whose origin is on the seabed. The log entry gives `plane_z` and the `seafloor_z_from_records` |
| `{stage: sas_image}`: `fov_factor`, `cull_block`, `interp`, `directivity_weight`, `replica_window`, `position_lookup`, `full_support`, `beamwidth_az` | passed to `recon.sas_image` / `recon.stripmap_grid` when given (their defaults: 1.13, 16, `sinc`, false, `none`, `analytic`, true, the element's) |
| `{stage: sas_image}`: `trajectory_allowance` | with a trajectory table (`position_lookup: software_table` or `texture`) and `fov_factor > 0`: how far the platform may stray from its first-order pose within a ping before the kernel's field-of-view pre-test rejects a pixel. Absent: 0.02 m (the fixed value). `table`: the largest residual in each ping's own table, at least 0.02 m (for long pings and turning platforms). A number: metres |
| `{stage: fan_image}`: `fov_factor`, `directivity_weight`, `stop_and_hop`, `replica_window` | passed to `recon.fan_image` when given, beside `interp` and `window` |
| `{stage: motion}`: `window_m`, `max_lag_m` | the correlation window and the largest change of range between pings, in metres of range; turned into samples from the record (n = 2 d fs / c). They replace `window_samples` (256) and `max_lag_samples` (12), which fit one sample rate. The log counts the windows dropped because the correlation peaked at the edge of the lag span |
| `{stage: motion}`: `seafloor_z` | z of the flat seabed (0.0), or `auto`: from the records, as `plane_z: seafloor` |
| `{stage: volume}`: `interface_z`, `sediment_speed` | refraction at the sediment: both or neither; neither has a default |

Function arguments with the same meaning, for code that does not go through a chain file:

| Argument | Meaning |
| --- | --- |
| `core.to_baseband(..., max_range_rate, doppler_allowance_hz, cw_band_factor)` (also through `recon.prepare`) | the band kept for a CW pulse is `cw_band_factor` / duration (4) and for a sweep the bandwidth × `band_margin`; neither allows for Doppler. `max_range_rate` (m/s, two-way closing or opening speed) widens the band on both sides by 2 v f / c; `doppler_allowance_hz` by that many Hz (a one-way arrival: v f / c). Default 0: a 1 s tone at 5 kHz keeps 4 Hz and loses an echo closing at 1 m/s (6.7 Hz) |
| `core.cfar_window(range_step, resolution, guard_resolutions, train_resolutions, guard_m, train_m)` | guard and training cells for `core.cfar` from a window stated in range-resolution cells (4 and 32) or metres, for a map that steps `range_step` metres. `core.cfar`'s own `guard` and `train` are cells and assume one cell per resolution cell |
| `core.backproject(..., trajectory_allowance)`, `core.BackprojectSession(..., trajectory_allowance)` | as the chain key above (`None`, metres or `"table"`); `session.trajectory_residual_max` is the largest residual tabulated |
| `core.synthesize_into(..., arrival_passes, arrival_residual_limit)` | passes of the arrival-time solve: `(4, 4)` (default; 8 for sampled pulses), a pair, or `"auto"` (fewest that leave under 1e-4 rad). With passes stated, the call raises when the bound T_end (v / c)^n 2π f_top on what they leave exceeds `arrival_residual_limit` (0.05 rad; 0: no check); with none stated (as the simulator calls it) the count is raised to what the bound needs and a `RuntimeWarning` says so |
| `core.sim_image_source_eigenrays(..., max_order, bottom_coefficient, bottom)` | `max_order` is required and a bottom needs `bottom_coefficient` or `bottom`: the C++ library has no default bounce count, reflection coefficient or sediment. `core.sim_image_series_truncation(...)` gives the strongest ray the bounce limit leaves out, relative to the direct path |
| `recon.tvg_parameters(record, spreading=, absorption=, ...)`, `recon.record_absorption(record)`, `recon.seafloor_z(records)` | what the chain keys above call |

Fixed capacities of the C++ library, and what happens at each:

| Limit | Value | At the limit |
| --- | --- | --- |
| knot frequencies of a sampled pulse's gain (`core.MAX_ENVELOPE_BANDS`) | 33 | error |
| absorption table over the transmit band (`core.ABSORPTION_TABLE_SIZE`) | 16 points | a frequency outside the table is read at its nearest end; the synthesis raises when that would change a level by more than 0.1 dB over the longest path in the record's window |
| amplitude knots along a swept pulse | 16 | beam patterns and loss are linear in frequency between knots; not checked (a pattern null crossing a wide band is under-sampled) |
| arrival-time passes | (4, 4), 8 for sampled pulses | above `arrival_residual_limit`: stated passes raise an error, default passes are raised with a warning; `"auto"` derives the count |
| bounces of an image-source path (`max_order`) | no default in C++ (the Python provider's is 2) | paths with more bounces are not computed; `sim_image_series_truncation` measures what is left out (0.9 of the direct path at 12 km in 100 m over a bottom of 0.9 with two bounces) |
| trajectory residual allowed by the culling pre-test | 0.02 m | pixels at the beam edge are dropped for a ping that strays further; `trajectory_allowance: table` removes the limit |
| time-varying gain | 120 dB | the law is flat beyond the range where it reaches the clamp; reported by the step |
| interpolator of a sampled pulse's envelope (`core.RESAMPLER_TAPS`, `core.RESAMPLER_KAISER_BETA`) | 16 taps, Kaiser 12 | the C++ and NumPy engines must use the same two numbers; a test compares them |

A record states its sound speed and its elements' boresights: `core.PingRecord()` holds NaN for both, validation rejects
a record that still does, and the imaging, gain and synthesis entry points raise. The motion solver's
`MotionSolveOptions` likewise has no default sound speed or residual scales (`motion.py` derives them from the carrier).

`nav` and `source` keys added for an inertial unit with a Doppler log and for decimating in the receiver (anything left out keeps its default):

| Key | Meaning |
| --- | --- |
| `nav.port`, `nav.host`, `nav.static` | as before: the UDP port the navigation datagrams arrive on, or a fixed pose |
| navigation datagram | `time_ns`, `x`, `y`, `z` (m north, east, down), `roll`, `pitch`, `yaw` (rad), `vx`, `vy`, `vz` (m/s over ground); optional `sound_speed` (m/s at the vehicle), `altitude` (m above the bottom), `bottom_lock` (0 or 1; absent: locked when the altitude is a positive number), `water_depth` (m), and one-sigma `roll_sigma`, `pitch_sigma`, `yaw_sigma` (rad), `position_sigma` (m), `velocity_sigma` (m/s), `depth_sigma`, `altitude_sigma`, `water_depth_sigma` (m), `sound_speed_sigma` (m/s). Used by the chain: `roll_sigma` (in place of `elevation.roll_sigma_deg` when larger), the water depth's sigma, the sound speed. The rest is carried into the record (`metadata.extra_json` under `nav`; attitude, depth and Doppler-log samples in `navigation.ins`, `.depth`, `.dvl`) |
| `nav.water_depth: {source, dvl_offset_m, altitude_sigma_m, depth_sigma_m, hold_s, hold_sigma_m_per_s, hold_sigma_m_per_m, max_altitude_m}` | `source: altitude`: water depth = the vehicle's depth (z − `nav.surface_z`) + the Doppler log's altitude + `dvl_offset_m`, with sigma = the two sigmas in quadrature (from the fix, else these keys; both unconfirmed until the datasheet is read). Without bottom lock (no altitude, `bottom_lock` 0, or an altitude above `max_altitude_m`) the last value is held for `hold_s` (60 s) with its sigma growing by `hold_sigma_m_per_s` × the time and `hold_sigma_m_per_m` (0.05, an assumed bottom slope) × the distance travelled, in quadrature; after that the depth is unknown again. A `water_depth` sent in the datagram takes precedence. The sigma reaches `elevation.water_depth_sigma_m` (in quadrature) |
| `nav.keep_s`, `nav.surface_z`, `nav.profile_step_m` | fixes are kept this long (120 s: a ping is processed after its window has closed, and its attitude history must still be there; the earlier limit was 600 fixes whatever their rate); the surface's z; a (depth, sound speed) sample is kept for the built profile every time the depth has changed by this (0.5 m) |
| `source.decimate` (`flank_udp`) | the same keys as `chain.decimate`, but done in the receiver as blocks arrive: the full-rate ping is never held and the recording holds the decimated pings (complex float32). The reference for the boards' clocks is taken from `chain.sync` |

`chain` keys of a deployment file (`gpusonar/live.py`; anything left out keeps its default):

| Key | Meaning |
| --- | --- |
| `sync` | several capture boards: `{sonar: popoto_4x7}` takes the reference description from that sonar; `{reference: {...}}` gives it directly; absent for one board or one clock |
| `engine` | `python` (default: stage by stage, every tap) or `cpp` (`libs/live` does everything after recording) |
| `beamformer`, `backend` | `backprojection` (the kernel; host or CUDA), `farfield` or `mvdr` (host; C++ or NumPy by engine); `auto`, `host` or `cuda` |
| `doa`, `sbl: {grid_step_deg, max_sources, min_snr_db, dynamic_range_db, max_gates, min_contrast_db, max_iterations, tolerance}` | `map` (default: bearings from the beamformed map) or `sbl` (sparse Bayesian bearings at each detected range; runs the C++ engine). `max_gates`: ranges solved per ping, strongest first. `min_contrast_db`: a gate whose conventional spectrum peaks less than this over its median keeps the map's detection |
| `bearings_deg`, `look`, `window` | `[first, last, step]` from broadside; `starboard` or `port`; aperture weighting |
| `range_cell_m` | power is averaged into cells this long before detection |
| `cfar: {pfa, guard_m, train_m}`, `min_snr_db` | detector windows in metres; detections below this SNR are dropped |
| `tracker: {range_sigma, bearing_sigma_deg, confirm_hits, max_misses, initial_speed_sigma, process_noise}` | the Kalman tracker |
| `classify: {min_speed, static_speed, max_fit_rms, multipath: {max_excess_range, bearing_deg, velocity}}` | `gpusonar/classify.py`; `multipath: null` switches echo grouping off |
| `sound_speed` | m/s at the vehicle, when the navigation datagrams do not carry `"sound_speed"`: replaces the record's value for every stage |
| `dead_channels` | channel rows (after synchronisation and the removal of dedicated reference inputs) left out of every stage |
| `elevation` | absent (default): the chain as it was. Present (even empty) with an array of stacked rows: elevation and depth (`gpusonar/elevation.py`, `DEPLOYMENT.md`). With a one-row array it changes nothing |
| `elevation.sector_deg` | `[first, last, step]`: elevation searched, in the vehicle's frame, up positive (default −30 to 30 in 0.25° steps) |
| `elevation.sector_from_depth: {bounces, margin_deg}` | search only where water depth and range let an arrival come from (straight rays with this many boundary bounces, plus the margin); needs the water depth. What makes two rows several wavelengths apart unambiguous in shallow water |
| `elevation.arrival`, `first_arrival_db`, `first_arrival_snr_db` | `first` (default): the earliest arrival of the detection that is within 8 dB of the strongest and 13 dB above the noise, the direct path; `strongest` |
| `elevation.range_window_m`, `max_detections` | half-width of the range search about a detection (default 1.5 range cells; it always reaches back to the detection's leading edge); detections given an elevation per ping, strongest first (32) |
| `elevation.min_quality` | no elevation below this quality (0.6) |
| `elevation.ambiguity_db`, `ambiguity_sigma`, `alternatives_db` | a second lobe within `ambiguity_db` (1 dB) of the first is never trusted; one further down must lose by `ambiguity_sigma` (3) sigma of the noise. Otherwise the detection is flagged and the lobes within `alternatives_db` (6 dB) are listed as candidates for the track to choose among |
| `elevation.correct_bearing`, `depth`, `fold`, `classify`, `classify_sigma` | switches: horizontal bearing instead of cone angle; depth on detections and tracks; folding into the water column; multipath from depth, at this many sigma (2) |
| `elevation.water_depth`, `water_depth_sigma_m`, `surface_z` | water depth (m; default: what navigation sends, else unknown), how well it is known (into the bottom test and bottom-bounce depths), and the surface's z in the navigation's frame |
| `elevation.attitude_time`, `attitude_rate_sigma_deg_s` | `transmit` (default): attitude as the record's pose has it; `arrival`: as it was when each echo arrived, from the attitude samples in the record (written from navigation). Per second of gap to the nearest sample the sigma grows by the rate given |
| `elevation.gradient_per_s`, `gradient_sigma_per_s` | an assumed mean vertical sound-speed gradient (1/s; 0.017 is isothermal water): `depth_m` then follows the ray through c(z) = local speed + g (z − own depth). And how well it is known (default 0.05): range² × this / 2c is added to every depth sigma |
| `elevation.ssp: {depth: [...], speed: [...]}` | a sound-speed profile: `depth_m` is then found along the ray through it, not the straight line |
| `elevation.ssp` (forms) | besides `{depth: [...], speed: [...]}`: a table `[[depth, speed], ...]`, or `{file: profile.csv}` (a text file of two columns, depth m and speed m/s; commas or blanks; lines that do not start with a number are skipped) |
| `elevation.profile` | absent (default): depth as before. Present (even empty): the sound-speed ladder (`libs/live` `environment.hpp`). Each ping the best rung that is available is used, the depth is traced through its profile, `depth_sigma_m` gets what that profile leaves open instead of the gradient prior alone, and the rung is named in the report: `depth_profile_source` with every depth of a detection and a track, and `profile: {rung, bins, span_m}` on the report line |
| `elevation.profile.ladder` | the rungs, tried in this order (default `[supplied, built, family, local, prior]`): `supplied` (`elevation.ssp`), `built` (the vehicle's own profile), `family` (an assumed shape), `local` (only the sound speed at the vehicle is measured: the straight line or `gradient_per_s`, sigma from `gradient_sigma_per_s`, as without the ladder), `prior` (the same with a nominal sound speed). The sound speed counts as measured when navigation sent it or `chain.sound_speed` is set |
| `elevation.profile.built: {bin_m, sample_sigma_m_s, age_sigma_m_s_per_hour, distance_sigma_m_s_per_km, max_age_s, max_distance_m, min_span_m, extrapolation_sigma_per_s, file, cast}` | the profile built from the vehicle's own (depth, sound speed) samples, which navigation leaves in each record. Samples are binned by depth (`bin_m`, 2 m; a bin is a running mean of its last 16 samples and takes the time and place of the newest). A bin's sigma is `sample_sigma_m_s` (the sensor's accuracy: **no default**, the rung refuses to work without it), plus, in quadrature, `age_sigma_m_s_per_hour` × its age and `distance_sigma_m_s_per_km` × the distance from where it was sampled (0.5 each: placeholders, unconfirmed); bins beyond `max_age_s` or `max_distance_m` are not used. The rung is used once the bins span `min_span_m` (5 m). **Outside the sampled depths** the profile continues with `gradient_per_s` (or flat) and its sigma grows linearly from the edge bin's by `extrapolation_sigma_per_s` per metre (default: `gradient_sigma_per_s`). `file`: the bins are kept there, so a restart keeps the profile. `cast`: a text file of `depth speed` lines to start from (a static array's cast at deployment; its bins do not age) |
| `elevation.profile.supplied: {sigma_m_s}` | how well `elevation.ssp` is taken to hold along the path; absent: no claim, and the depth sigma is the gradient prior's through that profile |
| `elevation.profile.family: {kind, ...}` | an **assumption**, hung on the sound speed measured at the vehicle: `{kind: isothermal, gradient_per_s}` or `{kind: mixed_layer, gradient_per_s, layer_depth_m, thermocline_gradient_per_s, thermocline_thickness_m}` (a layer with `gradient_per_s`, a linear thermocline below it, `gradient_per_s` again below that). No parameter has a default. `sigma_m_s`: how well it is taken to hold (absent: no claim). Reported as `family:isothermal` / `family:mixed_layer` |
| depth sigma on the rungs with a profile | the profile's error is taken as one tilt about the vehicle (where the sound speed is measured), e(z) = min(the rung's sigma at z, `gradient_sigma_per_s` × the distance in depth from the vehicle); `depth_refraction_sigma_m` is the sensitivity of the traced depth to that tilt. It is linear and not folded at the boundaries: a value beyond the water depth means the depth is not known. A sigma is no better than its prior: water with a steeper gradient than `gradient_sigma_per_s` allows is under-reported outside what was sampled |
| `blanking: {recovery_s, taper_s, mode}` | absent (default): nothing is blanked. The projector's own sound overloads the inputs for the length of the pulse and for as long after as they take to recover (`recovery_s`: hardware, to be measured). `mode: zero` (default): every sample from the transmit until pulse length + `recovery_s` is set to zero before baseband conversion and pulse compression, with a raised-cosine return to full gain over `taper_s`; `mode: ignore`: the samples are left alone. In both the map starts at the minimum range c (pulse + `recovery_s` + `taper_s`) / 2 (plus half the decimation filter when that stage ran): the detector neither reports nearer nor trains on the blanked cells, and the report line carries `blanking: {min_range_m, mode}`. With `ignore` what the clipped pulse leaks through the matched filter further out stays. When listening (`sonar_live --listen`) the same window is applied after every transmit the blocks mark |
| `sync.min_quality` (beside `sonar` or `reference`) | absent (default): every reference burst is used as found. Set (0.5 is a reasonable start): a burst whose correlation quality is below it, or that has more than 2% of its samples within 2% of the ADC's full scale, is not used. The board is then timed from its other burst alone, without a drift, listed in the report's `sync.flagged` (its number; −number − 1 when neither burst could be used and the board is left as it is), and the ping's `sync.ok` is false. The remedy is in the hardware timing: send the first burst after the blanked stretch (the reference's `lead`) |
| `decimate` | absent (default): the chain converts each ping to baseband itself. `{}` or keys below: a raw ping is first band-selected and decimated to complex baseband by one streaming FIR (`libs/live` `decimate.hpp`), its boards synchronised in the same stage from the raw stretches kept around the two reference bursts (the reference lies above the band, so it cannot be done after). The recording (written before) holds the raw ping. A record that is already complex is not touched |
| `decimate.f_low`, `f_high`, `band_margin` | the band of interest (default: the transmit band) and how much is kept flat about its centre (1.5 × the band, as the chain's own conversion) |
| `decimate.output_rate`, `oversample` | the output rate wanted (default `oversample` × the kept band, 2: the chain's own rule). Realised as the input rate over an integer, never below what was asked: 102.4 kHz and 4 to 6 kHz give a factor of 17 (6024 Hz); `output_rate: 4096` gives 25 |
| `decimate.ripple_db`, `stopband_db`, `taps` | pass-band ripple allowed (0.1 dB peak to peak), rejection of whatever would alias into the kept band (80 dB), and the filter length (default: Kaiser's estimate for those and the transition width; 171 taps for the case above, 0.83 ms of delay) |
| `elevation.roll_sigma_deg` | the navigation's roll accuracy; added to every elevation sigma |
| `elevation.depth_process_noise`, `row_tolerance_m` | how fast a track's depth may wander (0.5 m per √s); hydrophones within this of one line are one row (0.02 m) |

The C++ program `build/bin/sonar_live` takes the same settings as `key=value` arguments;
`tools/sonar_live.py CONFIG --export-template template.h5` prints them for a deployment
file. Passive listening adds `f_low`, `f_high`, `n_fft`, `hop`, `n_average`, `passive_method`
(`DEPLOYMENT.md`). The elevation settings are `elevation=1` and `elevation_<key>=value`
(`elevation_first_deg`, `elevation_last_deg`, `elevation_step_deg` for the sector;
`elevation_first_arrival=0|1`; `elevation_sector_from_depth=0|1` with `elevation_sector_bounces`
and `elevation_sector_margin_deg`; `elevation_attitude_arrival=0|1`; `elevation_use_gradient=0|1`
with `elevation_gradient`, `elevation_gradient_sigma`; `elevation_water_depth_sigma`;
`elevation_ssp_depth` and `elevation_ssp_speed` as comma-separated lists; the others as in the
table), `sound_speed=1490` and `dead_channels=5,40,41`. Navigation datagrams may carry
`"sound_speed"` and `"water_depth"`. `strftime` fields in its `--recording` and `--report` paths are filled
with the start time.

### Sizes the chain derives, and the keys of the programs around it

Settings whose right value follows from the sonar are **not typed anywhere**: left out of a deployment file they are
derived from the first ping's record (its waveform, its array and its sound speed) by one function, `libs/live`
`derive.hpp`, which both engines call. Each is the larger of a *standing value* (the number typed until now, tuned on
kilometre-range arrays with a band of 2 kHz or more) and a *floor* that follows from the sonar; a key in the file
always wins. Every shipped flank sonar gets exactly the standing values (`tests/python/test_hardcoded.py`). What was
resolved is printed once on stderr (`chain settings {...}`) and is `LiveChain.resolved` in Python, with the lists
`given` (set in the file), `scaled` (the floor decided) and `warnings`.

| Key | Left out, it is | Floor from the sonar |
| --- | --- | --- |
| `range_cell_m` | 2 m | one range resolution, c / 2B (B: the swept band, or 1 / pulse length for a pulse that does not sweep) |
| `cfar.guard_m` | 100 m | 4 range resolutions |
| `cfar.train_m` | 300 m | 12 range resolutions (three guard floors) |
| `tracker.range_sigma` | 15 m | one range resolution |
| `tracker.bearing_sigma_deg` | 1° | a sixteenth of the beamwidth (wavelength at the band centre over one row's length) |
| `classify.multipath.bearing_deg` | 3° | three of the bearing-sigma floors |
| `classify.max_fit_rms` | 60 m | four range resolutions |
| `elevation.row_tolerance_m`, `passive.tonal.row_tolerance_m` | 0.02 m | an upper limit instead: a quarter of the shortest wavelength when that is less (rows are at least half of one apart) |
| passive `n_fft`, `hop` | 1024, half of `n_fft` | the next power of two above two transit times of sound across one row, at the record's sample rate |

Not derived, because they are choices about targets, false alarms and the site, not properties of the sonar: `cfar.pfa`,
`min_snr_db`, `tracker.confirm_hits`, `max_misses`, `initial_speed_sigma`, `process_noise`, `classify.min_speed`,
`static_speed`, `multipath.max_excess_range`, `multipath.velocity`, `bearings_deg`, the elevation sector. Their defaults are
in `ChainOptions` (`libs/live` `chain.hpp`) and `live.DEFAULTS`, which agree.

Keys added so that nothing about one sonar, board or network is fixed in code:

| Key (deployment file) | `sonar_live` option | Meaning (default) |
| --- | --- | --- |
| `source.sonar` | (the template) | required: `flank_udp` has no default sonar |
| `source.port` | `--udp PORT` | required when the C++ command is exported: no default port in code |
| `source.host` | `bind` | the address the block and navigation sockets listen on (`0.0.0.0`) |
| `source.timeout` | `idle_seconds` | the run ends after this long without a ping (Python) or a block (C++); 0: never. (C++ default 30 s; was fixed) |
| `source.sample_bytes` | `stream_sample_bytes` | bytes of one sample in a block: 2, 3 or 4, a signed little-endian integer with full scale 2^(8 bytes − 1) counts (3). The block header does not carry it: sender and receiver are set alike |
| `source.buffer_bytes` | `stream_buffer_bytes` | the socket receive buffer asked of the system (the larger of 8 MB and half a second of the stream: channels × sample rate × `sample_bytes`). What was granted is printed, with a warning when it is less (`net.core.rmem_max`) |
| `source.queue_mb` | `stream_queue_mb` | blocks waiting for the chain may hold this much (1024 MB; 0: no bound). Beyond it arriving blocks are dropped and counted in the closing summary (`blocks dropped at the full queue`, `dropped_blocks`); the queue had no bound |
| `source.queue_pings` | — | Python: finished pings waiting for the chain (8); the oldest is dropped and counted (`overrun_pings`) |
| `output.report_udp_max_bytes` | `report_udp_max_bytes` | a report line longer than this (65000) goes out as several datagrams, each a complete report line of the same ping with a share of the detections and tracks and `"part": [k, n]`; it was cut at 65000 bytes. `gpusonar.live.join_report_parts` puts them together |
| `chain.cell_statistic` | `cell_statistic` | what a range cell holds of the finer map it is made from: `mean` power (default, as always) or its `peak` sample. Changes what `snr_db` means: `DEPLOYMENT.md`, "What `snr_db` is" |
| `chain.sync.search_time` | `sync_search_time` | seconds either side of where a reference burst is expected (0.002): boards that start further apart need more |
| `chain.sync.tolerance_us` | `sync_tolerance_us` | what the boards must agree within after resampling, microseconds (a twentieth of a period at the top of the transmit band: 8.3 µs at 6 kHz; rows measuring elevation want less) |
| `chain.elevation.min_row_elements` | `elevation_min_row_elements` | a line of fewer hydrophones than this is not a row (3) |
| `nav.history_margin_s` | `nav_history_margin_s` | a ping's attitude history reaches this far before its transmit and after its window (2 s) |

Report fields added (only present when the thing happened, so an existing reader sees the lines it always saw):
`"sbl_fallback": true` (active and passive: SBL was asked for on more channels than its kernels take,
`core.SBL_KERNEL_MAX_CHANNELS` = 64, and ran on the host reference instead of failing); `"rows_as_snapshots": n`
(passive on stacked rows without `elevation`: bearings are cone angles solved from the n rows as snapshots of one
line); `"part": [k, n]` (a report split over datagrams); `"bearing_corrected": false` on a detection (stacked rows,
`elevation.correct_bearing` on, the elevation ambiguous: bearing and range are the row's own, where they used to be
placed by whichever candidate elevation was picked).

### `passive.tonal` (optional): Goertzel lines and DEMON when listening

Off unless given; without it the passive report lines are what they were, byte for byte.
In a deployment file it is the `tonal` key of a `passive` section (beside `f_low`, `f_high`,
`n_fft`, `hop`, `n_average`); `gpusonar.tonal.TonalAnalyzer` takes the same dictionary, and
`gpusonar.tonal.cpp_options` (and `tools/sonar_live.py --export-template`) turns it into the
program's arguments: `tonal=1` and one `tonal_<key>=value` per setting below
(`tonal_goertzel_block_s=0.5`, `tonal_demon_pfa=1e-6`; lists comma-separated:
`tonal_goertzel_lines=300,450`; `bearings_deg` as `tonal_bearing_first_deg`, `_last_deg`,
`_step_deg`; `goertzel.track` as `tonal_goertzel_track=0|1` with `tonal_goertzel_alpha`, `_beta`,
`_max_misses`, `_max_shift_hz`; `demon.band` as `tonal_demon_f_low`, `tonal_demon_f_high`;
`shaft_hz` and `blades` as `tonal_demon_shaft_min_hz`, `_max_hz`, `tonal_demon_blades_min`,
`_max`; `tonal_demon=1` switches DEMON on, giving lines switches Goertzel on).
The method and what each number in the report means: `gpusonar/tonal.py`, `DEPLOYMENT.md`.

| Key | Meaning (default) |
|---|---|
| `bearings_deg` | first, last, step of the beams lines and modulation are looked for on (`[-80, 80, 1]`) |
| `window` | aperture weighting of each row of hydrophones: `hann` (default) or `rect` |
| `row_tolerance_m` | hydrophones within this of one line are one row (0.02 m). Rows are steered separately and their powers added, so stacked arrays need no elevation |
| `attach_deg` | a line or source within this of a passive bearing of the same frame is reported with that bearing's index as `peak` (3°); otherwise `peak` is -1 |
| `goertzel.lines` | the frequencies to watch, Hz absolute (required for Goertzel; there is no search for unknown lines) |
| `goertzel.on` | `beams` (default): the array steered over `bearings_deg`, the line reported at the bearing it is strongest at; `hydrophones`: channel powers added, no bearing |
| `goertzel.block_s`, `hop_s` | seconds of data per estimate (1.0): the bins are 1 / `block_s` apart and the bin's noise bandwidth is 1.5 / `block_s`; seconds between estimates (default: `block_s`) |
| `goertzel.search_bins`, `guard_bins`, `noise_bins` | the line is looked for this many bins either side of where it is expected (4); bins skipped (3) before the bins the noise is taken from, each side (8). The bank is 2 × (sum) + 3 filters per line |
| `goertzel.min_snr_db` | a line is detected, reported and tracked above this signal to noise in its bin (10 dB) |
| `goertzel.track` | `{alpha, beta, max_misses, max_shift_hz}`: alpha-beta filter on the line's frequency from block to block (0.5, 0.1), the search centred on its prediction; after `max_misses` (3) blocks without the line the search returns to the nominal frequency; a track further than `max_shift_hz` from nominal is dropped (0: no limit). `null`: the search stays on the nominal frequency |
| `demon.band` | `[f_low, f_high]` Hz absolute, the band whose envelope is analysed (default: the passive band, else the transmit band) |
| `demon.envelope` | `square` (default): the band's power in each block; `hilbert`: its square root (the RMS of the analytic signal over the block) |
| `demon.envelope_rate_hz`, `window_hops` | sample rate of the envelope (100 Hz: modulation up to 50 Hz, lines looked for up to 0.8 of that); each envelope sample is from a block `window_hops` (2) envelope samples long, Hann weighted. The block should be long against the travel time across the array (see `DEPLOYMENT.md`) |
| `demon.segment_s`, `segment_overlap`, `n_average`, `min_segments` | seconds of envelope per spectrum (8: modulation bins 0.125 Hz apart), the fraction shared with the next (0.5), how many of the latest spectra are averaged (8) and how many must be there before a result is given (4) |
| `demon.f_min_hz`, `f_max_hz` | modulation lines are looked for between these (1 Hz; 0.8 of half the envelope rate) |
| `demon.background_bins`, `guard_bins`, `clip` | the background of a bin is the mean of this many bins each side (16), skipping `guard_bins` (2), taken twice: the second time without the bins more than `clip` (3) times the first estimate |
| `demon.pfa` | false-alarm probability per bearing and modulation bin; the line threshold follows from it, the number of spectra averaged and the number of background bins (1e-6; a result of 161 bearings × 313 modulation bins has 50 000 cells). Measured on noise alone: 1.0e-3 to 1.5e-3 for 1e-3 asked for, about 1e-4 for 1e-4; at 1e-6 more than asked for (three false single lines in four minutes of 61 bearings), none of them forming a shaft rate |
| `demon.max_lines`, `max_sources`, `min_separation_deg` | lines kept per source (12), sources reported (4); lines closer in bearing than `min_separation_deg` (4°) are one source, and a line is reported only at the bearing it is strongest at within that distance |
| `demon.shaft_hz`, `max_harmonic`, `harmonic_tolerance_bins`, `score_fraction` | shaft rates considered (`[0.5, 30]` Hz), the highest harmonic of it a line may be (12), how far (in modulation bins, 1.0) a line may lie from a harmonic, and the fraction of the best explanation's score the highest candidate rate must reach to be taken (0.9) |
| `demon.blades` | blade counts considered (`[2, 9]`): the harmonic number of the strongest line of the set, when it is in this range |
| `demon.gram` | put the DEMONgram (dB above the background over bearing × modulation frequency, 0.1 dB steps) in every DEMON result of the report (`false`: it is large; `TonalAnalyzer.gram` and `core.TonalAnalyzerCpp.gram()` always hold the latest) |

The simulator uses its C++ path and reverberation code by default; `GPUSONAR_SIM_ENGINE=numpy`
(or `engine="numpy"` on `run_scenario`, `simulate_ping` and the providers) selects NumPy.
For a sampled transmit pulse the same switch also selects the echo synthesiser (C++ kernel
by default, NumPy reference); analytic pulses are synthesised by the C++ kernel only.

Source types and how to tap, inspect and re-inject data: `ARCHITECTURE.md`.

## Build specs, and what they add to the other files

`configs/builds/<build>.yaml` describes a build in spec-sheet terms; `tools/build_from_spec.py` generates
the sonar, deployment and scenario files from it. Fields: `REFERENCE_DESIGN.md`.

| Key | Where | Meaning |
| --- | --- | --- |
| `unconfirmed` | any sonar or scenario file | dotted keys whose values are assumptions. Through `base:` the lists **accumulate** (a child no longer hides its base's flags) |
| `confirmed` | a file with `base:` | takes flags off the inherited list: the entry itself and anything below it (`confirmed: [receive]` clears `receive.boards`). Naming a key under `unconfirmed` again puts it back |
| `platform: {name, kind, span_m, height_m, usable_span_m, usable_height_m, row_baseline_m, mounting, static, depth_m}` | sonar file | what the array is mounted on; read by the diagrams, the parts list and the validator, not by the simulator or the chain |
| `hardware: {hydrophone, boards, sync, projector, navigation, sound_speed, water_depth, compute, storage}` | sonar file | the build's parts as the spec names them (models, inputs per board, where the boards sit for cable lengths, the synchronisation scheme and its tolerance target): for the diagrams and the parts list only |
| `build` | sonar file | the spec the file was generated from |
| `receive.window` | every stage sonar file | stated in each file from the spec's `operation.range_m` (first range `null`: the pulse's blind range) |
| `truth` | generated scenarios, `popoto_12km_2row`, `_3row` | what the self-tests check reports against; the simulator ignores it |
| `GPUSONAR_CONFIG_PATH` | environment | further config trees searched after `configs/` (how a generated build is used in place); `config.add_config_dir(path)` does the same in code |
