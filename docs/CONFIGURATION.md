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

## Diagrams and checks that follow the files

| Tool | What it writes |
| --- | --- |
| `tools/design_report.py` | `docs/DESIGN.md`, `docs/design/*.png`: derived numbers, link budgets and the requirement checks |
| `tools/scenario_diagrams.py` | `docs/scenarios/<name>.png` for every scenario: plan and side view, and for Bellhop scenarios the sound-speed profile and transmission loss |
| `tools/popoto_range_study.py` | `docs/design/popoto_range_study.png`: source level against frequency for a given range (`--range`, `--ts`, `--pulse`, `--depth`) |
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
| `time_spread.surface.bubble_loss_db: {model: hall_novarini, wind_speed}` | the same loss from a wind-speed law (`sim/surface.py`): Hall-Novarini bubble population with resonant extinction, two slant crossings of the layer. `wind_speed` defaults to the section's. Warns outside 3.3-330 kHz (zero there), above 20 m/s, and above 10 kHz, where it is probably too high. No better-founded published law could be read (none is printed on arXiv; journal sites refused): a measured table remains the recommended input, and this law an upper estimate |
| `reverberation` item `fluid_sediment`: `sediment` | a name in `configs/sediments.yaml` (or in `sediment_file`) supplying the six parameters; keys in the item override it. Shipped: `medium_sand` and `very_fine_silt` (read from arXiv:1809.08459 Table 1, which attributes them to TR 9407; water 1000 kg/m3 and 1500 m/s assumed for the ratios) and `placeholder_sand` (unconfirmed, warns). An entry may give a `physical:` block in source units instead of the six parameters |
| `reverberation` item `fluid_sediment`: `roughness_model` | `perturbation` (default, as before), `kirchhoff` (near specular), `small_slope` (joins the two), `composite` (large-scale slope average, small grazing angles), `combined` (small-slope averaged over large-scale slopes: all three regimes). `small_scale_kh` (0.5) sets the composite split |
| `reverberation` item `{model: kraus_surface, boundary_z, wind_speed}` | empirical high-frequency surface backscattering strength (`sim/surface.py`) in place of Chapman-Harris. No validity range is stated in its source; warns below 10 kHz, held constant above 60 degrees |
| `reverberation` item `{model: searay_bottom, boundary_z, bottom_type}` | empirical bottom backscattering strength by type: `mud`, `sand`, `gravel`, `rock` (or 1-4) |
| `time_spread.angle_spread` | `true`: each tap of the tail leaves and arrives in the direction of the boundary facet that gives its delay, so an array sees the tail with reduced spatial coherence (`false`). Works on both engines |
| `time_spread.doppler` | `true`: the tail varies during a ping (frequency spreading); each tap becomes `doppler_lines` rays with their own frequency shift (`false`: the tail changes only from ping to ping). Works on both engines (the C++ path pairing carries a range rate per ray); paths per scatterer grow by about `doppler_lines` squared |
| `time_spread.doppler_spectrum` | `lorentzian` (SST's AR(1) process, half-width 2 f v sin(grazing) / c) or `gaussian` (rms width 2 f v sin(grazing) / c / sqrt(1 - coherent fraction), derived in `sim/timespread.py`). Also sets the ping-to-ping fading |
| `time_spread.doppler_lines` | sinusoids the process of each tap is made of (8). Changing it changes the realisation |
| `time_spread.decay_time` | `rule_of_thumb` (TR 0702: 2 mss t1 t2 / (t1 + t2), exponential) or `facet` (mss (1 + sin² grazing) t1 t2 / (t1 + t2) with the shape that Gaussian facet slopes give) |
| `time_spread.multi_bounce` | `convolve` (default: the convolution of the bounces' tails; a bounce delays the tail only when it scatters) or `sum` (the earlier rule: one exponential whose mean is the sum of the bounces' means; wrong when the bounces differ) |
| `environment` | `sound_speed`, `temperature`, `salinity`, `water_depth` |
| `absorption` | `francois_garrison`, `ainslie_mccolm` or `none` |
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
| `reverberation` | a list of boundaries: `{boundary_z, mu_db}` (Lambert) or `{boundary_z, model: chapman_harris, wind_speed}`. Fixed in the world: a moving sonar sees it Doppler shifted by bearing, through its own beam patterns. `offset_db` moves a boundary's strength whatever its model (what `tools/calibrate_sim.py` writes). A surface modelled by a `sea_surface` scene item must not also be listed here |
| `receiver.ambient_noise` | `{level_db}` flat, or `{wind_speed, shipping}` for the four-term ambient fit |
| `receiver.noise_rms`, `receiver.adc` | white noise at the ADC; whether to quantise |
| `synthesis.beam_floor_db` | skip paths this far down both beam patterns (speeds up narrow-beam scenes) |

## Processing chains and deployment files

| File | Read by | Keys |
| --- | --- | --- |
| `configs/chains/<name>.yaml` | `pipeline.run` | `source: {type, ...}`, `steps: [{stage, ...}]`, `backend`, `output: {geotiff, log, debug_dir, taps: {dir, stages}}` |
| `configs/deploy/<name>.yaml` | `tools/sonar_live.py` | `source`, `nav: {port, host}` or `nav: {static: {...}}`, `chain`, `output: {recording, report, report_udp, taps: {dir, pings}}` |

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
| `dead_channels` | channel rows (after synchronisation and the removal of dedicated reference inputs) left out of every stage |
| `elevation` | absent (default): the chain as it was. Present (even empty) with an array of stacked rows: elevation and depth (`gpusonar/elevation.py`, `DEPLOYMENT.md`). With a one-row array it changes nothing |
| `elevation.sector_deg` | `[first, last, step]`: elevation searched, in the vehicle's frame, up positive (default −30 to 30 in 0.5° steps) |
| `elevation.arrival`, `first_arrival_db`, `first_arrival_snr_db` | `first` (default): the earliest arrival of the detection that is within 8 dB of the strongest and 13 dB above the noise, the direct path; `strongest` |
| `elevation.range_window_m`, `max_detections` | half-width of the range search about a detection (default 1.5 range cells; it always reaches back to the detection's leading edge); detections given an elevation per ping, strongest first (32) |
| `elevation.min_quality`, `ambiguity_db` | no elevation below this quality (0.6); a second lobe within this of the first is flagged (1.5 dB) |
| `elevation.correct_bearing`, `depth`, `fold`, `classify`, `classify_sigma` | switches: horizontal bearing instead of cone angle; depth on detections and tracks; folding into the water column; multipath from depth, at this many sigma (2) |
| `elevation.water_depth`, `surface_z` | water depth (m; default: the record's, else unknown) and the surface's z in the navigation's frame |
| `elevation.ssp: {depth: [...], speed: [...]}` | a sound-speed profile: `depth_m` is then found along the ray through it, not the straight line |
| `elevation.roll_sigma_deg` | the navigation's roll accuracy; added to every elevation sigma |
| `elevation.depth_process_noise`, `row_tolerance_m` | how fast a track's depth may wander (0.5 m per √s); hydrophones within this of one line are one row (0.02 m) |

The C++ program `build/bin/sonar_live` takes the same settings as `key=value` arguments;
`tools/sonar_live.py CONFIG --export-template template.h5` prints them for a deployment
file. Passive listening adds `f_low`, `f_high`, `n_fft`, `hop`, `n_average`, `passive_method`
(`DEPLOYMENT.md`). The elevation settings are `elevation=1` and `elevation_<key>=value`
(`elevation_first_deg`, `elevation_last_deg`, `elevation_step_deg` for the sector;
`elevation_first_arrival=0|1`; `elevation_ssp_depth` and `elevation_ssp_speed` as
comma-separated lists; the others as in the table), and `dead_channels=5,40,41`. `strftime` fields in its `--recording` and `--report` paths are filled
with the start time.

The simulator uses its C++ path and reverberation code by default; `GPUSONAR_SIM_ENGINE=numpy`
(or `engine="numpy"` on `run_scenario`, `simulate_ping` and the providers) selects NumPy.
For a sampled transmit pulse the same switch also selects the echo synthesiser (C++ kernel
by default, NumPy reference); analytic pulses are synthesised by the C++ kernel only.

Source types and how to tap, inspect and re-inject data: `ARCHITECTURE.md`.
