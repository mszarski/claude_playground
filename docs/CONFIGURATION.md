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
| `array.layout` | `uniform_line` (default), `line` (positions along one axis) or `explicit` (x, y, z each) |
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
| `time_spread.surface.bubble_loss_db` | energy absorbed by near-surface bubbles at each surface bounce: dB, or `{grazing_deg: [...], loss_db: [...]}`. No wind law is built in |
| `time_spread.surface.bubble_loss_db: {model: hall_novarini, wind_speed}` | the same loss from a wind-speed law (`sim/surface.py`): Hall-Novarini bubble population with resonant extinction, two slant crossings of the layer. `wind_speed` defaults to the section's. Warns outside 3.3-330 kHz (zero there), above 20 m/s, and above 10 kHz, where it is probably too high |
| `reverberation` item `fluid_sediment`: `sediment` | a name in `configs/sediments.yaml` (or in `sediment_file`) supplying the six parameters; keys in the item override it. Only `placeholder_sand` (unconfirmed, warns) is shipped |
| `reverberation` item `fluid_sediment`: `roughness_model` | `perturbation` (default, as before), `kirchhoff` (near specular), `small_slope` (joins the two), `composite` (large-scale slope average, small grazing angles), `combined` (small-slope averaged over large-scale slopes: all three regimes). `small_scale_kh` (0.5) sets the composite split |
| `reverberation` item `{model: kraus_surface, boundary_z, wind_speed}` | empirical high-frequency surface backscattering strength (`sim/surface.py`) in place of Chapman-Harris. No validity range is stated in its source; warns below 10 kHz, held constant above 60 degrees |
| `reverberation` item `{model: searay_bottom, boundary_z, bottom_type}` | empirical bottom backscattering strength by type: `mud`, `sand`, `gravel`, `rock` (or 1-4) |
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

The C++ program `build/bin/sonar_live` takes the same settings as `key=value` arguments;
`tools/sonar_live.py CONFIG --export-template template.h5` prints them for a deployment
file. Passive listening adds `f_low`, `f_high`, `n_fft`, `hop`, `n_average`, `passive_method`
(`DEPLOYMENT.md`). `strftime` fields in its `--recording` and `--report` paths are filled
with the start time.

The simulator uses its C++ path and reverberation code by default; `GPUSONAR_SIM_ENGINE=numpy`
(or `engine="numpy"` on `run_scenario`, `simulate_ping` and the providers) selects NumPy.

Source types and how to tap, inspect and re-inject data: `ARCHITECTURE.md`.
