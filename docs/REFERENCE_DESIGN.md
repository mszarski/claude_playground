# Reference design: from a spec sheet to a simulated and a real system

For an engineer or an agent who is handed the spec sheets of a line-array sonar (a platform, hydrophones,
capture boards, a projector, a navigation unit) and has to stand it up: in simulation first, then ashore
with an emulated stream, then on the hardware. Nothing here needs the source to be read. The long-range
("Popoto") array's six builds are made this way (`configs/builds/`), and so is a deliberately different
example (`docs/examples/reference_design/example_static_2x24.yaml`: two rows of 24, 9–13 kHz, six-input
boards at 64 kHz, a static frame), which `tests/python/test_buildspec.py` takes through every step below.

What a build spec covers: one or more horizontal rows of omnidirectional hydrophones on a face or a frame,
capture boards of a fixed number of inputs streaming samples, one projector beside the array, a chirp, active
pinging with detection, tracking and (with two or more rows) elevation and depth, and passive listening.

## The steps

1. **Write the spec.** Copy the nearest file under `configs/builds/` (or the example) and fill it from the
   spec sheets, field by field with the table below. Every section may end with `confirmed: [field, ...]`:
   list there only what is measured, taken from a datasheet you hold, or stated by the owner. Everything else
   is carried as an assumption (`unconfirmed`) into every generated file. A value you do not have: leave the
   field out or `null`, or put a placeholder and do not list it under `confirmed`. Do not invent a figure and
   confirm it. If the build uses an existing row design, name it (`array.module`, see *Row modules*).
2. **Validate.** `tools/build_from_spec.py SPEC --validate`. Errors stop the generator and say which field,
   what is wrong, in what unit, and what to do: a missing or misspelt field; a pitch that puts a grating lobe
   in view; a sample rate that cannot carry the band; rows outside the usable zone; hydrophones against board
   inputs; a contradiction between two ways of giving the same thing (input full scale and ADC full scale);
   a target of the trial inside the blind range. Warnings do not stop it; read them: the projector's direct
   sound against the inputs' full scale (with the computed level and overload in dB), the first
   synchronisation burst falling right after an overload, two rows without a water depth, steering beyond
   the grating-lobe-free sector.
3. **Generate.** `tools/build_from_spec.py SPEC --out DIR`. In `DIR`: the sonar file and its `_clocks`
   variant, the deployment file, five simulated trials with their truth, the options of the standalone C++
   program (`chain.options`, `listen.options`, `COMMANDS.txt`; `--template` also writes `template.h5`), the
   hardware and timing diagrams, `parts.md`, `performance.json` and `BUILD.md` (numbers, expected performance,
   what is not confirmed, the bring-up checklist with this build's figures, the self-test list). The same spec
   always gives the same files. To use them without copying them into `configs/`:
   `export GPUSONAR_CONFIG_PATH=DIR/configs`.
4. **Run the simulation self-tests and read them.** Add `--self-test` (`--only free,rehearsal` for some).
   Each trial is simulated, run through the Python chain and the C++ chain, and checked against its truth:

   | Self-test | What is simulated | Pass |
   | --- | --- | --- |
   | `free` | free field, three targets above, level with and below the array | range within 3 m, bearing within 1°; three or more rows: depth within 3 m + 0.3% of range; two rows that repeat: the true elevation among the candidates |
   | `shallow` | surface and bottom echoes, the far range | the detection's first arrival within 6 m of the target's range (the simulator keeps bounces by energy, D-105, so a detection is the direct echo and tens of metres of bounces behind it, reported at their energy-weighted range; that range and the extent are recorded in `self_test.json`, not judged), bearing; on every ping the true depth within two of the stated sigma, and the sigma's multipath term no wider than the water column (D-133; a fixed allowance of 0.7° stood here before) |
   | `rehearsal` | raw samples at the boards' rate, every board on its own clock, the platform's attitude (rolling, or a tilted frame) | the boards synchronise under 1 µs; first arrival within 6 m, bearing, depth |
   | `passing` | a target crossing at 3 m/s | one mover, speed within 1 m/s |
   | `submerged` | a mover of the spec's first submerged target class, at speed, the chain given NO sound-speed profile and no water depth | a track classed mover, speed within 1 m/s, heading within 15°; any depth it reports has the truth inside two sigma; with two rows the track exists while every detection lists several possible elevations. Skipped without `targets:` |
   | `passive` | two radiating sources, the sonar silent (`sources:`, `pings.transmit: false`) | both bearings within 1° in 80% of frames |
   | `stream` | pings as board blocks at the build's channel count | both assemblers and UDP loopback return every sample |

   `DIR/self_test.json` holds the measured errors, not only pass or fail. A failure names the engine and the
   quantity. The usual causes: a trial range the pulse hides (lengthen `trial.ranges_m` or shorten the pulse);
   two rows far apart in water too shallow to bound them (the array cannot then tell which of several elevation angles an echo came from and reports the list of possible angles, not a depth: by design; bearing, range and the track are unaffected); a
   synchronisation reference outside the sampled band.
   **Plan the depth.** `tools/popoto_depth_planner.py SPEC` (after step 2; it needs only the spec): for each of the spec's environments and target
   classes it prints the detection margin against own depth, the depth to operate (a static frame: to hang) at, the band of depths that hold each class and
   two classes at once, the reach there, the target depths one depth leaves uncovered and how many depths cover all of them; `--out PREFIX` writes the table's
   JSON and a figure. An environment or a target can also be given on the command line (`--water-depth 80 --bottom medium_sand --sea-state 3 --rms-height 0.15
   --ts 10 --target-depth 20 60 --range 2500`). The free-field figures of `performance.json` do not contain the surface-reflected path; the planner's do.
   **What "detected" means in every one of these numbers.** No detection threshold is typed anywhere. The budget states the SNR of ONE row (peak signal over mean
   noise after the matched filter and an unweighted row beamformer; RMS source level). What that SNR must be is asked of the chain's own detector with the
   settings of the build's deployment file (`gpusonar.buildspec.detection`, from `gpusonar.live.budget_threshold_db`), for the detection probability on one ping
   the spec names in `operation.detection_probability` (0.5 unless given). The detector's figure already holds the aperture window, where an echo falls between
   the map's samples, the false-alarm budget over the build's own receive window and the rows added in power, so the budget adds no row gain. `performance.json`
   and `BUILD.md` give the threshold, the same for a probability of 0.9, and for each range the margin and the probability the detector gives at that SNR.
   Source levels are rms levels everywhere (`projector.source_level_db_rms`).
5. **Rehearse ashore with the emulated stream.** Command 2 of `DIR/COMMANDS.txt`: the rehearsal's pings are
   sent as board blocks over UDP to the deployment file's own port and come out as a recording, a report and
   taps, exactly as they will from the boards. The report must match the simulated run's.
6. **Bring up on hardware.** `DIR/BUILD.md` lists the steps with this build's numbers: the stream and the
   channel-to-board map on the bench; the synchronisation residual against the build's tolerance; the
   projector at low level and the recovery of the nearest inputs; a known target against the link budget;
   with rows, each row alone, the row offsets from a source at known depths, depth against truth.
7. **Measure, and feed back.** What to measure on the real system and where it goes in the spec:

   | Measure | How | Spec field, then `confirmed` |
   | --- | --- | --- |
   | Sample rate, word length, simultaneous sampling | a tone into all inputs; the recording's header and phases | `boards.sample_rate_hz`, `adc_bits`, `simultaneous` |
   | Full scale and gain | a calibrated voltage at an input | `boards.input_full_scale_vpp`, `gain_db` |
   | Clock offsets and drifts between boards | `sync` in the report over an hour | `boards.clock_tolerance_ppm`, `clock_offset_max_s` |
   | Hydrophone sensitivity and noise | against the reference hydrophone; a quiet recording's spectrum | `hydrophone.sensitivity_db`, `noise_db` |
   | Hydrophone positions and row offsets | the in-water calibration (`calibrate.estimate_channel_errors`) | `array.pitch_m` / `positions_m`, `out_m` |
   | Source level and the projector's position | the reference hydrophone at a known distance; a tape | `projector.source_level_db_rms` (an rms level), `position_m` |
   | Recovery after the direct sound | raw samples of the nearest input after the pulse | `projector.blanking_after_s` |
   | Attitude accuracy and time stamps | a source at a known depth with the platform level and rolled; `attitude_time: transmit` against `arrival` | `navigation.roll_sigma_deg`, `timestamp_sigma_s` |
   | Sound speed, water depth | the platform's sensors against a cast | `sound_speed.*`, `water_depth.*` |

   Then generate again and re-run the self-tests: the simulated trials now use the measured values.
   **A dive for a sound-speed profile is optional.** It improves the depth in metres at range and nothing else; a track of a submerged mover, its class,
   speed, heading, elevation angle and above-or-below never wait for it (the `submerged` self-test holds the software to that). For a static frame: a cast at
   deployment, equally optional.
   **Recording and compute.** `storage.record: raw` (the default: everything can be redone) or `decimated` (storage-limited runs: the synchronisation
   reference and everything outside the band are gone for good). `compute.option`: `gpu` for two or more rows, `decimate_first_cpu` allowed for one; the
   validator warns when a CPU option's measured time per ping exceeds the ping interval.
8. **What the design does not cover.** The projector's direct sound and the inputs' recovery are computed, not
   simulated. Reverberation-limited range is measured only in the shipped 12 km scenarios, not predicted for a
   new build (`performance.json` is noise-limited, spherical spreading). The bias multipath puts on elevation
   in a shallow channel is in the depth sigma only where `elevation.multipath_sigma` is on (every generated
   stacked-row file), with boundaries taken as perfect mirrors; it is not corrected. Depth beyond a few hundred metres needs a sound-speed profile
   nobody is assumed to have. Curved arrays, towed arrays, more than one projector, directional hydrophones,
   boards whose samples are not 24-bit, and a `port` look direction have not been run through the self-tests.
   The sparse Bayesian bearing kernels take at most 64 channels and raise an error for more (64 in a row; passive
   listening with stacked rows must run with `elevation=1`, as the generated `listen.options` do). Chain thresholds (`cfar`, `tracker`,
   `classify`) are the long-range array's defaults unless `operation.chain` changes them. Processing time is
   measured only for one, two and three rows of 32.

## Target classes and environments

Two optional top-level keys say what the build is planned against. They change no generated sonar or deployment file; `tools/popoto_depth_planner.py`,
`buildspec.performance(..., environment=...)` and the `submerged` self-test read them. Each is a list, or the name of a file under `configs/builds/targets/` or
`configs/builds/environments/` (shipped: `reference_classes`, `illustrative`; every figure in both is an assumed placeholder, and the validator says so in a warning
until a class is `status: confirmed` and an environment `illustrative: false`).

- `targets:` what is to be detected: a target strength (one value or the range over aspect; the planner works at the low end), the depths it may be at, its speeds.
  A class may give a `cylinder` instead of, or beside, a target strength: the geometric-acoustics figure of a bare rigid cylinder of that size is computed and printed
  by aspect. That is a model figure for sizing a study; a real hull is not a bare cylinder.
- `environments:` the water: depth and bottom (a name in `configs/sediments.yaml` or three numbers), wind or sea state, the rms wave height as its own input, an optional
  bubble law, an optional sound-speed profile (then Bellhop is needed), and the depths the platform may operate at. No water depth means no bottom return: the
  non-ducting case, direct path and surface image only.

This table is printed by `tools/build_from_spec.py --fields` after the spec's own.

| Field | Unit | Needed | Meaning | Where to find it | If unknown |
| --- | --- | --- | --- | --- | --- |
| `targets[].name` | — | yes | short name of the class (letters, digits, underscores): tables and file keys use it | the requirement | must be given |
| `targets[].title` | — | no | one line for headings | — | the name |
| `targets[].target_strength_db` | dB re 1 m | no | target strength in the sonar's band: one value, or the range over aspect. The planner works at the LOW end unless told otherwise | a measurement or a requirement document | give `cylinder` and take the model figure, or a placeholder with status: assumed |
| `targets[].cylinder` | — | no | {length_m, diameter_m, caps}: a rigid cylinder of that size stands in for the hull and the target strength is the geometric-acoustics figure for it (gpusonar.target_classes.cylinder_ts_db), over `aspects_deg` | the vehicle's outline | give target_strength_db |
| `targets[].aspects_deg` | deg | no | aspects from broadside (0) to end-on (90) over which the cylinder's figure is taken; the range is its lowest and highest band-averaged value | — | [0, 15, 30, 45, 60, 75] |
| `targets[].depth_m` | m | yes | depths the target may be at (a surface ship: its hull's draught range) | the requirement | must be given |
| `targets[].speed_m_s` | m/s | no | speeds it may move at: the scenarios use the top of the range for a mover | the requirement | leave it out: no mover scenario is derived |
| `targets[].status` | — | no | who vouches for the target strength: assumed (a placeholder for the owner to replace), confirmed (measured or stated by the owner), model (the cylinder figure) (one of: assumed, confirmed, model) | — | assumed |
| `targets[].note` | — | no | where the figures come from, in words | — | leave it out |
| `environments[].name` | — | yes | short name (letters, digits, underscores) | — | must be given |
| `environments[].title` | — | no | one line for headings | — | the name |
| `environments[].water_depth_m` | m | no | water depth. Left out or null: no bottom return at all (deep water, or a bottom that returns nothing): the surface image only | chart / echo sounder | leave it out: the non-ducting case |
| `environments[].bottom` | — | no | the sea bed: a name in configs/sediments.yaml, or {speed m/s, density re water, attenuation dB/wavelength}. Needs water_depth_m | site survey | leave it out with water_depth_m: the bottom then only limits own depth |
| `environments[].wind_speed_m_s` | m/s | no | wind at 10 m: ambient noise, the wind law for the surface's rms height (when rms_wave_height_m is not given) and the bubble law | forecast / trial plan | 5 (the link budget's) |
| `environments[].sea_state` | — | no | 0 to 9, in place of wind_speed_m_s: its median wind speed (acoustics.SEA_STATE_WIND_SPEED) | forecast | give wind_speed_m_s |
| `environments[].rms_wave_height_m` | m | no | rms displacement of the surface (a quarter of the significant wave height). An input of its own: measured heights did not follow the wind law (PHYSICS_NOTES Sec. 21) | wave buoy / forecast: significant height / 4 | leave it out: the Pierson-Moskowitz height for the wind speed |
| `environments[].bubble_model` | — | no | wind law for the loss at each surface bounce: apl_1994 (sim/surface.py), or null for none | — | null: no bubble loss |
| `environments[].sound_speed_m_s` | m/s | no | sound speed of isovelocity water | CTD | 1500 |
| `environments[].profile` | — | no | {depth_m: [...], speed_m_s: [...]}: a sound-speed profile. The factor then comes from Bellhop (GPUSONAR_BELLHOP) and needs water_depth_m | SVP cast / climatology | leave it out: isovelocity |
| `environments[].own_depth_limits_m` | m | no | depths the platform may operate at in this water (a static frame: depths it can be hung at); never deeper than the build's platform.max_depth_m | the operating rules | 5 m to the build's platform.max_depth_m (300 m when that is not given: a placeholder, flagged) |
| `environments[].bottom_clearance_m` | m | no | how close to the bottom the platform may come | operating rules | 10: a placeholder, flagged |
| `environments[].illustrative` | — | no | true: nobody measured this environment; it is an example | — | true |
| `environments[].note` | — | no | where the values come from, in words | — | leave it out |

## Row modules

A row module (`configs/builds/modules/<name>.yaml`) is a named row design that several builds share: the
count, the spacing, the hydrophone. A build says `array: {module: long_4p8m, rows: 3, heights_m: [...]}`;
fields given in the build win over the module's. Shipped: `long_4p8m` (32 at 0.15 m: the static array and
the face) and `testauv_short` (32 at 0.09 m: the test AUV only). A new row design is a new file there.

## The fields

Sections: `platform`, `array`, `hydrophone`, `boards`, `sync`, `projector`, `navigation`, `sound_speed`,
`water_depth`, `network`, `storage`, `compute`, `operation`, `trial`; at the top `build` (the name; file
names follow it), `title`, `stage`, `base_sonar` (a shipped sonar file the generated one extends: only what
differs is written) and `names: {sonar, deploy}` (file names, when they must differ from the build's).
Positions are in the body frame: x along the array (forward), y to starboard, z down, from the array's centre.
This table is printed by `tools/build_from_spec.py --fields`.

| Field | Unit | Needed | Meaning | Where a spec sheet has it | If unknown |
| --- | --- | --- | --- | --- | --- |
| `platform.kind` | — | yes | under way (navigation gives pose and attitude) or a frame left in place (a typed-in pose) (one of: vehicle, static) | the platform's general arrangement | must be given |
| `platform.name` | — | no | short name used in the diagrams | — | leave it out: it is listed as unknown |
| `platform.mounting` | — | no | what the array is fixed to, in words | general arrangement drawing | leave it out: it is listed as unknown |
| `platform.span_m` | m | yes | length of the face or frame along the array | general arrangement drawing | must be given: measure it |
| `platform.height_m` | m | yes | height of the face or frame | general arrangement drawing | must be given |
| `platform.usable_span_m` | m | no | length the rows may take (the rest is margin, fairing, projector) | integration notes | defaults to span_m |
| `platform.usable_height_m` | m | no | height the rows may take | integration notes | defaults to height_m |
| `platform.depth_m` | m | no | static only: the depth the frame hangs or stands at | trial plan | static: must be given (it is the typed-in pose) |
| `platform.max_depth_m` | m | no | deepest the platform operates: the depth planner searches own depths no deeper, and it is the depth rating required of hydrophones, cables, connectors, housings and the projector | the platform's depth rating / the owner | leave it out: the planner then uses each environment's limits (300 m when it has none: a placeholder) and the parts list has no rating |
| `array.module` | — | no | a row module under configs/builds/modules: a named row design (count, pitch, hydrophone, boards a row) that several builds share; fields given here win | the project's row design | leave out and give count and pitch_m |
| `array.rows` | — | yes | number of horizontal rows (1: bearing only; 2 or more: elevation and depth) | array drawing | must be given |
| `array.count` | — | yes | hydrophones in each row | array drawing | must be given |
| `array.pitch_m` | m | no | spacing along a row (uniform rows) | array drawing | give positions_m instead, or half a wavelength at the top of the band |
| `array.positions_m` | m | no | positions along the row from its centre, for a non-uniform row (instead of pitch_m) | array drawing | use pitch_m |
| `array.heights_m` | m | no | height of each row above the centre of the face, lowest first (rows > 1) | array drawing | must be given for rows > 1 |
| `array.out_m` | m | no | how far each row stands out from the face (a curved hull) | array drawing / survey | zero: flat face |
| `array.look` | — | no | which side the face looks to (one of: starboard, port) | general arrangement | starboard |
| `array.wiring` | — | no | row_by_row: every board serves one row. interleave: `interleave` hydrophones of each row in turn (rows share a board's clock) (one of: row_by_row, interleave) | wiring list | row_by_row |
| `array.interleave` | — | no | hydrophones of each row per turn when wiring is interleave | wiring list | half a board's inputs for two rows |
| `array.board_position_m` | m | no | where the capture boards sit, body frame [along, out, down] from the array's centre: cable lengths are measured to it | integration notes | the array's centre, behind the face: an assumption the parts list states |
| `array.cable_slack_m` | m | no | added to every cable run | wiring practice | 0.5 m assumed |
| `hydrophone.model` | — | no | part name | datasheet title | leave it out: it is listed as unknown |
| `hydrophone.sensitivity_db` | dB re 1 V/uPa | yes | open-circuit receive sensitivity (with its own preamp if it has one) | datasheet: receive sensitivity (OCV) | must be given: -208 is a small passive element; flag it assumed |
| `hydrophone.noise_db` | dB re 1 uPa^2/Hz | no | equivalent input noise at the band centre (self-noise) | datasheet: equivalent noise pressure | leave out: the parts list states the limit it must meet |
| `hydrophone.band_hz` | Hz | no | usable band | datasheet: frequency range | leave out: the validator then cannot check it covers the transmit band |
| `hydrophone.preamp_gain_db` | dB | no | gain of a preamp at the element (added to the board's gain) | datasheet | 0 |
| `boards.model` | — | no | part name | datasheet title | leave it out: it is listed as unknown |
| `boards.inputs` | — | yes | hydrophone inputs per board | datasheet: channels | must be given |
| `boards.sample_rate_hz` | Hz | yes | ADC sample rate per channel | datasheet / ICD: sample rate | must be given; flag it assumed until the maker confirms |
| `boards.adc_bits` | bit | yes | sample word length | datasheet | must be given |
| `boards.gain_db` | dB | no | preamp gain of the board's inputs | ICD: input gain | 0 |
| `boards.input_full_scale_vpp` | V peak-to-peak | no | largest signal at the INPUT connector (before the board's gain) | ICD: full-scale input | give adc_full_scale_v instead |
| `boards.adc_full_scale_v` | V peak | no | full scale referred to the gain stage's output: input peak x gain. What the records carry | derived from the ICD | derived from input_full_scale_vpp and gain_db |
| `boards.highpass_hz` | Hz | no | corner of the input's AC coupling | ICD | leave it out: it is listed as unknown |
| `boards.simultaneous` | — | no | all inputs of one board sampled on the same instant | ICD / ask the maker | leave out: it is an open question in the parts list |
| `boards.clock_tolerance_ppm` | ppm | no | how far a board's sample clock may be off: the drift the simulated boards are given | oscillator datasheet | 25 assumed |
| `boards.clock_offset_max_s` | s | no | largest start offset between boards the simulated boards are given | how recording is started | 0.3 ms assumed |
| `boards.jitter_rms_s` | s | no | sample-clock jitter of the simulated boards | oscillator datasheet | 20 ns assumed |
| `boards.clocks` | — | no | the simulated boards' clock errors: {seed: n} draws them inside the tolerance; {explicit: [[offset_s, drift_ppm], ...]} gives them, first board first | — | {seed: 1} |
| `boards.overload_damage_vpp` | V peak-to-peak | no | largest input the board survives | the board's absolute maximum ratings | leave out: an open question in the parts list |
| `boards.gain_switchable` | — | no | the input gain can be lowered during the transmit | ICD / ask the maker | leave out: unknown |
| `boards.shared_clock` | — | no | true: all boards run from one sample clock (no synchronisation stage) | ICD | false: every board on its own clock |
| `sync.scheme` | — | no | shared_reference: one electrical burst wired to every board and summed into its first input. none: one clock for all (one of: shared_reference, none) | system design | shared_reference when boards have their own clocks |
| `sync.reference` | — | no | the burst: {f_start_hz, f_stop_hz, duration_s, level_v, lead_s, tail_s} (level: V peak at the ADC) | system design | a 4 ms sweep in the top fifth of the Nyquist band, clear of the acoustic band |
| `sync.elevation_error_deg` | deg | no | elevation error allowed from clock error between rows: sets the required agreement | error budget | 0.05 |
| `projector.model` | — | no | part name | datasheet title | leave it out: it is listed as unknown |
| `projector.amplifier` | — | no | power amplifier part name | datasheet title | leave it out: it is listed as unknown |
| `projector.source_level_db_rms` | dB re 1 uPa at 1 m, rms | yes | source level in the band as an RMS pressure level (the usual datasheet convention; a peak figure is 3 dB higher for a tone) | datasheet: TVR x rms drive voltage, or SL | must be given: the link budget needs it; flag it assumed |
| `projector.band_hz` | Hz | yes | transmitted band | datasheet: usable band; waveform design | must be given |
| `projector.waveform` | — | yes | {type: lfm | cw, duration_s, window_alpha} | waveform design | must be given |
| `projector.position_m` | m | yes | body frame [along, out, down] from the array's centre | integration drawing | must be given: a placeholder is allowed if flagged assumed |
| `projector.blanking_after_s` | s | no | how long after the end of the direct sound the inputs are still unusable (recovery from overload): the first usable range follows from it | bench measurement of the boards | leave out: unknown, and the validator says so when the direct sound overloads the inputs |
| `navigation.unit` | — | no | part name of the navigation / attitude source | platform documentation | leave it out: it is listed as unknown |
| `navigation.adapter` | — | no | what turns its messages into the chain's navigation datagram (docs/DEPLOYMENT.md) | interface document | json_udp: the datagram itself |
| `navigation.roll_sigma_deg` | deg | no | roll (or frame tilt) accuracy, one sigma: goes into every elevation and depth sigma | datasheet: roll/pitch accuracy | vehicle 0.2, frame without a tilt sensor 1.0: placeholders, flagged |
| `navigation.rate_hz` | Hz | no | output rate of attitude | interface document | leave it out: it is listed as unknown |
| `navigation.timestamp_sigma_s` | s | no | accuracy of its time stamps against the boards' time | interface document | leave it out: it is listed as unknown |
| `navigation.attitude_time` | — | no | use attitude as it was when each echo arrived (needs samples through the ping) or at the transmit (one of: arrival, transmit) | — | vehicle: arrival; typed-in pose: transmit |
| `navigation.attitude_rate_sigma_deg_s` | deg/s | no | sigma added per second an echo lies beyond the attitude samples | — | 0.5 assumed |
| `navigation.static_pose` | — | no | static only: {x, y, z, roll, pitch, yaw} north, east, down in m; radians | trial plan | level at depth_m, heading north |
| `navigation.tilt_sensor` | — | no | static only: the frame sends navigation datagrams from a tilt and compass sensor | frame design | false: the typed-in pose |
| `sound_speed.nominal_m_s` | m/s | no | value used when nothing better is known | — | 1500 |
| `sound_speed.rung` | — | no | the best knowledge available: local_sensor (navigation sends it), typed (measured once), gradient_prior (a mean gradient and its sigma), profile (a measured profile) (one of: local_sensor, typed, gradient_prior, profile) | platform sensors / trial plan | gradient_prior |
| `sound_speed.value_m_s` | m/s | no | typed: the measured local sound speed | CTD / SVP cast | leave it out: it is listed as unknown |
| `sound_speed.gradient_per_s` | 1/s | no | assumed mean vertical gradient along the path (0.017: isothermal) | climatology | 0.017, flagged |
| `sound_speed.gradient_sigma_per_s` | 1/s | no | how wrong that may be: range^2 x this / 2c into every depth sigma | climatology | 0.05, flagged |
| `sound_speed.profile` | — | no | {depth: [...], speed: [...]} a measured profile | SVP cast | leave it out: it is listed as unknown |
| `water_depth.source` | — | no | where the chain gets it: the navigation datagram, a typed value, or nowhere (one of: navigation, typed, unknown) | platform sensors (altitude + depth) | unknown: no folding of bottom bounces, no depth bound for two rows |
| `water_depth.value_m` | m | no | typed: the water depth | chart / echo sounder | leave it out: it is listed as unknown |
| `water_depth.sigma_m` | m | no | how well it is known | — | 5, flagged |
| `network.stream_port` | — | no | UDP port the boards' blocks arrive on | ICD | 5700 |
| `network.nav_port` | — | no | UDP port of the navigation datagrams | ICD | 5702 |
| `network.report_udp` | — | no | [host, port] the report lines go to | ICD | [127.0.0.1, 5701] |
| `storage.recording` | — | no | path of the raw recording (strftime fields allowed) | — | /data/sonar/<build>_%Y%m%d_%H%M%S.h5 |
| `storage.report` | — | no | path of the report | — | the same with .jsonl |
| `storage.taps_dir` | — | no | directory for the stage taps of the first pings | — | /data/sonar/taps |
| `storage.taps_pings` | — | no | pings tapped | — | 2 |
| `storage.capacity_gb` | GB | no | size of the recording volume | computer's specification | leave it out: it is listed as unknown |
| `storage.record` | — | no | what the recording holds. raw: every sample of every board, the synchronisation reference included (everything can be redone; the chain decimates when compute.option says so). decimated: the receiver decimates as blocks arrive (`source.decimate`) and only the band is kept: the reference and everything outside the band are gone for good; for storage-limited runs (one of: raw, decimated) | the trial plan | raw: the safe default during trials |
| `compute.option` | — | no | how real time is reached at the boards' full rate. decimate_first_cpu: the chain converts to baseband and decimates first (`chain.decimate`), on a CPU. gpu: the map and the elevation search on a GPU. cpu: the chain as it is (one of: decimate_first_cpu, gpu, cpu) | — | two or more rows: gpu (decimate-first on a CPU is slower than the ping interval over the full window); one row: decimate_first_cpu |
| `compute.unit` | — | no | part name of the computer | datasheet title | leave it out: it is listed as unknown |
| `compute.engine` | — | no | which engine tools/sonar_live.py runs (the standalone C++ program's command is always generated) (one of: python, cpp) | — | python |
| `compute.backend` | — | no | kernels on the CPU or the GPU (one of: auto, host, cuda) | — | auto |
| `operation.range_m` | m | yes | [first, last] range the receive window covers; first null: the blind range of the pulse | requirement | must be given |
| `operation.bearings_deg` | deg | no | [first, last, step] from broadside | requirement | [-60, 60, 0.5] |
| `operation.chain` | — | no | settings of the chain laid over the defaults (docs/CONFIGURATION.md) | — | the defaults |
| `operation.coverage` | — | no | {environment: one of the build's `environments` by name, or an environment mapping}: the deployment file gets `chain.coverage` (what the sonar can see from where it is) for that site. The site must state its water depth, bottom and wind: none has a default | the site the build is deployed at | leave it out: no coverage picture |
| `operation.detection_probability` | — | no | the probability of detecting a steady point echo on one ping that the link budget's thresholds, margins and reaches are stated for (the threshold follows from the chain's detector for it: buildspec.detection) | requirement | 0.5 (buildspec.DESIGN_DETECTION_PROBABILITY) |
| `trial.water_depth_m` | m | yes | water depth of the simulated trial | trial plan | must be given |
| `trial.own_depth_m` | m | yes | depth of the array in the trial | trial plan | must be given |
| `trial.ranges_m` | m | yes | target ranges: the first for the free-field truth check, the last for the shallow-water one | trial plan | must be given, beyond the pulse's blind range |
| `trial.target_depths_m` | m | no | target depths for the truth checks | trial plan | above, level with and below the array |
| `trial.target_strength_db` | dB | no | strength of the simulated targets | — | 10 |
| `trial.wind_speed_m_s` | m/s | no | sea state as a wind speed, for the ambient noise | trial plan | 5 |
| `trial.sound_speed_m_s` | m/s | no | sound speed of the simulated water (isovelocity) | trial plan | nominal |
| `trial.pulse_s` | s | no | pulse length used in the long-range scenarios, when the sonar's is too short for them | — | the projector's |
| `trial.roll_deg` | deg | no | roll (frame: tilt) in the rehearsal | — | vehicle 3, frame 2 |
| `trial.roll_rate_deg_s` | deg/s | no | vehicle only: roll rate during the ping in the rehearsal | — | 2 |
| `trial.speed_m_s` | m/s | no | vehicle only: speed | — | 1.5 |
| `trial.bottom` | — | no | {speed, density, attenuation} of the sea bed for the shallow scenarios | site survey | sand: 1650 m/s, 1.9, 0.8 dB/wavelength: textbook values, flagged |
| `trial.seed` | — | no | seed of the scenarios' noise | — | 1 |

## Where things are

| | |
| --- | --- |
| `python/gpusonar/buildspec.py` | the schema, the validator, the generated files, the derived numbers, expected performance |
| `tools/build_from_spec.py` | the command: validate, generate, self-test; `--write-shipped`, `--check-shipped` for the shipped builds |
| `tools/build_diagrams.py`, `tools/parts_list.py` | the diagrams and the parts list of a build, from its files |
| `configs/builds/`, `configs/builds/modules/` | the shipped specs and row modules |
| `docs/design/popoto_parts.md`, `docs/design/build_*.png` | the staged Popoto plan's parts list and diagrams |
| `DESIGN.md` Sec. 3 | the staged plan, the trade space, the projector's direct sound, the 12 km design point |
