# Deploying the long-range array on a vehicle

For the Popoto-board array (`popoto_long_range`, or any `popoto_*` / `flank_*` sonar).
Nothing here has run on a vehicle or against real boards. What is tested is the software
path, end to end, on simulated data, on recordings and on an emulated board over UDP.

![Deployment](architecture/deployment.png)

## What runs

One YAML file, and either of two programs that read the same settings:

```bash
tools/sonar_live.py configs/deploy/popoto_vehicle.yaml            # Python: taps, simulator, rehearsal
tools/sonar_live.py configs/deploy/popoto_vehicle.yaml --export-template template.h5
build/bin/sonar_live --udp 5700 --template template.h5 --nav-port 5702 ...   # C++: no Python on the vehicle
```

`--export-template` writes one empty ping record of the configured sonar (array geometry,
waveform, receive window, ADC scale), which is all the C++ program needs to know about the
array, and prints the complete `build/bin/sonar_live` command for the file's `chain`,
`nav` and `output` sections. The C++ program takes the same options as `key=value`
arguments or from `--options FILE` (the list is at the top of `apps/sonar_live.cpp`); it
also replays a recording (`--replay`) and listens passively (`--passive`, below). It
depends on the suite's C++ libraries, HDF5 and, for a GPU, CUDA.

`tools/sonar_live.py --engine cpp` runs the C++ chain behind the Python program, for
taps and rehearsal with exactly the code that will be deployed.

| Part of the file | What it sets |
| --- | --- |
| `source` | where pings come from: `flank_udp` (blocks from the boards), `wav_watch` (a directory the boards write recordings into), `container` (a recording) |
| `nav` | where the vehicle's pose comes from: JSON datagrams on a UDP port, or a fixed pose |
| `chain` | the processing: board synchronisation, bearings, range cell, CFAR, tracker, classifier, CPU or GPU |
| `output` | the raw recording, the report file, the report's UDP destination, taps |

Per ping, in order (`libs/live/src/chain.cpp`; `python/gpusonar/live.py` is the same in Python):

1. Stamp the pose from navigation into the record.
2. **Write the raw ping to the recording.** This happens before any processing, so a
   processing fault cannot lose data.
3. Synchronise the capture boards on the shared reference and drop or blank it (`sync.py`).
4. Baseband conversion and pulse compression.
5. Beamform to range × bearing with the backprojection kernel.
6. CFAR and clustering: detections with range, bearing and SNR. With `doa: sbl`, the
   bearings at each detected range are then solved for by sparse Bayesian learning, which
   separates arrivals inside one beam.
7. Track in a world-fixed frame, so ownship's motion is not put on the targets.
8. Classify each track: mover, static, multipath of another track, uncertain.
9. Emit one JSON object: detections, tracks (position, velocity, kind), sync state,
   processing time. To a file, and to the vehicle computer by UDP.

## Rehearsing it ashore with the same program

```bash
tools/sonar_live.py CONFIG --scenario popoto_12km --out run1/     # the simulator feeds the chain
tools/sonar_live.py CONFIG --replay run1/recording.h5 --out run2/ # the recording gives the same reports
tools/sonar_live.py CONFIG --emulate flank_target_1km --port 0    # simulated pings sent as board blocks over UDP
tools/inspect_tap.py run1/taps/ping0000 --stage prepare --plot p.png
```

Measured on the simulated 6 km scenario (`popoto_far_mover`, 32 hydrophones, 50,000
complex samples a ping, four CPU cores): 0.55 s of processing per ping against a 12 s ping
interval. From ping 4 on the report holds one mover (the ship, −5.8 m/s against a true
−6) and four tracks marked as its multipath.

## What the vehicle has to provide

| Interface | Form | State |
| --- | --- | --- |
| Samples from the boards | **either** WAV files with a YAML sidecar per recording, dropped into a directory (`wav_watch`), **or** blocks over UDP in the format of `sources/flank_stream.py` | The file path needs only what the boards already do. The UDP path needs a forwarder on each board that is **not written**; the block format here was designed for the flank array's capture board and is unconfirmed for Popoto boards. |
| Ping timing | the sidecar's `pings.times`, or the block headers | The transmit trigger's timing relative to the boards' clocks is unconfirmed. |
| Board synchronisation | a reference signal wired to every board (`configs/sonars/popoto_4x7.yaml`: a dedicated input per board, the plan's scheme; `popoto_4x8`: summed into a hydrophone input) | Tested in simulation with 300 µs offsets and 25 ppm drifts. The reference generator and its wiring are hardware to build. |
| Navigation | UDP JSON: `{"time_ns", "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz"}`, north-east-down metres and radians, any rate; optionally `sound_speed`, `altitude`, `bottom_lock`, `water_depth` and one-sigma fields (Stage 2 below); interpolated to each ping's transmit time | `tools/nav_adapter.py` converts a vendor stream by a mapping in YAML (a generic CSV / JSON example is shipped). The mapping for the vehicle's own unit is **a template only**: its interface document and datasheet have not been seen. Without navigation, pings are processed where the record says they are and counted in `pings_without_nav`. |
| Clock | transmit times and navigation times on one clock | Assumed. A difference shows up as tracks that move with ownship. |
| Results | UDP JSON, one datagram per ping, and a `.jsonl` file | Field names in `live.py`. A report too long for one datagram goes out as several complete lines, each with `"part": [k, n]` (an open wire-format question for the owner, `STATUS.md`). |
| Compute | any Linux computer with Python 3 and the built library; a GPU is optional | Not measured on a Jetson. At the boards' full 102.4 kHz real sampling the conditioning step is heavier than in the simulated runs, which start at baseband. |

## First time on the real hardware

1. Bench, boards only: record a few pings to WAV with a known signal on every input.
   `tools/inspect_tap.py` on the result: check sample rate, channel order, clipping, the
   reference bursts on the inputs that should carry them.
2. Run `sonar_live.py` on that directory (`wav_watch`) with `nav: {static: ...}`. Check
   `sync.ok` and the offsets and drifts in the report against what the bench predicts.
3. In water, static: a known target at a known range and bearing. Check its detection's
   range and bearing; a bearing mirrored about broadside means `chain.look` is wrong.
4. Calibrate the array's channel gains, phases and positions from that target at three
   bearings (`calibrate.py`), and store the result with the sonar.
5. Under way: keep every recording. Each replays through the same chain with different
   settings (`--replay`), and its background can be used to calibrate the simulator
   (`SIM_CALIBRATION.md`) so later rehearsals match the site.

## The staged build: one row, then stacked rows (elevation and depth)

The array is built in stages, each run by the same programs from its own files. What each stage is for,
its numbers, and the choice between two rows and three: `DESIGN.md`, Sec. 3.

| Stage | Sonar file | Deployment file | Rehearsal ashore |
| --- | --- | --- | --- |
| 1: test AUV, one row of 32 over 3 m (no room for a second row) | `popoto_testauv_1row` | `configs/deploy/popoto_testauv_1row.yaml` | `--scenario popoto_testauv_trial_1row` |
| 2: static drop-in array, two rows of 32, 1.5 m apart | `popoto_static_2row` | `configs/deploy/popoto_static_2row.yaml` | `--scenario popoto_static_trial_2row` |
| 3: static drop-in array, three rows | `popoto_static_3row` | `configs/deploy/popoto_static_3row.yaml` | `--scenario popoto_static_trial_3row`; `popoto_static_free`, `popoto_static_passing` (baseband pings: see the tests) |
| 4: 2 m × 9 m face, two rows 1.5 m apart within 7 m of its length | `popoto_long_range_2row` | `configs/deploy/popoto_face_2row.yaml` | `popoto_face_2row_shallow` (baseband pings: see the tests) |
| 5: the face, three rows (recommended) | `popoto_long_range_3row` | `configs/deploy/popoto_vehicle_elevation.yaml` | `configs/deploy/popoto_elevation_rehearsal.yaml --scenario popoto_elevation_rehearsal` |

```bash
tools/sonar_live.py configs/deploy/popoto_testauv_1row.yaml                          # the boards, over UDP
tools/sonar_live.py configs/deploy/popoto_static_3row.yaml --scenario popoto_static_trial_3row --out run/
tools/sonar_live.py configs/deploy/popoto_elevation_rehearsal.yaml --scenario popoto_elevation_rehearsal --out run/ --engine cpp
tools/sonar_live.py configs/deploy/popoto_vehicle_elevation.yaml --export-template template.h5   # prints the C++ command
build/bin/sonar_live --udp 5700 --template template.h5 --nav-port 5702 ... elevation=1 elevation_attitude_arrival=1
```

The `*_clocks` sonar files (which the deployment files name) describe every board on its own clock with
the shared synchronisation reference; the rehearsal scenarios are raw 102.4 kHz samples with clock offsets
to 0.3 ms and drifts to 25 ppm, the vehicle rolled and rolling during the ping (the static frame: tilted and
still). For a 2 m span on the test AUV: `sonar_overrides: {array: {pitch: 0.0625}}`; for another row
separation or another frame: the `rows` of the sonar file.

**The static drop-in array** (stages 2 and 3) is the same programs with three differences, all in its
deployment file. Its pose is typed in: `nav: {static: {x, y, z, roll, pitch, yaw}}` (north, east, down in
metres, radians; `z` the depth it hangs at, `yaw` where its length points), or `nav: {port: ...}` when a tilt
and compass sensor on the frame sends the same datagrams a vehicle would. Without a sensor the frame is
taken as plumb: `elevation.roll_sigma_deg` says how plumb (1° shipped, a placeholder), `attitude_time` is
`transmit`. Sound speed and water depth are measured once at deployment and typed in (`chain.sound_speed`,
`elevation.water_depth`). Nothing in the software needs to know the array is not moving. Its dimensions,
depth, projector and site are not given by the owner: parameters, flagged `unconfirmed`.

A deployment file with rows differs from a one-row file in `source.sonar` and in having `chain.elevation`
(`CONFIGURATION.md` lists its keys). The rows are found from the hydrophone positions in each record;
nothing else says how many there are, and a one-row array is processed exactly as before. Per ping, with rows:

- each row is beamformed over range × bearing on its own and the rows' maps are added;
- at each detection (the strongest 32) the whole array is steered over elevation at that range and bearing,
  and the first arrival of the detection is taken: the direct path;
- the detection's line in the report gains

| Field | Meaning |
| --- | --- |
| `elevation_deg`, `elevation_sigma_deg` | elevation above the horizontal, with the vehicle's roll, pitch and yaw **as they were when the echo arrived** (`attitude_time: arrival`), and its one-sigma: the bound at the measured SNR, `roll_sigma_deg`, and the attitude's timing. `null`: none found |
| `elevation_quality` | 1 for one clean arrival; lower when several arrivals share the range cell. Below `min_quality` no elevation is reported |
| `elevation_ambiguous`, `elevation_alternatives_deg` | rows several wavelengths apart repeat their response every few degrees. When another lobe is within 1 dB of the best, or the noise could swap them, the detection is flagged and the other candidates are listed. Two rows on the face: always, unless `sector_from_depth` bounds the search; three rows: at low SNR only |
| `bearing_deg`, `cone_bearing_deg` | the horizontal bearing, corrected for the elevation, and what the line alone reported |
| `depth_m` | depth: own depth + range × sin(elevation), folded into the water column; along the ray through `gradient_per_s` (or `ssp`) when given |
| `depth_sigma_m`, `depth_refraction_sigma_m` | one sigma of the depth, and the part of it that is the unknown sound-speed gradient (range² × `gradient_sigma_per_s` / 2c) |
| `image_depth_m`, `path` | the straight-line depth before folding, and what it says of the path: `direct`, `surface` (the image is above the surface), `bottom`, `multiple` |

- each track gains `depth_m`, `depth_sigma_m` and `path`. A track takes its depth from unambiguous detections;
  once it has one it picks, from an ambiguous detection, the candidate that fits. A track whose straight-line
  depth stays outside the water column by more than two sigma is classified `multipath`.

Each use can be switched off on its own (`correct_bearing`, `depth`, `fold`, `classify`). Passive listening takes
`elevation=1` too: every bearing in a frame then has an entry in `elevations_deg` and `elevation_quality`.

What the vehicle must provide in addition to the list above:

| | Why | State |
| --- | --- | --- |
| Roll and pitch in the navigation datagrams at a few a second, accurate to 0.1–0.2° | depth error = range × roll error: 0.2° is 3.5 m at 1 km and 42 m at 12 km (`DESIGN.md` tabulates 0.02° to 1°); with rows 1.5 m apart roll is the first limit of the angle | The owner: configurable; the vehicle has a SPRINT-Nav ("good in general"), whose figures are not known here. `elevation.roll_sigma_deg` is the accuracy; `attitude_time: arrival` uses the samples around each echo's arrival (an echo from 12 km arrives 16 s after the ping), `transmit` the ping's. The sensor's accuracy and rate are unconfirmed. |
| Own depth (`z`), and where the surface is in that frame (`elevation.surface_z`) | reported depth is own depth plus the measured difference | as for one row |
| Sound speed at the vehicle: `"sound_speed"` in the navigation datagrams, or `chain.sound_speed` | it sets the beamforming and the angle at the array | The owner: the AUV measures it. Without either, the sonar file's nominal 1500 m/s is used |
| Water depth: `"water_depth"` in the datagrams, or `elevation.water_depth`, and how well it is known (`water_depth_sigma_m`) | folding bottom bounces, recognising them, and bounding the search for two rows | The owner: "some idea of water depth". 5 m assumed |
| A sound-speed profile | none will exist. The chain assumes a mean gradient (`gradient_per_s`, isothermal water by default in the vehicle's file) and carries its uncertainty (`gradient_sigma_per_s`) into every depth sigma: 17 m at 1 km, 600 m at 6 km for 0.05 /s | depth beyond a few kilometres is limited by this, not by the array. `elevation.ssp` takes a measured profile if there ever is one |
| Boards of different rows synchronised to about 1 µs | 1 µs is 0.06° with rows 1.5 m apart (0.87 µs for 0.05°; `DESIGN.md` has the curve against separation) | 0.5 µs in simulation with the shared reference; real boards unconfirmed, and the question is open |
| An in-water calibration with sources at several depths | a row 1 mm further out from the face than drawn is 0.04° | `calibrate.estimate_channel_errors(..., planar=False)` |
| A failed hydrophone or board named in `chain.dead_channels` | so that it is left out, not summed | the chain does not detect dead channels by itself |

First time in water with rows (the static array first, then the face), after step 4 above:

1. Per row: with `chain.dead_channels` naming every channel of the other rows, each row alone must give the
   same detections (range, bearing) as the others. A row that differs is miswired or misplaced.
2. Row offsets: lower a source (or use the support boat's echo sounder) to three known depths at two or three
   bearings; run the calibration with `planar=False`; store it with the sonar.
3. Elevation: the same source again. `depth_m` against its depth with the vehicle level, and again rolled 5°.
   A depth error that changes sign with the side the source is on is a row offset; one that follows the roll is
   the attitude sensor or its time stamp (try `attitude_time: transmit` against `arrival`).
4. Two rows 1.5 m apart (frame or face): check `elevation_alternatives_deg` on that source (candidates 11.5° apart) and that
   `sector_from_depth` with the site's water depth leaves the right one.

## From a build spec: generated deployment files, diagrams, parts and checklists

The stage sonar and deployment files are generated from build specs (`configs/builds/<build>.yaml`) by
`tools/build_from_spec.py`; a test fails if a committed file is not what its spec generates, so change the
spec, then `tools/build_from_spec.py --write-shipped`. `REFERENCE_DESIGN.md` is the guide for a new build.

| Build | Row module | Sonar file | Deployment file | Hardware and timing diagrams |
| --- | --- | --- | --- | --- |
| `popoto_testauv_1x32` | `testauv_short` (the one exception) | `popoto_testauv_1row` | `popoto_testauv_1row` | `design/build_testauv_1x32_*.png` |
| `popoto_static_2x32`, `popoto_static_3x32` | `long_4p8m` | `popoto_static_2row`, `_3row` | `popoto_static_2row`, `_3row` | `design/build_static_2x32_*.png`, `_3x32_` |
| `popoto_face_1x32` | `long_4p8m` | `popoto_face_1row` | `popoto_face_1row` | `design/build_face_1x32_*.png` |
| `popoto_face_2x32` | `long_4p8m` | `popoto_long_range_2row` | `popoto_face_2row` | `design/build_face_2x32_*.png` |
| `popoto_face_3x32` | `long_4p8m` | `popoto_long_range_3row` | `popoto_vehicle_elevation` | `design/build_face_3x32_*.png` |

- **One row design.** The owner: "Yes treat auv special case and have static array the longer row." The static
  array and the face carry the same row module (32 hydrophones over 4.8 m, four boards); the test AUV's
  short row exists only because of that vehicle's span. The face starts with one row, then two, then three.
- **Each sonar file states its receive window** (`receive.window`): from the pulse's blind range to 2 km on
  the test AUV, 3 km on the static array, 12.7 km on the face. Before, every stage silently inherited the
  eight-hydrophone board's 50 m to 2 km. A full-window ping on the face is 1.7 million samples a channel
  (666 MB recorded with three rows): for a bench run give `source.sonar_overrides: {receive: {window: {range: [...]}}}`.
- **For a build, per stage:** `tools/build_from_spec.py <build> --out DIR` writes `BUILD.md` (the bring-up
  checklist with that build's tolerances), `COMMANDS.txt` (simulate, emulate, run, the C++ program with
  `chain.options`, listening with `listen.options`), the diagrams and `parts.md`; `--self-test` runs the
  simulated trials through both engines against their truth.
- **What the generated deployment files switch on** (the owner's decisions; `buildspec.chain_config`, tests in
  `tests/python/test_deployment_defaults.py`):
  - every build of two or more rows: `elevation.sector_from_rays: {}` (the elevation search is limited to where a ray
    with two bounces or fewer can arrive from). The hand-written `popoto_elevation_rehearsal` has it too;
  - every build of two or more rows: `elevation.multipath_sigma: {}` (the depth's sigma grows, and
    `depth_multipath_limited` says so, where direct, surface and bottom echoes share the range cell and the rows read a
    boundary instead of the target). The hand-written `popoto_elevation_rehearsal` has it too;
  - a build that switches depth from multipath timing on (`operation.chain.multipath_depth`): `rays: true` and
    `withhold_flagged: true`. No shipped build has it on (it is off on all six);
  - a build whose spec names a site (`operation.coverage: {environment: ...}`): `chain.coverage`, with that site's water
    depth and bottom, the spec's target classes and `buildspec.budget`'s levels at the site's wind. The three face builds
    have it, and the hand-written `popoto_vehicle` and `popoto_elevation_rehearsal`; the static builds and the test AUV do
    not. **Water depth, bottom and wind are required of the site and have no default**: the generator stops on a site
    that lacks one, and a chain given `coverage` without `water_depth_m` or `bottom` does not start. No site is chosen:
    the shipped files carry the illustrative `shelf_100m_sand` (100 m over sand, 5 m/s), flagged `unconfirmed`. **Put the
    real site in before a deployment** (the spec's `operation.coverage.environment`, then `--write-shipped`).
- **Parts:** `design/popoto_parts.md` (generated; quantities per stage, what the design asks of each part,
  what fixes it, the plan's own words, where the plan and the files disagree).

### The projector's direct sound and the first usable range

At every placeholder position the projector overloads the nearest inputs (19 dB over full scale on the face,
34 to 36 dB on the test AUV and the static frame; `DESIGN.md` Sec. 3 has the table and the options). Until
the inputs' recovery time is measured: record one ping at the lowest source level and look at the raw samples
of the nearest hydrophone (`tools/inspect_tap.py`), then at full level; put the recovery in the spec
(`projector.blanking_after_s`), and move the first synchronisation burst after it (`sync.reference.lead_s`;
the validator warns while it is not). The chain ignores everything before the receive window, which opens at
the blind range; it does not protect an input.

### The 12 km design point

`DESIGN.md`, *The 12 km design point*. To run it: `popoto_12km` (one row), `popoto_12km_2row` with
`configs/deploy/popoto_face_2row.yaml`, `popoto_12km_3row` with `configs/deploy/popoto_vehicle_elevation.yaml`
(leave the `sync` section out: these pings are at baseband; set `chain.elevation.water_depth: 100`).
`tools/popoto_12km_study.py` runs all three row counts in each shipped environment and writes
`design/popoto_12km.json`. Bellhop is needed for the ducted cases; image-source paths run anywhere.

## Stage 2: navigation unit, sound-speed ladder, compute, blanking

Everything in this section is off unless its key is in the deployment file
(`CONFIGURATION.md` lists the keys). Numbers come from `docs/design/popoto_stage2.json`
(`tools/popoto_stage2_study.py`, `tools/popoto_elevation_bench.py --full-rate`).

### Navigation from an inertial unit with a Doppler log

The vehicle carries a SPRINT-Nav (the owner: "sprintnav IMU and dvl"). **We hold neither its
datasheet nor its interface document**: nothing here states its message format, rates or
accuracies, and every such figure is a field marked unconfirmed.

    unit -> tools/nav_adapter.py MAPPING ... -> UDP JSON datagrams -> tools/sonar_live.py (nav.port)
                                                                   -> build/bin/sonar_live (--nav-port)

- **The adapter** (`python/gpusonar/navadapter.py`, `tools/nav_adapter.py`) turns one vendor
  record into one datagram. The vendor's side is a mapping in YAML: field names or columns,
  units, signs, frames (north-east-down or east-north-up; body forward-right-down or
  forward-left-up), Euler order or quaternion, velocity in world or body axes, time unit and
  offset, the Doppler log's lock flag and "no value" codes, output rate.
  `configs/nav/generic_csv.yaml` is a worked example on a line-oriented CSV stream (JSON
  lines work the same way); `--replay FILE [--speed 1]` replays a logged file, `--in-udp` and
  `--stdin` run live. `configs/nav/sprintnav_template.yaml` is the template to fill in; if
  the unit's output is binary, a small reader that yields one dict per record replaces the
  line parser and the mapping stays.
- **What a real SPRINT-Nav adapter must supply**, each fix: time of validity on the boards'
  time base; roll, pitch and heading; position (north, east) and depth; and, for what this
  stage adds, velocity over ground, the Doppler log's altitude with its bottom-lock state, and
  the sound speed at the vehicle if the unit or a separate sensor reports it. Its own
  accuracy estimates (roll, depth, altitude) are used when sent.
- **To read from its datasheet into the build spec** (all unconfirmed, listed with empty
  values under `navigation.from_the_datasheet_into_the_build_spec` in the JSON): attitude
  output rate; roll, pitch and heading accuracy; time-stamp accuracy, output latency and time
  base; message format; velocity, altitude and depth accuracy and the altitude range;
  whether and how well it gives sound speed; and the alignment between the unit and the
  array, which has to be surveyed, not read.
- **Attitude rate.** `attitude_time: arrival` interpolates linearly between attitude samples:
  a roll of amplitude A at frequency f sampled every dt is wrong by at most A (2 pi f dt)^2 / 8.
  For a roll of +-5 degrees with an 8 s period that is 0.39 degrees at 1 Hz, 0.015 at 5 Hz and
  0.004 at 10 Hz, against 0.2 degrees of roll accuracy assumed so far: 5 to 10 Hz is enough,
  and `output: {rate_hz}` in the mapping thins a faster stream. Fixes are now kept by time
  (`nav.keep_s`, 120 s), not by count: the earlier 600-fix buffer held six seconds of a 100 Hz
  stream, less than one ping's receive window.
- **Water depth from the Doppler log**: `nav: {water_depth: {source: altitude}}`. Water
  depth = vehicle depth + altitude (+ the log's offset), sigma from both. When bottom lock is
  lost the last value is held with a sigma that grows with time and distance travelled, then
  dropped (`hold_s`). It feeds every consumer of the water depth: the search sector from
  depth, folding, the bottom test, the bottom-bounce sigma. It is the depth under the
  vehicle, not under a target kilometres away: `elevation.water_depth_sigma_m` still has to
  cover the bottom's relief over the ranges of interest.
- **In the record** (no schema change): attitude, depth and Doppler-log samples around the
  ping in `navigation.ins`, `.depth`, `.dvl`; what navigation said about its own numbers, and
  the (depth, sound speed) samples since the last ping, in `metadata.extra_json` under `nav`.
  A replayed recording therefore gives the same water depth, sigmas and profile.

### The sound-speed ladder

The owner: no long-range profile, but the vehicle measures sound speed where it is, the
water depth is roughly known, and the water will not always be isothermal. `elevation.profile`
turns that into a ladder. Each ping the best available rung is used and named with every
depth (`depth_profile_source`):

| Rung | What is known | Depth by | Sigma from |
| --- | --- | --- | --- |
| `supplied` | a profile in the file (`elevation.ssp`: lists, a table, or a CSV file) | ray through it | its `sigma_m_s`, else the prior |
| `built` | the profile the vehicle built from its own (depth, sound speed) samples as it changed depth; a static array: a cast at deployment (`built.cast`) | ray through it | sensor accuracy, age and distance of each depth bin; widening outside the sampled depths |
| `family` | an assumed shape (isothermal; mixed layer over a linear thermocline) through the local sound speed | ray through it | its `sigma_m_s`, else the prior |
| `local` | the sound speed at the vehicle | straight line (or `gradient_per_s`) | the gradient prior: range^2 x `gradient_sigma_per_s` / 2c |
| `prior` | nothing measured | the same | the same |

What knowing the profile buys: depth error from refraction alone (angle and travel time
exact), median / worst in metres over four **illustrative** profiles (defined by parameters
in the study tool, not data: isothermal; mixed layers of 30 and 50 m over thermoclines of
-0.15 and -0.05 /s; a -0.05 /s gradient from the surface) in 200 m of water, vehicle at 50 m,
targets at 20 and 120 m:

| Rung | 1 km | 3 km | 6 km | 12 km |
| --- | --- | --- | --- | --- |
| prior / local: straight line | 10.4 / 25 | 50.9 / 122 | 80.4 / 156 | 71.6 / 163 |
| family: isothermal assumed | 8.9 / 30 | 31.4 / 115 | 43.0 / 95 | 45.7 / 145 |
| family: mixed layer, layer 25% too deep | 3.2 / 18 | 8.5 / 105 | 45.8 / 88 | 20.3 / 119 |
| family: mixed layer, right parameters | 0.0 / 0 | 0.1 / 0 | 0.1 / 0 | 0.4 / 1 |
| built: only 10 m about the vehicle's depth sampled | 12.6 / 16 | 46.7 / 115 | 51.9 / 167 | 71.0 / 167 |
| built: the whole column sampled | 0.0 / 1 | 0.3 / 5 | 0.3 / 19 | 2.1 / 16 |
| supplied: the true profile | 0.0 / 0 | 0.1 / 0 | 0.1 / 0 | 0.4 / 1 |

- A profile that covers the water between vehicle and target brings the median depth error
  to a metre or two at every range tested (the worst case of the built profile, binned at
  2 m, is 19 m, at 6 km). Anything less leaves tens of metres from 3 km on, in 200 m of
  water: little better than not knowing the depth at all.
- A dive through the water column is therefore worth more than any assumption. A vehicle that
  stays at one depth has the `local` rung and should treat depths beyond a few kilometres as
  indications.
- An assumption is only as good as its parameters: a mixed layer assumed 25% too deep still
  leaves tens of metres from 6 km on, and assuming isothermal water over a thermocline does
  little better than the straight line.
- Beyond 3 km in this water the first arrival has usually touched a boundary: the ray code
  mirrors at the surface and at the bottom, and the answer then also hangs on the water depth
  along the path.
- The sigma reported is honest only as far as the prior is: outside what was sampled it grows
  at `gradient_sigma_per_s` (0.05 /s), and a thermocline of -0.15 /s is three times that. In the
  study the error stayed within two reported sigmas in all cases with the true or the fully
  built profile, and in 5 of 8 at 3 km for the rungs that extrapolate.

### Compute at the boards' full rate: two options

Measured here on 4 CPU threads of a shared machine with a load average of 9 to 17 while it
ran: runs of one configuration differed by up to 30%, each figure is the shorter of two, and
the ratios mean more than the seconds. One full-length ping, 0.5 to 13 km at 102.4 kHz, ideal
clocks, C++ chain (`tools/popoto_elevation_bench.py --full-rate`). The ping interval is 17.9 s.
(On the idle machine the chain as it is took 15.5 / 23.5 / 37.7 s with the earlier geometry.)

| | 1 row (32) | 2 rows (64) | 3 rows (96) |
| --- | --- | --- | --- |
| Chain as it is (converts to baseband itself), s per ping | 13.3 | 25.2 | 53.8 |
| **Option (i)** decimate first, factor 17 (6.0 kHz), s per ping | 13.1 | 22.9 | 44.2 |
| ... of which the stage itself | 0.6 | 1.0 | 2.9 |
| ... chain on records a decimating receiver hands it | 6.6 | 15.9 | 53.9 |
| Conditioning alone (baseband and pulse compression): raw -> decimate first, s | 4.7 -> 1.3 | not run | 11.9 -> 3.1 |
| decimate first, factor 25 (4.1 kHz), s per ping | 7.2 | 19.1 | 47.8 |
| Streaming stage in the receiver, s of one core per s of stream | 0.03 | 0.08 | 0.12 |
| Recording per ping, raw -> decimated, MB | 232 -> 27 | 463 -> 55 | 695 -> 82 |
| Recording per hour at 17.9 s a ping, raw -> decimated, GB | 44 -> 5.2 | 93 -> 11.0 | 140 -> 16.4 |

**Option (i), decimate first** (`chain.decimate` or `source.decimate`; `decimate=1` for
`sonar_live`). One streaming FIR mixes the band to its centre, low-passes and keeps one sample
in 17 (171 taps; 0.1 dB ripple over the kept 3 kHz, 80 dB rejection of what would alias; band
and rate follow the sonar file, nothing is fixed to 102.4 kHz or 4 to 6 kHz). It is the same
conversion the chain did per ping by FFT, done once, early.

- *Order of operations.* The synchronisation reference (38 to 48 kHz) is above the band, so
  the filter removes it: boards cannot be synchronised after decimation. Both bursts are
  needed for offset and drift, and the second closes the window, so they cannot be corrected
  before it either without holding the whole ping at the full rate. The reference is
  therefore **carried through**: the raw samples around the two bursts are kept aside (a few
  milliseconds per board), the clocks are estimated from them exactly as before, and the
  correction is applied to the baseband samples (a resampling plus a turn of the carrier
  phase). The estimates equal the full-rate ones to 0.1 ns; the corrected baseband equals
  "synchronise, then decimate" to a part in a thousand.
- *Results.* On the one-, two- and three-row stage scenarios with board clocks (4, 8 and 12
  boards), detections above 25 dB through the stage against the full-rate path: range within
  0.5 m, bearing 0.05 degrees, level 0.3 dB, elevation 0.05 degrees, depth 0.3 m, tracks within
  0.06 degrees, clock estimates within 0.02 us; listening gives the same bearings. Weak
  detections near threshold (sidelobes of a strong echo) can differ. At a 25th of the board
  rate (4.1 kHz) the same detections come out with bearings up to 0.2 degrees off: the
  default stays at 17.
- *What it saves, and what it does not.* Samples, memory and storage by 8.5 (a raw sample is
  4 bytes in the container, a complex one 8, and one in 17 is kept): 695 -> 82 MB a ping and
  140 -> 16 GB an hour for three rows; the stream from the receiver 236 -> 37 Mbit/s.
  Conditioning (baseband conversion and pulse compression) is 3.7 times faster, and board
  synchronisation works on a seventeenth of the samples. **The range x bearing map does
  not shrink**: it costs per range cell, bearing and hydrophone whatever the sample rate, and
  on the full 0.5 to 13 km window it is most of the ping's time. The whole ping is therefore
  only 1.0 to 1.2 times faster, which for three rows is inside the scatter of these runs, and
  the stage alone does not make two or three rows real time on this CPU.
- *Latency added*: half the filter, 0.83 ms. The streaming stage's share of one core is in
  the table.
- *Recording.* `chain.decimate`: the recording holds the raw ping (everything can be redone;
  140 GB an hour for three rows) and the chain decimates. `source.decimate` (or
  `decimate_streaming=1`): the receiver decimates as blocks arrive, the full-rate ping is
  never held, and the recording holds the decimated pings (16 GB an hour): the band of
  interest is all there, the reference and everything outside the band are gone for good.

**Option (ii), a GPU.** No new code. Measured on an NVIDIA L4
(`docs/benchmarks/gpu_benchmark_all_l4.md`): the C++ chain with the backprojection map takes
0.010 / 0.021 / 0.031 s per ping for 1 / 2 / 3 rows, 14 to 15 times faster than that host
(0.15 / 0.30 / 0.44 s on 8 threads), on scenario-size pings (15,467 complex samples at 8 kHz,
241 bearings, a 1.4 km window).

- What a GPU gives: the map, which is the part option (i) cannot shrink, and the elevation
  search (the same kernel).
- **Not measured**: a full-length ping at the boards' rate on a GPU (the L4 table is on
  decimated, scenario-size data: about a ninth of the range cells of the 0.5 to 13 km
  window); the present 1.5 m geometry (the table was run on the tree before it); a Jetson.
  MVDR and far-field beamformers had no CUDA path in that table.
- The command for that measurement, in a GPU job with a CUDA build:
  `PYTHONPATH=python python3 tools/popoto_elevation_bench.py --full-rate --backend cuda`
  (add `--clocks` for boards on their own clocks). It writes
  `decimation.timing.cuda` into `docs/design/popoto_stage2.json`.

| | Option (i): decimate first, CPU | Option (ii): GPU |
| --- | --- | --- |
| New hardware | none | a GPU module, its power and cooling |
| Makes the map faster | no | yes (14 to 15x on an L4, scenario size) |
| Shrinks storage and the stream | 8.5x | no (unless (i) is used too) |
| Three rows, full window, in real time | not on this 4-thread CPU | expected, **not measured** |
| State | built, tested, measured here | built; full-rate figure pending a GPU job |

The two are not alternatives in the end: (i) is what makes the stream and the recording
small and is worth having in any case; (ii), or a shorter window, coarser cells or fewer
bearings, is what the map needs.

### Blanking the projector's own sound

The projector is on the face, a metre or two from the hydrophones: its direct sound is far
above the inputs' full scale for the whole pulse, and the inputs then need time to recover.
`chain.blanking: {recovery_s, taper_s, mode}`:

- Samples from the transmit until pulse length + `recovery_s` are zeroed before baseband
  conversion and pulse compression (inside the decimate-first stage when that runs: after
  the reference is read, before the filter), with a raised-cosine return over `taper_s`. Not
  blanked, a clipped pulse correlates with the replica: in the test it is "detected" 113 dB
  above the noise where it sits, and the matched filter carries it to the far end of the
  window at 41 dB. Blanked, the target's level and the noise floor are what they are without
  the overload.
- The detector does not report nearer than c (pulse + recovery + taper) / 2, the map starts
  there, and the report line says so (`blanking.min_range_m`). With a 5 ms taper:

| Recovery, s | 0 | 0.005 | 0.02 | 0.05 | 0.1 | 0.2 | 0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 s pulse: minimum range, m | 79 | 83 | 94 | 116 | 154 | 229 | 454 |
| 1 s pulse: minimum range, m | 754 | 757 | 769 | 791 | 829 | 904 | 1129 |

- `recovery_s` is a property of the inputs that nobody has measured: unconfirmed until the
  boards are on the bench with the projector.
- **Board synchronisation.** With the window opening during the pulse, the first reference
  burst sits in clipped samples. Unchecked, the search for it either fails (the ping is lost)
  or could lock onto a harmonic of the clipped pulse. With `sync: {..., min_quality: 0.5}` a
  burst that correlates badly or sits in railed samples is not used: the board is timed from
  its other burst alone, without a drift, flagged in the report (`sync.flagged`), and the ping
  is marked not ok (in the test the offsets are then right at one end of the window and up to
  17 us off at the other, against a tolerance of 8 us). The remedy is timing, not software:
  send the first burst after the blanked stretch (the reference's `lead`); then both bursts
  are clean and synchronisation is as usual (tested).
- **Listening on a platform that also pings** (`sonar_live --listen` with `blanking=1`): the
  same window is zeroed after every transmit the blocks mark.

## Refraction in operation

What the vehicle should do about the water it is in, and how far to trust each depth it reports. Built so far: the
second depth estimate (`chain.multipath_depth`) and the ray code under it (`python/gpusonar/raytrace.py`). Not built
yet: the coverage summary on the report line and the profile ensemble (see "What is not there yet").

**Measure sound speed continuously, and dive when it can.** Everything below turns on the profile between the vehicle
and the target. The ladder (`elevation.profile`) climbs as the vehicle samples depths; a rung that only knows the water
at the vehicle's own depth leaves the gradient prior in every depth, and at 12 km that is the whole water column.

**Where depth comes from.**

| Estimate | Needs | Its own error (isovelocity or a known profile) | What an unknown gradient σ_g adds |
| --- | --- | --- | --- |
| elevation (`depth_m`) | two or more rows; with two, several candidates | r σ_θ: 0.41° at 13 dB, 0.18° at 20 dB is 7 / 3 m at 1 km and 85 / 38 m at 12 km (`docs/design/popoto_elevation_study.json`) | r² σ_g / 2c: 17 m at 1 km, 2400 m at 12 km for 0.05 /s |
| multipath timing (`depth_multipath_m`) | one row; a surface echo; a target deeper than 3 r c / (4 B z_own) | 0.02 to 0.7 m rms at 1 to 6 km, 0.1 to 2.8 m rms at 12 km (worst 9 m near the threshold: one delay cell), strongest arrival at 17 or 26 dB; a track of 6 pings 0.25 to 1.25 m (`docs/design/multipath_depth_study.json`) | z_t r² σ_g / (2 c (z_own + z_t)): 7 m at 1 km, 960 m at 12 km for 225 m and 150 m |

So the multipath depth is not free of the profile: the same gradient that bends the ray under the elevation estimate
stretches the surface echo's delay, by the smaller factor z_t / (z_own + z_t). What it is free of is the angle: no rows,
no roll, no elevation ambiguity. With the profile known (a dive, a cast) it is the far finer of the two; with only the
prior, both are indications at long range and the report's sigma says so.

**How to set it.** `multipath_depth: {}` switches it on. Give `surface.rms_height_m` (and `surface.loss_db` at a high sea
state), the vehicle's `own_depth_sigma_m`, and in shallow water `bottom` and the water depth. Give `gradient_per_s` when a
mean gradient is known (0.017 for isothermal water) and `gradient_sigma_per_s` for what is not known of it. Read
`depth_multipath_sigma_m`, `multipath_ambiguous`, `multipath_note` with every depth.

**When it gives nothing**, and what that means: `merged` (the sonar is too shallow: at 10 m no depth is resolvable at
12 km; at 225 m targets above 30 m are not), `no_surface_path` (bubbles or roughness), `single_arrival` (no surface echo
found: a target above the resolvable depth, a surface path the water does not carry, or too little SNR; a track adds
pings up), `window` (edge of the record). Through a thermocline the surface path or the direct path may not exist at
all (`tools/multipath_depth_study.py`: no direct path beyond 1 to 3 km from 50 m under the stage-2 thermoclines); the
flat pattern then matches whatever does arrive, or nothing.

**With two rows** the multipath depth picks among the rows' candidate elevations (`elevation_selected`): at 6 km the
candidates are 12° apart, kilometres in depth, and the multipath depth is known to under a metre in the same water.

**Cost.** 20 to 45 ms per examined detection at 6 to 12 km, 80 ms at 3 km, 0.2 s at 1 km with the search to 500 m (one
core); 0.50 against 0.54 s per ping on the 12 km test window with one detection.

### Rays through the profile the vehicle holds (`multipath_depth.rays`, `elevation.ray_method: arcs`)

Both depths can come from one ray calculation through whatever the sound-speed ladder holds this ping. Switch it on
with the ladder:

```yaml
chain:
  elevation:
    ray_method: arcs
    profile:
      built: {sample_sigma_m_s: <the sound-speed sensor's accuracy>}
      ensemble: {gradients_per_s: [<the gradients the unsampled water may have>], extent_m: [<how far they may run>]}
  multipath_depth: {rays: true, withhold_flagged: true, surface: {rms_height_m: <rms wave height>}}
  coverage: {water_depth_m: <the site's>, bottom: {...}, classes: [...], budget: {...}}     # REQUIRED: water_depth_m, bottom. Generated from the spec's site (operation.coverage)
```

**Where each depth comes from and how far to trust it** (numbers: `tools/refraction_depth_study.py`, illustrative
waters, the measurement itself taken as exact; `docs/design/refraction_depth_study.json`):

| What the vehicle knows (rung) | One row, `depth_multipath_m` | Stacked rows, `depth_m` |
| --- | --- | --- |
| a cast, or its own profile from a dive through the water between it and the target (`supplied`, `built` over the column) | the true depth; sigma under 1 m at 1 km, a few metres at 3 to 6 km | the true depth; sigma 1 m at 1 km, 10 to 20 m at 3 to 12 km |
| its own sound speed only (`local`, `prior`), or a profile from a few metres about its depth | median error 3 m at 1 km, 18 m at 3 km, 30 to 50 m at 6 km, with a sigma that says so (6, 60, 300 m) | median error 6 to 9 m at 1 km, 40 to 50 m at 3 km, 160 m at 6 km, 800 m at 12 km (sigma 15, 130 to 150, 400, 1500 m) |
| an assumed family | as good as the assumption: exact when it is the water's, tens of metres off when it is not | the same |

- A depth beyond a kilometre or two is a measurement only with a profile that covers the water between vehicle and
  target. **Dive through the column before a search** (or load a cast); without it treat long-range depths as indications
  and read the sigma, the ensemble's span and the rung (`depth_profile_source`, `multipath_profile_source`).
- `multipath_reason: no_direct_path` / `no_surface_path`: the profile says no such path reaches that range from the
  vehicle's depth, so there is no multipath depth to give. In the study's waters with the vehicle at 50 m in 200 m, or at
  225 m in 1000 m, a direct path existed in 10 of 10 cases at 1 km, 5 of 10 at 3 km, 3 of 10 at 6 km and 2 of 10 at 12 km,
  and a direct with a surface path in none at 12 km. The flat pattern used to fit whatever arrived there.
- `multipath_note: shadow` with `multipath_shadow_m`: the depth fits, but part of the column is not reached by the
  direct path at that range; a target there could give the same echo by other paths. Treat the depth as one explanation.
- `depth_ensemble_m`, `depth_multipath_ensemble_m`: where the depth would lie if the unsampled water were any of the
  ensemble's profiles. Much wider than the sigma means the sigma's single tilt does not describe what is not known.
  The span is only as wide as the configured gradients and extents make it; it is not a bound.

- **Stacked rows: `elevation.sector_from_rays: {}` is on in every shipped stacked-row deployment file** (two and three
  rows, generated and hand-written). The reason is the rows with unequal spacing: their far lobes are only a few dB down
  (4 dB for the three-row static array); where several arrivals share a range cell one can win outright, unflagged, and its
  bearing correction moves the detection by a degree (`ssp_thermocline_static_trial_3row`: -31 degrees on every ping, no
  track). The key limits the search to elevations a ray with two bounces or fewer can arrive from.
- **Stacked rows: read `depth_multipath_limited` before a depth.** With `elevation.multipath_sigma: {}` (on in every
  shipped stacked-row file) a detection's and a track's depth carry `depth_unresolved_sigma_m` inside `depth_sigma_m`.
  Where the flag is true the rows are reading the surface or the sea bed, not the target: in 100 m of water at 3 km from
  30 m the two-row face reads 0 to 11 m or 93 to 99 m for a target anywhere between 10 and 95 m, and states 19 to 48 m
  of sigma for it (the generator's `rehearsal` self-test, and the sweep in `PHYSICS_NOTES.md` Sec. 29). At 1 km with
  three rows the same target is resolved and the term is under 2 m. `depth_unresolved_span_m` gives the depths that
  could have given the reading. Without a water depth the chain has no bottom in this picture and says so in
  `depth_unresolved_assumed`: give it one (`elevation.water_depth`, or navigation) wherever the bottom is within reach.
- **Shallow water without `multipath_depth.bottom`: `withhold_flagged: true`** (what the generator writes, with
  `rays: true`, for a build that switches `multipath_depth` on), or read `multipath_note` before the depth. A flagged depth's sigma is the fit's and is small even when the fit is to the bottom's echoes.

### The coverage picture (`chain.coverage`)

One member of the report line says what the sonar can see from where it is (keys: docs/CONFIGURATION.md):

```
"coverage": {"own_depth_m": 225, "water_depth_m": 1000, "profile": "built", "pd": 0.5, "age_pings": 3,
  "ranges_m": [750, 1767, ..., 12700],
  "classes": [{"name": "xl_auv", "max_range_m": 4893, "pd": [1, 1, 1, 1, 0.4, 0, ...], "direct_to_m": 11683,
               "blind_m": [[2530, 3040, 20, 65]], "max_range_ensemble_m": [3643, 4922], "recommended_max_range_m": 6142}, ...],
  "recommend": {"own_depth_m": 60, "gain_m": 1249}}
```

- **An empty sector or a blind one?** `pd` at the range in question (`gpusonar.coverage.could_have_seen(report["coverage"], range_m)`):
  near 1, a target of that class there would have been reported, so the sector is empty; near 0, the sonar could not
  have seen it from this depth, whatever the bearing.
- `max_range_m` is for the worst of the class's depths; `blind_m` lists what is not seen nearer than that (range band,
  depth band); `direct_to_m` is where the direct path stops reaching every depth of the class.
- `recommend`: change depth when `gain_m` is worth the transit. It looks at detection range only, within
  `own_depth_limits_m` and the bottom clearance.
- `profile` is the rung all of it rests on, and `max_range_ensemble_m` how far the range moves over the profiles the water
  may also be. On `prior`, `local` or `none` the picture is that of isovelocity water (or the assumed gradient).
- **The water depth and the bottom are required** (`coverage.water_depth_m`, `coverage.bottom`): a chain given
  `coverage` without either does not start, and says which is missing. They have no default because without them the
  picture is the deep-water one: on the popoto_12km scenario (sonar at 10 m, 100 m of water over sand, ship at 12 km
  detected at 20 to 23 dB, the record without a water depth) it said the ship could not be seen; with them, probability 1.
  `water_depth_m` is the site's depth and is used on a ping that carries none (`"water_depth_from": "site"` in the member);
  navigation's altitude or `elevation.water_depth` comes first when it is there.
- It is on in the shipped face and vehicle deployment files, for an ILLUSTRATIVE site (100 m over sand): replace the site
  before a deployment (see "From a build spec").
- Before a mission: `tools/coverage_map.py --build <build> --environment <name> --own-depth <m> [--profile built:LO-HI]
  [--figure out.png]` draws the same numbers against range and target depth.
- It costs time only when it is refreshed: 0.03 s in deep water without a bottom, 0.3 to 0.8 s with a bottom or an
  ensemble (one core, ten own depths tried).

## A trial in a fjord or a harbour: what to expect and how to configure

From the rehearsal in fjord-type water (`tools/fjord_rehearsal.py`, `docs/figures/fjord_rehearsal.md`,
`PHYSICS_NOTES.md` Sec. 26, `RESULTS.md`). The rehearsal's profile, seabed section and seabed are illustrative, its
late reverberation is fitted to one fjord on one night at 4 to 8 kHz, and it is four pings and one seed per case: read
the numbers as what kind of thing to expect, not as predictions for another site. Enclosed water differs from the
open-water rehearsal in one main way: sound keeps arriving for about a second after every strong arrival (27 to 30 dB
of decay per second in the measured fjord).

1. **Range.** The reported range is that of the strongest arrival, which can be a bottom bounce 20 to 40 m behind the
   target, with 2 to 6 more detections following on its bearing. Check range against a transponder or a surveyed
   target, not against the first ping.
2. **SNR.** A strong echo reads 22 to 30 dB, not the 40 to 70 dB of open water: the echo's own reverberation fills the
   detector's background window and caps the reading. That is the water, not a fault of the array.
3. **Clutter.** With the tail there were 20 to 47 detections per ping at other bearings and 16 to 32 false tracks at
   short range with the 0.1 s pulse; none above 15 dB. Use `chain.min_snr_db: 15` (the deployed value is 9): it left
   0.5 detections per ping and no false track, with the target's SNR unchanged.
4. **A long pulse or a weak target.** Use a short CFAR window instead, `chain.cfar: {guard_m: 30, train_m: 100}`. It
   costs 3 to 6 dB of reported SNR, but at 2.6 km with the 1 s pulse it kept all four pings (12.4 dB) where
   `min_snr_db: 15` lost two.
5. **Measure the ping's own reverberation first, and set the blanking from it.** Nothing in the repository measures
   it. Ping with no target and plot level against time; expect a fall of 27 to 30 dB per second if the water is like
   the measured fjord. Set `chain.blanking` to the time at which it has fallen below the expected echo, and plan the
   first targets beyond it. In the rehearsal's what-if a 10 dB target was undetectable out to 1 km with one row
   whatever the chain settings, and unaffected at 2 km.
6. **Depth from stacked rows needs a profile.** Cast one and give it as `chain.elevation.ssp`. Without it the
   three-row depth at 150 m was 3 to 4 m off (21.3 to 21.8 m for a target at 18.0 m, with a stated sigma of 2.6 m);
   with it, 18.0 to 18.6 m. With two rows at kilometres there is no usable depth in either case.
7. **Hanging depth.** Hang the array above the shallowest ground between it and the target (a 53 m sill removed the
   direct path to a target at 60 m in the rehearsal), and on or above the sound-speed maximum rather than far below
   it. Own depth is otherwise worth at most 3 to 4 dB at these ranges (`tools/fjord_depth_plan.py`).
8. **Noise near infrastructure.** Record a minute of it and fit it (`tools/fit_noise.py`, then
   `receiver.ambient_noise: {model: fitted, file: ...}` in the rehearsal scenario) rather than use the wind curve. The
   measured site read 63 against 44 dB re 1 µPa²/Hz at its two hydrophones, and the quieter one had heavy-tailed
   clicks. Under the louder noise the tail's clutter is hidden and the target kept 24 dB.

Rehearse with `configs/scenarios/fjord_*_trial_*.yaml`, replacing the profile, the section and the tail's three
numbers by the site's as soon as a cast, a chart and one sounding of a pinger exist
(`tools/basin_reverberation_fit.py` shows the fit). `tools/fjord_rehearsal.py` fails on three of its six variants at
the present head (`STATUS.md`, open item 18); the committed table is from before the deployment-default change, which
was measured not to move any fjord figure.

## Listening without transmitting

The array is for active pings; the same backend also gives passive bearings from any
stretch of recorded samples:

```bash
build/bin/sonar_live --passive recording.h5 --band 4000 6000 n_fft=256 hop=128 n_average=12
```

One JSON line per frame: time, the bearings of the sources found, their powers and the
noise level, in the units of the samples squared. Options: `passive_method=conventional`
for the plain beamformer's bearing-time record, `sbl_min_snr_db`, `sbl_max_sources`, the
bearing grid.

From the boards, without transmitting:

```bash
build/bin/sonar_live --udp 5700 --template template.h5 --listen --band 4000 6000 \
    --recording listen.h5 --report listen.jsonl n_fft=4096 hop=2048 n_average=12
```

The stream is cut into stretches of one frame each (`n_fft + (n_average − 1) × hop`
samples), recorded, and solved as they complete; a lost block discards its stretch.

## As a Holoscan application

```bash
cmake -S . -B build -DSONAR_ENABLE_CUDA=ON -DSONAR_BUILD_HOLOSCAN=ON     # needs the Holoscan C++ package, CMake >= 3.30
build/bin/sonar_live_holoscan --udp 5700 --template template.h5 --nav-port 5702 --report out.jsonl backend=cuda
```

Same flags, options and reports as `build/bin/sonar_live`. The operators are in
`libs/holoscan_ops` (UDP or replay source → recorder → chain or passive bearings → report
sink); `tools/gpu_job.sh` with `GPUSONAR_HOLOSCAN_CPP=1` shows the install on a clean
Ubuntu 24.04 CUDA image. Run and compared with the standalone program on an L4
(`RESULTS.md`).

## Through Holoscan Sensor Bridge's receiver

```bash
tools/build_hsb.sh /opt/hsb          # Sensor Bridge 2.7.0: emulator, socket receiver, Python bindings, against Holoscan 4.6.0
tools/sonar_live_hsb.py serve --scenario flank_target_1km --ip 127.0.0.1             # the simulator behind Sensor Bridge's emulator
tools/sonar_live_hsb.py receive configs/deploy/popoto_vehicle.yaml --hololink 127.0.0.1 --report out.jsonl
```

`serve` cuts the board stream into fixed-size frames (one block of the UDP format each)
and sends them with Sensor Bridge's own emulator; `receive` is a Holoscan application of
Sensor Bridge's `LinuxReceiverOp`, an operator that turns frames back into pings
(`python/gpusonar/sources/hsb.py`), and the chain. On a real board only `receive` runs.

State (`RESULTS.md`, "Sensor Bridge loopback"): Sensor Bridge 2.7.0 builds natively
against Holoscan 4.6.0, and our operators, run inside Holoscan behind a stand-in that emits
what the receiver emits, give the direct chain's reports exactly; a withheld frame drops
exactly its ping.

The emulator opens a raw socket, which Hugging Face containers refuse. With
`HSB_UDP_PATCH=1 tools/build_hsb.sh /opt/hsb` and `HSB_EMULATOR_UDP_SOCKET=1` in the
emulator's environment it sends through an ordinary UDP socket instead
(`third_party/patches/hsb-2.7.0-emulator-udp-socket.patch`, `DECISIONS.md` D-79);
`tools/gpu_job.sh` sets both when `GPUSONAR_HSB=1` and raw sockets are refused. Without the
variable the patched emulator behaves as stock.

With the patch the loopback ran on T4 and L4 jobs. At half real time (`serve --pace 0.5`;
the default `--pace` is 1.0, real time) all frames arrive and the reports equal the direct
chain's. Two things are not settled. The loopback test's final assertions have not yet
passed in a job: they were corrected after the last job and await the GPU job on the
merged tree. And at real-time pace each run loses one burst of datagrams, which costs one
frame, for a reason not established; the receiver's socket buffer is the system default
(208 KB), which those containers cannot raise.

On a machine or container with CAP_NET_RAW (plain Docker grants it) the stock emulator
should run unpatched; that is untested. So is the RoCE receiver, which needs real network
cards.

## What `snr_db` is, and how it relates to the link budget

A detection's `snr_db` is 10 log10 of **its strongest sample over the detector's background there**, on the beamformed,
pulse-compressed range × bearing map (the rows' maps added in power when `elevation` stacks rows). The background is the
mean power of the cells `cfar.train_m` either side of it along range, beyond `cfar.guard_m`. A track's `mean_snr_db` is
the mean of that over the detections it took. This is the default detector, `cell_statistic: multiscale`
(`libs/live` `detector.hpp`); `mean` is the detector until now, for which `snr_db` is the strongest range cell's **mean**
power over the same background, and `peak` its strongest sample over the mean of the cells' largest samples.

A link budget (`gpusonar.sonar_equation`, `buildspec.performance`, `tools/popoto_range_study.py`) states something
else: the **peak signal power over the mean noise power at the output of an unweighted beamformer of one row**. For a
point echo the chain reads lower, by `gpusonar.live.snr_loss_db(record, range_cell_m, window, cell_statistic)`
(measured in `tests/python/test_detector.py` and `test_hardcoded.py`, a free-field point target against the beam output
computed separately):

| Part | `multiscale` | `mean` (until now) | Why |
| --- | --- | --- | --- |
| aperture window | 1.76 dB | 1.76 dB | Hann weighting against noise that is uncorrelated between hydrophones: N Σw² / (Σw)². It buys −31 dB bearing sidelobes where an unweighted row has −13 dB |
| range cell | 0.5 dB | 7.7 dB | the map is computed every 0.25 m for a 0.375 m resolution (4–6 kHz). `multiscale` reads the sample nearest the echo's peak; `mean` averaged it with the other seven samples of a 2 m cell |
| total | 2.3 dB (measured 2.4) | 9.4 dB (measured 9.4) | |

### The detector

**What was wrong.** The cell-averaging threshold for 10⁻⁵ (10.7 dB) is the threshold for ONE exponentially distributed
sample. It was applied to the mean of the 8 correlated samples of a cell (5.8 independent looks), which exceeds 10.7 dB
in noise about once in 10²¹ cells: the false-alarm probability typed was never the one obtained, the threshold was 4.8 dB
higher than that mean needs for 10⁻⁵, and a point echo had lost 7.7 dB in the same mean. Rows added in power changed
nothing: the threshold on the mean stayed where it was, so two or three rows detected no further than one.

**What it is now** (both engines; `LiveChain.detector` and the program's `detector {...}` line print it):

- Every range cell is tested on its **strongest sample** and on its **mean**, each against the threshold of its own
  statistic (`libs/detect` `statistic.hpp`: Erlang for a sample of the added rows, a weighted sum of exponentials for a
  mean of correlated samples; exact or saddlepoint, held to Monte Carlo). Longer means (`cfar.integrate_cells: [1, 4, 16]`)
  can be added for echoes tens of metres long; see below for why they are not on by default.
- The false-alarm budget is stated **per ping** (`cfar.false_alarms_per_ping`, 10⁻³: one false detection in a thousand
  pings of noise, the whole map and both tests together) and the thresholds follow from the map's size: 13.0 dB for a
  sample and 7.3 dB for a cell's mean on one row of the 1.4 km window at 12 km; 10.6 dB for a sample with two rows
  added, 9.3 dB (and 4.8 dB for the mean) with three. `cfar.pfa` belongs to `mean` and `peak` only.
- **One echo, one detection.** A flagged cell that a stronger cell's sidelobes explain is dropped: the compressed pulse's
  along range (1/(π x)² for a swept pulse), the aperture window's along bearing (the pattern of the hydrophones the
  record has, over the band: a dead board's gap raises it; not below `cfar.sidelobe_floor_db`, −40 dB), with a margin
  of `cfar.sidelobe_margin_db` (10 dB). A cell inside a stronger echo's own core (its half-power beam, the next range
  cell) is never dropped. Flagged cells are then
  grouped when they are no further apart in range than `cfar.merge_m` (default: the tracker's range sigma, 15 m) and no
  further apart in bearing than half the beam's half-power width (2.6° for a 32-hydrophone row at 5 kHz); touching cells
  always are. So a target's highlights and the arrivals of one multipath group are one detection. It reports its
  strongest sample as `snr_db`; its range, the energy-weighted mean of its samples; its bearing, the mean of its beams
  weighted by their energy above half the strongest one's (within 0.06° of truth on `popoto_elevation_free`, as before;
  from ping to ping it scatters by 0.02° where the old centroid scattered by 0.01°); and `extent_m`, the range it spans.
- **What a detection holds of its echo.** `first_range_m` and `extent_m` (it covers the range cells from the one centred
  on `first_range_m` over `extent_m`), and, when there are two or more, `arrivals`: `[range_m, snr_db]` of each local
  peak along range inside it that stands above the single-sample threshold (the strongest sample over its bearings;
  at most the 8 strongest, nearest first). A submerged target's direct echo and its echoes by the surface are ONE
  detection and one track, with those arrivals inside it (`tests/python/test_detector.py`: a target at 150 m seen from
  225 m at 6 and 12 km, arrivals 2.8 to 11 m apart). The merge window: flagged cells up to 16 m apart in range (21 ms of
  two-way delay) are joined, so arrivals up to 16 m apart are always in one detection and arrivals more than 20 m apart
  never. The complex samples behind them, at any range sampling over the detection's window at its bearing:
  `beam_samples` (`libs/live` `chain.hpp`; `core.live_beam_samples` in Python), one row at a time.
- **What is resolved.** Two echoes are two detections when they are more than 20 m apart in range (the merge distance
  and a cell either side), or more than one and a half half-power beamwidths apart in bearing (7.9°). A weaker echo
  less than 10 dB above a stronger one's sidelobe envelope at its place is not told apart from those sidelobes: on
  the same bearing that is an echo more than 14 dB below the stronger one 4 m away, 34 dB below it at 20 m, 44 dB at
  60 m. `doa: sbl` resolves bearings inside a beam.
- **The background at the ends of the map.** Near either end the training window has cells on one side only, and
  reverberation falls with range: the plain mean of cells that all lie beyond the cell under test reads low (1.7 dB at
  the near end of `flank_patrol`). With 6 dB of unintended threshold that went unnoticed; with a matched threshold it is
  a row of false alarms. The mean is therefore moved along the background's slope there (the slope between the medians
  of the first, or last, two training lengths of the same bearing).

**Probability of detection**, point echo, steady, one ping, 12 km window (`live.detection_probability`; measured through
the C++ chain on 50 simulated pings per level for one row and 20 for three, `tests/python/test_detector.py` repeats one
level):

| Budget SNR of one row | 11 dB | 12.5 | 14 | 15.5 | 17 | 18.5 | 20 | 21.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| one row, `multiscale`: measured (closed form) | 0.00 (0.01) | 0.08 (0.05) | 0.32 (0.22) | 0.56 (0.58) | 0.98 (0.92) | 1.00 | 1.00 | 1.00 |
| one row, `mean`: measured (closed form) | 0 | 0 | 0 | 0 | 0.00 (0.002) | 0.24 (0.08) | 0.58 (0.62) | 0.92 (0.95) |
| three rows, `multiscale` | 0.45 (0.44) | 0.95 (0.83) | 0.95 (0.99) | 1.00 | 1.00 | | | |
| three rows, `mean` | 0 | 0 | 0 | 0 | 0 | | | |

The budget SNR for a probability of one half (`live.budget_threshold_db(record, chain, pd=..., rows=...)`), and for 0.9:

| Rows added | `multiscale` Pd 0.5 / 0.9 | `mean` Pd 0.5 / 0.9 | Gain at 0.5 |
| --- | --- | --- | --- |
| 1 | 15.1 / 16.8 dB | 19.7 / 21.0 dB | 4.6 dB |
| 2 | 12.6 / 14.3 dB | 19.7 / 20.7 dB | 7.1 dB |
| 3 | 11.2 / 12.8 dB | 19.7 / 20.6 dB | 8.5 dB |

Against the best this architecture can do for a point echo at the same false-alarm budget (an unweighted row, the echo
on a sample, one test: 12.5 dB for one row), the detector is 2.6 dB short: 1.76 dB is the Hann window (kept: an
unweighted row gave 15 to 52 detections a ping), 0.5 dB the echo falling between samples, 0.3 dB the second test and the
search over an oversampled map. Of the 7.7 dB the cell's mean cost, `snr_db` has 7.2 dB back. The detection threshold
has 4.6 dB of it back for one row, because the honest threshold for a sample (13.0 dB at one false detection in a
thousand pings) is 2.3 dB above the 10.7 dB that was typed for a statistic it did not fit.

**Echoes that are not points.** Total echo energy needed for Pd 0.5 when it is spread evenly over an extent in range
(one row, Monte Carlo of the statistic alone, the energy stated as the peak SNR the same echo would have as a point):

| Extent | `mean` (until now) | sample + cell (default) | + 8 m | + 8 m + 32 m |
| --- | --- | --- | --- | --- |
| point | 18.0 dB | 13.0 | 13.1 | 13.2 |
| 5 m | 20.8 | 16.3 | 15.0 | 15.1 |
| 20 m | 25.0 | 20.2 | 17.9 | 16.9 |
| 100 m | 30.5 | 25.4 | 23.0 | 20.6 |

Integration over range IS right for an extended echo: 1.3 dB at 5 m, 3.3 dB at 20 m, 4.8 dB at 100 m. The longer means
are nevertheless an option (`cfar.integrate_cells: [1, 4, 16]`), not the default: their thresholds are 4.0 and 2.1 dB
over the background, which a strong, long echo's bearing sidelobes (measured −28 dB on `flank_patrol`) and any error of
the background reach. On `flank_patrol` they give 3 to 6 detections a ping and a fourth, `uncertain` track against
3 targets (9 to 14 a ping before the background's slope was handled). Use them in noise that is known to be flat.

**Measured on the shipped scenarios** (both engines give the same lines; before → after; detections per ping, strongest
`snr_db`, tracks at the last ping):

| Scenario | True targets | `mean` (until now) | `multiscale` (default) |
| --- | --- | --- | --- |
| `popoto_testauv_trial_1row` | 1 | 1 a ping, 48.7 dB, 1 static | 1 a ping, 52.7 dB, 1 static |
| `popoto_static_trial_2row` | 1 | 1, 32.7 dB, 1 static | 1, 37.0 dB, 1 static |
| `popoto_static_trial_3row` | 1 | 2 to 3, 68.7 dB, 2 static | 1, 73.8 dB, 1 static |
| `popoto_elevation_free` | 3 | 3, 38.5 dB, 3 static | 3, 44.7 dB, 3 static |
| `popoto_12km` (image-source stand-in) | 1 | on 3 of 6 pings, 13.7 dB, `uncertain` from ping 5 | every ping, 17.5 dB, `mover` from ping 3 |
| `popoto_12km_2row` (stand-in) | 1 | never | every ping, 14.7 dB, `mover` from ping 3 |
| `popoto_12km_3row` (stand-in) | 1 | on 1 of 6 pings, 11.9 dB, no track | every ping, 15.9 dB, `mover` from ping 3 |
| `popoto_elevation_shallow` (multipath) | 1 | 1, 43.4 dB, 1 static | 1, 49.3 dB, 1 static |
| `popoto_face_2row_shallow` (multipath) | 1 | 1, 34.5 dB, 1 static | 1, 37.7 dB, 1 static |
| `flank_target_1km` (Bellhop, one ping) | 1 | 4 detections, 18.5 dB | 1 detection, 27.0 dB |
| `flank_patrol` (Bellhop, 0.2 s pulse, 8 arrivals) | 3 | 3 to 5, 30.9 dB; mover, static, `multipath`, `uncertain` | 3 every ping, 36.5 dB; mover, mover, static |

On noise alone (the 12 km window, 220 pings of one row): no detection from either detector. At budgets large enough to
count the new one keeps its budget: 3.3, 1.3, 0.35 and 0.12 false detections a ping at budgets of 3, 1, 0.3 and 0.1.
With three rows the detector runs about twice over its budget (3.8, 1.7, 0.47, 0.22, 0.05 and 0.033 a ping at budgets
of 3, 1, 0.3, 0.1, 0.03 and 0.01, on 30 and 60 pings) and reported one false detection in 90 pings at the default budget;
the old detector none. Not explained; `cfar.search_factor: 5` would bring it back for 0.15 dB.

**For a range prediction.** Use `live.detection_probability(snr_of_one_row, record, chain, rows=...)`, or
`live.budget_threshold_db(record, chain, pd=0.5, rows=...)` as the threshold on the SNR of ONE row, without a separate
gain for the rows (it is in the threshold). For the shipped builds, with their own receive windows (a longer window is
more cells to search: 0.5 dB between 1.4 km and 12 km of window):

| Build | Rows | Window | One-row budget SNR for Pd 0.5 / 0.9 | What the rows are worth at the detector (one row's threshold less theirs) |
| --- | --- | --- | --- | --- |
| `popoto_face_1x32` | 1 | 0.75–12.7 km | 15.6 / 17.3 dB | — |
| `popoto_face_2x32` | 2 | 0.75–12.7 km | 13.1 / 14.7 dB | 2.6 dB |
| `popoto_face_3x32` | 3 | 0.75–12.7 km | 11.6 / 13.2 dB | 4.0 dB |
| `popoto_static_2x32` | 2 | 75–3000 m | 12.8 / 14.5 dB | 2.5 dB |
| `popoto_static_3x32` | 3 | 75–3000 m | 11.4 / 13.0 dB | 4.0 dB |
| `popoto_testauv_1x32` | 1 | 75–2000 m | 15.2 / 16.8 dB | — |

**The link budgets use exactly these figures.** `buildspec.detection(d)` computes them from the build's own template
record and the chain settings of its deployment file, for the detection probability the spec names
(`operation.detection_probability`, 0.5 unless given); `buildspec.performance`, `surface_term`, the depth planner
(`gpusonar.depthplan`, `tools/popoto_depth_planner.py`), `tools/reference_plan.py`, `tools/popoto_range_study.py`,
`tools/popoto_elevation_study.py` and `tools/projector_requirements.py` take their threshold from it. The budget's SNR
is ONE row's and no row gain is added to it: the rows are in the threshold. The typed 13 dB they used before, with a
closed-form row gain on top (2.4 and 3.7 dB), stood for 15.5 dB at a probability of one half (17 dB for nine in ten);
with the detector until now it was 19.7 dB of one row's SNR whatever the number of rows. The reference ship (+25 dB,
free field, `buildspec.budget`) at 12 km on the face builds, margin at Pd 0.5 and the probability of detection on one
ping:

| Rows | The margin as it was quoted (13 dB) | `mean`: margin, Pd | `multiscale`, and the budget now: margin, Pd | Pd at 10 / 12 / 14 km, before → after |
| --- | --- | --- | --- | --- |
| 1 | +1.6 dB | −5.1 dB, 0.00 | −1.0 dB, 0.24 | 0.24 → 1.00 / 0.00 → 0.24 / 0.00 → 0.00 |
| 2 | +4.0 dB | −5.0 dB, 0.00 | +1.6 dB, 0.89 | 0.17 → 1.00 / 0.00 → 0.89 / 0.00 → 0.05 |
| 3 | +5.3 dB | −5.0 dB, 0.00 | +3.0 dB, 0.99 | 0.14 → 1.00 / 0.00 → 0.99 / 0.00 → 0.24 |

The source level is RMS in the budget and in the simulator alike (D-102), so no convention stands between them. What
is left between the budget and the chain for a point echo in free field at 12 km, measured term by term
(`tools/surface_image_validation.py --only-chain`, `through_chain.accounting` in
`docs/design/surface_image_validation.json`): the aperture window and the echo's place between the map's samples,
both inside the threshold above, and about half a decibel the budget leaves out (the sweep's taper with the
absorption's slope across the band, the noise spectrum's slope across the band, the row's gain across the band).

`cell_statistic: mean` (with `range_cell_m` and `cfar` as before) is the detector until now, line for line:
`tests/python/test_detector.py` recomputes its detections from the map with the cell-averaging detector alone.

Not done, and why: a weighted replica. Measured with the library's pulse compression (`recon.prepare(window=...)`) at
time-bandwidth products of 200 and 2000: Hamming costs 1.1 dB and widens the compressed pulse 1.41 times for −36 dB
range sidelobes, Hann 1.5 dB and 1.53 times for −31 dB (unweighted: −13.3 dB); at a product of 20 the sidelobes are
only −28 and −27 dB, so a rule that picks the window would weight nothing below a product of about 100. The chain
does not weight: the sidelobe envelope removes a strong echo's range sidelobes at no loss, at the price of the
masking stated above, and a weighted replica would need its own sample correlation and envelope in the detector's
plan; a Taylor aperture window (computed for 32 hydrophones: 0.69 dB of loss at −30 dB sidelobes against the Hann
window's 1.76 dB at −31.5 dB, so 1.1 dB back; it needs the window in `flank.py` and in the elevation estimate); training cells that leave out other detections (a second echo
100 to 400 m from a strong one is still desensitised by it, as before).

Related report fields: with stacked rows a detection whose elevation is ambiguous carries `"bearing_corrected": false`:
its `bearing_deg` and `range_m` are the row's own (cone angle and slant range), not corrected by an elevation that is
one of several candidates. A track never waits for a depth, a sound-speed profile or an unambiguous elevation: those
add `depth_m` to a track that exists without them. A track is called `multipath` only of a track that is itself
classed `mover`, `static` or `line`.

## Keys a new sonar must review

Bringing up another array, band, board or platform is a matter of files: a sonar file (`configs/sonars`), a deployment
file (`configs/deploy`) and, for the C++ program, the options `tools/sonar_live.py CONFIG --export-template` prints.
Nothing below is fixed in code any more. Run one ping first and read the line the programs print once,
`chain settings {...}`: it lists what the chain derived from the sonar, which values the file gave (`given`), which the
sonar's floor decided instead of the standing value (`scaled`), and `warnings`.

**From the sonar file, nothing to type.** Channel count, rows, pitch, band, pulse, sample rate, receive window, ADC
scale and the boards come from the sonar file through the record. The chain derives from them: range cell, CFAR guard
and training, tracker sigmas, the multipath bearing gate, the classifier's fit limit, the row tolerance and the passive
block length (`CONFIGURATION.md`, "Sizes the chain derives"). For an array like the ones shipped (kilometres of range,
2 kHz of band or more, a beam under 16°) these are the numbers used so far. Check them when the line says `scaled`.

**Review for every new sonar** (deployment file → `sonar_live` option):

| Key | Why |
| --- | --- |
| `source.sonar`, `source.port`, `nav.port`, `output.report_udp` | required or site-specific; there is no default sonar and no default port in code |
| `source.sample_bytes` → `stream_sample_bytes` | the boards' sample word (3 bytes). The block header does not say it; a wrong value is a counted bad block, never wrong samples |
| `source.buffer_bytes` → `stream_buffer_bytes` | derived from the data rate; read the granted size at start-up and raise `net.core.rmem_max` if a warning says it is smaller |
| `source.queue_mb` → `stream_queue_mb` | 1024 MB of waiting blocks: seconds of stream = this / (channels × rate × bytes). A chain slower than real time fills it; the summary then counts dropped blocks |
| `source.timeout` → `idle_seconds` | longer than the longest silence of the stream (a board that sends only inside its receive window is silent for the rest of the ping interval); 0 for a service that must never end by itself |
| `source.host` → `bind` | the sonar network's interface, when the computer has several |
| `chain.bearings_deg` | the sector is a requirement; the step should be at most half a beamwidth (a warning says when it is not) |
| `chain.cfar.guard_m`, `train_m` | the standing 100 m and 300 m are for echoes spread by a waveguide at kilometres; a warning says when the receive window is shorter than both sides of them. A short-range sonar should set them |
| `chain.classify.multipath.max_excess_range` (800 m), `velocity` | the site: the longest extra path a bounce can have at the water depth and range worked |
| `chain.classify.min_speed`, `static_speed`, `chain.tracker.process_noise`, `initial_speed_sigma`, `confirm_hits`, `max_misses` | the targets and the ping interval, not the sonar |
| `chain.cfar.false_alarms_per_ping`, `chain.min_snr_db` | the false-alarm rate wanted: expected false detections per ping in noise (10⁻³), and a floor on `snr_db` (9 dB). `chain.cfar.pfa` is for `cell_statistic: mean` and `peak` only |
| `chain.cfar.merge_m`, `sidelobe_floor_db` | what is one detection: echoes of one bearing within the tracker's range sigma (15 m) are merged; the bearing sidelobe floor (−40 dB) is an assumption about the array's gain and phase errors until it is measured |
| `chain.sync.reference`, `search_time`, `tolerance_us`, `min_quality` | several boards: the reference burst above the band, how far apart the boards can start (2 ms), how well they must agree (a twentieth of a period; tighter for rows) |
| `chain.elevation.*` (sector, `min_row_elements`, `water_depth`, sigmas, gradient) | stacked rows only; the row tolerance is derived |
| `chain.doa: sbl` on a row of more than 64 hydrophones | runs on the host reference (about half the speed); every report line then says `"sbl_fallback": true` |
| `chain.blanking.recovery_s`, `chain.decimate.*` | hardware recovery time (to be measured); the decimation follows the band and rate by itself |
| passive `f_low`, `f_high`, bearing sector, `n_average` | what is listened for; `n_fft` and `hop` are derived |
| `output.report_udp_max_bytes` | 65000: lower it if the receiving side's buffer is smaller |

**Wire formats are contracts.** The block stream (`sources/flank_stream.py`: `GSFB`, version 1, 128-byte header) and the
report line are unchanged. Two things were added in ways an existing peer does not notice: the sample word length is a
parameter set alike on both ends (the header has no field for it; adding one is a format change that has not been
made), and a report line too long for one datagram is sent as several complete lines with `"part": [k, n]` (a reader
that parses each datagram as JSON sees reports of the same ping with fewer detections; one that knows the field joins
them). The Holoscan report sink sees text, not lists: it groups whole lines into datagrams and refuses, counts and
reports a single line that is too long, where `build/bin/sonar_live` splits it.

## What is not there yet

- **Refraction in operation: what is still missing.** The coverage picture, the profile ensemble and rays through the
  ladder's profile are built (sections above). Not there: rays through water that changes with range; reverberation as
  a limit in the coverage picture; a depth sigma for two rows that owns up to unresolved boundary echoes (the
  `popoto_face_2x32` rehearsal self-test fails on it); a multipath depth for the test AUV in 30 m of water (the ray
  mode answers `window`). `STATUS.md`, "Open and known wrong", has the list.
- The board forwarder. The navigation adapter exists (`tools/nav_adapter.py`), but the mapping for the
  vehicle's own unit is a template: its interface document has not been seen.
- A service wrapper beyond the example unit `configs/deploy/sonar_live.service` (paths and
  user are placeholders; the vehicle's operating system is not known).
- Range correction for paths that arrive by the bottom (in deep water the simulated ship
  shows 300 m long in range, `SCENARIOS.md`).
- Telling a moored object from a rock, or a ship from a whale: the classifier uses the
  track only.
- Any run at the 12 km range on real water. The design's link budget and five simulated
  oceans are in `DESIGN.md` and `SCENARIOS.md`; the ocean changes the answer by tens of dB.
