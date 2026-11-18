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
| Results | UDP JSON, one datagram per ping, and a `.jsonl` file | Field names in `live.py`. Datagrams are cut at 65 kB (hundreds of tracks). |
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

A detection's `snr_db` is 10 log10 of **the power of the strongest range cell of its cluster over the detector's
background for that cell**, on the beamformed, pulse-compressed range × bearing map. The background is the mean power
of the cells `cfar.train_m` either side of it along range, beyond `cfar.guard_m`. A cell is reported when that ratio
exceeds both the detector's own factor for `cfar.pfa` (10.7 dB at 10⁻⁵) and `min_snr_db` (9 dB).
A track's `mean_snr_db` is the mean of that over the detections it took.

A link budget (`gpusonar.sonar_equation`, `buildspec.performance`, `tools/popoto_range_study.py`) states something
else: the **peak signal power over the mean noise power at the output of an unweighted beamformer**. For a point echo
the chain reads lower by two things, both measured (`tests/python/test_hardcoded.py`, a free-field point target
against the beam output computed separately) and both given by `gpusonar.live.snr_loss_db(record, range_cell_m,
window, cell_statistic)`:

| Part | Size, as shipped (4–6 kHz, 2 m cell, Hann) | Why |
| --- | --- | --- |
| aperture window | 1.76 dB | Hann weighting against noise that is uncorrelated between hydrophones: N Σw² / (Σw)². `window: rect` has none, and −13 dB sidelobes instead of −31 dB |
| range cell | 7.7 dB (measured 7.6) | the map is computed every 0.25 m and the cell holds the **mean** power of its 8 samples; an echo one range resolution long (0.375 m) fills about one and a half of them |
| total | 9.4 dB (measured 9.4) | |

So the chain as shipped reports a point echo from **20.1 dB** of budget SNR (`live.budget_threshold_db`), not from the
13 dB the plan's range figures use. A third difference is a matter of convention, not of the chain: a budget that
takes the source level as a peak where the chain's pings carry rms (or the reverse) is another 3 dB. Together these are
the 12 dB by which the detector was found to read under the budget for the same target; nothing is left unexplained.

What can be had back, measured on the shipped stage scenarios and the 12 km stand-in (four pings each, C++ engine):

| Change | Point echo, free field | On the scenarios | Cost |
| --- | --- | --- | --- |
| `cell_statistic: peak` (the cell holds its largest sample; the background is then the mean of such maxima, 4.3 dB above the mean noise for 8 samples) | +2.8 dB | −0.9 to +3.1 dB on the strongest detection; same detections and tracks, except that the 12 km mover was still `uncertain` after four pings where it was `mover` | none in compute |
| `range_cell_m` at 0.75 of the resolution (one sample per cell, no averaging) | +7.2 dB | +3.4 to +6.3 dB on the strongest detection, but 5 to 16 detections where there was 1 (range sidelobes of the compressed pulse and the target's own highlights become detections of their own), 4 to 8 tracks where there was 1, and the 12 km mover `uncertain` | tracks fragment |
| `window: rect` | +1.8 dB | 20 to 50 detections per ping where there were 1 to 3 (bearing sidelobes) | false tracks |

Neither is the default: the 2 m cell of mean power also merges an echo's range sidelobes and highlights into one
detection, which is what the tracker needs, and the measured gain of `peak` on echoes that are not points is small.
Getting the 7 dB back without the fragmentation needs work in the detector itself (a weighted replica to bring the
range sidelobes down, detections merged over the guard width, and a threshold matched to the cell's statistic); it is
not done. Until then a range prediction should use the chain's threshold in the budget's units
(`live.budget_threshold_db`), and state the target's source-level convention.

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
| `chain.min_snr_db`, `chain.cfar.pfa` | the false-alarm rate wanted |
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
