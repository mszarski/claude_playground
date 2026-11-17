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
| Navigation | UDP JSON: `{"time_ns", "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz"}`, north-east-down metres and radians, any rate; interpolated to each ping's transmit time | The adapter from the vehicle's own navigation message to this one is **not written** (the vehicle's interface is not known). Without navigation, pings are processed where the record says they are and counted in `pings_without_nav`. |
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

## Stacked rows: elevation and depth (optional)

With two or three rows of hydrophones at different heights (`popoto_long_range_2row`, 64
hydrophones on 8 boards; `popoto_long_range_3row`, 96 on 12) the chain also reports each
detection's elevation and depth. What the option can and cannot do, in numbers, and the
choice between two rows and three: `DESIGN.md`, Sec. 3. One row needs none of this and is
processed exactly as before.

```bash
tools/sonar_live.py configs/deploy/popoto_vehicle_elevation.yaml                         # the boards, over UDP
tools/sonar_live.py configs/deploy/popoto_vehicle_elevation.yaml --scenario popoto_elevation_free --out run/
tools/sonar_live.py configs/deploy/popoto_elevation_rehearsal.yaml --scenario popoto_elevation_rehearsal --out run/
tools/sonar_live.py configs/deploy/popoto_vehicle_elevation.yaml --export-template template.h5   # prints the C++ command
build/bin/sonar_live --udp 5700 --template template.h5 --nav-port 5702 ... elevation=1 elevation_roll_sigma_deg=0.2
```

The deployment file differs from the one-row file in two places: `source.sonar` names the
stacked array, and `chain.elevation` is present (`CONFIGURATION.md` lists its keys). The
rows are found from the hydrophone positions in each record; nothing else says how many
there are. Per ping, with rows:

- each row is beamformed over range × bearing on its own and the rows' maps are added
  (a row's bearing response does not depend on elevation, so arrivals from above and
  below are not lost);
- at each detection (the strongest 32) the whole array is steered over elevation at that
  range and bearing. The first arrival of the detection is taken: the direct path;
- the detection's line in the report gains

| Field | Meaning |
| --- | --- |
| `elevation_deg`, `elevation_sigma_deg` | elevation above the horizontal (roll and pitch from navigation applied), and its one-sigma: the bound at the measured SNR plus `roll_sigma_deg`. `null`: none found |
| `elevation_quality` | 1 for one clean arrival; lower when several arrivals share the range cell. Below `min_quality` no elevation is reported |
| `elevation_ambiguous` | a second lobe within 1.5 dB (two rows, steep arrival): the elevation may be the wrong one and is not passed to the track |
| `bearing_deg`, `cone_bearing_deg` | the horizontal bearing, corrected for the elevation, and what the line alone reported |
| `depth_m`, `depth_sigma_m` | depth: own depth + range × sin(elevation), folded into the water column; with `elevation.ssp`, along the ray through that profile |
| `image_depth_m`, `path` | the straight-line depth before folding, and what it says of the path: `direct`, `surface` (the image is above the surface), `bottom`, `multiple` |

- each track gains `depth_m`, `depth_sigma_m` and `path` (a filter over its detections'
  depths), and a track whose straight-line depth stays outside the water column by more
  than two sigma is classified `multipath`.

Each of these can be switched off on its own (`correct_bearing`, `depth`, `fold`,
`classify`). Passive listening takes `elevation=1` too: every bearing in a frame then has
an entry in `elevations_deg` and `elevation_quality`.

What the vehicle must provide in addition to the list above:

| | Why | State |
| --- | --- | --- |
| Roll and pitch in the navigation datagrams, accurate to 0.1–0.2° | depth error = range × roll error: 0.2° is 3.5 m at 1 km and 42 m at 12 km | The fields exist in the datagram; what the vehicle's attitude sensor delivers is unconfirmed. `elevation.roll_sigma_deg` must be set to it. Roll is applied as at the transmit time, not at each echo's arrival. |
| Own depth (`z`) in the same datagrams, and where the surface is in that frame (`elevation.surface_z`) | reported depth is own depth plus the measured difference | as for one row |
| Water depth (`elevation.water_depth`) | folding bottom bounces and recognising them | from the altimeter plus depth sensor, or the chart; without it only surface bounces are recognised |
| A sound-speed profile (`elevation.ssp`) beyond a few kilometres | a straight ray is 51 m off at 3 km in merely isothermal water | from a CTD cast or a model; without it depths are straight-ray |
| Boards synchronised to better than 1 µs between rows | 1 µs is 0.23° of elevation | 0.4 µs in simulation with the shared reference; or interleaved wiring (`array.interleave: 4`), which removes the term for two rows |
| An in-water calibration with sources at several depths | a row 1 mm further out than drawn is 0.15° | `calibrate.estimate_channel_errors(..., planar=False)` |
| A failed hydrophone or board named in `chain.dead_channels` | so that it is left out, not summed | the chain does not detect dead channels by itself |

First time in water with rows, after step 4 above: lower a source (or use the support
boat's echo sounder) at a known depth at two or three bearings; check `depth_m` against
it with the vehicle level and again rolled 5°. A depth error that changes sign with the
side the source is on is a row offset; one that follows the roll is the attitude sensor.

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

State (`RESULTS.md`): Sensor Bridge builds natively and imports on an L4; our operators,
run inside Holoscan behind a stand-in that emits what the receiver emits, give the direct
chain's reports exactly, and a withheld frame drops exactly its ping. The loopback through
the emulator itself has not run on Hugging Face: the emulator opens a raw socket, which
those containers do not permit. It should run unchanged on a GPU machine or container with
CAP_NET_RAW (plain Docker grants it). The RoCE receiver needs real network cards.

## What is not there yet


- The two adapters named above (board forwarder, navigation).
- A service wrapper beyond the example unit `configs/deploy/sonar_live.service` (paths and
  user are placeholders; the vehicle's operating system is not known).
- Range correction for paths that arrive by the bottom (in deep water the simulated ship
  shows 300 m long in range, `SCENARIOS.md`).
- Telling a moored object from a rock, or a ship from a whale: the classifier uses the
  track only.
- Any run at the 12 km range on real water. The design's link budget and five simulated
  oceans are in `DESIGN.md` and `SCENARIOS.md`; the ocean changes the answer by tens of dB.
