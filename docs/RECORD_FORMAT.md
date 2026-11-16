# Record format

One ping record is the unit every source produces and every stage consumes. This file
documents the on-disk container. The in-memory struct is `sonar::PingRecord` in
`libs/core/include/sonar/core/record.hpp`; the two have the same fields.

- Container: HDF5 (why: `DECISIONS.md` A-01).
- Schema: `gpusonar.record`, version **1.0.0**.
- Code: `libs/io` (`RecordWriter`, `RecordReader`, `validate`, `validate_file`).
- The files are plain HDF5. `h5py` reads them with no gpusonar code
  (`tests/python/test_record.py::test_file_is_readable_with_plain_h5py`).

**Changing this schema needs the project owner's sign-off.**

## Conventions

- SI units: metres, seconds, radians, hertz. Angles in the sidecar and in configs are
  radians too.
- Frame: north-east-down. Body axes: x forward, y starboard, z down. Attitude is roll,
  pitch, yaw with the ASASIN Eq. 2 conventions.
- Times inside a ping are seconds relative to that ping's transmit instant. Fields ending
  `_ns` are integer nanoseconds since the Unix epoch, UTC.
- "Unknown" for an optional float is NaN. An absent dataset means an empty table.

## Versioning

- Root attributes `schema_version_major`, `_minor`, `_patch`.
- Major: a reader refuses a file whose major version differs from its own.
- Minor: fields were added. Old readers ignore them; new readers supply defaults.
- Patch: documentation or validator changes only.

## File layout

```
/                                   attributes: schema_name, schema_version_{major,minor,patch},
                                                created_by, config_yaml
/pings/00000000/                    one group per ping, zero-padded index, in write order
/pings/00000001/
...
/truth/                             optional; simulator ground truth (free-form float64 arrays
                                    and string attributes)
```

`config_yaml` holds the resolved config that produced the file, verbatim.

### One ping group

Scalars are attributes; tables are datasets. `C` = channels, `N` = samples.

**Samples** (attributes on the ping group, plus one dataset)

| Name | Type | Meaning |
| --- | --- | --- |
| `samples` | dataset `[C, N]` | float32, int16 or int32 for real data; a compound `{r, i}` of the same base type for complex data (h5py shows float32 pairs as complex64) |
| `n_channels`, `n_samples` | int64 | shape, repeated so a header-only read needs no dataset access |
| `sample_kind` | int64 | 0 real passband, 1 complex baseband, 2 complex analytic (carrier present) |
| `sample_rate` | float64 | Hz |
| `center_frequency` | float64 | Hz; the removed carrier for baseband data |
| `start_time` | float64 | time of sample 0 relative to transmit, s; may be negative |
| `storage_type` | int64 | 0 float32, 1 int16, 2 int32 |
| `storage_scale` | float64 | physical value = stored integer × scale (integer storage only) |

Integer storage keeps raw ADC counts compact: 12-bit ultrasound samples as int16,
24-bit hydrophone samples as int32. In memory, samples are always float32.

**`transmit/`**

| Name | Type | Meaning |
| --- | --- | --- |
| `time_ns` | int64 attr | transmit instant |
| `kind` | int64 attr | 0 point source (real or virtual), 1 plane wave |
| `direction` | `[3]` | body-frame unit vector of travel, plane wave only |
| `waveform_type` | int64 attr | 0 CW, 1 LFM, 2 Gaussian, 3 arbitrary |
| `f_start`, `f_stop` | float64 attr | Hz; equal for CW; a Gaussian pulse is centred on their mean and `f_stop − f_start` is its −6 dB bandwidth |
| `duration` | float64 attr | s |
| `window_alpha` | float64 attr | Tukey taper fraction of the envelope, 0 = rectangular |
| `source_level_db` | float64 attr | dB re 1 µPa at 1 m, NaN if unknown |
| `amplitude` | float64 attr | peak amplitude at 1 m in the sample units |
| `replica_sample_rate` | float64 attr | Hz, arbitrary waveform only |
| `replica` | `[M]` | replica samples, arbitrary waveform only |

**`geometry/`**

| Name | Shape | Meaning |
| --- | --- | --- |
| `tx_pose`, `rx_pose` | `[4, 3]` | rows: position (NED, m), roll/pitch/yaw (rad), velocity (NED, m/s), attitude rates (rad/s), all at the transmit instant. Two poses because the projector can be on a different body from the array. |
| `tx_lever_arm`, `tx_boresight`, `tx_beamwidth` | `[3]`, `[3]`, `[2]` | transmitter in the body frame; beamwidth is (azimuth, elevation) FWHM, NaN = omnidirectional |
| `rx_lever_arm`, `rx_boresight`, `rx_beamwidth` | `[C, 3]`, `[C, 3]`, `[C, 2]` | one row per channel |
| `frame_origin_lla` | `[3]` | latitude (deg), longitude (deg), altitude (m) of the NED origin; NaN if not georeferenced |

Position of a body point at time `t` after transmit:
`position + velocity·t + R(rpy + rates·t) · lever_arm`. This first-order model is the
paper's kinematic model and is what the backprojection kernel evaluates.

**`navigation/`** (rows are samples inside this ping's interval; `t` relative to transmit)

| Name | Columns |
| --- | --- |
| `ins` `[K, 13]` | t, position (3), roll/pitch/yaw (3), velocity NED (3), attitude rates (3) |
| `dvl` `[K, 5]` | t, body velocity (3), altitude above the seafloor (NaN without bottom lock) |
| `gps` `[K, 5]` | t, latitude, longitude, altitude, horizontal accuracy |
| `depth` `[K, 2]` | t, depth (positive down) |

**`calibration/`** (one entry per channel)

| Name | Meaning |
| --- | --- |
| `gain` `[C]` | linear amplitude gain of the channel; conditioning divides by it |
| `phase` `[C]` | rad at `center_frequency`; conditioning removes it |
| `delay` `[C]` | s; conditioning advances the channel by it |
| `position_offset` `[C, 3]` | body-frame correction added to the nominal lever arm |

**`environment/`**

| Name | Meaning |
| --- | --- |
| `sound_speed` | m/s at the sonar |
| `temperature`, `salinity`, `water_depth` | °C, PSU, m; NaN if unknown |
| `profile_depth`, `profile_speed` | optional sound-speed profile, depth strictly increasing |

**`metadata/`**

| Name | Meaning |
| --- | --- |
| `array_id` | name of the sonar config that describes the array |
| `mode` | 0 unknown, 1 FLS, 2 SAS, 3 flank |
| `sequence` | ping counter from the source |
| `sub_sequence` | index within a multi-transmit scan: mux phase, compounding angle |
| `channel_element` `[C]` | index of each channel's element in the full array. This is how the two halves of a 64-of-128 mux capture are told apart. |
| `channel_board` `[C]` | index into `boards` |
| `boards` `[B, 5]` | per capture board: clock offset (s), clock drift (ppm), sample index of the 1PPS edge (NaN if none), reference channel (−1 if none), synchronised flag |
| `hardware_gain_db` | fixed analogue gain |
| `tvg_time`, `tvg_gain_db` | knots of the hardware gain ramp, time since transmit and dB above the fixed gain |
| `adc_bits`, `adc_full_scale` | converter resolution, and the sample value at full scale (for clipping checks) |
| `extra_json` | source-specific settings as a JSON object |

## Validator

`validate(record)` and `validate_file(path)` return a list of issues, each with a
severity, a dotted field path and a message. They never throw on bad data.

Errors (the record is unusable):

- shapes: sample buffer size, one geometry element and one calibration entry per channel,
  channel maps sized to the channels, table pairs of equal length;
- values: non-finite samples or poses, non-positive sample rate, sound speed, gain or
  pulse duration, non-unit boresight or direction vectors;
- band: the transmit band does not fit in a complex baseband sample rate;
- order: navigation times, profile depths or gain-ramp times not increasing; transmit
  times going backwards in a file;
- file: wrong schema name, unsupported major version.

Warnings (usable, but look):

- real sampling below Nyquist for the band (legitimate for deliberate bandpass sampling);
- samples at ADC full scale, with the count;
- sound speed outside 1300–1700 m/s;
- empty array ID, unknown mode, pitch within 1° of gimbal lock, sequence going backwards.

## WAV sidecar

A `.wav` file cannot carry geometry or ping timing, so a YAML sidecar supplies them.
`gpusonar.sources.WavSource(sidecar_path)` reads Popoto recordings and generic
multichannel `.wav` this way and yields ordinary ping records.

```yaml
array_id: popoto_8
mode: flank                       # fls | sas | flank | unknown

# One multichannel file, or one file per channel (same rate and length).
file: trial_2026_11_03.wav
# files: [ch0.wav, ch1.wav, ...]

sample_rate: 96000                # optional; must agree with the audio if given
scale: 1.0                        # sample value of a full-scale input (default: ±1.0)
adc_bits: 24                      # optional; default taken from the file
hardware_gain_db: 0.0

# Element lever arms in the body frame: an explicit list of [x, y, z] ...
elements: [[-0.0945, 0, 0], [-0.0675, 0, 0], ...]
# ... or a uniform line, centred on `origin`:
# elements: {count: 8, pitch: 0.027, axis: [1, 0, 0], origin: [0, 0, 0]}
boresight: [0, 1, 0]
beamwidth: {az: 3.14, el: 3.14}   # optional, rad

transmit:
  lever_arm: [0, 0, 0.5]
  waveform: {type: lfm, f_start: 20000, f_stop: 35000, duration: 0.01,
             window_alpha: 0.1, source_level_db: 190}

pings:
  times: [0.10, 2.10, 4.10]       # transmit times in seconds from the start of the file
  # times: {first: 0.10, interval: 2.0, count: 50}   # or a regular train
  window: [0.0, 1.5]              # start after transmit, and duration, s

recording_start_ns: 1700000000000000000   # UTC of the first sample, optional
pose: {position: [0, 0, 5], rpy: [0, 0, 0], velocity: [0, 0, 0]}
environment: {sound_speed: 1480, temperature: 18, salinity: 0, water_depth: 6}
frame_origin_lla: [-31.95, 115.86, 0]

# Several boards with their own clocks (for the sync stage):
boards:
  - {channels: [0, 1, 2, 3, 4, 5, 6, 7], reference_channel: 7}
  - {channels: [8, 9, 10, 11, 12, 13, 14, 15], reference_channel: 15}
```

Behaviour worth knowing:

- Ping windows are snapped to whole samples. `samples.start_time` reports where the
  window really starts, so the snap costs no timing accuracy.
- A ping whose window runs off the recording is dropped and counted in
  `WavSource.dropped_pings`.
- Contradictions are errors, not guesses: element count against channel count, sidecar
  sample rate against the audio, per-channel files of different lengths or rates.
- 24-bit PCM is supported (it arrives left-justified in 32 bits and is scaled to ±1).
