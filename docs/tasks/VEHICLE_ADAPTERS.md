# Task: vehicle adapters for the live chain (board samples and navigation)

Issue `sonar-o7m` (P1). Written 2026-11-23 at commit `6600df8` on branch `docs-handoff` for an
agent with no history of this project. Read `docs/HANDOFF.md` first, then this file.

## 0. What is not known, and blocks the work

**Nothing in this task can be finished from the repository alone.** Both interfaces the
adapters have to speak are unconfirmed: no interface document, datasheet, sample capture or
packet capture of the real Popoto boards or of the vehicle's navigation unit has been seen.
Every format statement in the code about them is either ours (the `GSFB` block stream) or a
placeholder (`configs/nav/sprintnav_template.yaml`). Do not write a decoder for a format you
have not been given. Ask the owner the questions in section 3 first, and record the answers
(section 6) before any code.

What can be done before the answers arrive: read sections 1 and 2, run the existing tests
named there, and prepare the questions. Nothing else.

## 1. Goal and done-criteria

Goal: the live chain (`tools/sonar_live.py` and `build/bin/sonar_live`) takes samples from
the real Popoto capture boards and the pose from the vehicle's real navigation unit, with
no change to the chain's processing, and both engines (Python `python/gpusonar/live.py`,
C++ `libs/live`) give the same reports on the same input.

Done means all of the following:

1. **Board samples.** One working path from the boards to ping records, chosen with the
   owner (section 4, step 2): either (A) the boards' recordings as WAV with a YAML sidecar
   picked up by `wav_watch`, (B) a forwarder that turns the boards' native output into
   `GSFB` version 1 blocks for the existing `flank_udp` receivers, or (C) a new native
   source in both engines. Each hardware fact it relies on is in a config file, with the
   source of the fact named in a comment, and removed from that file's `unconfirmed` list
   only when the owner or a document confirmed it.
2. **Navigation.** A mapping for the vehicle's unit, filled from its interface document
   (`configs/nav/<unit>.yaml`, from `configs/nav/sprintnav_template.yaml`), with a reader
   if its output is not line-oriented text, delivering the chain's datagram
   (`NavListener` in `python/gpusonar/live.py`, `Navigation` in
   `libs/live/include/sonar/live/udp_source.hpp`).
3. **Tests**, written before or with the code, all passing: decoding of synthetic fixtures
   that mimic the confirmed formats byte for byte; an end-to-end run of the chain fed
   through each new adapter that equals the chain fed directly; engine parity (Python and
   C++) on the adapter's output. No test is weakened to pass; if one has to change, say so
   plainly in the report.
4. **Docs**: `docs/DEPLOYMENT.md` ("What the vehicle has to provide", "Navigation from an
   inertial unit with a Doppler log", "What is not there yet") updated to what is now
   true; `docs/CONFIGURATION.md` for any new key; `docs/RECORD_FORMAT.md` only for the WAV
   sidecar section and only if the owner approved a change (section 5).
5. **Measured and recorded** (numbers name tool and commit): on the synthetic fixtures,
   samples recovered (max error in LSB), records per second the adapter sustains on this
   machine's CPU against the boards' data rate (32 channels x 3 bytes x 102,400 Hz =
   9.83 MB/s if the configured rate is confirmed), and for navigation the time offset and
   latency used. If a bench capture from the real hardware is supplied: dropped, late and
   bad block counts (`dropped_blocks`, `overrun_pings`, `bad_blocks` printed by
   `tools/sonar_live.py`), and the first report line's `sync` object.

Never mark the issue done with a failing test.

## 2. Current state (checked in the code at `6600df8`)

### 2.1 How samples reach the chain

`tools/sonar_live.py CONFIG` reads the deployment file's `source` section and builds a
source with `make_source` (`python/gpusonar/sources/base.py`; a class registers itself
by `type_name` with `@register`). Every source yields `PingRecord`s; the chain
(`LiveChain.process`) does not know where they came from. Per ping the chain stamps the
pose from navigation, writes the raw ping to the recording, then synchronises the boards,
compresses, beamforms, detects, tracks and classifies (`docs/DEPLOYMENT.md`, "What runs").

Sources that matter here:

| `type` | File | What it reads | Engines |
| --- | --- | --- | --- |
| `wav_watch` | `python/gpusonar/sources/wav_watch.py` (`WavWatchSource`) | a directory; takes each `*.yaml` sidecar once its audio files have stopped growing for `settle` s, reads it with `WavSource` | Python source only. The C++ chain runs behind it with `tools/sonar_live.py --engine cpp`; `build/bin/sonar_live` has no WAV input (`apps/sonar_live.cpp` takes `--replay`, `--udp`, `--passive`) |
| `wav` | `python/gpusonar/sources/wav.py` (`WavSource`) | one sidecar; format in `docs/RECORD_FORMAT.md`, "WAV sidecar" | as above |
| `flank_udp` | `python/gpusonar/sources/flank_stream.py` (`FlankUdpSource`, `PingAssembler`, `encode_block`, `decode_block`, `FlankBoardEmulator`) | `GSFB` v1 blocks as UDP datagrams | both: the C++ counterpart is `libs/live/src/udp_source.cpp` (`PingAssembler`, the UDP receiver), used by `build/bin/sonar_live --udp PORT --template FILE` |
| `container` | `python/gpusonar/sources/container.py` | a recording (`--replay`) | both |

Facts about these that an adapter must respect:

- **The `GSFB` block is ours and provisional.** It was designed for a flank capture board
  that does not exist (`flank_stream.py` docstring; `docs/HOLOSCAN_NOTES.md` section 8.1).
  Little-endian, 128-byte header (magic, version 1, channel count, samples in the block,
  index of the first sample since capture start, sample index of the latest transmit or
  -1, ping index of that transmit, gain code, flags with bit 0 = ADC overflow, capture
  start in ns since the Unix epoch), then signed samples, sample by sample. The sample word
  (`sample_bytes`, default 3) is not in the header; it is `source.sample_bytes` in the
  deployment file and `stream_sample_bytes` for the C++ program. `docs/DEPLOYMENT.md` ("Wire
  formats are contracts") treats this layout as a contract: changing it is a format change
  and needs the owner (STATUS question 13).
- **`PingAssembler` needs the transmit's sample index** in the block header to cut a ping:
  each transmit opens the sonar config's receive window. A board stream that does not say
  when the projector fired cannot be cut into pings by this assembler.
- **The WAV sidecar carries the array itself** (`elements`, `transmit`, `pings.times`,
  `boards` with `reference_channel` for synchronisation, `recording_start_ns`). `WavSource`
  does not read `configs/sonars/*.yaml`, so a sidecar writer must reproduce the sonar
  file's geometry. Without `recording_start_ns` each ping's `transmit.time_ns` counts from 0
  (`WavSource.read`), which cannot be matched to navigation time. `metadata.sequence` is the
  ping's index within its file, so it restarts with every file.
- **The flank_udp path takes the array from the sonar config** (`source.sonar`, e.g.
  `popoto_long_range`) and, for the C++ program, from a template record written by
  `tools/sonar_live.py CONFIG --export-template template.h5`.
- **Sensor Bridge** (`python/gpusonar/sources/hsb.py`, `tools/sonar_live_hsb.py`) carries
  the same `GSFB` blocks as fixed-size frames. Feeding Sensor Bridge's receiver from real
  hardware is a separate issue (`sonar-0ra`); do not do it here.

Hardware facts already in config, all unconfirmed (`unconfirmed:` lists in the files):
`configs/sonars/popoto_8.yaml` (102.4 kHz, 24 bits, 2.61 V full scale at the ADC, 43 dB
gain, -208 dB sensitivity, simultaneous sampling of the 8 inputs: each listed unconfirmed),
`configs/sonars/popoto_array.yaml` (`receive.board_size: 8`), `configs/sonars/popoto_4x7.yaml`
(`receive.boards`, `receive.reference` unconfirmed), `configs/sonars/popoto_long_range.yaml`
(`transmit`, `array.pitch` unconfirmed), and the deployment file
`configs/deploy/popoto_vehicle.yaml` (`source: {type: flank_udp, sonar: popoto_long_range,
host, port: 5700, timeout}`).

### 2.2 How navigation reaches the chain

- The chain takes one fixed datagram: JSON over UDP, `{"time_ns", "x", "y", "z", "roll",
  "pitch", "yaw", "vx", "vy", "vz"}`, north-east-down metres, radians (rotation z-y-x from
  world NED to body forward-right-down), velocity over ground in m/s; optional
  `sound_speed`, `altitude`, `bottom_lock`, `water_depth` and one-sigma fields. Python:
  `NavListener` (`python/gpusonar/live.py`; `from_config` reads `nav.port`, `host`,
  `static`, `water_depth`, `keep_s`, `surface_z`, `profile_step_m`, `history_margin_s`).
  C++: `Navigation` (`libs/live/include/sonar/live/udp_source.hpp`), `--nav-port` and the
  `nav_*` options of `apps/sonar_live.cpp`.
- The pose is interpolated to each ping's `transmit.time_ns` between the fixes around it,
  held from the nearest one outside them. Fixes are kept for `keep_s` (120 s) behind the
  newest one. A fix without finite `vx`, `vy`, `vz` is taken at rest in what it lacks and
  counted in `pings_without_velocity`.
- **The adapter exists and is generic**: `python/gpusonar/navadapter.py` (`Mapping`,
  `Mapping.convert`, `parse_line`, `replay`, `run_udp`, `udp_sender`) and
  `tools/nav_adapter.py` (`--replay FILE [--speed 1]`, `--in-udp PORT`, `--stdin`,
  `--out-udp HOST:PORT`, `--out-file`). A YAML mapping describes the vendor side: format
  (`csv` or `json` lines), fields, units, ENU/NED, Euler order and signs or quaternion,
  body FRD/FLU, velocity frame, time unit and `offset_s`, Doppler-log lock and invalid
  codes, sigmas, `output.rate_hz`. `configs/nav/generic_csv.yaml` (with
  `generic_csv_example.csv`) is a worked example. `configs/nav/sprintnav_template.yaml` is
  the vehicle unit's template (the owner named the unit as a SPRINT-Nav; "sprintnav IMU
  and dvl"): every vendor-side entry is `FILL_IN` and listed `unconfirmed`, and
  `tests/python/test_nav_adapter.py::test_the_unit_template_is_marked_unconfirmed_and_claims_nothing`
  holds it so.
- Binary or non-line output: the module's docstring and the template say a small reader
  that yields one dict of fields per record, then calls `Mapping.convert`, replaces the
  line parser; the mapping stays.
- **Not found in the code**: a key for the offset (lever arm and rotation) between the
  navigation unit's reference point and the array's body origin. The records' pose is
  taken as the body origin of the element lever arms. I searched `configs/`,
  `navadapter.py` and `live.py` for it and found none; check again before relying on this,
  and ask the owner before adding one (it touches both engines).
- `tools/sonar_live.py` ignores `nav` for `--replay`, `--scenario` and `--emulate`
  (recordings carry their pose already).

### 2.3 Tests that already hold these paths

- `tests/python/test_flank_stream.py`: `test_block_round_trip`,
  `test_assembler_recovers_the_pings`, `test_a_lost_block_costs_one_ping_only`,
  `test_wrong_channel_count_is_refused`, `test_emulated_board_over_udp`.
- `tests/python/test_hsb_receive.py`: assembler engine parity
  (`test_both_assemblers_make_the_same_decisions`, `test_both_assemblers_reorder_before_decimating`,
  `test_review_cases_in_both_assemblers`, `test_finish_counts_what_is_still_awaited_and_nothing_else`).
- `tests/python/test_live_cpp.py`: `test_cpp_assembler_matches_python`,
  `test_program_takes_blocks_over_udp_with_navigation`,
  `test_one_deployment_file_runs_either_engine`, `test_report_fields_match_the_python_chain`,
  `test_exported_template_and_command_run_the_program`.
- `tests/python/test_wav_watch.py`: `test_the_factory_builds_it`,
  `test_a_late_recording_is_picked_up_once`, `test_a_bad_recording_goes_to_failed_and_the_rest_are_read`.
- `tests/python/test_nav_adapter.py`: conventions, frames, units, time offset and rate,
  csv and json lines, the template, water depth from altitude, and parity:
  `test_python_and_cpp_receivers_put_the_same_navigation_into_a_record`,
  `test_chain_fed_through_the_adapter_equals_the_chain_fed_directly` (both engines),
  `test_both_receivers_mark_a_pose_without_velocity_and_take_a_missing_or_non_finite_one_as_zero`.

Run them one file at a time before you start, so you know the baseline (section 7).

## 3. Questions for the owner (ask before any code)

Ask in one message. Say which answers you need for which path; the owner may not have all
of them. Where an answer is "not known yet", that item stays in config and `unconfirmed`.

**Board samples**

1. What do the boards actually output during a ping train: files on a card or disk
   (format? WAV? one file per channel or per board? per ping or continuous?), a network
   stream (UDP or TCP? packet layout?), a serial or USB stream, or something else? Is there
   an interface document, and can you supply a short real capture (a few pings, any signal)?
2. Sample format: word length and signedness, byte order, channel interleaving, channel
   order against the hydrophone positions, full scale in volts, and the real sample rate.
   Are all inputs of one board sampled at the same instant?
3. How many boards, how are they numbered, and how does the host tell them apart (IP and
   port, file names, a header field)?
4. Timing: what clock stamps the samples (free-running per board, PTP, GPS PPS, the
   vehicle's clock)? Is there a sample counter since capture start? What happens to it at a
   restart or a dropout?
5. The transmit: how does the host learn when the projector fired, to the sample (a
   trigger line recorded on an input, a field in the stream, a fixed schedule from a known
   start, the direct blast found in the data)? Who triggers the projector?
6. Board synchronisation: is the reference burst of `configs/sonars/popoto_4x7.yaml`
   (one dedicated input per board) what will be wired, or the summed input of
   `popoto_4x8`, or none?
7. Gain: can the boards switch gain during a ping, and is the gain state reported with the
   samples?
8. Where would a forwarder run (on each board, on a board host, on the vehicle computer),
   with what OS, language and CPU? Is any code on the boards themselves ours to change? (Any
   FPGA or firmware change needs your approval first.)
9. Which path do you want: (A) WAV + sidecar files into a watched directory, (B) a
   forwarder that converts the boards' output to our `GSFB` UDP blocks so both engines stay
   unchanged, or (C) a native decoder of the boards' format inside both engines?

**Navigation**

10. The SPRINT-Nav interface document and datasheet (STATUS question 6): which output
    message(s) will be used, text or binary, which fields, at what rate, over which link
    (serial, UDP, TCP)? A short real log?
11. Time base of its time stamps (UTC, GPS, since power-up), time-stamp accuracy and output
    latency. Is it on the same clock as the boards (DEPLOYMENT.md assumes one clock)?
12. Frames and conventions: position as latitude/longitude or local north-east, depth sign,
    heading reference (true or magnetic), attitude convention (Euler order, signs) or
    quaternion, velocity frame (body or world, over ground or through water).
13. Does it (or a separate sensor) report sound speed? Altitude and bottom-lock state?
    Its own accuracy estimates?
14. The surveyed offset and rotation between the navigation unit and the array (no key for
    it exists; see 2.2). Should the adapter apply it before the datagram, or should both
    engines take a new key?
15. The test AUV's navigation unit (`popoto_testauv_1row`): the same unit or another?

**Data and privacy**

16. May confirmed hardware facts go into the committed `configs/` files, or must they stay
    in private config files outside the repository (loaded with `GPUSONAR_CONFIG_PATH`,
    `python/gpusonar/config.py`)?
17. Where will real captures be kept, and may their derived numbers (rates, counts, error
    statistics) be committed?

## 4. Plan

1. **Baseline.** Build (section 7) and run the test files of 2.3, one at a time. Note
   anything already failing; do not fix unrelated failures without asking.
2. **Record the answers.** Put every confirmed fact in the config file it belongs to with a
   comment naming its source (document and section, or "the owner, <date>"), and take it
   out of `unconfirmed`. Facts still open stay in config, marked `unconfirmed`. Nothing about
   the hardware is a constant in code. Choose path A, B or C with the owner.
3. **Fixtures first.** Write a generator for synthetic fixtures that mimic the confirmed
   format exactly (byte layout, framing, headers, time stamps, dropouts), from simulated
   pings: as the `pings` fixture of `tests/python/test_flank_stream.py` does
   (`SonarModel.load("flank_32", ...)`, `simulate_ping` on a one-point `Scene`, fixed
   `default_rng` seed), with the Popoto sonar the owner confirms in place of `flank_32`.
   Keep the generator in the tests (for example a fixture in
   `tests/python/conftest.py` or the new test file), small, and deterministic (fixed seed).
   If the owner supplied a real capture, keep it outside the repository and compare the
   generator's output with it field by field; never commit the capture.
4. **Board path.** Write the failing tests, then the code:
   - Path A (`wav_watch`): a sidecar writer that turns the boards' files plus the sonar
     config into `docs/RECORD_FORMAT.md` sidecars (same geometry as `source.sonar`, real
     `recording_start_ns`, `boards` with `reference_channel`). Natural place:
     `python/gpusonar/sources/` (a helper next to `wav.py`) and a tool in `tools/`. Test that
     `WavWatchSource` over the written files gives records equal to the simulated pings
     (samples within half an LSB, times, geometry), and that a run through
     `tools/sonar_live.py --engine python` and `--engine cpp` gives the same reports.
   - Path B (forwarder, recommended when the boards' format carries a sample counter and the
     transmit sample): a decoder of the native format plus `encode_block` from
     `flank_stream.py`, in a new module (e.g. `python/gpusonar/sources/popoto_board.py`) and
     a tool (e.g. `tools/board_forwarder.py`), modelled on `tools/nav_adapter.py` (replay a
     capture file, or read live, and send datagrams to `HOST:PORT`). Both engines are
     unchanged, so parity is inherited; test it anyway: native fixture -> forwarder ->
     `FlankUdpSource` and -> `build/bin/sonar_live --udp` give the same records and reports
     as `FlankBoardEmulator` on the same pings. If the forwarder must run where Python is not
     available, ask the owner before writing it in C++.
   - Path C (native source in both engines): a Python source registered with `@register`
     in `python/gpusonar/sources/` and the C++ counterpart in `libs/live` beside
     `udp_source.cpp`, with an option of `apps/sonar_live.cpp` to select it. Every rule
     (reordering, loss counting, restarts) must match between the two; extend the parity
     tests in `tests/python/test_hsb_receive.py` and `tests/python/test_live_cpp.py` to the
     new source. This path changes `libs/live`, a shared area: keep the change minimal.
5. **Navigation.** Copy `configs/nav/sprintnav_template.yaml` to a mapping for the unit,
   fill each entry from the interface document, delete each `unconfirmed` entry only when
   checked. If the output is binary or not one record per line, write a reader that yields
   one dict per record and calls `Mapping.convert` (in `navadapter.py` or a module beside
   it), and an input switch for it in `tools/nav_adapter.py`. Tests in
   `tests/python/test_nav_adapter.py`: a synthetic fixture in the unit's exact format with a
   known trajectory (including a turn, a roll, a loss of bottom lock, a time wrap or restart
   if the unit has one) must convert to the known datagrams; then
   `test_chain_fed_through_the_adapter_equals_the_chain_fed_directly` extended to this
   mapping on both engines. Keep `test_the_unit_template_is_marked_unconfirmed_and_claims_nothing`
   passing (the template stays a template; the filled mapping is a new file).
6. **Clock.** Whatever the answers to questions 4, 5 and 11, write the offset between the
   boards' time and the navigation's time as config (the mapping's `time.offset_s` and, for
   the boards, a key in the source config), never in code. Add a test where a known offset
   is applied and the pose lands at the right transmit time; DEPLOYMENT.md says an error
   here shows as tracks that move with ownship.
7. **Docs.** Update the DEPLOYMENT.md sections named in section 1, item 4, the deployment
   file example (`configs/deploy/popoto_vehicle.yaml`) if the source type changes, and
   `docs/CONFIGURATION.md` for any new key. Do not edit `docs/STATUS.md`, `docs/RESULTS.md`
   or `docs/DECISIONS.md` unless the owner or the orchestrator tells you to; otherwise put
   the proposed text in your report.
8. **Measure** what section 1, item 5 lists and put the numbers, with tool and commit, in the
   report (and in `docs/RESULTS.md` only if you are told you own it).

## 5. Rules you must follow

From the owner's brief (`docs/HANDOFF.md` section 2) and the project's working rules:

- Tests before or alongside every change. Never mark a milestone or issue done with failing
  tests. Never weaken a test to make it pass without saying so plainly.
- **Stop and ask the owner before**: adding a large or copyleft dependency; changing the
  record schema (`docs/RECORD_FORMAT.md`, `libs/core/include/sonar/core/record.hpp`; frozen
  since milestone 2) or the `GSFB` block layout; changing any FPGA (or board firmware) code;
  acting when a paper and a cited reference disagree in a way that changes results; and
  changing code beyond this task.
- GPL and LGPL code only as a separate test oracle or optional tool, never linked into the
  core. Do not copy code from any closed or GPL source. Record every open-source decision in
  `docs/OPEN_SOURCE.md`.
- Unconfirmed hardware facts live in config, not code. Do not invent numbers a datasheet or
  paper reports; a number with no source is marked as ours or unconfirmed.
- The two live-chain engines stay equal: any behaviour change in one is made in both and
  held by engine-parity tests.
- Real data is never committed: vehicle and board recordings, navigation logs, and any
  licensed or owner-supplied data (UACR, Watermark, Smart Ocean, PICMUS and the like). Keep
  it outside the repository and reference it by path in a config or an environment
  variable, the way `GPUSONAR_UACR_DIR` and `GPUSONAR_PICMUS_DIR` are used in the tests
  (a test that needs it skips without it). Only derived numbers and figures are committed,
  and only if the owner agrees.
- Privacy: the owner's identity and project details never go into a web query or any
  external request. Search with generic terms only (e.g. "24-bit little-endian interleaved
  PCM UDP framing", not the project or vehicle). No accounts, forms or emails. Paywalled,
  login-gated or bot-challenged pages: move on, never circumvent. APL-UW TR 9407 is
  distribution-limited: never fetch or read it. This task needs no web access.
- GPU work runs only on the owner's approved Hugging Face Jobs setup (`docs/HANDOFF.md`
  section 4): `git archive` the tree to a tarball, upload it to the owner's private dataset
  named there, and launch a job on an NVIDIA CUDA devel image that downloads the tarball and
  runs `tools/gpu_job.sh` (`bash tools/gpu_job.sh --help` lists its modes and variables; the
  default mode builds with CUDA and runs ctest and pytest). The job reads the Hugging Face
  token as a job secret; never print, paste or commit it. One job at a time, `l4x1`, a
  timeout of at most 3 h. This task should need a GPU run only to confirm that the CUDA
  backend gives the same reports through the new adapter (the default test mode does it).
- Issue tracking is `bd` (beads), prefix `sonar-` (`export PATH=$PATH:$HOME/.local/bin:$HOME/go/bin`;
  `bd show sonar-o7m`, `bd update sonar-o7m --claim`, `bd create` for follow-ups, `bd close`
  when done). Every `bd` write re-exports `.beads/issues.jsonl`, which needs its own commit.
  If an orchestrator tells you not to run `bd`, list follow-ups in your report instead.
- Commits: small and topical, with the trailer lines the harness gives you. Push only where
  the owner says.
- Writing style: plain statements; what failed, got worse or is uncertain first; every
  number names the tool and commit it came from.

## 6. Where answers and decisions go

- Hardware facts: the sonar file (`configs/sonars/`), the deployment file
  (`configs/deploy/`), the navigation mapping (`configs/nav/`), each with its source, or
  private files on `GPUSONAR_CONFIG_PATH` if the owner says so (question 16).
- The owner's answers and the path chosen: proposed `docs/DECISIONS.md` entry in your
  report (the next free D-number), and the answered STATUS questions (3, 6, 13) as proposed
  `docs/STATUS.md` text.

## 7. Build and test on this machine

15 GB RAM, 4 cores, no GPU, shared with other work. From `docs/HANDOFF.md` section 3:

```bash
export OMP_NUM_THREADS=1 PYTHONPATH=$PWD/python
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSONAR_BUILD_TESTS=ON -DSONAR_BUILD_PYTHON=ON \
      -DSONAR_ENABLE_CUDA=OFF -DSONAR_BUILD_HOLOSCAN=OFF \
      -DSONAR_CERES_DIR=/opt/ceres-nosuitesparse -DSONAR_ALLOW_SUITESPARSE=OFF
cmake --build build -j1                               # builds build/bin/sonar_live and python/gpusonar/_gpusonar
python3 -m pytest tests/python/test_flank_stream.py -q -p no:cacheprovider     # one file at a time
ctest --test-dir build -R <name>
```

Never run the whole suite at full width here; it takes about two hours.
