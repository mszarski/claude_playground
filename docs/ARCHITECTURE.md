# Architecture

How the pieces fit, where each runs, and how to get at the data between them.
Diagrams are drawn by `tools/architecture_diagrams.py`.

## One record, many sources, the same stages

![Data flow](architecture/dataflow.png)

- **Everything between stages is a ping record** (`RECORD_FORMAT.md`): the samples of one
  transmit, with the array geometry, pose, waveform and time. Simulated and real data are
  the same thing from the first stage on.
- **Sources** yield records and are chosen by name in a config:

| `type` | What it reads | File |
| --- | --- | --- |
| `scenario` | the simulator, from a scenario YAML | `sim/` |
| `container` | a recording or a tap (`.h5`) | `sources/container.py` |
| `tap` | the records after a named stage of an earlier run | `taps.py` |
| `wav` | a WAV recording with a YAML sidecar | `sources/wav.py` |
| `wav_watch` | a directory such recordings are dropped into | `sources/wav_watch.py` |
| `flank_udp` | capture-board blocks arriving over UDP | `sources/flank_stream.py` |
| `nvus_scanner` | the NVIDIA ultrasound reference's runtime | `sources/nvus_scanner.py` |
| `picmus` | the PICMUS benchmark files | `sources/picmus.py` |

- **Stages** are library functions on records. Three things call them, and nothing else
  holds processing logic:
  - `pipeline.run(config)`: a batch chain from a YAML file (`configs/chains/`), for imaging;
  - `live.LiveChain`: one ping at a time, for the long-range arrays on a vehicle
    (`DEPLOYMENT.md`);
  - `holoscan_ops`: the same functions wrapped as Holoscan operators.
- **Configuration is data.** A sonar is `configs/sonars/<name>.yaml`; a scenario is
  `configs/scenarios/<name>.yaml`; a chain is `configs/chains/` or `configs/deploy/`.
  `CONFIGURATION.md` lists the keys and `DESIGN.md` checks the owner's designs against them.

## Looking at the data at every stage

| To get | Do |
| --- | --- |
| The records after every stage of a batch run | `output: {taps: {dir: taps/}}` in the chain's YAML |
| The same for chosen stages | `taps: {dir: taps/, stages: [prepare, whiten]}` |
| Every stage of the first pings of a live run | `output: {taps: {dir: ..., pings: 2}}` in the deploy YAML |
| The samples after each stage inside the simulator | `source: {type: scenario, name: ..., sim_taps: dir/}`, or `with taps.sim_capture(dir): run_scenario(...)` |
| What a tap directory holds | `tools/inspect_tap.py taps/` |
| Numbers per channel (RMS, peak, clipping, strongest line) | `tools/inspect_tap.py file.h5` or `taps/ --stage prepare` |
| Waveform, spectrum, all channels, spectrogram | add `--plot out.png` |
| The samples for another tool | add `--wav out.wav` (32-bit float; I and Q as separate channels for baseband) or `--npy out.npy` |
| To run later stages from a tap | `source: {type: tap, dir: taps/, stage: whiten}` and list the remaining steps |
| To put your own waveform in | write records with `gpusonar.RecordWriter` (or a WAV and a sidecar) and name the file as the source |

![inspect_tap.py on a tapped Popoto ping](figures/inspect_tap_example.png)

- A tap file is an ordinary container file, so it works as a source on any machine.
  `test_taps.py` checks that a chain restarted from a tap gives the same image.
- Simulator stages, in order: `echoes`, `reverberation`, `reference_and_clocks`, `noise`,
  `hardware_gain`, `adc`. The last equals the record the simulator returns.
- The live chain always writes the raw pings (with the pose) before processing them. That
  recording replays through the same chain and gives the same reports (`test_live.py`).

## What runs on a GPU

![Stage by backend](architecture/backends.png)

- The two heavy kernels, echo synthesis in the simulator and backprojection in
  processing, are written once and compiled for the host (OpenMP) and for CUDA. So are
  conditioning (cuFFT on the device) and CFAR. Select with `backend=` or
  `backend: cuda` in a config; `auto` uses the GPU when the build has it and one is present.
- **Simulation on a GPU** means synthesis on a GPU: `run_scenario(name, backend=core.Backend.Cuda)`,
  or `tools/gallery.py --backend cuda`. Path geometry, reverberation and noise are
  generated on the CPU; for scenes with many scatterers or long pulses synthesis is nearly
  all of the time (91 of about 100 s per ping in `flank_patrol` on four CPU cores).
  Measured on an L4: the 445,000-scatterer SAS scene in 96 s against 818 s on four CPU
  cores, with the same numbers out; scenes whose background is statistical reverberation
  gain little (`RESULTS.md`).
- **Beamforming for the long-range arrays** goes through the backprojection kernel
  (`flank.backprojection_map`, the live chain's default), not the NumPy far-field
  beamformer: same detections, 11 times faster on a CPU (0.55 s against 6.1 s per ping at
  32 channels and 50,000 samples), and it runs on a GPU unchanged.
- **Sparse Bayesian bearings** run as three kernels (`libs/live/include/sonar/live/sbl_kernel.hpp`)
  on CPU threads or on the device, selected by the chain's `backend`. On an L4 the whole
  chain takes 18 ms a ping with bearings from the map and 0.48 s with SBL (`RESULTS.md`).
- **Not on a GPU, and why.** Clustering, tracking and classification handle tens of
  objects per ping. MVDR, whitening, median TVG, DRC, board synchronisation and motion
  estimation are CPU code; none is the bottleneck of a chain that uses them. Reverberation generation is NumPy FFTs and is the largest CPU-only cost in the
  simulator.
- The CPU path is complete: everything runs without a GPU and gives the same results.
  `RESULTS.md` has the GPU timings and the comparison.

## What is C++ and what is Python

The long-range array's live chain exists twice, on purpose.

- **C++ (`libs/live`, program `build/bin/sonar_live`)**: board blocks over UDP to ping
  records, navigation, recording, board synchronisation, conditioning, backprojection to
  range × bearing, CFAR, clustering, sparse Bayesian bearings, tracking, classification,
  the JSON report. No Python at run time. This is what goes on a vehicle.
- **Python (`gpusonar/live.py`, `tools/sonar_live.py`)**: the same chain stage by stage in
  NumPy on top of the same C++ kernels, with taps after every stage and the simulator as a
  source. `chain: {engine: cpp}` puts the C++ chain behind the same class.
- The tests hold the two together: clustering, tracker state, classes, synchronisation
  offsets and whole reports are compared (`tests/python/test_live_cpp.py`), and
  `tests/cpp/test_live.cpp` tests the library without Python.

Everywhere else the split is: kernels and file format in C++ (CUDA where listed above),
the simulator's scene and propagation, SAS motion estimation glue, calibration and all
tools in Python. Holoscan operators exist in both: Python wrappers of the stages
(`holoscan_ops.py`) and C++ operators around the live library (`libs/holoscan_ops`,
program `build/bin/sonar_live_holoscan`), built and run on a GPU machine with Holoscan 4.6.

### Bearings: the map, or sparse Bayesian learning

`chain: {doa: map}` (default) takes bearings from the beamformed map: one detection per
beam-wide blob. `doa: sbl` keeps the map and CFAR for *where in range* something is, then
solves for *which bearings* at each detected range with multi-frequency sparse Bayesian
learning (`libs/live/src/doa.cpp`). It separates arrivals closer than a beamwidth,
including coherent ones (multipath of one ping), from a single snapshot. Where a gate
holds no discrete arrival the map's detection is kept. The same solver on averaged
spectra of a recording gives passive bearing-time records (`sonar_live --passive`,
`core.doa_passive`). `RESULTS.md` has the measurements, `DECISIONS.md` D-57 the reasons.

## Calibrating the simulator to real data

`SIM_CALIBRATION.md`. Separate from array calibration (`calibrate.py`), which estimates a
real array's channel errors.

## Deployment

`DEPLOYMENT.md`: the Popoto array on a vehicle.

## Layout

| Path | What |
| --- | --- |
| `libs/core`, `libs/dsp`, `libs/recon`, `libs/sim`, `libs/detect`, `libs/io` | C++20 / CUDA: record, conditioning, backprojection, synthesis, CFAR, container file |
| `libs/live`, `apps/sonar_live.cpp` | C++20: the long-range array's live chain and the standalone program `build/bin/sonar_live` |
| `libs/holoscan_ops`, `apps/sonar_live_holoscan.cpp` | the same chain as C++ Holoscan operators and an application (`-DSONAR_BUILD_HOLOSCAN=ON`) |
| `python/gpusonar/` | bindings and everything else: `sim/`, `sources/`, `recon/`, `pipeline.py`, `live.py`, `taps.py`, `detect.py`, `classify.py`, `flank.py`, `motion.py`, `sync.py`, `calibrate.py`, `simcal.py`, `holoscan_ops.py` |
| `configs/` | `sonars/`, `scenarios/`, `chains/`, `deploy/` |
| `tools/` | `test.sh`, `gallery.py`, `sonar_live.py`, `inspect_tap.py`, `calibrate_sim.py`, `design_report.py`, `scenario_diagrams.py`, `architecture_diagrams.py`, `popoto_range_study.py`, `benchmark.py`, `gpu_job.sh` |
| `tests/` | GoogleTest and pytest, one command: `tools/test.sh` |
