# Status

Updated at the end of each milestone. Newest facts first within each section.

## Where things stand

| Milestone | State | Evidence |
| --- | --- | --- |
| 1 Notes and survey | done | `PAPER_NOTES.md`, `HOLOSCAN_NOTES.md`, `OPEN_SOURCE.md`, `DECISIONS.md`; `tools/test.sh` builds and runs GoogleTest and pytest |
| 2 Record format and file replay | done | `RECORD_FORMAT.md`; `tests/cpp/test_record_io.cpp`, `tests/python/test_record.py`, `tests/python/test_sources.py` |
| 3 Simulator, point targets, all sonar configs | done on CPU; GPU half not run | `tests/python/test_sim.py`: delay and amplitude against analytic values on every sonar, an independent frequency-domain oracle, Doppler, receiver motion, surface image, board clocks, ADC. `test_gpu_forward_model_matches_cpu` is **skipped** here (no device). |
| 4 Backprojection kernel, FLS mode | in progress | |
| 5–13 | not started | |

## What is next

1. Milestone 4: backprojection kernel and FLS mode, PICMUS comparison.
2. On the first GPU host: run `tools/test.sh` and confirm the skipped GPU tests pass.

## Environment this was built in, and what that limits

- The work so far was done in an x86_64 cloud container with **no GPU**. The brief's
  development host is a DGX Spark.
- Every kernel has a host backend and a CUDA backend from the same source
  (`DECISIONS.md` A-03). Here the CUDA backend is **compiled** with nvcc 12.0
  (`cmake -DSONAR_ENABLE_CUDA=ON`, host compiler g++-12) but has **never been executed**.
- Consequences until someone runs `tools/test.sh` on the Spark:
  - GPU-versus-CPU agreement tests are skipped, not passed.
  - No GPU timing exists. Nothing in this repo reports a GPU speed.
  - Holoscan cannot be installed or run here (it needs CUDA and Python 3.12 wheels).

## Open questions for the project owner

1. **Plan PDF.** `docs/plan/GPU-FLS-Phased-Plan.pdf` is not in the repo yet. Hardware
   numbers currently come from the brief alone.
2. **Ceres and SuiteSparse (copyleft).** Ubuntu's `libceres-dev` links SuiteSparse
   CHOLMOD/SPQR, which are GPL. Ceres itself is BSD and builds without SuiteSparse. Motion
   estimation (milestone 8) will use a Ceres built with `-DSUITESPARSE=OFF` unless you say
   the distro package is acceptable. The distro package is installed in this container but
   nothing links it yet.
3. **Holoscan versions.** The ultrasound reference at `24d405e` pins Sensor Bridge 2.7.0
   and Holoscan 4.4.0 with six local SDK patches. Sensor Bridge HEAD is 3.0.0 on Holoscan
   5.0, which is not API compatible. Proposed: build our ultrasound source against the
   reference's pinned pair, and keep the low-rate flank source on stock operators so it
   can follow either line. Confirm.
4. **Jetson receive path.** Per the Sensor Bridge source, AGX Orin is Linux-sockets only
   and AGX Thor is built without RoCE (CoE or sockets). 38.4 Gbit/s ultrasound capture
   therefore looks like DGX Spark / IGX only; the 49 Mbit/s flank source is fine anywhere.
5. **`popoto_4x8` reference channel.** Four 8-input boards give 32 inputs. A shared
   reference "wired into one input on every board" leaves 28 hydrophones, not 32.
   Proposed default: 32 hydrophones, with the reference summed into one hydrophone input
   per board as an out-of-band pilot tone; the dedicated-input variant (28 hydrophones)
   stays available. Confirm which matches the hardware plan.
6. **Copyleft test oracles.** Bellhop / Acoustics Toolbox and bellhopcuda (GPL-3), k-Wave
   (LGPL-3) and PyMUST (LGPL) are proposed as separate-process oracles only, as the brief
   allows. Nothing copyleft is installed in the build or linked. Say if any of these should
   not be used even as an oracle.
7. **Blocked reading.** The container's network policy blocks `apl.washington.edu`
   (TR 0702, TR 9407), `oalib-acoustics.org` and the original PICMUS host. Either allow
   those hosts in the environment or drop the PDFs in `docs/refs/`. PICMUS data itself is
   reachable through a Hugging Face mirror (`zeahub/picmus`, a converted layout); the
   published PICMUS reference numbers could not be fetched, so none are quoted yet.

## Discrepancies between the brief and the references

From reading the reference code (`HOLOSCAN_NOTES.md` section 1 has file and line for each):

- The reference's live imaging app is a native C++ loop around the non-operator
  `RoceReceiver`, not a Holoscan operator graph.
- The 1024-frame receive ring is the reference's own setting (`rx_pages`), settable from
  Python or C++ with no FPGA change. Stock `RoceReceiverOp` defaults to 2 pages.
- "Odd/even" mux: per the code, "odd" is elements 0–63 and "even" is 64–127. It is two
  contiguous halves, not an interleave. The mux is the TX73L64's own T/R switch select.
- Gain ramp: 12 dB and about 122 µs are the encoder's maxima, not the shipped settings
  (shipped: 12 dB, start 2 µs, ramp 10 µs). No AFE5932 attenuator register is written
  anywhere in the reference; the "unused 0–24 dB attenuator" was not found.
- A 150k-sample receive window exceeds two host-side caps (65,536 samples in the C++
  sequence compiler, 32,768 in the capture CLI). Neither is an FPGA limit;
  `Evm.init(samples=150000)` has no cap. Sustained 38.4 Gbit/s for 3 ms through the
  FPGA FIFO is untested by the reference, which streams about 60 µs per ping.
- No FPGA timestamp or ping counter reaches the application in the reference; the
  Sensor Bridge metadata has PTP times but the reference drops them.
- The reference transmits 5, 6.25 or 8.33 MHz only. Nothing in it supports sonar carrier
  frequencies; stages 3–6 need their own transmit chain.
- Docs say probe "L12-5 D"; code says "L7-3".

## Simulator limits so far

- Scenes are point scatterers only. Seafloor, objects, walls as surfaces, sediment and
  moving-target tracks come with the later milestones that need them.
- Tiers `free_field` and `image_source` exist. `eigenray` (Bellhop) is milestone 12 and
  raises `NotImplementedError` until then.
- Noise is white Gaussian at a given RMS. Ambient spectra, self-noise and array coherence
  are milestone 12.
- Arbitrary (sampled) transmit replicas cannot be synthesised yet; CW, LFM and Gaussian can.
- `fls_300k` needed an analogue gain ramp to register a 300 m echo in 16 bits; the ramp in
  its config is a placeholder, flagged unconfirmed.

## Known deviations from the paper

None in implemented code yet. Planned resolutions of the paper's gaps are in
`DECISIONS.md` (D-01 to D-25), each tied to a numbered gap in `PAPER_NOTES.md` Part 2.

## Issue tracker

`bd` holds one issue per milestone (prefix `sonar-`). Its database is local to the
checkout; this file is the durable record.
