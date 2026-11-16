# ASASIN paper notes

Source: Gerg, Brown, Wagner, Cook, O'Donnell, Benson, Montgomery, "GPU Acceleration for
Synthetic Aperture Sonar Image Reconstruction", arXiv:2101.05888v2 (26 May 2021), 9 pages.
Read from the arXiv PDF (text and figures). There is no public ASASIN code or data
(see `OPEN_SOURCE.md`), so everything below is implemented from the text alone.

How to read this file:

- Part 1 lists every equation and algorithm with its section and how we implement it.
- Part 2 lists every place the paper is vague, inconsistent or silent, and points to the
  entry in `DECISIONS.md` that resolves it.
- Part 3 is the scope checklist. Each item gets a file and a test when it is implemented;
  until then it says `pending`.

Symbols follow the paper unless noted. Our frame is the paper's: north-east-down (NED),
x along-track, y cross-track, z down.

## Part 1: equations and algorithms

### Sec. II: framework (Fig. 1)

| Item | Paper says | Our implementation |
| --- | --- | --- |
| Config-driven chain | One config file holds all processing parameters; guarantees reproducibility. | YAML config, loaded and validated in Python, resolved config copied into the processing log and every output file. |
| Data readers | "All data readers are derived from a base class and implement the necessary methods to read data and navigation." | `Source` base class producing ping records; subclasses for our container, Popoto/generic .wav, the simulator and the Holoscan sources. |
| Outputs | Image products (GeoTIFF; multi-layer GeoTIFF for 3D), debug products, processing log with "performance metrics, processing path choices, and data issues". | GeoTIFF writer, per-stage debug dumps, JSON-lines processing log with timings, choices and validator warnings. |

### Sec. II-A: forward model, Eq. 1

```
e(t, x_RX(t)) ≈ ∫ σ(x) / (‖x_TX − x‖ ‖x_RX(t) − x‖) · q(t − (‖x_TX − x‖ + ‖x_RX(t) − x‖)/c) dx
```

- `e`: array time series; `x_TX`: transmitter position, assumed stationary during transmit;
  `x_RX(t)`: receiver position, moving during reception; `σ`: scattering cross-section
  function (ref [6]); `q`: transmitted waveform; `c`: sound speed.
- The paper never writes the inverse. It says only that the model "is inverted to solve
  for σ(x) for each pixel" under "very simplistic assumptions on the form of σ".
- Implementation: Eq. 1 is the simulator's `free_field` forward model, evaluated per
  scatterer and per channel with the receiver moving during reception. The inverse is the
  standard delay-and-sum backprojection

  ```
  I(x) = Σ_records w(x, record) · s_record(τ(x, record)) · exp(+j 2π f_c τ)   (baseband input)
  τ(x, record) = (‖x_TX − x‖ + ‖x_RX(t_TX + τ) − x‖) / c
  ```

  where `s` is the conditioned, pulse-compressed series and `w` holds aperture weighting,
  element directivity and optional spreading compensation. `τ` is implicit because the
  receiver moves; we solve it by fixed-point iteration (stop-and-hop is the zero-iteration
  case, behind a config switch).

### Sec. II-B: coordinates, Eq. 2

```
p̃ = R(φ, θ, ψ) p,   R = Rz(ψ) Ry(θ) Rx(φ) =
[ cθ cψ    cψ sθ sφ − cφ sψ    cφ cψ sθ + sφ sψ ]
[ cθ sψ    cφ cψ + sθ sφ sψ    cφ sθ sψ − cψ sφ ]
[ −sθ      cθ sφ               cθ cφ            ]
```

- φ, θ, ψ are roll, pitch, yaw in radians. Positive roll lowers starboard, positive pitch
  raises the bow, positive yaw rotates to starboard.
- Checked: the printed matrix is exactly the yaw-pitch-roll (Z-Y-X) body-to-NED rotation,
  and the three sign conventions follow from it with body axes x forward, y starboard,
  z down. No inconsistency.
- Implementation: `core/geometry` `rotation_ned(roll, pitch, yaw)`. Element position in
  the scene is `p_vehicle + R · lever_arm`. Tests check each printed matrix entry and each
  of the three sign conventions.

Scene conventions:

- Origin on the seafloor directly below the first ping, so the first ping sits at
  `(0, 0, −altitude)`.
- Stripmap grid (Fig. 3, Fig. 4): the grid does not start at the origin. The first and last
  pings lie outside it so the far-range corners get the full azimuth beam. The paper gives
  no formula; Fig. 4 shows the grid edge where the leading edge of the start-ping beam
  reaches far range. We use `x_start = x_first + y_max · tan(θ_az / 2)` and
  `x_end = x_last − y_max · tan(θ_az / 2)`, with `θ_az` the FWHM azimuth beamwidth.

### Sec. II-C: run-time optimisations

1. **Slab allocator.** Every algorithm derives from a base class with a method returning
   the GPU memory it needs. One slab is allocated to cover all requests as algorithms are
   instantiated; the stated purpose is to avoid fragmentation next to other GPU processes.
   Implementation: `core/allocator` slab with a two-phase protocol (`reserve` during
   pipeline construction, `commit` once, then bump allocation with alignment). Each stage
   implements `memory_required()`.
2. **Texture position lookup.** Sensor positions are upsampled and stored in GPU texture
   memory; hardware linear interpolation returns the position at a query time.
   Implementation: a 1D float texture per position component with linear filtering in the
   CUDA backend, and an explicit software path used as the default and as the oracle.
   The limits of hardware filtering (8-bit interpolation weights, float32 values) are
   quantified in `DECISIONS.md` D-07; positions are stored relative to a local origin so
   the precision budget holds.
3. **Ray culling (Fig. 5).** Split the image into `b × b` pixel blocks. Test the four
   corners of each block for transmitter/receiver visibility. If no corner passes, discard
   the block; if any passes, process every pixel in the block with an explicit per-pixel
   test. Fig. 5 uses `b = 5`. Reported speedup: "a factor of two".
   Implementation: a per-record block mask computed first, then the kernel visits only
   live blocks. `b` is a config value. Our corner test adds a margin so that a beam
   narrower than a block is never missed (see Part 2, item 9). The benchmark reports our
   measured speedup next to the paper's 2×.

### Sec. II-D: motion estimation, Eq. 3–8

- **Sources per degree of freedom.** Sway and heave: estimated, from a "derivative work"
  of the displaced phase centre (DPC) method of ref [5] (Cook, 2007). Surge: DVL, or the
  along-track estimator of ref [9]. Roll, pitch, yaw: the INS.
- **Eq. 3, Huber loss.** `h(a) = a²/2 for |a| ≤ 1, |a| − 1/2 otherwise`.
- **Eq. 4, residual for one delay measurement.**

  ```
  g = c⁻¹ [ (‖p_TX1 − p_s‖ + ‖p_s − p_RX1(t)‖) − (‖p_TX2 − p_s‖ + ‖p_s − p_RX2(t)‖) ] + δ(t)
  ```

  `p_s` is the common seafloor footprint at reference time `t`; `δ(t)` is the measured
  time difference of arrival between ping one and ping two. Transmission is instantaneous;
  the platform moves during reception.
- **Eq. 5, DPC loss.** `L_DPC = Σ_{i,k} h( g(p_TXi, p_RXi(t_k), p_TXi+1, p_RXi+1(t_k), p_s, δ_{i,k}[t_k], c | v_i, v_i+1, Ω_i, Ω_i+1) )`.
  `Ω_i` holds lever arms and attitude for ping `i`. Positions follow first-order
  kinematics, `p = v t`. Unknowns are `v_y`, `v_z` per ping. All pings are solved jointly.
  Residuals may be weighted by the correlation coefficient of each delay. Known bathymetry
  enters as `p_s,z`; flat seafloor means `p_s,z = 0`.
- **Eq. 6, smoothness.** `L_smooth = Σ_i (v_y''[i])² + Σ_i (v_z''[i])²`.
- **Eq. 7, DVL altitude.** `L_DVL = Σ_i [ (p_z,1(0) + Σ_{n=1..i} v_z[n] Δ_n) − z_DVL[i] ]²`,
  with `Δ_n` the time between transmissions `n` and `n+1`.
- **Eq. 8, total.** `argmin over v_y,1..n, v_z,1..n of L = L_DPC + λ1 L_smooth + λ2 L_DVL`.
- **Solver.** Ceres, Levenberg–Marquardt, automatic differentiation.
- **Delay measurement (ref [11], Sæbø 2010).** Coarse delay from quadratic interpolation
  of the peak of the magnitude of the correlation; fine delay from the phase of the
  complex correlation at the interpolated point.
- **Two-pass unwrapping.** Minimise `L_DPC` with coarse delays. Unwrap the fine delays
  around that solution. Minimise again with the unwrapped fine delays. Then minimise Eq. 8.

Implementation:

- `motion/delay_estimation`: windowed complex cross-correlation between overlapping
  phase-centre channels of consecutive pings, per range window `k`; returns coarse delay,
  fine delay, correlation coefficient.
- `motion/dpc_solver`: Ceres problem with one `AutoDiffCostFunction` per delay (Eq. 4),
  `HuberLoss` or trivial loss (config), optional correlation weighting, Eq. 6 and Eq. 7 as
  extra residual blocks, parameters `v_y[i]`, `v_z[i]`. Positions come from integrating
  velocities from the first ping.
- `motion/surge`: DVL surge by default; an along-track estimator in the spirit of ref [9]
  (correlation against element shift with sub-element interpolation) as the alternative.
- `motion/pipeline`: the two-pass sequence above.

### Sec. II-E: pre- and post-processing, Eq. 9–12

- **Eq. 9, spectral whitening.** `G(f) = h( 1 / (γ P̄ + P̂(f)) )`. `P̂(f)`: power estimate at
  `f`; `P̄`: mean power over all frequencies; `γ`: system-dependent calibration factor;
  `h`: normalisation "ensuring the minimum attenuation is 0 dB", i.e. `max G = 1`.
- **Eq. 10, time-varying gain.** `G(t) = (α P̄~ + P~(t))⁻¹`. `P~(t)`: population median power
  at sample time `t` across a batch of pings; `P̄~`: mean over `t` of those medians;
  `α`: system-dependent calibration factor.
- **Eq. 11, dynamic range compression (Schlick, ref [12]).**
  `I_DRC = q I / ((q − 1) I + 1)`, `I` normalised to [0, 1].
- **Eq. 12, brightness parameter.** `q = (b − b Ĩ) / (Ĩ − b Ĩ)`, `Ĩ` the median of the
  normalised input, `b ∈ (0, 1)` the wanted output median.
  Checked: substituting Eq. 12 into Eq. 11 at `I = Ĩ` gives exactly `b`, and Eq. 11 is
  monotonic, so the output median equals `b`.

Implementation: `dsp/whiten`, `dsp/tvg`, `dsp/drc`, each a library function with a
Holoscan wrapper. Tests: whitening flattens a known transfer function; TVG flattens
intensity over range; DRC output median equals `b`.

### Sec. III-A, III-B: imaging results and 3D near-field imaging

- Stripmap example (Fig. 7): no parameters or metrics given.
- 3D sub-bottom imaging (Fig. 8) changes three things:
  1. bistatic ray culling (the Fig. 5 mask assumes a monostatic pair);
  2. refraction at the sediment–water interface in the travel time;
  3. 3D output built as 2D layers over focus depths, viewed as planar slices and maximum
     intensity projections.
- Navigation there is RTK-GPS; DPC is not used.
- The paper gives no refraction equation. Implementation: for a transmitter or receiver in
  water at height `h` above a flat interface and a focus point at depth `d` below it, find
  the interface crossing that satisfies Snell's law (equivalently, Fermat's least time);
  travel time is `ℓ_water/c_w + ℓ_sed/c_s`. Solved per path with a few Newton iterations
  on the crossing offset. Config holds `c_sediment` and the interface depth.

### Sec. III-C: performance model (Fig. 9)

- Method: one small representative SAS dataset, processed three times in a row, last run
  reported (to take cache effects out).
- Fit: runtime against single-precision GFLOPS (taken from Wikipedia's list of NVIDIA
  GPUs, ref [14]) follows `y ≈ 485307 · x^−0.755`.
- Fig. 9 axes: x is "Performance [GFLOPS]", y is "Time [s]"; a second axis gives
  performance relative to real time. The real-time line sits at roughly 1900 s, so the
  dataset is roughly that long. GPUs plotted: TX2, Xavier, GTX 970, Tesla K80, Titan X
  Pascal, RTX 2080 Ti, Tesla V100. Xavier is the slowest that beats real time.
- The paper has no timing table. The only numbers are the fit and what can be read off
  the plot, and plot read-offs are not quoted anywhere in this repo as paper results.
- Eight V100s in one machine: "performance scales linearly until the limit of the
  hard-disk throughput is reached". No numbers.

Implementation: benchmark harness (three runs, report the last), a fit of `y = a·x^b` to
our own measurements, the paper's curve plotted for comparison only, real-time factor per
mode, and multi-GPU by image tiles.

## Part 2: where the paper is vague, inconsistent or silent

Each item names the decision that resolves it. Decisions live in `DECISIONS.md`.

| # | Where | Problem | Resolution |
| --- | --- | --- | --- |
| 1 | Eq. 1, Sec. II-A | The backprojection (inverse) equation is never written. | D-01 |
| 2 | Sec. II-A | Reflectivity weighting in the inverse is unspecified ("very simplistic assumptions"). Eq. 1 has a `1/(R_tx R_rx)` spreading term; whether it is compensated is not said. | D-02 |
| 3 | whole paper | Matched filtering / pulse compression is never mentioned, though `q` appears in Eq. 1. | D-03 |
| 4 | whole paper | Baseband or passband data is never stated. | D-04 |
| 5 | whole paper | The range interpolation kernel is never stated. | D-05 |
| 6 | Sec. III-C | Numeric precision is never stated. "Single-precision GFLOPS" suggests float32 arithmetic. | D-06 |
| 7 | Sec. II-C | Texture lookup: upsampling factor and method not given; hardware filtering precision not discussed. | D-07 |
| 8 | Sec. II-C | Allocator: alignment, lifetime and failure behaviour not given. | D-08 |
| 9 | Sec. II-C, Fig. 5 | Ray culling: (a) block size `b` not given beyond the figure's 5; (b) the caption swaps the colours relative to the text and the figure (text and figure: green = in the field of view; caption: red = in the field of view); (c) the corner test is justified by beam convexity, but a beam narrower than a block can pass between its corners, so a strict corner test can drop live pixels near the sonar; (d) "coarse-to-fine" is described with two levels only; (e) the field-of-view test itself (beamwidth definition, elevation) is not given. | D-09 |
| 10 | Sec. II-B vs Eq. 2 | φ and θ denote the elevation and azimuth beamwidths in Fig. 3 and Fig. 4, and roll and pitch in Eq. 2. | D-10 |
| 11 | Fig. 4 | The grid start offset is "a function of the sensor beamwidth" with no formula. | D-11 |
| 12 | Eq. 5 caption vs text | The caption defines `h(a) = a²`; the text recommends replacing it with the Huber loss of Eq. 3. | D-12 |
| 13 | Eq. 3 | Huber threshold is fixed at 1 with no scale. The residual `g` is in seconds, so a threshold of 1 would never engage; the residual needs scaling. | D-12 |
| 14 | Eq. 4 | Sign convention of `δ` is not stated. `g = 0` requires `δ = τ2 − τ1`. | D-13 |
| 15 | Sec. II-D | "recall v = [v_z, v_y, v_z]" is a typo for `[v_x, v_y, v_z]`. | noted |
| 16 | Eq. 6 | Discretisation of `v''` and handling of non-uniform ping intervals not given. | D-14 |
| 17 | Eq. 7 | (a) `z_DVL` is called an altitude (positive up from the seafloor) while `p_z` is NED (negative above the seafloor): the signs disagree as written. (b) The sum to `i` gives the position of ping `i+1`, not `i`. (c) Flat seafloor is implied. | D-15 |
| 18 | Eq. 8 | λ1 and λ2 not given. | D-16 |
| 19 | Sec. II-D | The "derivative work" of DPC is not specified: which channels overlap, window lengths, number of range windows, how `p_s` is placed. Ref [5] describes classic DPC. | D-17 |
| 20 | Sec. II-D | Surge by ref [9] is cited, not described. | D-18 |
| 21 | Sec. II-D | Unwrapping "around this solution" is not specified further. | D-19 |
| 22 | Eq. 9 | Whether `G` multiplies power or amplitude; how `P̂(f)` is estimated; behaviour out of band; γ value; whether whitening runs before or after pulse compression. | D-20 |
| 23 | Eq. 10 | Same power/amplitude question; what a "batch" is; α value. | D-21 |
| 24 | Eq. 11–12 | How `I` is normalised to [0, 1] (peak? percentile?), and whether `I` is magnitude or power. | D-22 |
| 25 | Sec. III-B | No refraction equations; interface assumed where; sediment sound speed. | D-23 |
| 26 | Sec. III-C | Dataset size, image size, units of the fit, and how multi-GPU work is split are not given. | D-24 |
| 27 | Eq. 1 | Transmitter stationary during transmit; no treatment of long pulses or Doppler. | D-25 |

## Part 3: scope checklist

Status values: `pending`, `done`. A `done` row names the file and the test.

### Framework (Sec. II)

| Item | File | Test | Status |
| --- | --- | --- | --- |
| YAML config-driven chain | | | pending |
| Readers from a common base class | `python/gpusonar/sources/base.py` | `tests/python/test_sources.py` | done (container, WAV); simulator and Holoscan sources pending |
| GeoTIFF output | | | pending |
| Debug products per stage | | | pending |
| Processing log with performance metrics | | | pending |

### Geometry (Sec. II-B, Eq. 2)

| Item | File | Test | Status |
| --- | --- | --- | --- |
| NED frame, Eq. 2 rotation, lever arms | `libs/core/include/sonar/core/vec.hpp`, `libs/core/src/record.cpp` | `tests/cpp/test_geometry.cpp` (`RotationNed.MatchesEquation2EntryByEntry`, `WorldPosition.*`) | done |
| Origin on the seafloor below the first ping | | | pending |
| Stripmap grid sized from the beamwidth | | | pending |
| Sign conventions for roll, pitch, yaw | `libs/core/include/sonar/core/vec.hpp` | `tests/cpp/test_geometry.cpp` (`PositiveRollLowersStarboard`, `PositivePitchRaisesBow`, `PositiveYawRotatesToStarboard`) | done |

### Backprojection and run-time (Sec. II-A, II-C, Eq. 1)

| Item | File | Test | Status |
| --- | --- | --- | --- |
| Eq. 1 forward model (simulator) | `libs/sim/include/sonar/sim/synth_kernel.hpp` | `tests/python/test_sim.py::test_gaussian_echo_delay_and_amplitude`, `::test_time_domain_synthesis_matches_frequency_domain_oracle`, `::test_receiver_moves_during_reception` | done (CPU; GPU run pending) |
| Time-domain backprojection, continuous receive motion | | | pending |
| Stop-and-hop option | | | pending |
| Slab allocator, stages report need up front | | | pending |
| Texture position lookup | | | pending |
| Ray culling on b×b blocks, speedup reported | | | pending |

### Motion estimation (Sec. II-D, Eq. 3–8)

| Item | File | Test | Status |
| --- | --- | --- | --- |
| Coarse and fine delay estimation | | | pending |
| Eq. 4 residual, first-order kinematics | | | pending |
| Eq. 5 loss, Huber (Eq. 3) or squared | | | pending |
| Correlation-coefficient weighting | | | pending |
| Eq. 6 smoothness | | | pending |
| Eq. 7 DVL term | | | pending |
| Eq. 8 solve with Ceres LM and autodiff | | | pending |
| Two-pass phase unwrapping | | | pending |
| Surge from DVL | | | pending |
| Surge from along-track estimation | | | pending |
| Attitude from INS | | | pending |
| Known bathymetry option | | | pending |

### Pre- and post-processing (Sec. II-E, Eq. 9–12)

| Item | File | Test | Status |
| --- | --- | --- | --- |
| Spectral whitening, Eq. 9 | | | pending |
| Median time-varying gain, Eq. 10 | | | pending |
| Dynamic range compression, Eq. 11 | | | pending |
| Closed-form brightness, Eq. 12 | | | pending |

### 3D near-field imaging (Sec. III-B)

| Item | File | Test | Status |
| --- | --- | --- | --- |
| Bistatic ray culling | | | pending |
| Refraction at the sediment–water interface | | | pending |
| Volume as 2D layers over focus depth | | | pending |
| Slices and maximum intensity projections | | | pending |

### Performance model (Sec. III-C, Fig. 9)

| Item | File | Test | Status |
| --- | --- | --- | --- |
| Three-run benchmark harness | | | pending |
| Power-law fit against GFLOPS | | | pending |
| Real-time factor | | | pending |
| Multi-GPU by image tiles | | | pending |
