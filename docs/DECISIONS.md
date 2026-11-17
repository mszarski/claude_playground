# Decisions

Every choice made where the ASASIN paper, the brief or a reference is silent or
inconsistent. Each one sits behind a config option unless marked "fixed".
`D-nn` entries resolve the numbered gaps in `PAPER_NOTES.md` Part 2. `A-nn` entries are
architecture choices for the suite as a whole.

Status: `planned` until the code and a test exist, then `done` with the test name.

## Architecture

### A-01 Container format: HDF5 (fixed)

One HDF5 file holds a sequence of ping records plus optional ground truth.

- HDF5 over Zarr because the core is C++ and must write records from a live pipeline on
  Jetson. HDF5 has a maintained C library packaged for aarch64 (`libhdf5-dev`) and h5py
  for Python; Zarr's C++ story is third-party and thinner. See `OPEN_SOURCE.md`.
- Costs accepted: single-writer files, and no cloud-native chunk store. Neither matters
  for vehicle logging and bench work.
- The schema is versioned (`/schema_version` attribute, semantic versioning) and has a
  validator in the `io` library.

### A-02 Language split

- C++20 for everything on the real-time path: record, geometry, allocator, I/O,
  conditioning, pulse compression, backprojection, far-field beamformer, simulator
  synthesis, delay estimation, the Ceres motion solve, CFAR.
- CUDA for the kernels, written once and compiled for both host and device (A-03).
- Python (pybind11 bindings) for config loading, scenario building, pipeline
  orchestration, tools (sonar equation, ambiguity reports), tracking glue, plots, tests
  that compare against Python oracles, and the Holoscan application layer.
- YAML is parsed in Python only. C++ stages take plain parameter structs, so there is one
  config schema and no YAML dependency in the core.

### A-03 One kernel source, two backends; this development container has no GPU

The brief's development host is a DGX Spark. The container this code was first written
in is x86_64 with no GPU, so:

- Each kernel body is a functor in a header, marked `SONAR_HD` (`__host__ __device__`
  under nvcc, nothing otherwise). `parallel_for(backend, n, functor)` runs it on CPU
  threads or as a CUDA kernel. The arithmetic is the same source line for line.
- The **host backend** makes every stage runnable and testable without a GPU.
- The **CUDA backend** is compiled here with nvcc 12.0 as a build check only. It has not
  been executed in this container. Every GPU-versus-CPU test exists and is skipped with
  an explicit message when no device is present. `STATUS.md` tracks what still needs a
  first run on the Spark.
- The **reference implementation** required by the brief is separate from both: naive,
  double (or long double) precision, no shared code with the kernel, used only in tests.

### A-04 Repository layout

The suite lives at the repository root next to the existing `trajectory_classifier`
package, because the brief names root-level paths (`docs/…`). Python package name:
`gpusonar`. C++ namespace: `sonar`.

### A-05 Simulator synthesis in the time domain with analytic waveforms

Echoes are synthesised as `Σ a · q(t − τ(t))` with `q` evaluated analytically (CW burst,
LFM, Gaussian-windowed pulse), and `τ(t)` evaluated at each receive sample from the
moving geometry. This is exact band-limited variable-delay synthesis for these waveforms:
no interpolation filter, and Doppler and continuous motion come for free. Frequency
dependence along a path (absorption) is applied at the instantaneous frequency of the
waveform; channel transfer functions are applied per channel afterwards as filters.
A frequency-domain synthesiser (`Σ a · exp(−j2πfτ)` per bin) is kept as an independent
test oracle for static geometry.

## Paper gaps

### D-01 Backprojection equation
`I(x) = Σ_r w_r(x) · s_r(τ_r(x)) · exp(+j2π f_c τ_r(x))` for complex baseband input; the
exponential is dropped for analytic passband input. `τ` includes receiver motion during
reception (fixed-point solve); `stop_and_hop: true` freezes the receiver at transmit time.

### D-02 Reflectivity weighting
Default: `w = aperture_weight × directivity`, no spreading compensation, since time-varying
gain already flattens range. Option `spreading_compensation: none | one_way | two_way`
multiplies by `(R_tx R_rx)^p`. Option `normalise: none | weight_sum` divides each pixel by
the sum of weights so partially covered pixels are not darker.

### D-03 Matched filtering
Pulse compression runs before backprojection by default (`pulse_compression.enabled: true`),
because Eq. 1 contains the transmitted waveform `q` and delay-and-sum of uncompressed data
does not resolve range. It can be turned off for impulsive transmits.

### D-04 Baseband or passband
Default: complex baseband, decimated to about 1.5× the signal bandwidth, with carrier
phase restored in the kernel. Analytic passband and real passband inputs are accepted.

### D-05 Interpolation kernel
`interpolation: nearest | linear | sinc`, default `sinc` with 8 taps and a Kaiser window
(β = 6). `linear` is the fast option and is adequate when the baseband series is
oversampled 2× or more; the tests quantify the image error of each.

### D-06 Numeric precision: how phase error stays under 1° at 10 km and at 5 MHz
Samples are float32. Delay and phase use a split:

- Records are grouped by ping. Each group has a transmit position and a receive
  reference position (array centre), held in float64.
- Per output point and group, the reference two-way delay is solved in float64, including
  receiver motion. It is split into an integer sample index, a fractional sample and a
  fractional carrier cycle before anything is narrowed to float32.
- Per element, only the *differential* path `|x − a − d| − |x − a|` is computed in float32,
  in the cancellation-free form `(d·d − 2 w·d) / (|w − d| + |w|)`, where `d` is the element
  offset from the reference. Its magnitude is bounded by the aperture, so float32 relative
  error (about 6e-8) gives 2e-7 m for a 3 m array and 3e-9 m for a 38 mm probe.
- Receiver motion over the differential arrival time is applied in closed form:
  `ΔR' = ΔR / (1 − ṙ/c)`, with `ṙ` the range rate of the reference.

Budget: 1° is 0.82 µm of path at 5 MHz and 0.41 mm at 10 kHz. A plain float32 range at
10 km has a resolution of about 1 mm and fails; the split leaves less than 0.01° from
arithmetic. Test: kernel delays and phases against a long-double reference at 10 km /
10 kHz and 3 m / 5 MHz.

### D-07 Texture position lookup
Hardware linear filtering uses 8 fractional bits for the interpolation weight and float32
values. So the texture holds only the *residual* of each ping's trajectory about its
first-order model `p0 + v·t` (the paper's kinematic model).
The residual is millimetres to centimetres, which float32 and 8-bit weights resolve to
well under a micron. The table is upsampled by `texture.upsample` (default 8) with cubic
interpolation. `position_lookup: analytic | software_table | texture`; default `analytic`
(first-order only), with the other two for trajectories that are not linear within a ping.
The table is built from the ping's INS samples (uniformly spaced). On the device the
texture is 2-D, one row per ping. On the host backend `texture` runs the same 8-bit-weight
arithmetic in software, which is how the precision claim is tested without a GPU.

### D-08 Allocator
Two phases. Stages call `reserve(bytes, alignment)` while the pipeline is built; `commit()`
makes one allocation; `allocate()` then hands out aligned bump-pointer blocks and fails
loudly if a stage asks for more than it reserved. `reset()` rewinds for the next run.
Default alignment 256 bytes.

### D-09 Ray culling
- `culling.block: 16` pixels by default; `0` disables culling.
- Field-of-view test: the angle between the element or array boresight and the direction
  to the pixel, against `culling.beam_factor × FWHM` in azimuth and elevation
  (default factor 1.5, so the processed sector is wider than the FWHM beam).
- A block is kept if any corner passes, as in the paper's text and figure (the Fig. 5
  caption has the colours reversed). To stay exact when a block is wider than the beam,
  the corner test is widened by the half-angle the block subtends from the sonar. Pixels
  in kept blocks are tested individually without the margin.
- Two levels, as described. Bistatic mode tests transmitter and receiver separately and
  requires both.
- Acceptance: image identical to the unculled image to within float rounding.

### D-10 Symbols
Beamwidths are `beamwidth_az`, `beamwidth_el`. Attitude is `roll`, `pitch`, `yaw`.
φ and θ are not used as names anywhere.

### D-11 Stripmap grid start
`x_start = x_first_ping + y_max · tan(beamwidth_az / 2)`, and symmetrically at the end.
`grid.full_support: true` by default; `false` images from the first ping to the last.

### D-12 Loss function
`motion.loss: huber | squared`, default `huber`. Residuals are divided by
`motion.residual_scale_s` (default one sixteenth of a carrier period) before the loss, so
Eq. 3's threshold of 1 means "one scale unit". The paper's caption (`a²`) and text (Huber)
disagree; both are available.

### D-13 Sign of δ
`δ = τ2 − τ1`: positive when the second ping's echo from the common footprint arrives later
relative to its own transmit. This is the sign that makes Eq. 4 zero at the truth.

### D-14 Smoothness term
`v''[i] = v[i−1] − 2 v[i] + v[i+1]`, in velocity units, for interior pings. Non-uniform ping
intervals are handled by the three-point second-derivative stencil times the mean interval
squared, which reduces to the above for uniform pings.

### D-15 DVL term
Altitude of ping `i` is `−p_z,i`. Positions integrate as `p_z,i = p_z,1 + Σ_{n=1..i−1} v_z[n] Δ_n`.
The residual is `(−p_z,i) − altitude_DVL[i]`. `p_z,1` is `−altitude_DVL[1]`, which also fixes the
origin on the seafloor below the first ping.

### D-16 λ1, λ2
Config values `lambda_smooth`, `lambda_dvl` in `MotionOptions`, both 1 by default with the
residual scales of D-31. The planned sweep on simulated data has not been run.

### D-17 DPC details
Classic displaced-phase-centre overlap: with `N` receivers at pitch `d` and an along-track
advance `Δx` per ping, the trailing phase centres of ping `i+1` overlap the leading phase
centres of ping `i`. The overlap count comes from the surge estimate. Overlapping channel
pairs are correlated in `motion.range_windows` windows; delays from pairs within a window
are averaged with coherence weights. The footprint `p_s` for window `k` is placed on the
seafloor at the ground range matching the window's centre time. Details are finalised in
milestone 8.

### D-18 Surge
`motion.surge: dvl | ate`. The along-track estimator finds the fractional element shift
that maximises inter-ping coherence, interpolating the coherence-against-shift curve.
Ref [9] is not open access; ours follows the idea, not the paper's kernels.

### D-19 Phase unwrapping
After the coarse solve predicts `δ_pred` for each measurement, the fine delay is
`δ_fine + n / f_c` with the integer `n` that brings it closest to `δ_pred`.

### D-20 Whitening (Eq. 9)
`G(f)` is treated as a power gain; the amplitude spectrum is multiplied by `sqrt(G)`.
Reason: multiplying amplitude by `1/(γP̄ + P)` would turn a spectrum `P` into `P/(γP̄ + P)²`,
which is not flat, whereas `sqrt(G)` gives `P/(γP̄ + P) → 1` wherever `P ≫ γP̄`.
`whiten.apply_as: power_gain | amplitude_gain` keeps the literal reading available.
`P̂(f)` is a Welch-style average over pings and channels, smoothed over `whiten.smooth_hz`.
`γ` defaults to 0.1. `h` scales so `max G = 1`. Whitening runs after pulse compression.

### D-21 Time-varying gain (Eq. 10)
Same reading: `G(t)` is a power gain, amplitude is multiplied by `sqrt(G)`. A batch is all
pings and channels handed to the stage in one call. `α` defaults to 0.01.

### D-22 Dynamic range compression (Eq. 11–12)
`I` is magnitude, normalised by `drc.normalise: max | percentile` (default percentile
99.99, clipped to 1). Target median `b` defaults to 0.2.

### D-23 Refraction
Flat interface at a configured depth, two fluid half-spaces, Snell's law solved per path
by Newton iteration on the crossing point. `recon.refraction: off | flat_interface`.

### D-24 Benchmark
Our dataset, image size and timing method are defined in the benchmark config and printed
with every result. The paper's fit is shown for comparison only; its dataset is unknown,
so only the exponent is comparable.

### D-25 Long pulses and Doppler
The transmitter is not assumed stationary during long pulses in the simulator. On receive,
`pulse_compression.doppler_bank` holds a small set of time-scaled replicas; the best
hypothesis per channel block (or a fixed hypothesis from navigation) is used.

### D-26 What "resolution within 10% of theory" is measured against
The brief asks for range resolution within 10% of `c/2B` and beam width within 10% of
theory. Both need a level and a pulse shape to be testable:

- **Range.** The −3 dB width of the point response along range, after pulse compression.
  Theory is computed for the actual pulse: a Gaussian spectrum of −6 dB width `B`
  compresses to a −3 dB width of `0.88 · c/2B`; a flat (LFM) spectrum gives `0.886 · c/2B`.
  The test requires the measurement within 10% of that, and also checks it against the
  nominal `c/2B`. For reference, `c/2B` exactly is the −3.9 dB width of a flat spectrum.
- **Bearing.** The −3 dB width of the point response across bearing, uniform aperture
  weighting, against `0.886 · λ/L` at the centre frequency, with `L = N · pitch`.

Measured values are in `RESULTS.md`.

### D-27 Pulse reference
Backprojection samples each record at the two-way delay, so the record's time axis must
mark delay. Pulse compression does that: an echo of delay τ peaks at t = τ. A record
remembers it has been compressed in `metadata.extra_json` (`gpusonar.compressed`), so
`recon.prepare` is idempotent. This uses the free-form field and leaves the schema alone.

### D-28 Array attitude during reception
The kernel evaluates the receive body's attitude exactly at each point's arrival time
(`R(rpy + rates · τ)`), not to first order in τ. A first-order offset model put 74° of
phase error on a 3 m array yawing at 0.02 rad/s over a 10 km round trip. Each element's
differential delay uses that element's own range rate. Neglected: the change of attitude
during the differential delay itself (at most the aperture over c, about 2 ms).

### D-29 What "along-track resolution within 10% of D/2" is measured against
`D/2` is the stripmap resolution when each point is integrated over the angular extent
`λ/D` of an element of length `D`. `sas_image` therefore limits each ping to
`fov_factor = 1.13` (= 1/0.886) times the FWHM beam, which is that extent, and the test
compares the −3 dB along-track width with `D/2`. Processing more of the main lobe is
allowed and gives a finer result (15.6 mm against 20 mm null to null in the test scenario).
For the rail, where the element beam is far wider than the rail subtends, the reference
is the aperture-limited `0.886·λR/(2L)`.

### D-30 Unknowns of the motion solve
The paper's unknowns are per-ping velocities, with positions obtained by integration.
The Ceres problem here uses the sway and heave positions at each transmit as its
parameter blocks, with `v_i = (p_{i+1} − p_i)/Δt_i`. This is a linear, one-to-one change
of variables (the first position is fixed), so the loss of Eq. 8 and its minimiser are
unchanged. The gain is sparsity: a delay residual touches three neighbouring pings
instead of every earlier one. One extra node after the last ping gives it a velocity.

### D-31 Scales and weights in the motion solve
- Delay residuals are divided by `residual_scale_periods / f_c` (default 1/16 period)
  before the loss, so Eq. 3's fixed threshold of 1 is one such unit.
- Eq. 6 residuals are divided by `smooth_scale` (default 2 mm/s) and Eq. 7 residuals by
  `dvl_scale` (default 5 mm); `lambda_smooth` and `lambda_dvl` then multiply those terms
  and both default to 1. A sweep to tune them (D-16) has not been run; the defaults
  passed the acceptance scenario unchanged.
- The first ping is the anchor: its sway is zero and its height is minus its DVL altitude.
- Ceres is built without SuiteSparse and the problem uses the dense QR solver, which is
  adequate for hundreds of pings. A sparse solver would be needed for very long runs.

### D-32 Shared reference for board synchronisation
The brief asks for a shared reference signal wired into one input on every board, and
for 32 hydrophones on four 8-input boards. Both hold if the reference is *summed* into a
hydrophone input instead of replacing it. The reference is two short chirp bursts per
ping, above the acoustic band (38–48 kHz against 20–35 kHz), one just after the receive
window opens and one just before it closes. One burst gives a board's clock offset; the
spacing of the two gives its drift. Burst timing is estimated from the correlation
envelope and refined with the correlation phase. Boards are then resampled onto the
first board's timeline with a windowed-sinc interpolator, and the bursts are measured
again as the residual check. All parameters are in `configs/sonars/popoto_4x8.yaml`.
A CW pilot tone was considered and rejected: it gives the offset only modulo its period.

### D-33 MVDR training data
The adaptive beamformer estimates each range block's covariance from the other range
blocks (excluding a guard). With the echo itself in the training data, MVDR at high SNR
treats the echo's small mismatch from its steering vector as interference and cancels it;
that happened in the first version here. The cost: it does not separate two point targets
at the same range.

### D-34 Stages still in Python
`flank.py`, `detect.py`, `calibrate.py`, `sync.py`, `sonar_equation.py`, `processing.py`
and the delay estimator in `motion.py` are NumPy. The brief asks for a C++/CUDA core with
each stage as a library function and a Holoscan operator. These were written in Python
first to get tested behaviour; porting the per-ping ones (far-field beamformer, CFAR,
sync resampler) to the kernel framework is outstanding work, listed in `STATUS.md`.
Update: cell-averaging CFAR and the split-window normaliser are now a C++ kernel for both
backends (`libs/detect`), identical to the NumPy version on the test maps. The flank
range × bearing map can already be formed by the backprojection kernel (the benchmark does
so); the FFT beamformer, clustering, tracking, calibration and sync remain NumPy.

### D-35 Float32 pre-test ahead of the float64 solve
Measured on GPUs, the float64 reference-delay solve (D-06) dominated run time on every
card with slow float64. The kernel now tests each (point, ping) pair against the ping's
beam in float32 first, using the transmit-time geometry widened by the array extent, the
platform's motion and attitude change during the flight time, and 2 cm for trajectory
residuals. It only rejects pairs the exact per-element test would also reject, so images
are unchanged. Pings with zero attitude rates reuse one rotation matrix. The precision
argument of D-06 is untouched: accepted pairs go through the same split-precision path.

### D-36 Bellhop time convention
Bellhop's arrival phases follow exp(−iωt). This suite uses exp(+iωt) (an echo of delay τ
carries exp(−j2πfτ)), so arrival amplitudes are conjugated on reading.

### D-37 Two reverberation methods
Imaging modes use point scatterers on the boundary (`scene.seafloor_patch`): the same
scatterers are seen from every ping, which synthetic aperture processing needs. Long-range
modes use the scattering-function method (`sim/reverb.py`): reverberation power against
time from eigenray pairs over range rings, realised as Gaussian sequences per azimuth
sector and delayed to each hydrophone. It costs the same at 10 km as at 100 m but draws
new noise each ping, so it must not be used for SAS. Only paths travelling towards the
boundary count as incident. The projector is taken as omnidirectional in azimuth, the
platform as stationary, and arrival elevation is ignored when steering to the hydrophones;
all three are limits to lift, not physics choices. The two methods agree within 2 dB on a
flat uniform seabed (`test_reverberation.py`).

### D-38 Sea-surface models
Backscatter: Chapman and Harris (1962), S = 3.3β log10(θ/30) − 42.4 log10(β) + 2.6,
β = 158 (v f^(1/3))^(−0.58), θ in degrees, v in knots, f in Hz. The coefficients were
checked against two published forms (158 with knots, 107 with m/s; they are the same
formula). The fit covers 400–6400 Hz; at 8 kHz it is a mild extrapolation and at 27.5 kHz
a large one (bubbles dominate there; the APL-UW model of TR 9407 is the right one and is
not implemented because the report could not be obtained). For a path pair with different
grazing angles the strength is taken at their mean.

Forward loss: the coherent reflection coefficient exp(−Γ²/2), Γ = 2 k h sin θ, with h the
Pierson–Moskowitz rms height (significant height 0.21 U²/g, a quarter of it rms), applied
per surface bounce to Bellhop's flat-surface arrivals (`BellhopProvider(wind_speed=...)`).
It removes all scattered energy from the forward path, which overstates the loss when Γ is
large, so it is off by default and the shipped scenarios use a flat surface. Bubble
absorption is not modelled.

### D-39 Arrival cap of the eigenray provider
The provider keeps the strongest `max_arrivals` rays per point. The first long-range
scenarios used 8, which the KRAKEN cross-check showed drops 4.9 dB of one-way energy at
10 km in 100 m of water. The scenarios now set the cap per range (24 to 80) so the
one-way energy is within 0.1 dB of the normal-mode sum. Cost: the number of two-way paths
is the square of the cap, so the 10 km case takes minutes to synthesise on a CPU.

### D-40 Sessions: collection data stays on the device
`recon::BpSession` (Python `core.BackprojectSession`) builds the kernel inputs once and,
on the CUDA backend, uploads samples, ping and element tables and the interpolation table
to each device on first use; they stay until the session is destroyed. A call then moves
only its points and its image. The one-shot `backproject` is a session used once. Volume
layers and multi-GPU tiles use a session. This departs from the paper's single allocation
per run: there are now two slabs per device (the resident collection, and one per call
for points, image and culling mask), each still reserved and committed in one piece.

### D-41 Extended objects are clouds of surface points
Spheres, boxes, plates, rocks and pool walls are random points on their surfaces, each
with a normal and a complex Gaussian amplitude scaled for a Lambert strength, synthesised
by the same point-scatterer kernel as everything else. A face turned away returns nothing
(cosine from the normal clipped at zero). Shadows come from testing the straight line
from the projector and from the array centre to every scatterer against sphere and box
primitives, once per ping, on the host. This gives highlights, shadows and speckle, which
is what imaging and detection tests need; it does not give specular glints, diffraction
or resonances, and it is not a substitute for a boundary-element or k-Wave model of a
target.

### D-42 Conditioning on the device with cuFFT
Baseband conversion and pulse compression run on the CUDA backend with cuFFT for the
transforms and small kernels between them (`libs/dsp/src/dsp_cuda.cu`); the host path
keeps pocketfft. The arithmetic is the same and results agree to 2 × 10⁻⁷ of the peak.
Plans are cached per (device, length, batch) for the life of the process, and a lock is
held while a plan executes, so two threads conditioning on one device take turns. cuFFT
ships with the CUDA toolkit (NVIDIA licence, linked only into the CUDA build). Whitening,
median TVG, calibration and DRC remain host code: they are cheap next to the transforms.

### D-43 Reverberation for a moving sonar with beams
The scattering-function generator (D-37) now weights each azimuth and each path by the
transmit and element beam patterns (the same sinc model as the simulator and the kernel),
and shifts each azimuth's reverberation by the Doppler of the sonar's own velocity towards
it, pulse included. The boundary is therefore fixed in the world while the sonar moves.
Still approximate: no range walk within a ping, arrival elevation is ignored when steering
to the elements, and each ping draws new noise (so it suits detection and tracking, not
synthetic aperture or ping-to-ping coherence).

### D-44 Design traceability
The config files are the only place a design value lives. `tools/design_report.py` derives
the rest and writes `docs/DESIGN.md`, including a table of the owner's requirements, the
config key that meets each, and a check. A test regenerates the report and compares, so the
document cannot drift from the configs.

### D-45 Patch to the ultrasound reference
The owner chose to keep the reference's fast native path for long windows. The change is a
patch file applied to a copy of the pinned checkout (`third_party/patches/`), three lines
of host C++ raising one limit; the reference repository itself is not modified or vendored.

### D-46 SAS surveys advance half the array per ping
With one projector and N receivers the phase centres of a ping span half the array. Survey
scenarios set speed × ping interval to exactly that (0.544 m for `sas_minsas120`), so the
phase centres of successive pings are evenly spaced. At the earlier 0.4 m they overlapped
unevenly and a point target had along-track ghosts at −15 dB; at 0.544 m the worst is
−36 dB (`test_minsas_resolution.py`). A real vehicle cannot hold this exactly; redundant
phase centres are also what motion estimation uses, so a real survey trades one for the other.

### D-47 Weighting in the survey figures
The MINSAS survey figures use a Hamming-weighted replica and an aperture 1.6 beamwidths long
weighted by the beam patterns. That costs range resolution (2.6 cm against 1.9 cm) for range
sidelobes of −41 dB, and keeps 2.0 cm along track with −36 dB sidelobes. The unweighted
numbers are what the design table quotes; both are measured in the same test. The library
defaults are unchanged.

### D-48 One model per boundary
A scenario models the sea surface either statistically (a `reverberation` boundary) or as
explicit wave facets (a `sea_surface` scene item), never both. With both, the first rough-sea
example counted the surface twice. The facets show wave structure; the statistical boundary
gives Doppler-shifted reverberation without structure.

### D-49 The 12 km Popoto design
`popoto_long_range` is sized by `tools/popoto_range_study.py` on spherical spreading, the
cautious case: 4–6 kHz, 32 hydrophones at half a wavelength, 1 s sweep, and 197 dB, which is
2 dB over the 195 dB the study gives for 13 dB at 12 km. Shallow-water ducting helps and a
summer thermocline over mud hurts by more than that margin (`SCENARIOS.md`), so the number
is a starting point for a trial, not a guarantee. No projector is chosen.

### D-50 MINSAS-class receive array: 48 elements at 2.27 cm
Kraken does not publish the element count. With 32 elements over 1.09 m the receive phase
centres are 1.7 cm apart, which meets the usual half-projector-length sampling rule for the
4.2 cm projector but leaves an along-track grating lobe at 7.6 degrees that the beams only
take down to −42 dB; the survey images showed it as ghost cables. 48 elements put it at 11
degrees and −47 to −54 dB. More elements were not pursued: the receive element narrows with
the pitch, so the lobe stays near its first null, and the projector's sidelobes fall slowly.

### D-51 Synthesis kernel: amplitude interpolated over the pulse, phasor by recurrence
Per sample the kernel evaluated two beam patterns, a power and a sine and cosine. The
frequency-dependent amplitude is now evaluated at 17 emission times across the pulse and
interpolated (it is smooth in frequency and the sweep is linear), and the phasor is
advanced by a second-order recurrence re-seeded exactly every 16 samples. Host and CUDA
share the kernel, so both change. 5.7 times faster on the 120 kHz transit; the simulator's
tests against analytic delays, amplitudes, Doppler and the frequency-domain oracle are
unchanged and pass.

### D-52 A sea surface that travels, with new speckle each ping
Wave facets stay where they are in the water and the wave shape moves through them at each
train's deep-water speed (omega squared = g k), from a Pierson-Moskowitz spectrum with
random phases and directional spread. The facets' complex amplitudes are drawn afresh for
every ping, because the small scatterers on a real surface lose their phase in
milliseconds at these frequencies. Without the first, crests stood still; without the
second, the image texture was a pattern nailed to the water and wave faces made tracks
that lasted for ever. A tilt-independent term stands for small-scale roughness and bubbles.

### D-53 Classification from the track alone
`classify.py` uses only what a track has: fitted velocity and its uncertainty, how often
it was seen, where the other tracks are. That is enough to separate movers from static
objects, to mark detections sliding along a pipeline or wake, and to attach multipath
echoes to their source. It is not enough to tell a buoy from a rock; that needs image
features and is not attempted.

### D-54 Taps are container files
A tap writes the ping records after a stage with the ordinary record writer, so a tap is
also a source and no second format exists. Products that are not records go beside them as
.npz and .json. The simulator's internal stages are saved as arrays, since a record is not
complete until the last of them.

### D-55 The live chain beamforms with the backprojection kernel
For the long-range arrays the range x bearing map is made by the same kernel as every
other image, on a grid of ranges and bearings. It is exact in the near field, 11 times
faster on a CPU than the NumPy far-field beamformer at 32 channels, and already has a CUDA
build. The NumPy far-field and MVDR beamformers stay as alternatives and as cross-checks.

### D-56 Calibrating the simulator by simulating its parts
The simulator's output power is linear in each boundary's strength and in the noise
level. `simcal.py` simulates each alone through the same imaging as the log and fits the
log's range profile as a non-negative sum. Nothing has to be known about processing gains,
because the log and the simulated parts go through the same processing. With relative
levels only differences between terms are reported. A closed-form sonar-equation fit was
the alternative; it would need every gain in the chain to be modelled a second time.

### D-57 Sparse Bayesian learning for bearings; the map and CFAR stay for range
Asked for: source-separated bearings from the long-range array to hand to a downstream
batch tracker, by "ILRMA or normal correlation or whatever is state of the art". Chosen:
multi-frequency sparse Bayesian learning (SBL; Gerstoft, Mecklenbräuker, Xenaki and
Nannuru 2016), one source-power vector shared by every frequency in the band, noise
re-estimated from the residual.
- The active case is the hard one for the usual high-resolution methods: echoes of one
  ping from two bearings (two targets, or a target and its bottom bounce) are coherent
  and there is one snapshot per range gate. MVDR and MUSIC need many snapshots and
  incoherent sources. SBL works on one snapshot and does not care about coherence.
- ILRMA and other blind separation recover source *waveforms* without array geometry. The
  array's geometry is known and the tracker wants bearings, so a geometric method uses
  more of what is known. Cross-correlation (GCC-PHAT) gives pairwise delays, good for two
  hydrophones, weaker than a 32-element solution.
- SBL is not a detector over range. The beamformed map and CFAR say at which ranges
  something is; SBL then says from which bearings at each of those ranges. Where it finds
  no discrete arrival the map's detection is kept, so the option never loses a detection.
- The same solver takes averaged spectra of unsynchronised listening and gives passive
  bearing-time records with source powers and the noise level in the samples' units.
- Cost control: frequencies subsampled to 24, grid points pruned as their power collapses,
  a coarse grid then a fine one around what survived, peaks closer than 0.2 beamwidths
  folded together, and SBL skipped on gates whose conventional spectrum has no peak 8 dB
  over its median. All are options (`doa.hpp`, `chain.hpp`).
- The iteration is three kernels written once for the host and CUDA (`sbl_kernel.hpp`):
  the model covariance per frequency, its Cholesky factor, and one triangular solve per
  bearing in play. On the host this is 2.3 times faster than the first version (Eigen, a
  full solve per bearing), which is kept as a reference the kernels are tested against.
  The choice of bearings in play and the noise estimate stay on the host.

### D-58 The live chain is a C++ library with a standalone program
`libs/live` holds every stage after the kernels: stream assembly, navigation, board
synchronisation, clustering, the tracker, the classifier, the report. `build/bin/sonar_live`
is the deployable: no Python interpreter on the vehicle. The Python chain stays, because
taps, the simulator and the tools hang off it, and because two implementations compared by
tests catch errors in both. The deployment file stays the one place settings live:
`tools/sonar_live.py --export-template` turns it into the C++ program's template record
and command line. A YAML reader in C++ would have added a dependency for the same result.

### D-59 Holoscan operators in C++ are shells around the live library
`libs/holoscan_ops` has six operators (replay source, UDP source, recorder, chain, passive
bearings, report sink) and `apps/sonar_live_holoscan.cpp` composes them. Each calls the
same class the standalone program uses, and the tests require the two programs' reports
to be identical. Messages are shared pointers to ping records and to report text, passed
in process; nothing is converted to tensors, because no operator of the SDK needs to see
inside a ping. The chain is one operator, not one per stage: the stages share a tracker
and state, and splitting them would add copies without adding anything to schedule.
Built against Holoscan 4.6.0 (the C++ package from NVIDIA's apt repository, which needs
CMake 3.30 or newer); the Python wheel has no headers.

### D-60 Listening is a second way to cut the same stream
The boards send one continuous stream of blocks either way. For active pings the
assembler cuts a window after each transmit mark; with `--listen` it cuts fixed stretches,
each long enough for one passive frame, and ignores transmit marks. A gap discards the
stretch it falls in. Stretches are recorded as ordinary records (start time 0, the time
of the first sample in the transmit time field), so a listening run replays through
`--passive` unchanged.

### D-61 Ported stages keep their NumPy versions, and NumPy keeps the random numbers
Far-field and MVDR beamforming, array calibration, the non-Bellhop path providers and the
reverberation sum now have C++ implementations, used by default (`engine: cpp` in the live
chain; `GPUSONAR_SIM_ENGINE=numpy` or `engine="numpy"` selects the old simulator code).
The NumPy versions stay as the reference each port is tested against. The simulator's
random numbers are still drawn by the NumPy generator, in the same order as before, and
handed to C++: a scenario's seed gives the same ping to 3.5e-7 with either engine, and the
generator ends in the same state. A counter-based generator on the device would be
faster and would change every seeded result; not done. Reverberation's FFTs and its
power stage (which calls Python strength and beam-pattern functions) remain on the CPU;
the sum over azimuth sectors into every channel is the kernel with a CUDA build.
