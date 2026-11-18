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

### D-62 Time spreading: SST's algorithm with open substitutes for what TR 9407 supplies
`sim/timespread.py` wraps any eigenray provider and gives each bounce path a coherent part
and a random, slowly changing tail of later arrivals (TR 0702 Sec. 12.4). TR 0702 takes
three things from APL-UW TR 9407: the coherent/incoherent split, the shape of the
intensity impulse response and the frequency spread. TR 9407 carries a limited-distribution
marking on DTIC and the download was refused in this environment, so it was not used.
- Split: Rayleigh-parameter coherent reflection, exp(-Gamma^2) in intensity. The rest of
  the flat-boundary intensity is the incoherent part, so energy is conserved on average.
- Decay time: TR 0702's own rule of thumb. Its two cases (mid-path, near one end) are the
  two ends of T = 2 mss t1 t2 / (t1 + t2), which is used for any bounce position.
- Shape: exponential, a handful of taps with their delays scaled so the mean excess delay
  is the model's. Dahl's measured shapes are not exponential; this is a placeholder shape.
- Time variation: exponential correlation, which is what SST's AR(1) filter gives, at the
  Doppler width of a mirror moving at the boundary's rms vertical speed. Realised as a sum
  of sinusoids with Cauchy-distributed frequencies keyed by a hash of (seed, place, ray,
  tap), so no state is carried between pings.
- Surface statistics from wind: Pierson-Moskowitz height and vertical speed, Cox-Munk slope.
- Reverberation keeps using the flat-boundary paths: it is incoherent already, and its
  backscatter geometry is far from the specular direction, so nothing is counted twice.
- Cost: each bounce ray becomes 1 + taps rays one way, so a two-way path through bounces
  on both legs becomes (1 + taps)^2 paths.
- A user who holds TR 9407 can replace the three substitutes; they are three functions.

### D-63 The bottom scattering model is derived here, not transcribed
SST's bottom models (`JacksonBottom`, `APLBottom`) are defined in documents that could not
be read (TR 9407; Jackson and Richardson's book). The owner said to finish without them.
`sim/bottom.py` therefore derives the two terms those models share: interface roughness by
first-order perturbation theory for a fluid sediment with a power-law spectrum, and
sediment volume scattering by reciprocity. The derivation is in the module's docstring.
It is tested against results that do not depend on any handbook: the pressure-release and
rigid limits, no contrast, reciprocity, the reflection coefficient at normal incidence and
at the critical angle, and the depth integral of the volume term.
- It uses the published models' six parameters, so values from the literature can be put
  straight in. No table of named sediments is shipped; `PLACEHOLDER_SAND` is illustrative
  and says so.
- Not included, and present in the published models: the Kirchhoff regime near vertical
  incidence, the composite-roughness correction, elastic and layered bottoms. The module
  states its range (grazing below about 60 degrees, small k x relief).
- It feeds the scattering-function reverberation (`model: fluid_sediment`). Seabeds made
  of point scatterers (SAS, close-range FLS) still use Lambert's law;
  `bottom.equivalent_lambert_mu_db` gives the matching strength at one grazing angle.
- Surface bubble loss is a number or a table per bounce (`time_spread.surface.bubble_loss_db`).
  No wind-speed law is built in: the usual one is in TR 9407.
- Later additions, all opt-in, leave this entry's default unchanged: the Kirchhoff,
  small-slope and composite regimes (D-66), two named sediments (D-67), a wind-speed
  bubble law of limited standing (D-68), and the model for point-scatterer seabeds (D-64).

### D-64 Targets and model-driven seabeds are per-ping scene modifiers, not a wider kernel
SST's targets are sets of highlights with aspect tables, delays and an `ExternalTarget`
hook; ours were fixed point clouds. `Scene.evaluate(t, tx, rx, frequency, sound_speed)`
(`sim/scene.py`) now calls each modifier once per ping with the projector and receiver
positions, and returns points, amplitudes and velocities that enter the eigenray pairing
as ordinary scatterers. `sim/targets.py` supplies the modifiers: `HighlightTarget` (a
rigid set of highlights on a trajectory, with `AspectTable`, per-highlight `delay`,
`common_center`, `random_position`), the Kirchhoff shapes of D-65, and
`SedimentScatterers` (a `seafloor` item with `model: fluid_sediment`).
- Why: both engines, every provider and the record stay unchanged, and the cost is one
  evaluation per ping, as with SST's `ExternalTarget`. Without modifiers `evaluate`
  returns exactly what `at_time`/`amplitudes_at` returned; sample hashes of six scenarios
  were identical before and after on both engines.
- Rejected: an amplitude and delay per (path, frequency) in the kernel or the record. It
  would touch the C++ and CUDA kernels and the schema.
- The price: aspect, glint position and sediment angles come from straight lines to the
  sonar. A bounce pair gets the direct geometry's target amplitude; a seabed scatterer
  gets the model's strength at the direct angles times Lambert's ratio for its own. Only
  the value and group delay of a `response` at the centre frequency are carried, and the
  seabed model is evaluated at the centre frequency only.
- Sediment scatterers keep the Lambert flag: their amplitude is multiplied by
  sqrt(sigma / (sin g_in sin g_out)), so the existing per-pair Lambert factor in both
  engines makes the direct pair exact without touching the C++ pairing.
- A highlight's `delay` and `common_center` are displacements along the bisector of the
  two directions, with the amplitude rescaled so the spreading loss is that of the true
  place. This keeps "one amplitude and one apparent position per path". `common_center`
  reproduces SST's behaviour but saves no eigenray work here.
- Ground truth in the record holds nominal values for targets (glint amplitudes are
  written as 0): the real ones are per ping.

### D-65 Specular shapes by the Kirchhoff approximation, with a `resolved` mode; cylinder obliquity is cos, not cos squared
`SphereGlint`, `CylinderGlint`, `PlateGlint`, `DiscGlint` (rigid, or scaled by
`reflection`) follow the Kirchhoff backscatter integral of Lavia, Gonzalez and Blanc,
arXiv:1805.06945v2, eqs. (1), (5), (6), (7). The large-argument form of eq. (6), the
rectangle closed form, the bistatic element n.(s + r)/2, the phases for the simulator's
exp(+i omega t) convention and the resolved shapes are derived here and flagged so in the
code.
- Two modes for every shape but the sphere. Default: one highlight carrying the far-field
  pattern, for long-range sonars. `resolved: true` with `spacing`: a row or grid of
  coherent points at half a wavelength or less with the Kirchhoff element amplitude. The
  imaging sonars see man-made objects inside L^2/lambda, where a closed-form target
  strength is wrong; the resolved row gives the far-field pattern (0.3% from eq. (6) at
  45 degrees) and the near field from the same code.
- Cylinder obliquity factor: cos(theta), which is what eqs. (1) and (6) of the opened
  source give. The often-quoted cos^2 form in Urick's table of simple shapes is not used:
  that table could not be opened (core.ac.uk, apps.dtic.mil and www.fao.org were outside
  the allowed hosts), and nothing is taken from Urick.
- Warnings once each for ka < 10, the unresolved mode used inside L^2/lambda, and spacing
  above half a wavelength. The ka threshold is ours, from the exact rigid-sphere series
  (0.63 dB from a^2/4 for ka >= 10).
- Unconfirmed: eq. (6)'s phase as read through the web fetch differs from eq. (1) by a
  quarter turn. Magnitudes are tested against eq. (6), phases against eq. (1) by
  quadrature and a large-plate mirror test; whether the difference is a transcription
  problem or the paper's convention was not settled.
- Not modelled: creeping waves, elastic resonances, reflections between shapes, the
  sphere's finite-range factor (r - a)/r; plates cast no shadow; `occlude` is for static
  targets only.

### D-66 Rough-bottom regimes: small-slope as the interpolation, `combined` as the full model, our own composite split
`bottom.py` gains `roughness_model`: `perturbation` (default, bit-identical to D-63),
`kirchhoff`, `small_slope`, `composite`, `combined`. The published interpolation and
composite criteria (Jackson, Winebrenner and Ishimaru 1986; Mourad and Jackson 1989;
Williams and Jackson 1998) could not be opened, so these are documented substitutes.
- Interpolation between regimes is the small-slope approximation (perturbation factor x
  Kirchhoff integral / (2 pi q_z^2)), not a blend. It equals perturbation theory for small
  relief and Kirchhoff in the specular direction, and its hemisphere energy stays within
  0.87 to 1.03 of |V|^2.
- Rejected: a smooth minimum of Kirchhoff and composite. For a sediment the Kirchhoff
  result is below perturbation theory at 20 to 30 degrees (it carries the normal-incidence
  reflection coefficient), so the minimum pulled the valid perturbation result down by
  about 6 dB.
- `combined` is the small-slope result averaged over Gaussian large-scale slopes with
  shadowed facets dropped: one expression for near-specular, intermediate and small
  grazing angles with no switch. Cost: large-scale slopes are counted twice near specular,
  widening the peak by about the rms slope. `composite` alone (Bragg scattering only) is
  meaningful only at low and moderate grazing angles.
- The split between small and large scales is at k x h_small = 0.5, exposed as
  `small_scale_kh`. It is our choice, not a published value, and the low-angle level
  depends on it (for `medium_sand` at 100 kHz the composite correction to the roughness
  term is +24 dB at 2 degrees).
- Confirmed from opened sources: the perturbation factor against Jackson's form in Olson
  and Jackson, arXiv:2009.12420, eqs. (13)-(16), (20) (agreement 1e-10, used only in a
  test); the spectrum convention and the form of the 2-D Kirchhoff integral from Olson,
  arXiv:2106.08287. The Kirchhoff prefactor, the structure constant, the integral's
  evaluation on a rotated complex path and the small-slope prefactor are derived here and
  each tested independently.

### D-67 Named sediments come from arXiv secondary sources; TR 9407 is still unread
The first pass shipped no named sediments: the only tables found (in arXiv papers) are
attributed by their authors to TR 9407, and were treated as copies of the restricted
handbook. The owner then ruled: "Use anything from arxiv that cites things you need".
- `configs/sediments.yaml` now holds `medium_sand` and `very_fine_silt`
  (`status: confirmed`), read from Brown, Johnson and Brownstead, arXiv:1809.08459,
  Table 1, plus `placeholder_sand` (`status: unconfirmed`, warns when used). Scenario
  items take `sediment: <name>` and `sediment_file`.
- Rows are stored in the units their source prints (a `physical:` block with `source`
  and `assumptions`) and converted in code (`bottom.parameters_from_physical`), so the
  numbers can be checked against the paper by eye and the conversions are tested.
- Two assumptions sit inside the "confirmed" rows, recorded per entry: the water
  reference for the ratios (1000 kg/m^3, 1500 m/s; the paper does not print it), and the
  reading of "Vol. Scat. Strength [dB/m]" as 10 log10 of the volume cross section per
  unit volume in 1/m.
- Generic rock is not a row. The only arXiv source (Olson, Lyons and Saebo,
  arXiv:1601.06078, Table II) gives elastic parameters with no roughness or volume terms,
  and the fluid model would be the wrong one. The values are in a comment in the YAML.
- TR 9407 itself was not read, and the Hamilton regressions were not found. Other arXiv
  tables seen (arXiv:1611.00608, arXiv:2003.11156) give only sound speed, density and
  attenuation.

### D-68 Bubble loss from the Hall-Novarini population; two empirical laws; arlpy as an oracle only
`sim/surface.py` adds `bubble_loss_db` (`time_spread.surface.bubble_loss_db:
{model: hall_novarini, wind_speed}`), `kraus_surface_db` (reverberation item
`{model: kraus_surface, wind_speed}`) and `searay_bottom_db`
(`{model: searay_bottom, bottom_type}`).
- Bubble loss: the Hall-Novarini bubble population with all its constants as printed in
  Nennouche et al., arXiv:2603.23727v2, eqs. (20)-(23), with resonant-bubble extinction
  and two slant crossings of the layer derived here. It is the only wind-speed law that
  could be built from an opened source. The owner's arXiv ruling (D-67) did not help: no
  arXiv paper found prints a bubble-loss law, and pubs.aip.org, core.ac.uk and
  ieeexplore.ieee.org refused automated access.
- Status: weak above about 10 kHz. It grows too fast with frequency (2.3 to 7.5 times
  the arlpy value in dB at 10 m/s; `RESULTS.md`) and should be read as an upper estimate
  there; the code warns. A measured table per bounce (D-63) remains the better input.
  The unit of the population density n is not printed in the source; m^-4 was taken, the
  unit that makes its eqs. (19) and (24) consistent.
- The empirical surface law and the SEARAY bottom law are taken as printed in Morency
  and Stilwell, arXiv:2210.06535, Sec. II-E, II-F (they quote D. Kraus's lecture notes).
  The source gives no validity range. Ours: the Kraus law is held constant above 60
  degrees, the bubble slant path has a 2 degree grazing floor, and warnings fire for wind
  above 20 m/s, the bubble law above 10 kHz and the Kraus law below 10 kHz. The SEARAY
  law was not asked for; it came with the surface law in the same source.
- `tools/boundary_models_study.py` calls `arlpy.uwa.bubble_surface_loss` (BSD; its
  docstring says "APL model (1994)") to print a comparison column. Its formula was not
  read or copied, no value from it is used by any code (the ratios to it are quoted in
  the `surface.py` docstring and in `RESULTS.md`), and the core does not import it.
  If the owner judges that too close to the restricted handbook, the oracle block in the
  tool and two sentences in `surface.py` and `PHYSICS_NOTES.md` are what to delete.
- Not built: a physical rough-surface plus bubble-layer backscatter model.

### D-69 Bistatic reverberation by components
`geometry: bistatic` (`sim/reverb_bistatic.py`) lifts the monostatic assumption of the
scattering-function method (D-37, D-43): transmitter and receiver at different places,
each moving.
- Power is computed on a receiver-centred polar grid and binned by two-way time, rather
  than by constructing SST's iso-time ellipse. The same code then serves any eigenray
  provider, and the arrival azimuth is the grid's own coordinate.
- Each leg's eigenrays are taken along one line of horizontal distance (stratified
  medium, flat boundary), the transmit leg from a half-ring table with its delay
  corrected by the horizontal slowness: two provider calls instead of sectors x rings.
- Strength is evaluated at both true grazing angles and the bistatic azimuth
  (`fluid_sediment` uses the azimuth; Lambert and volume do not depend on it;
  Chapman-Harris is still a backscatter fit, used at the mean grazing angle).
- Contributions are grouped into components by (azimuth sector, arrival-elevation bin;
  9 bins by default), each with a power-weighted mean 3-D direction and Doppler shift
  f/c (V_S.u_Sq + V_R.u_Rp), eq. (71) to first order. No separate Doppler bins: TR 0702
  notes there is typically one Doppler cell per direction, and this keeps the generator a
  list of delayed, shifted Gaussian sequences that the existing C++ and CUDA steering
  kernel handles unchanged. Components more than 120 dB below the strongest are dropped.
- **Known defect of this choice** (found by the validation of D-78; not fixed in the
  merged tree): pairs with different transmit legs that share a receive ray share one
  shift, whereas eq. (71) gives each pair its own. Two strict xfail tests pin it
  (`test_sst_validation.py::test_reverberation_gives_each_eigenray_pair_its_own_doppler`,
  `::test_bottom_paths_change_nothing_before_they_can_arrive`). The fix proposed is to
  add Doppler or the transmit path to the component key; it is in progress.
  *Later: fixed, see D-80.*
- Opt-in, because random draws and steering differ from the sector generator. Default
  outputs are bit-identical to before on both engines (16 arrays compared).
- `strength_map` (a callable of world x, y, or a table of dB against range and optionally
  azimuth) makes the boundary's strength vary with place; it needs `geometry: bistatic`.
- Left alone so that old results do not change: the nearest-bin ripple of the default
  `reverberation_power` (-1.4 to +1.7 dB between neighbouring 1 ms bins; the ten-bin mean
  is right; the new code deposits linearly and has none), and the default mode's Doppler,
  which treats every path as horizontal.
- Not done: per-element receive patterns. The first element's pattern stands for all; a
  gain per direction and channel would have to enter the steering kernel on host and
  CUDA. No shipped sonar has differing elements, and a `RuntimeWarning` fires if they do.

### D-70 K-distributed reverberation by a gamma texture
`statistics: {model: k, shape, cell_time, cell_azimuth_deg}`.
- Compound representation: a unit-mean gamma texture per resolution cell of time
  (1/bandwidth by default) and azimuth (wavelength/extent), multiplying the Gaussian
  deviates. No engine change, and mean power is untouched. The moments it is tested
  against are derived from gamma and exponential moments; no external value was used.
- `shape` has no default: it is a seabed property with no confirmed value. Infinite
  shape or `model: rayleigh` is bit-identical to the default.
- Statistics after beamforming come from the array structure: N cells give a second
  moment of 2(1 + 1/(N x shape)).
- Limits: the texture is independent between pings and between neighbouring cells, and
  components in one azimuth cell share it even when they come from different path pairs.

### D-71 Leakage along long records: `band_taper` exists and stays opt-in
The generator's ideal (frequency-domain) delays and its interpolation at the reduced rate
have sinc responses, and the pulse spectrum is not zero at Nyquist, so the strong start
of a record leaks into its weak far end. `band_taper: true` rolls the pulse spectrum off
with a raised cosine from half the generation Nyquist to Nyquist and generates two pulse
lengths of silence after the record (`guard: seconds` sets that length).
- Measured on a 30 m to 3 km CW record (100 dB of level change): up to +54 dB excess at
  the far end without the taper, 0.2 dB or less with it (`RESULTS.md`).
- The coordinator's first plan was to make the taper the default, as a bug fix. It was
  then measured on four shipped scenarios (`fls_120k_transit`, `fls_170k_transit`,
  `fls_330k_transit`, `popoto_trial_100m`): it changes nothing beyond 0.06 dB on the FLS
  scenarios and 0.3 dB on `popoto_trial_100m`. None of those shows leakage, so the
  default is unchanged and every earlier result stands. The other shipped scenarios were
  not measured. The validation scenario `sst_pursuit`, added afterwards (0.5 to 5 s of
  record at one go), sets `band_taper: true` itself.
- Use it for a short pulse on a record whose reverberation falls by more than about
  50 dB (the first measurement's threshold; the follow-up puts the practical limit at
  about 60 dB).
- With the taper and a moving platform the small anti-causal part of the tapered pulse
  gets a constant phase error in the Doppler-shifted copy; identical on both engines and
  not visible in the level tests.

### D-72 Scatterer Doppler in reverberation
`scatterer_motion` on a reverberation item (needs `geometry: bistatic`). TR 0702 gives
only the split into a platform factor (71) and a scatterer factor (72), bins of total
Doppler (129) and the coherence-time result (96); it gives no distribution for surface
or volume.
- Mean velocity (`velocity`) enters each cell as -f/c V.k, k the sum of the two legs'
  travel directions at the cell: eq. (72) to first order.
- Random motion is a Gaussian spread, sigma_f = (f/c) sqrt(rms_speed^2 |k|^2 +
  rms_vertical_speed^2 k_z^2): an isotropic part and a vertical-only part, the two cases
  with a clear physical parameter (volume scatterers; a heaving surface, whose rms
  vertical speed `wind_speed` sets from the existing `acoustics.rms_surface_vertical_speed`).
- Realised as unresolved Doppler lines (SST's bins of total Doppler), Gaussian-weighted,
  spaced at most half the pulse resolution 1/T, at most 41, each an independent
  sequence. Rejected: a time-varying filter. With lines the generator and the CUDA
  kernel stay unchanged and the mean power is exact. A spread below 1/20 of 1/T is left
  as one line.
- No drift law from wind is built in (none was confirmed from an opened source), nor
  orbital wave motion or Bragg phase speed; `velocity` and `rms_speed` carry them by hand.
- Ping-to-ping coherence stays outside the scattering-function method: it draws new
  sequences every ping, which is consistent whenever there is any spread and remains
  wrong for a still sonar over a still bottom.
- The component path projects each platform's full velocity on its own ray (heave and
  the cos(elevation) factor are tested); attitude rates times lever arm are not included.

### D-73 Reverberation over relief
`bathymetry` on a reverberation item (needs `geometry: bistatic`) or on a scene
`seafloor` item. `sim/bathymetry.py` describes the bed (a plane, a grid, depth against
range and optionally azimuth, or a callable); `sim/reverb_relief.py` computes the power.
- Each cell sits at its own depth with eigenrays asked for at the cell itself, its true
  facet area, grazing angles to the facet normal on both legs (a facet turned away
  returns nothing), the bistatic azimuth in the facet frame, and arrival direction and
  Doppler from the true position.
- Shadowing: a horizon map per apparent origin (the platform, or its image in the
  surface) along straight lines, with a 0.2 mrad tolerance; `shadowing: false` turns it
  off. It is skipped for refracted paths.
- Legs with a bottom bounce before the cell are excluded by default, because image paths
  over relief are not physical. The cost was measured on a level bed: under 0.2 dB within
  about two water depths, 5 to 6 dB beyond five with a reflective bottom (`RESULTS.md`).
  `bottom_bounces: flat` keeps them as the provider's flat-bottom images: exact on a
  level bed, error not bounded over relief.
- With Bellhop, `BellhopProvider.along_profile` runs it once per azimuth sector over that
  bearing's depth profile (N x 2-D). Transmitter and receiver must be over one ground
  point, otherwise `NotImplementedError`; no out-of-plane reflection.
- A bathymetry that is exactly level routes to the flat code, so results cannot drift.
- One description serves reverberation items and scene seafloors. The scene's scatterer
  seabed is not shadowed by its own relief, so the two methods differ behind a ridge.

### D-74 A sampled transmit pulse is carried in the record's existing fields
`transmit.waveform: {type: sampled, source: lfm | hfm | stepped_fm | barker | mseq | code
| samples | file}` (`sim/waveform.py`, `sonar.py`). The record already had
`WaveformType::Arbitrary` (3), `replica` and `replica_sample_rate`, written, read and
validated; nothing was added to the schema.
- `replica` holds the real passband pulse Re{b(u) exp(2 pi i F u)}, sample 0 at emission
  time 0; `replica_sample_rate` defaults to four times the top of the band;
  `f_start`/`f_stop` are the occupied band and their mean is the carrier F
  (`RECORD_FORMAT.md`). A real passband series fully determines the envelope, so
  processing rebuilds it (analytic signal, mixed down) and a record read back from a file
  compresses identically.
- Rejected: a complex replica (it would change the meaning of an existing field), and
  regenerating the pulse from the scenario config (a record must be processable alone).
- Cost: the replica is stored with every ping as float64 (a 1 s pulse at 6 to 10 kHz is
  40,001 samples).
- The matched filter for an arbitrary waveform is built in C++
  (`libs/dsp/src/pulse_compression.cpp`, `build_arbitrary`, including `doppler_scale` and
  replica windows), so one place serves Python, the live chain and the CUDA compress
  path, which takes the host-built filter. Before this, `pulse_compress` threw for
  `Arbitrary`.
- CW cannot be given as `sampled` (zero bandwidth is refused); `ComplexAnalytic` output
  is refused, as in the kernel.

### D-75 Echoes of a sampled pulse: the envelope interpolated at solved emission times, Kaiser window, NumPy for now
- Per (path, channel) the synthesiser follows the analytic kernel's own steps: exact
  arrival times of pulse start, middle and end, emission time as the quadratic through
  them, echo b(u(t)) exp(2 pi i F u(t)). Envelope and carrier are both time-scaled, so
  the kernel's exact delay and Doppler are kept. SST's `VarDelay` interpolates a slow
  delay track linearly; that approximation is not needed here.
- Interpolation is on the envelope, so its accuracy depends on bandwidth / envelope
  rate whatever the carrier. Two stages: a 16-tap windowed sinc onto a table 8 times
  finer, then a 4-point cubic.
- Window: Kaiser, 16 taps, shape 12, by default; Hann (SST's, TR 0702 eqs. 15-17)
  available. Measured at the same length: -117 dB against -62 dB. The shape 12 is from
  our own sweep.
- Absorption and beam patterns are applied across the spectrum, piecewise linear between
  `gain_knots + 1` frequencies (default 8 intervals), because a sampled pulse has no
  instantaneous frequency. Agrees with the analytic kernel to under 1e-3 for an LFM.
- Sampled-pulse synthesis runs in NumPy only, with a visible fallback: the record's
  `extra_json` gets `{"gpusonar": {"echo_synthesis": "numpy_sampled"}}`, and asking for
  the C++ engine or the CUDA backend warns once. Reason: the kernel is a plain-data
  struct mirrored in a `.cu` file that cannot be compiled in the development container.
  Path pairing and reverberation keep their C++ engine. A C++/CUDA port is in progress.
- Found on the way, outside this work: the analytic kernel's arrival-time fixed-point
  solve stops after 3 passes outbound and 4 on return, leaving about (R/c)(v/c)^3
  (0.007 rad at 8 kHz for 12 m/s closing at 400 m). The NumPy path runs 8 passes; the
  kernel was left as it is (an open question in `STATUS.md`).
- *Later: the synthesis was ported to the C++ kernel, see D-81.*

### D-76 Angle and frequency spreading of bounce paths from one facet model
New opt-in keys under `time_spread` (`sim/timespread.py`): `angle_spread`, `doppler`,
`doppler_spectrum`, `doppler_lines`. One model, tilted mirrors with Gaussian slopes,
gives the iso-delay geometry, the tap directions, a decay time and the multi-bounce
rule (D-77), so the options are consistent with each other. All formulas are derived in
the module docstring and `PHYSICS_NOTES.md` Sec. 16 and tested against exact geometry or
Monte Carlo; no coefficient comes from any paper and nothing from TR 9407. With the new
keys absent the eigenrays are identical to D-62's.
- Angle: a tap of excess delay tau lies on a circle of radius rho, rho^2 = 2 c tau r1 r2 /
  (r1 + r2), seen along the ray; its position on the circle is drawn per tap from the
  density Gaussian facet slopes give. The direction rides in the eigenray's
  `launch`/`arrival`, so both engines carry it with no fallback.
- Frequency within a ping: the existing ping-to-ping process is already a sum of
  sinusoids; with `doppler` each sinusoid becomes its own ray with a path-length rate, so
  the same process continues through the ping and ping-to-ping statistics and seeds are
  unchanged. The C++ pairing returns one velocity per scatterer, so rays with a range
  rate are paired in NumPy with a one-time `RuntimeWarning` (synthesis stays in C++). A
  C++/CUDA port of that pairing is in progress. *Later: done on the host, see D-81.*
- **Finding: the Doppler width goes as sin g, not 1/sin g.** A boundary displaced by zeta
  changes the path by -2 zeta sin g; for a Gaussian surface the scattered part's spectrum
  has second moment exactly sigma_nu^2 = (2 f v sin g / c)^2 / (1 - exp(-Gamma^2)). TR
  0702 p. 101 prints the rule with a division by sin g. The derivation settles note 23 of
  `PHYSICS_NOTES.md` Sec. 11. TR 0702's remark that the ASIAEX surface spread is "less
  than 10 Hz" does not decide between the two: both readings are under 10 Hz for that
  geometry (`RESULTS.md`).
- `doppler_spectrum: gaussian` uses the derived rms width (the Gaussian shape is exact
  only for Gamma >> 1; at small Gamma only the rms width is right). The default
  `lorentzian` is SST's AR(1), unchanged; it has no rms width.
- With `angle_spread` on, platform-motion Doppler comes from the tap directions through
  the kernel, so the V sqrt(mss) term is dropped within a ping to avoid counting it twice.
- Limits: second-order (small-angle) geometry, with a turn towards the horizontal
  clamped at 0.9 of the ray's elevation; no shadowing; isotropic slopes (no up-wind /
  cross-wind difference) and no mean surface Doppler shift; taps cut at three mean
  delays, so coherence falls faster than Gaussian beyond about 1.5 coherence lengths; tap
  directions fixed across pings; with several rough bounces the sonar-end direction comes
  from the first and the far-end from the last. Not compared with measured spreads
  (Dahl's papers were not opened).
- Found and not fixed: the realisation key uses floor(position / coherence_length). A
  sonar exactly on a cell edge (x = 0, where the array mean can be -1e-17) puts projector
  and array in different cells, so out-leg and back-leg tails are independent. Fixing it
  changes existing realisations (an open question in `STATUS.md`).

### D-77 The facet decay time and convolved bounces are offered, not made the default
- `decay_time: facet`: the facet model gives T = mss (1 + sin^2 g) t1 t2 / (t1 + t2), with
  a shape that is not exponential. TR 0702's rule of thumb (D-62, the default) equals it
  at normal incidence and is twice as long at low grazing. The rule stays the default: it
  is what SST documents and what earlier results used, and the facet value was checked
  against a Monte Carlo of slopes, not against data.
- `multi_bounce: convolve`: each bounce is c_i delta + (1 - c_i) h_i, convolved on the
  tap grid, so a bounce delays the tail only when it scatters. The default `sum` adds
  the mean delays into one exponential, the error TR 0702's author describes for SST.
  `sum` stays the default for reproducibility of earlier results.
- What the validation found (D-78): with the ASIAEX parameters `sum` does not show the
  defect as TR 0702 words it but one of the opposite sign, because the bottom there is
  93% coherent with the longer decay and `sum` gives all scattered energy the summed
  delay. The validation recommends `convolve` whenever the two boundaries differ, and
  suggests considering it as the default. That is the owner's call (`STATUS.md`).
- *Later: the owner decided for `convolve`, see D-82.*

### D-78 Validation examples carry our own parameters, marked unconfirmed; a relay stands in for a one-way source
`tools/sst_validation.py`, `tests/python/test_sst_validation.py` and the scenarios
`sst_pursuit`, `sst_asiaex`, `sst_echo_level`, `sst_sediment_reverb` check the simulator
against TR 0702 itself, section by section (`docs/figures/sst_validation.md`).
- TR 0702 gives no complete numerical test case. For its two worked examples (Pursuit,
  Fig. 2; ASIAEX, Figs. 3-5) everything it does not print is our choice and sits under
  `unconfirmed` in the scenario file, each chosen to place one feature the text names
  (for Pursuit: carrier 9.9 kHz, vehicle 20 m/s at 350 m, water 1275 m, target 2.5 m/s
  away at 2124 m with three -10 dB highlights, 0.1 s pulse, 20 degree sinc beams, wind
  8 m/s). Assertions are against closed forms for our geometry, never against the
  figures. Neither is a numerical reproduction and neither may be presented as one.
- The Pursuit source level of 230 dB is not a design value: it is only there to put
  reverberation about 10 dB over wind noise in the displayed band.
- One-way path: the simulator has no one-way source (SST's `DirectSignal`). In
  `sst_asiaex` a -20 dB point relay 0.1 m from the projector makes one; the outbound
  multipath is 55 dB down. No simulator change was needed. Not done: a real one-way
  source, so target-radiated noise and tonals cannot be checked.
- The reverberation hook is checked against the first-order formula the generator
  implements; its 3.6 Hz gap to the exact eq. (71) at 20 m/s and 9.9 kHz is reported
  separately as a documented limit (echoes use the exact factor). *Later: the bistatic
  mode uses the exact factor and the check is against eq. (71) itself, see D-80.*
- The defect it found (D-69) is recorded with strict xfail tests rather than fixed
  inside the validation package, which edits no simulator module.
- The readings chosen for TR 0702's inconsistent passages (items 17 to 24 of
  `PHYSICS_NOTES.md` Sec. 11) are each pinned by a named test; the table is at the end
  of `docs/figures/sst_validation.md`.

### D-79 Sensor Bridge's emulator gets an opt-in UDP-socket patch; the loopback test runs at half real time
The loopback through Sensor Bridge 2.7.0's board emulator and `LinuxReceiverOp` could not
run on Hugging Face Jobs: the emulator's data plane opens a raw IP socket, which needs
CAP_NET_RAW, and those containers refuse it.
- `third_party/patches/hsb-2.7.0-emulator-udp-socket.patch` (27 changed lines in three
  files under `src/hololink/emulation/src/linux/`): with `HSB_EMULATOR_UDP_SOCKET=1` in
  the emulator's environment the data socket is an ordinary UDP socket bound to the
  board address and port 12288, and the send skips the emulator-built IP and UDP
  headers. Without the variable nothing changes.
- Opt-in twice: `tools/build_hsb.sh` applies the patch only when `HSB_UDP_PATCH=1`, and
  the patched emulator behaves as stock without the run-time variable.
  `tools/gpu_job.sh` sets both only when `GPUSONAR_HSB=1` and a raw socket cannot be
  opened. The patch is applied at build time to a separately built third-party tree;
  nothing of Sensor Bridge is vendored.
- On the wire the UDP payload (BTH, RETH, immediate, data, frame metadata, iCRC) is the
  same bytes from the same source address and port. The IP and UDP headers are the
  kernel's (TTL, identification, DF, a filled-in UDP checksum). The iCRC is still
  computed over the emulator's own header fields; `LinuxReceiver` does not check the
  iCRC or the sender, and accepted the packets. A receiver that checked could differ.
- Rejected: loosening the report comparison to tolerate the frame lost at real-time
  pace. The sender was slowed instead: the test's pace is 0.5 (`GPUSONAR_HSB_PACE`
  overrides), the default frame is 400 samples (38,528 bytes) and the receiver queue is
  256 frames. Why real time loses one burst per run is not established (`RESULTS.md`).
- Not covered by the patch or the runs: the RoCE receiver, and the unpatched emulator on
  a machine with CAP_NET_RAW.

### D-80 Reverberation components are keyed by total Doppler (follow-on to D-69)
D-69 grouped bistatic reverberation by (azimuth sector, arrival-elevation bin) with one
Doppler shift each, the power-weighted mean over every eigenray pair and every two-way
time arriving from there. The validation (D-78) showed that to be a defect. It is fixed:
the component key gains a third part, a bin of total Doppler shift, SST's own cell of
TR 0702 eq. (129).
- Chosen over keying by transmit path: a Doppler bin also separates early from late
  returns of one pair, and needs no ray identity, which Bellhop does not give.
- Bins span every shift the speeds allow (f v / c per platform, 2 f v / c for scatterers,
  plus a 5% margin) and are half the pulse resolution wide (0.5 / T), since nothing finer
  is observable within a pulse. There are at most `max_doppler_bins` (default 128, a new
  key on the reverberation item); they widen if more would be needed, which bounds the
  count for long pulses.
- Stored sparsely: only components that receive power exist, since most (direction,
  Doppler) cells are empty. In practice a few per direction.
- The shift is now the exact factor of eqs. (71)/(72), (Gamma x gamma - 1) f, not its
  first-order form. This removes the 3.6 Hz gap at 20 m/s and 9.9 kHz that D-78 recorded
  as a limit.
- What changed and what did not: default (monostatic) outputs are bit-identical (the same
  16-array check); bistatic with nothing moving is unchanged (one bin, same code path and
  summation order); bistatic realisations with moving platforms or scatterers changed,
  from the keying and from the exact factor. No C++, kernel or `.cu` change was needed.
- The two strict xfail tests of D-69 pass with their markers removed.
- Cost on the Pursuit ping (72 sectors, 90 elevation bins, 0.1 s pulse, 20 m/s): 2484
  sequences and 5.8 s with one bin (the old behaviour), 12770 sequences and 28.3 s with
  the default 128 (`RESULTS.md`). With Doppler bins, fewer elevation bins would give the
  same Doppler fidelity for less; the scenario was not retuned.
- Still so: within one Doppler bin the shift is a mean (error at most half a bin, which a
  pulse cannot resolve); the direction is a mean within an elevation bin; one shift per
  echo for the whole pulse, no range walk.

### D-81 Sampled pulses and per-ray range rates move into the C++ engine (follow-on to D-75 and D-76)
D-75 synthesised echoes of a sampled pulse in NumPy only, and D-76 paired rays that carry
a range rate in NumPy with a warning. Both fallbacks are removed on the host. The CUDA
side is written; it had not been compiled or run when this was recorded.
- Sampled pulses use the same kernel header as analytic pulses
  (`synth_kernel.hpp`, a second pair function `pair_sampled`), with the envelope table as
  plain data: built once in NumPy (`SampledWaveform.kernel_table()`), passed as a pointer
  in the parameter block, uploaded by the CUDA driver. One kernel header then serves host
  and device, and the table is the reference engine's own, so the engines differ only in
  rounding. Rejected: a C++ table builder, a second implementation of the interpolator to
  keep in step. Consequence: a C++-only caller must supply the table.
- Frequency-dependent gain is the per-sample sum of gain-weighted bands, the same linear
  expression as NumPy's, so there is no approximation to bound. Cost: about three times
  the analytic kernel with beams (attributed to the nine-band sum at every sample; not
  profiled).
- `table_factor: 0` and more than 32 `gain_knots` stay in NumPy, with a warning and the
  `echo_synthesis: numpy_sampled` note in the record: the kernel reads a table and holds
  a fixed-size gain array. The note is now written only when NumPy ran.
- Arrival-time pass counts are parameters (`SynthParams::arrival_passes_out/back`;
  `arrival_passes=(out, back)` on `core.synthesize_into`). The analytic default stays
  (3, 4), so existing analytic records stay bit-identical (7 of 7 scenarios compared);
  sampled pulses use (8, 8), as the NumPy reference does. Whether to raise the analytic
  default to (4, 4) is the owner's decision (`STATUS.md`): it removes the 0.007 rad offset
  at under 1% cost and changes every existing record with moving geometry at the 1e-3
  level or below.
- The C++ pairing (`sim::two_way_paths_doppler`) takes a range rate per eigenray on each
  leg and returns a velocity for each end of each pair. The kernel already took two
  velocities per path, so nothing else had to change. The NumPy branch remains as the
  reference engine. Path pairing has no CUDA backend and never had one.
- The extension must be rebuilt after this change: the new Python calls
  `core.synthesize_into(..., envelope=...)` and `sim_two_way_paths(..., out_range_rate=...)`,
  which an old build does not have.

### D-82 `time_spread.multi_bounce` defaults to `convolve` (follow-on to D-77; the owner's decision)
D-77 kept `sum` as the default for reproducibility and left the choice to the owner. The
owner asked: "should we do what the validation agent asked? Why wouldn't we? Which is
physically realistic". The default is now `convolve`.
- Reason: a cascade of independent bounces is the convolution of their responses. `sum`
  gives every scattered part the summed delay, which is wrong when the bounces differ: in
  the ASIAEX set-up it is -2.3 dB just after the pulse and +3.7 dB later, relative to
  `convolve` (`RESULTS.md`).
- `sum` remains as an option, as the earlier rule.
- The facet decay time (`decay_time: facet`) remains opt-in: it was checked against a
  Monte Carlo of slopes, not against data, and TR 0702's rule is what SST documents.
- Consequence for earlier numbers: results for paths with two or more rough bounces that
  were produced before this change were produced under `sum`. In `RESULTS.md` that is the
  section "Time spreading of bounce paths and volume layers" and the rows labelled `sum`
  in the spreading study; they were not re-measured. Single-bounce rays are bit-identical
  between the two modes.

### D-83 The echo kernel runs four outbound passes of the arrival-time solve

The kernel finds when sound meets a moving point by fixed-point passes, each gaining a
factor v/c. It ran three outbound and four on return, which leaves (R/c)(v/c)^3 outbound:
0.14 us, a phase of 0.007 rad at 8 kHz, for a target closing at 12 m/s from 400 m
(measured 6.9e-3 against the closed form; `RESULTS.md`, "Sampled pulses and per-ray Doppler
in the C++ engine"). It is a constant phase offset of the echo; the Doppler factor was
already right. With (4, 4) the error is 5.5e-5 for under 1 % of a (path, channel) pair's
cost. The owner chose to switch ("Switch the echo kernel"). Consequence: records of scenes
with motion differ from earlier ones at the 1e-3 level or below; static scenes are
unchanged. `arrival_passes=(3, 4)` reproduces the old kernel; sampled pulses keep (8, 8).

### D-84 Surface Doppler spreading defaults to the Gaussian spectrum

`time_spread.doppler_spectrum` now defaults to `gaussian`; `lorentzian` (SST's AR(1) process)
stays selectable. Reason: in the first comparison with measured channels (Underwater Acoustic
Channel Repository, SPACE'08 site, 12.5 kHz; one fitted parameter per recording, the wind)
the measured Doppler spectra of surface arrivals have the Gaussian shape (width ratio
w90/w50 of 2.0 to 3.1; a Gaussian has 2.44) and the derived rms width; the Gaussian option
is inside the measured spread for 13 of 18 arrivals in width and 15 of 18 in shape, the
Lorentzian for 4 and 7. One site and one frequency: evidence, not proof; the comparison is
being extended to the repository's other sites. The owner chose to switch ("Yes change to
Gaussian"). Consequence: realisations of runs with time spreading change (the spectrum also
sets the ping-to-ping fading). The decay-time rule is not changed: the same data do not
settle it.

### D-85 Time spreading: the decay time defaults to the facet rule

`time_spread.decay_time` now defaults to `facet` (mean excess delay
mss (1 + sin^2 g) t1 t2 / (t1 + t2), with the shape Gaussian facet slopes give);
`rule_of_thumb` (TR 0702's 2 mss t1 t2 / (t1 + t2), one exponential) stays selectable.
Evidence: the measured channels of `PHYSICS_NOTES.md` Sec. 21 (21.4 for the three
recordings): 27 surface arrivals at the SPACE'08 site, 12.5 kHz, three recordings in two
sea states, one fitted number per recording (the wind). Measured decay time over model:
median 0.47 for the rule of thumb (about twice too long) against 0.99 for `facet`; above
12 degrees of grazing 0.75 against 1.28, below it 0.28 against 0.59. Limits: one site, one
band; the scatter is a factor 5 per arrival for `facet` (3 for the rule), so the median is
supported and a single arrival is not; neither rule is supported in detail. The two rules
agree at normal incidence and differ by a factor two at low grazing angles. The owner was
shown the recommendation and answered "Take your recommendations".
Consequence: runs with `time_spread` and no `decay_time` key get tails up to half as long
(shipped: `sst_asiaex`, `flank_sst_shallow`; `flank_sst_spread` already names `facet`).
The comparison with TR 0702 (`tools/sst_validation.py`) names `rule_of_thumb`, as it names
`multi_bounce: sum` and `doppler_spectrum: lorentzian`: that report's rule is what it tests.
D-84's last sentence ("the decay-time rule is not changed") is superseded by this entry.
