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

Correction 2026-11-20: the call is `reserve(bytes)` (`libs/core/include/sonar/core/allocator.hpp`
`SlabAllocator::reserve`); the alignment is one per allocator, given to its constructor
(`SlabAllocator(Backend backend = Backend::Host, std::size_t alignment = 256)`), and every
reservation and block is rounded up to it.

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

Correction 2026-11-20: there is no `culling.beam_factor`. The parameter is `fov_factor`
(`BpOptions::fov_factor` in `libs/recon/include/sonar/recon/backproject.hpp`; a `sas_image` /
`fan_image` stage key): a direction is accepted when its angle from the boresight is within
`fov_factor · FWHM / 2` in azimuth and in elevation (a half-angle), separately per axis. The
C++ default is 0 (no limit, and no culling); `recon.sas_image` defaults to `fov_factor` 1.13 and
`cull_block` 16 (`python/gpusonar/recon/sas.py`). Culling is `cull_block` (C++ default 0, off)
and needs `fov_factor > 0`. The factor 1.5 above is not a default anywhere in the code.

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
Amended in D-164: `p_z,1` starts at `−altitude_DVL[1]` and is held there only with
`fix_first_height`; by default it is an unknown with its own Eq. 7 residual.

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

### D-86 Bottom roughness scattering defaults to the small-slope approximation

`roughness_model` of the fluid-sediment bottom model (`sim/bottom.py`: reverberation items,
sediment seafloors, `sediment_fit`, the study and fit tools) now defaults to `small_slope`;
`perturbation` stays selectable and is unchanged. Evidence is the literature pass
(`PHYSICS_NOTES.md` Sec. 23.2, `REFERENCES.md`), not a measurement of ours: the lowest-order
small-slope result is what current work uses; against boundary-element solutions it is valid
wherever first-order perturbation theory is, and beyond (Olson, J. Acoust. Soc. Am. 160,
1918, 2026, hdl 10945/76181: "PT is accurate whenever SSA(2) is accurate"); it equals the
perturbation result for small relief and the Kirchhoff result in the specular direction
(our tests), where perturbation theory with a power-law spectrum diverges. Limits: that
comparison is for one-dimensional surfaces and backscatter; no data of ours decide between
the two. The owner was shown the recommendation and answered "Take your recommendations".
Consequence, measured on the shipped cases: below 60 degrees of grazing the backscattering
strength changes by 0.0 to +0.3 dB; at 75 degrees by +0.0 to +0.6 dB; at 85 degrees by
+0.3, -2.8 and -3.3 dB (placeholder sand at 8 kHz, medium sand at 20 kHz, placeholder
sand at 300 kHz), and at 90 degrees it is finite (+17, -1, -2 dB) where the perturbation
result is not. In records: `sst_sediment_reverb` +0.07 dB (grazing 8 to 53 degrees),
`fls_glints_on_sediment` unchanged to 0.00 dB (difference -82 dB re the record); in
`flank_sst_shallow` and `flank_sst_spread` the first 0.2 s of the record, where the long
pulse's return from the bottom below the sonar (normal incidence) still arrives, falls by
6 to 8 dB (record energy -6.5 dB) and the rest is unchanged to 0.00 dB: that part of the
old records came from the diverging perturbation result as sampled by the angle grid.
Cost: 1.3 times the perturbation evaluation (a cached table; 0.21 s against 0.16 s for
200 000 angle pairs), nothing measurable in scenario run time, so no further tabulation.
`sim/sediment_fit.py` and `tools/fit_sediment.py` keep naming `perturbation` as their own
default: a fit records its model in the row it writes, its tests assert what first-order
theory can and cannot separate, and with a free exponent the small-slope table would be
rebuilt at every step; `--roughness-model small_slope` is there for data above 60 degrees. The small-slope result needs 2 < gamma2 < 4; outside that range, with no
model named, the perturbation result is used and a warning says so. Sediment seafloors now
pass `roughness_model` and `small_scale_kh` on, so the earlier amplitudes stay selectable
there too (they were not before, because there was nothing to select).

### D-87 Surface backscatter: the bubble-layer model in its band, a warning and an opt-in cap outside it

In 12 to 70 kHz and 3 to 15 m/s, `bubble_layer_surface` is the model to use: it is the only
one here whose source states a comparison with data (+-5 dB; Gallaudet 2001, pp. 124-125,
`PHYSICS_NOTES.md` Sec. 23.3). The owner was shown the recommendation and answered "Take
your recommendations". What that changes:
- No code default: a reverberation item without `model` is a Lambert boundary, so every
  surface item names its law. There was nothing to flip.
- No shipped scenario was switched, because none has a surface item in that band: the
  surface items are at 5 to 9.9 kHz (flank, long-range Popoto, `sst_pursuit`) and at 120 to
  330 kHz (FLS); the `popoto_8` trials at 27.5 kHz have no surface item. Switching the ones
  outside the band would be an extrapolation of the bubble model, so they keep their law.
- `chapman_harris` and `kraus_surface` are unchanged and now warn, once per law, when used
  at 12 kHz or above where they exceed the level a saturated bubble layer returns
  (delta_r sin t / (8 pi delta)) at 10 degrees of grazing. Computed crossing frequencies
  for 3 / 5 / 6 / 8 / 10 / 15 m/s: Chapman-Harris 416 / 127 / 84 / 43 / 26 / 10 kHz; Kraus
  never / 511 / 223 / 74 / 36 / 12 kHz. Shipped scenarios that now warn (records
  unchanged): `fls_170k_traffic`, `fls_170k_transit` (+1.8 dB above the level),
  `fls_330k_transit` (+5.9 dB). `fls_120k_*` is 0.4 dB below it.
- `cap: bubble_saturation` on such an item (opt-in) takes the smaller of the law and that
  level.
Limits: literature only. The level is the thick-layer limit of the bubble term, not its
maximum (the term itself peaks 3.7 dB above it at an optical depth of 0.6 along the ray,
e.g. -25.4 dB against -29.1 dB at 27.5 kHz, 10 m/s, 10 degrees), and it says nothing about
facets near normal incidence. Above 70 kHz nothing that could be opened validates any of
the three models; the warning says so.

### D-88 Rock: the measured Lambert value by name; the type law warns

The type law `searay_bottom` with `bottom_type: rock` gives -16.9 / -7.2 / -3.2 dB at
10 / 20 / 45 degrees at every frequency. The brightest Lambert surface energy conservation
allows (mu = 1 / pi, -4.97 dB) gives -20.2 / -14.3 / -8.0 dB, so the law is above the bound
by up to 7.5 dB between 10 and 45 degrees (the literature pass found the same of the
handbook's own rock curve, arXiv:1601.06078, introduction). Measured rock is described by
Lambert fits of mu = -23 to -10 dB (`configs/sediments.yaml`, `rock`: -16 dB, our choice of
the middle of the published range). The owner was shown the recommendation to move
scenarios that use the type law for rock to that value and answered "Take your
recommendations". Done: the reverberation item `{boundary_z, sediment: rock}` (Lambert's
law with the table's value; `pebbles` likewise, flagged unconfirmed), and
`searay_bottom: rock` warns once that it exceeds the bound and names the alternative; the
law itself is unchanged for whoever names it. Scenarios changed: none. No shipped scenario
uses `searay_bottom` at all (the scene items of type `rocks` are objects with their own
`mu_db`, not this law). For anyone who moves, the level falls by 14.3 / 18.1 / 15.8 dB at
10 / 20 / 45 degrees. Limits: literature only; -16 dB is a representative value inside a
13 dB range of site fits, and smooth abraded rock is lower and not Lambertian.

### D-89 The default Lambert parameter is -29 dB, and says what it is

A reverberation boundary that gives neither `mu_db` nor a model nor a sediment now gets
-29 dB (`bottom.DEFAULT_LAMBERT_MU_DB`); it was -27 dB, written from memory as "the
Mackenzie value". The one open print found says -29 dB: Muggleworth, MS thesis, Naval
Postgraduate School 1994, hdl 10945/28529, p. 34, quoting Mackenzie (1961) for deep water
at 530 and 1030 Hz and 20 to 60 degrees. The owner was shown the recommendation and
answered "Take your recommendations". Limits: one secondary print; Mackenzie's paper was
not opened; and either number is a low-frequency deep-water value, not one for a sonar at
tens or hundreds of kHz (measured sands at 45 degrees and 100 kHz and above are near
-25 dB, mu about -22 dB). So the definition cites the print and says to name a sediment
instead, and the bare default warns once when used at 10 kHz or above. Same default in
`reverb_bistatic.directional_power` and `reverb_relief.directional_power`.
Consequence: a boundary that relied on the default is 2.0 dB lower. No shipped scenario
did: all 21 Lambert boundaries in `configs/scenarios` give `mu_db` (-27.0 in 18 of them,
by choice of their authors, unchanged). `tools/design_report.py` and `DESIGN.md` state
"-27 dB Lambert" as an assumed seabed for the SAS cell: an explicit assumption, not this
default, left as it is.

### D-90 `medium_sand` keeps its source's volume parameter

`configs/sediments.yaml`, `medium_sand`: `sigma2` stays 0.001, the value its source
tabulates for that sediment (arXiv:1809.08459, Table 1: -20.0 dB/m volume scattering
strength at 10.0 dB/m of attenuation). The grain-size rule (Snellen et al. 2011, Table II;
2013) gives 0.002 for everything coarser than 5.5 phi, and the same arXiv table agrees
with the rule for very fine silt, so the difference is real and known; nothing read says
which number the handbook holds for medium sand. The owner was shown the choice and
answered "Take your recommendations": the recommendation was to keep the tabulated value,
because a value printed for the sediment itself in a cited source outranks a general
rule. No number changes. Cost of the choice: the row sits 0 to 2.5 dB below its
neighbours' trend (most at 6 kHz and 45 degrees). The row's provenance and note now say
so; `sigma2: 0.002` beside the name gives the rule's value. Limit: literature only, one
print; a second print of the table would settle it (`REFERENCES.md` wanted list).

### D-91 The gain step derives absorption from the record by default

`{stage: tvg}` of a chain file and `recon.tvg_parameters` now take absorption from the
record unless told otherwise: Francois-Garrison at the record's centre frequency for the
temperature and salinity in `record.environment` (`absorption: auto`, the default). The
law was `40 log10 R` with no absorption, which is right at a few kHz and silently wrong
for the imaging sonars (`HARDCODED_AUDIT.md` K-4). The owner was shown the numbers and
answered "Take your recommendations". Evidence, at the far end of each sonar's receive
window in sea water at 12 deg C and 35 PSU: the old law was short by 51.4 dB for `fls_330k`
(307 m, 0.084 dB/m), 49.2 dB for `fls_170k` (465 m), 46.6 dB for `fls_300k`, 25.4 dB for
`fls_120k`, 21.9 dB for `fls_750k` (50 m), 34.1 dB for `sas_minsas120` (204 m), 17.8 dB
for `sas_auv`, and 0.0 dB for the 20 kHz sub-bottom sonar.
A record that states no temperature and salinity (every record made before this) does not
fail: it gets no absorption and one `RuntimeWarning` per process that names the missing
fields. `absorption: record` asks for the derivation by name and raises on such a record;
`absorption: none` is the old law exactly (a test compares the samples with the bare C++
call); a number is dB/m one way.
What did not change: `core.apply_tvg` itself (C++ and binding) still defaults to no
absorption, because the library has no model of the water; the Holoscan `PrepareOp`, whose
gain is off unless `tvg_db_per_decade` is given, calls it directly and keeps the old law (since D-151 through the shared step, `absorption: none`).
The 120 dB clamp stays as it is (a hardware question of its own): with absorption in the
law it is reached inside the window of `fls_120k` (273 m), `fls_170k` (237 m), `fls_300k`
(188 m), `fls_330k` and `sas_minsas120` (179 m), and the step's log entry says where and by
how much.
Consequence for shipped material: none. No file under `configs/chains`, `configs/deploy`
or `configs/scenarios`, no tool and no gallery scenario has a `tvg` step (searched for
`stage: tvg` and `apply_tvg`); the gallery applies its own display gain. A user chain with
a bare `{stage: tvg}` changes by 2 alpha R dB at range R.

### D-92 The culling pre-test allows for each ping's own trajectory residual

Backprojection with a field-of-view limit rejects a (pixel, ping) pair early when the
pixel cannot be in the beam, using the transmit-time geometry widened by what may change
before the echo arrives. One of those terms was a typed 2 cm "trajectory residual
allowance" (`HARDCODED_AUDIT.md` R-1). It is now, by default, the largest residual in the
ping's own trajectory table and never less than 2 cm (`recon::make_inputs`,
`trajectory_allowance = kTrajectoryAllowanceFromTable`; `None` or `table` in Python and
chain files). `fixed` (or 0.02) is the behaviour before, and a number of metres is taken
as given. The owner was shown the evidence and answered "Take your recommendations".
Why it is safe as a default: the allowance only widens a pre-test that is followed by the
exact per-element test, so it can add back pairs that test keeps and cannot add a wrong
one; it is never smaller than before, so nothing kept before is lost. Evidence: a ping
with a 1.3 m residual and a projector wider than the receive beam lost more than 20
beam-edge pixels out of 701 under the fixed 2 cm that the exact test keeps
(`tests/python/test_hardcoded_defaults.py`); the default now equals an allowance of 50 m
on that case, pixel for pixel. With equal transmit and receive beams the transmit test
cuts the same pixels and the 2 cm cost nothing.
Consequence for shipped material: none. No shipped scenario carries INS samples (0 per
ping in the eight SAS, rail and buried-object scenarios checked) and the gallery images
with `position_lookup: analytic`, so no table exists and the allowance is the same 2 cm:
`wreck_sas`, `minefield_sas`, `rail_objects` and `sas_minsas_survey` (the gallery
scenarios that image with a field-of-view limit) give the same numbers to every digit
before and after. The forward-looking scenarios set no field-of-view limit and do not run
the pre-test.
CUDA: the value travels in `BpGroup::traj_allowance`, which `make_inputs` fills on the
host and the shared kernel header reads; no `.cu` file and no kernel line changed for
this decision. Not compiled or run on a GPU here.

### D-93 The chain detects at the range resolution: `cell_statistic: multiscale` is the default

The live chain's detector (`libs/live` `detector.hpp`, `gpusonar.detect.multiscale`) tests every
range cell on its strongest sample and on its mean, each against the threshold of its own
statistic; `cell_statistic: mean` is the detector until now, line for line.
Why: the old detector applied the cell-averaging threshold for ONE exponential sample
(10.7 dB at 1e-5) to the mean of the 8 correlated samples of a 2 m cell. That mean exceeds
10.7 dB in noise about once in 1e21 cells, so the typed false-alarm probability was never
the one obtained; a point echo lost 7.7 dB in the same mean; and rows added in power gained
nothing, because the threshold on the mean did not move. Measured through the C++ chain on
simulated pings: a probability of detection of one half needed 19.7 dB of link-budget SNR
with one row and still 19.7 dB with three; the new detector needs 15.1 / 12.6 / 11.2 dB with
one / two / three rows (`live.budget_threshold_db`; closed form held to 50 pings a level).
Made the default because it passed every acceptance point on the eleven cached scenarios on
both engines: detections per ping equal to the true targets, the same number of tracks as
truth (the old detector had one extra track on `popoto_static_trial_3row` and
`flank_patrol`), every class at least as good (the 12 km stand-in is a `mover` from ping 3
with one, two and three rows; before it was `uncertain` with one row and not tracked with
two or three), no detection on 220 noise-only pings from either detector. Consequence:
every report line of a chain that does not say `cell_statistic` changes (`snr_db` 3 to 9 dB
higher, `extent_m` added to detections); `DEPLOYMENT.md`, "What `snr_db` is", lists before
and after per scenario. Limits: Gaussian noise is assumed for the thresholds; the scenario
set has no non-Rayleigh reverberation; the 12 km cases are the image-source stand-in.

### D-94 The detector's false-alarm budget is stated per ping; its thresholds follow from the map

`cfar.false_alarms_per_ping` (0.001) is shared between the tests and the map's independent
cells (beams across the sector x range resolutions along the window x 2.5 for searching an
oversampled map); the thresholds are the inverse tails of the statistics
(`libs/detect` `statistic.hpp`: Erlang for a sample of rows added in power, a weighted sum of
exponentials for a mean of correlated samples, by the saddlepoint formula), times the
cell-averaging correction for a background estimated from a finite window. `cfar.pfa` stays
the key of `mean` and `peak` only. Why per ping: a per-cell probability means nothing until
the number of cells is known, and the old 1e-5 was neither applied nor wanted (1e-5 a cell
is one false detection a ping on this map). Why 0.001: at a 20 s ping that is one false
detection in five and a half hours, and a track needs three in a gate; a decade costs
0.45 dB of threshold. The factor 2.5 is measured, not derived: on 60 + 160 noise pings of
the 12 km window, budgets of 3, 1, 0.3, 0.1, 0.03 and 0.01 a ping gave 3.3, 1.3, 0.35,
0.12, 0.044 and 0.006 with a factor of 2.0 to 2.5. Three rows ran 1.3 to 3 times over
their budget on 30 and 60 pings (0.22, 0.05, 0.033 a ping at 0.1, 0.03, 0.01) and gave one
false detection in 90 pings at the default budget: about a factor of two, not explained and
not corrected (the factor is one number for all row counts).

### D-95 One echo is one detection: sidelobes by their envelope, merging within what the tracker resolves

Before grouping, a flagged cell no more than 10 dB (`cfar.sidelobe_margin_db`) above a
stronger cell's sidelobe envelope is dropped: 1 / (pi x)^2 along range for a swept pulse, the
row's own Hann pattern averaged over the band along bearing, not below -40 dB
(`cfar.sidelobe_floor_db`), computed from the hydrophones the record has (a dead board's gap
raises it). A cell inside a stronger source's own core (its half-power beam, the next range
cell) is that detection's and is never dropped. Flagged cells are then one detection when they are within
`cfar.merge_m` in range (default: the tracker's range sigma, 15 m) and within half the
beam's half-power width in bearing. Why not a weighted replica: measured with the library's
pulse compression, Hamming costs 1.1 dB and widens the pulse 1.41 times for -36 dB
sidelobes, Hann 1.5 dB and 1.53 times for -31 dB; the envelope removes a 62 to 74 dB echo's
sidelobes at no loss. The price is stated: a weaker echo less than 10 dB above the stronger
one's sidelobe envelope is not reported (34 dB down at 20 m on the same bearing). Why merge
at the tracker's sigma: two detections nearer than that cannot be told apart by the tracker
anyway, and without it the arrivals of one multipath group (Bellhop, `flank_patrol`: three
groups 11 and 15 m apart for the small vehicle) became three tracks. Stated resolution: two
echoes are two detections beyond 20 m in range or one and a half half-power beamwidths in
bearing (`tests/python/test_detector.py`, `tests/cpp/test_detector.cpp`). The -40 dB floor
is an assumption about the array's gain and phase errors, not a measurement.

### D-96 The detector's background follows the slope where its window is one-sided

Near either end of the map the training cells all lie on one side, and reverberation falls
with range: the plain mean read 1.7 dB low at the near end of `flank_patrol`, which with a
threshold matched to the statistic was a row of false detections (9 to 14 a ping against 3
targets). The mean is moved by exp(slope x offset of the training cells' centroid), the
slope taken between the medians of the first (last) two training lengths of the same
bearing; the estimate's larger variance goes into the threshold. Medians, so that a target
in one of the two lengths does not turn the slope. Balanced windows are untouched. Only the
multiscale detector does this; `mean` and `peak` are as they were.

### D-97 Integration beyond the range cell is an option, not the default

The means of 4 and 16 cells (`cfar.integrate_cells: [1, 4, 16]`) gain 1.3, 3.3 and 4.8 dB
on echoes 5, 20 and 100 m long (Monte Carlo of the statistic, total energy at Pd one half)
and cost a point echo 0.2 dB. They are off by default because their thresholds are 4.0 and
2.1 dB over the background, which any error of the background reaches: on `flank_patrol`
they give 3 to 6 detections a ping and a fourth, `uncertain` track against 3 targets (the
bearings beside the submarine's 36 m echo, whose bearing sidelobes, measured at -28 dB, add
up over 32 m; 9 to 14 a ping before D-96). A target's highlights and
multipath arrivals are joined by the merge of D-95 instead, which costs nothing in
threshold. Limit: an echo spread evenly over tens of metres with no sample or cell standing
out is detected 3 to 5 dB later than it could be.

### D-98 Both engines take the CFAR window in cells from `detect::cfar_window`

`cfar.guard_m` and `train_m` are turned into cells by the library's one function (rounded
up, the guard at least one range resolution) in the C++ chain and in the Python one, in
place of each engine's own round-to-nearest. A window that is a whole number of cells does
not move: the eleven cached scenarios (2 m cells, 100 m and 300 m) gave identical report
lines on both engines before the detector changed. A window that is not moves up by a cell
(100 m over a 3.7 m cell: 28 cells, was 27).

### D-99 Left as they are in the detector package, with the measurement behind each

The Hann aperture window: it is 1.76 dB of the 2.6 dB the detector stands above the best
this architecture can do for a point echo; a Taylor window (computed for 32 hydrophones:
0.69 dB of loss at -30 dB sidelobes with nbar 4, against Hann's 1.76 dB at -31.5 dB) would
give 1.1 dB back, and needs the window in `flank.py`, the elevation estimate's noise weight
and the detector's envelope. The tracker's range sigma (15 m) and bearing sigma (1 degree):
the strongest detection of a
still target scatters by 1.1 m and 0.4 degrees or less from ping to ping on the five static
scenarios, the same before and after, so the data would support smaller ones; but the
sigmas also set the gate against a manoeuvre, the merge distance of D-95 and the
classifier's fit limit, and the scenarios hold no manoeuvring target to set them against.
Not derived from the cell and SNR. Training cells still include
other detections: an echo 100 to 400 m from a strong one is desensitised by it, as before.

### D-100 One environment per site, one schema for scenarios and build specs

The owner: "Remove hardcoded assumptions where we can" (audit G-1, G-2, M-1, P-1, P-2, P-3, P-9).
A scenario typed its site in up to eight places and the ray tracer's frequency beside the
sonar. Now `environment:` describes the site once (`python/gpusonar/environment.py`): inline,
a name under `configs/environments/`, or `{base: name, ...}`; propagation, time spreading,
reverberation items, ambient noise, Bellhop and absorption take their site values from it,
and Bellhop's frequency is the sonar's band centre. Choices made:

- **One schema, the build layer's.** The depth planner already had `water_depth_m`,
  `wind_speed_m_s`, `profile: {depth_m, speed_m_s}`; those names (with their units) are the
  schema, extended with the water (`temperature_c`, `salinity_psu`, `ph`,
  `reference_depth_m`), `absorption`, `shipping`. `depthplan` now delegates loading and
  checking; a scenario can name an entry of the build layer's list file (the three
  `popoto_env_*` profile scenarios and `popoto_12km*` do). The scenario layer's older
  spellings (`sound_speed`, `water_depth`, ...) are aliases of the same fields, not a second
  format: one field under two spellings with different values is an error.
- **A typed value that contradicts the environment is an error**, unless the scenario names
  the key under `environment_overrides` with a reason. An override that no longer differs
  warns. Rejected: letting the block's value win silently (today's failure), and forbidding
  typed values (a bank at another depth, a flat surface for the rays, are legitimate).
- **The absorption MODEL is not a site value.** A scenario's own `absorption` wins over the
  environment's without an override: comparisons run one site under several formulas. The
  water the formula is evaluated in is the site's.
- **The simulated site is not what the sonar knows.** `water_depth_known: false` keeps the
  depth out of the record (the chain reads `record.environment.water_depth` for elevation):
  21 shipped scenarios simulated a bottom the record never mentioned, and stay so.
  The live chain's `libs/live/environment.hpp` (the sound-speed ladder) is untouched.
- **The loader expands.** `load_scenario_config` returns the scenario with the site values
  written into the blocks (`expand=False`: as written), because about a hundred places in
  tools and tests read `cfg["environment"]["sound_speed"]` or `cfg["propagation"]["bottom_z"]`.
  To move the site of a loaded scenario, pass the change as an override to the loader.
- **Sound speed:** the value given; else the profile at `reference_depth_m`; else
  Mackenzie/Marczak from temperature and salinity; else `acoustics.NOMINAL_SOUND_SPEED`,
  the one definition of 1500 (about 60 bare literals in signatures now name it).

Every shipped scenario was converted with records unchanged (evidence in the report: the
explicit configuration of all 69 is equal before and after; record hashes on both engines).
Inconsistencies found were NOT resolved: each is kept through a commented override and
listed for the owner (D-101). Not reached by the environment: positions typed in `scene` and
`platform`, the `sediment` tier's `sediment_speed`/`interface_z`, a scene surface's own
`spectrum`. Limit: `max_range` of the ray grid defaults to 1.1 x the window's far edge
when left out; the shipped scenarios keep theirs.

### D-101 Inconsistent scenarios are kept as they were, each named

Converting the scenarios showed four kinds of inconsistency. None was fixed, because fixing
any changes records and each is the owner's call:

1. **A nominal sound speed beside a temperature that implies another** (audit P-3): 47
   scenarios through `coastal_12c` (1500 against 1496.8 m/s), `coastal_10c` (1489.8),
   `tank_20c_1480` (1482.3), `sst_asiaex` (1505.5), `sst_pursuit` (1482.4), and
   `popoto_elevation_rehearsal` (1490 typed). Geometry uses the typed speed, absorption the
   temperature. Kept by `overrides: {sound_speed_m_s: reason}` in the environment; a
   scenario without the mark gets one warning per process.
2. **Rays from a flat surface in a windy sea**: nine Bellhop scenarios use the wind for
   surface reverberation and noise but trace eigenrays with no surface loss. Kept by
   `propagation.wind_speed: 0.0` under `environment_overrides`.
3. **A bottom the vehicle is not told about**: 21 scenarios (the 15 Bellhop ones, the two
   `flank_sst_*`, the four `sst_*`) simulate a water depth that the record does not carry.
   Kept by `water_depth_known: false`.
4. **Two sands** (audit S-2): 1650 m/s, 1.9, 0.8 dB/wavelength in the Popoto and FLS
   scenarios; 1600, 1.8, 0.5 in the flank and Popoto-trial ones (Bellhop's in-code
   fallback). Kept apart by name: `coastal_12c_sand`, `coastal_12c_bellhop_sand`.

Recommendation, for when records may change: drop the typed 1500 (let the water give the
speed), give the rays the site's wind through `time_spread` rather than the coherent loss
alone, tell the vehicle the depth where a real one would know it, and pick one sand from
`configs/sediments.yaml` by name.

### D-102 Source levels are RMS; the square root of two is applied in one place

The owner's decision (audit P-5). A configured source level is dB re 1 uPa at 1 m RMS,
under a key that says so: `transmit.waveform.source_level_db_rms` (the build spec already
had `projector.source_level_db_rms`). `source_level_db` is a deprecated alias: read as RMS
with a `FutureWarning`, so nothing silently keeps the old meaning. Amplitudes are peak
values; `sonar.peak_pressure_at_1m` applies the sqrt(2) and nothing else does. A RECORD
keeps its own convention, stated in `core/record.hpp`: `source_level_db` and `amplitude`
there are peak values, so `make_record` writes the configured level + 3.01 dB and
`sonar.record_source_level_db_rms` goes back. Rejected: a `source_level_reference: peak |
rms` switch (two conventions for ever), and changing the record's meaning (old files).

What moves, by construction: every echo, reverberation and direct-arrival amplitude of a
calibrated sonar rises by 20 log10 sqrt(2) = 3.01 dB against unchanged noise (uncalibrated
sonars, which give `amplitude`, do not move). Overload figures rise by the same 3.01 dB
(nearest hydrophone of the three-row face: 19.1 to 22.1 dB over full scale; no overload
beyond 21.6 m instead of 15.3 m). The link budgets do not move: they always took the
number as RMS. The chain-versus-budget gap for a point echo in free field at 12 km went
from 12.2 dB to 9.1 dB (detector 35.5 dB, budget 44.6 dB): exactly the 3.01 dB, so that
part of the gap was the convention and the rest is the detector's (cell averaging and
aperture window, measured by the chain's owner at 7.7 and 1.8 dB). The shipped
`popoto_testauv_direct_overload_gain_switch` scenario keeps its -40 dB switch and now has
2.8 dB of headroom instead of 5.8 dB.

### D-103 Baseband noise has the spectrum and the array's coherence

The owner's decision (audit P-8, P-6). A baseband record's ambient noise was white at the
centre frequency's level and independent between hydrophones, so a row closer than half a
wavelength got the full 10 log N against sea noise. It now has the configured spectrum
across the record's band and the coherence the real-passband path has always had
(isotropic, sin(kd)/(kd) per frequency band; `coherence: surface` selects the Cron-Sherman
matrix that was in `sim/noise.py` unused; both paths take the option). The complex envelope
is drawn in the frequency domain with E|x|^2 = 2 x the integral of the spectrum, mixed
per band with a factor of the coherence matrix at the passband frequency.
`coherence: independent, spectrum: flat` is the earlier generator, sample for sample.
Alpha-stable noise and K noise without a correlation time stay white and independent (a
filter would change the distribution they are asked for); a K texture with a correlation
time multiplies the shared field, as in the real path. Measured against the closed form
N^2 / sum(Gamma) for 32 hydrophones at 5 kHz: 13.0 dB at the test AUV's 0.09 m pitch
(2.1 dB below 10 log N = 15.05 dB), 15.0 dB at 0.15 m (half a wavelength: unchanged).
Through the C++ chain with the shipped deployment files: the target's SNR on
`popoto_testauv_trial_1row` (0.09 m pitch) went from 53.2 to 50.8 dB (-2.4 dB; four pings,
one track either way); on `popoto_face_2row_shallow` (0.15 m) from 37.0 to 36.7 dB
(-0.3 dB). The record's own noise power rose 1.2 dB on both: the 8 kHz-wide baseband
record reaches down to 1 kHz, where the sea is louder than at the centre, and a flat draw
missed that; the matched filter removes most of it again.
Limit: `buildspec.directivity_db` already used the closed form in the budgets, so the
budgets do not move; `sonar_equation.SonarEquation.from_sonar` still takes 10 log N unless
it is given the gain.

### D-104 A bottom needs a description; no reflection coefficient is assumed

The owner approved changing the default (audit S-1). An image-source bottom with nothing
said about it reflected 0.5 at every grazing angle: a made-up number, 6 dB a bounce where a
sand bottom below its critical angle loses almost nothing. Now the bottom is the
environment's (`bottom`: a name in `configs/sediments.yaml` or `{speed, density,
attenuation}`: the Rayleigh coefficient at each ray's angle), and a `bottom_z` with no
description at all is an error that says what to give. An explicit `bottom_coefficient`
still works; a wall needs its `coefficient` likewise (0.8 was the same kind of number). The
C++ structs lost the same defaults in the core batch. No shipped scenario relied on the
default: `sst_echo_level` and `sst_pursuit` state `bottom_coefficient: 0.5` (SST's own
runs used a constant; kept and listed for the owner).

### D-105 Bounces and arrivals are kept by energy, with a cap that speaks

The owner approved changing the default (audit S-3, M-2). Multipath was cut at two bounces
(image source) or the strongest eight arrivals (Bellhop), silently. Left to the provider,
the cut is now by energy: bounce orders, or arrivals, are kept while one lies within
`arrival_floor_db` (20 dB) of the strongest at some field point, up to a cap
(`max_order_cap` 8, `max_arrivals_cap` 64). When the cap cuts something still above the
floor a warning says how much, and the provider keeps the figure (`truncation`,
`order_used`, `arrivals_used`, Bellhop's `dropped_energy`). A stated `max_order` or
`max_arrivals` is honoured as before. Why 20 dB: an arrival 20 dB down adds at most
0.9 dB coherently and 0.04 dB in energy; why a cap: a duct over a hard bottom has no end
(path pairs grow as the square of the arrivals). The order is chosen in Python for both
engines, on up to 256 of the field points. Every shipped scenario states its order or its
arrival count, so no shipped record moves. What the criterion would choose instead, at each
scenario's own targets (arrivals per leg; incoherent two-way level against the shipped
cut): the FLS transits (60 m, `max_order: 1`) 3 to 5 bounces, 7 to 11 arrivals, +1.3 to
+2.8 dB; the Popoto shallow scenarios (`max_order: 1` or 2) 5 to 8 bounces, 11 to 17
arrivals, +3.8 to +12.3 dB, three of them at the cap (`popoto_face_2row_shallow`,
`popoto_static_trial_2row`, `popoto_testauv_trial_1row`); `flank_sst_*` 6 bounces,
+3.9 dB; the SST cases 3 to 5, under +1.6 dB. Bellhop: `popoto_12km*` state 40 arrivals
and leave out 9.7 % of a point's arrival energy (the criterion: the cap of 64, 1.2 % left
out, +0.8 dB); `popoto_env_gravel_30m` 34 % (cap, 13.6 % left out, +2.4 dB);
`popoto_far_mover` 4.0 % (42 arrivals, +0.35 dB); the other ten leave out under 1 % and
would keep fewer arrivals than they state. So the shipped shallow-water Popoto records
under-state the multipath by 4 to 12 dB of incoherent level: the owner decides scenario
by scenario whether to drop `max_order` there.

### D-106 Late reverberation of a basin: an exponential tail only; what `energy_db` and `own_ping_db` mean

`basin_reverberation` (`sim/basin.py`) adds a late, diffuse tail behind each arrival. Only
one law is implemented, an exponential, with all five keys required and no defaults
(`energy_db`, `onset_s`, `decay_s`, `coherence_m`, `repeatable`).

- **Why only the exponential.** The law was fitted to the energy still to come against
  delay at the two hydrophones of the one measured data set in the long-range band (SFI
  Smart Ocean, 4 to 8 kHz, one fjord, one night; `docs/figures/basin_reverberation_fit.md`).
  The exponential fits to 0.5 to 0.8 dB rms (1.3 to 2.1 dB at worst) with the same decay at
  274 m and at 1020 m (0.145 to 0.162 s, 27 to 30 dB per second). A power law (three
  numbers) and two exponentials (four) fit the 274 m curve better (0.18 and 0.17 dB rms)
  but with an exponent of 11 to 14, which is an exponential with slight curvature, and at
  1020 m both converge to the exponential. Two more numbers for 0.4 dB at one hydrophone
  were not worth it.
- **`energy_db` is relative to the modelled arrivals, not a level of the basin.** The late
  field fell 18 dB from 274 m to 1 km where spreading alone gives 11 dB, so one basin level
  does not describe both hydrophones. The fitted values are -4.2 dB at 274 m and +0.2 dB
  at 1020 m; a scenario states which it uses.
- **`own_ping_db` has no default and is labelled a what-if.** It adds the basin's answer to
  the ping itself. The data were taken 274 m and 1 km from a source, not at one, so they do
  not give this level. Nothing in the repository measures it.
- **Assumed, not measured:** the tail's coherence across the array (the two hydrophones
  were 1.2 km apart), its elevation spread, and that an echo's tail is two one-way legs
  (2E + E² for a one-way share E).

Tests: `tests/python/test_basin.py`. Limits: an arrival before the window opens leaves no
tail; the projector's direct sound and the boundary reverberation of `reverb.py` get none;
the tail has no Doppler.

### D-107 The reach of a facet tail is a stated share of its energy, opt-in

`time_spread.tail_quantile` (default absent: taps span three mean decay times, as before).
With it the taps span the stated quantile of the tail's energy, computed from the shape's
own mean and variance; for one rough bounce each tap is an exact interval of delay, so
energy and mean excess delay are exact to 1e-9 without rescaling. Reason: against the
measured tail of the one clean surface bounce in the Smart Ocean data the shipped rule read
1.34, 2.05 and 0.46 times the measurement in three fits (the tail was cut too early for the
long window); with `tail_quantile: 0.99` the ratios are 0.74, 0.52 and 0.60, an error of
one kind that a single factor on the unmeasured surface slope would remove
(`docs/figures/facet_tail_smartocean.md`). Why opt-in: the reach grows 2.1 to 2.2 times, so
keeping the tap spacing needs 13 taps for six and four times the ray pairs between two
bounce rays; with `taps` unchanged there is no cost but coarser taps. Proposed and not
applied: `tail_quantile: 0.99` with `taps: 12` as the default.

### D-108 The coherent surface arrival wanders in phase; the Fresnel limit of 1 rad is ours

`time_spread.coherent_phase` splits the surface's height into long waves that heave the
reflecting patch and the rest, which scatters as before. The heave is applied as a change
of the coherent ray's delay and its rate, so the phase is proportional to frequency and
continuous within a ping and from ping to ping; energy is unchanged to 1e-9. Each wave's
effect turns from phase into amplitude with the angle
Θ = ρK²(cos²a/sin²g + sin²a)/2k (range along the path, wavenumber of the wave, its
direction, grazing angle); waves with Θ above `fresnel_limit` are left to the scattering
part, because their diffraction orders arrive as separate later arrivals. **The limit of
1 rad and the expression for Θ are our derivation, not from a paper**; they are tested
against closed forms (`tests/python/test_coherent_phase.py`), not against data. What
supports the model: phase variance 0.010 rad² with the measured 0.8 cm height against
0.0084 measured (a consistency check, since the height was read from the same arrival),
and the half-band estimator reads 1.42 on the model against 1.57 ± 0.07 measured. Not
verified: `legs: 1` (one-way sources) through the kernel. Phase is linear within a ping;
the curvature left out is computed separately in `tools/coherent_phase_study.py`.

### D-109 Depth from multipath timing: a pattern fit on the beam samples, one C++ stage

`chain.multipath_depth` (opt-in; `libs/live` `multipath_depth.cpp`, NumPy reference
`python/gpusonar/multipath_depth.py`) estimates a target's depth from the delays between
its direct and surface-reflected echoes.

- **A pattern fit, not arrival picking.** The detector merges arrivals that are close
  (D-95), so there are no separate arrivals to pick. For each candidate depth the whole
  two-way pattern (direct-direct, direct-surface, surface-direct, surface-surface) is
  correlated with the compressed beam samples; one statistic covers resolved and
  overlapping echoes.
- **A minimum delay × bandwidth of 3.** Below it the tilt of absorption across the band or
  a tapered replica fits the pattern as well as a second arrival does: a single 31 dB echo
  passed in a third of pings when 1 was allowed.
- **Carrier fringes within a ping, the envelope along a track.** Depth and surface move by
  more than a fringe between pings, so pings are combined on the envelope only.
- **One C++ stage for both chain engines**, writing its own report members, so the fields
  cannot diverge between engines.
- **The gradient prior goes into the depth's sigma**, and is treated as one error shared
  with the stacked-row depth when the two are fused. The surface echo's delay is not
  independent of the profile: a mean gradient g stretches it by about
  1 + r²g/(2c(z_own + z_target)).
- **A bottom that is not modelled is flagged (`bottom_unmodelled`), not guessed.**

Tests: `tests/python/test_multipath_depth.py`, `test_raytrace.py`. Numbers:
`RESULTS.md`, "Depth from multipath timing".

### D-110 Water that changes with range is Bellhop's own 2-D field, measured from the sonar

`environment.profiles_by_range` gives sound-speed profiles on one depth grid at ranges from
the sonar. They go to Bellhop as its SSP option `Q` with a `.ssp` file; nothing is
approximated on our side. Range is horizontal distance from the source of each eigenray
call, the same on every bearing, because that is how the provider's bathymetry section was
already defined and a second convention would be worse than a limited one. The cost: the
water is rings about the sonar and moves with the vehicle; a front lying across a map, and
horizontal refraction, are not modelled. The first profile must be at range 0 and is the
local profile. No value is filled in: differing depth grids, a grid that does not reach the
bottom, or ranges that do not increase are errors. Beyond the last range the last profile
holds, as a bathymetry's last depth does. With one profile the Bellhop input and the
records are byte-identical to before. Tests: `tests/python/test_ssp_range.py` (closed forms
for a constant range gradient and a front). The file format was read from the Acoustics
Toolbox documentation and reader (GPL; read for the input format only, nothing copied, run
as a separate process as before).

### D-111 The record carries the local sound speed; everything without a path uses the local profile

The record schema is unchanged: one sound speed, the local profile at the sonar's depth,
which is what the vehicle would measure. Apparent positions of eigenrays, the boundary
speeds of time spreading, the reverberation strength laws, noise coherence and absorption
use the local water. The provider's delay correction from a grid node to a field point uses
the speed at the point's own range. Tiers without a profile (`image_source`, `free_field`)
use the one local speed and warn that the change with range is not simulated. The depth
planner plans with the local profile and says so in its output. The provider's
rough-surface loss takes the surface grazing angle by Snell's law from the launch angle
through the local profile, which is not valid in range-dependent water; the three shipped
`ssp_*` scenarios therefore reflect eigenrays from a flat surface until that is replaced.

### D-112 One ray calculation for both depth estimators, as a fan; the stepped ray stays the default

`libs/live` `raytrace.hpp` traces closed-form arcs through a layered profile and reads one
fan of rays at the ranges wanted; arrivals at any depth come from cubic Hermite
interpolation along the wavefront (dt/dz = s/c, dt/dr = p), and where neighbouring rays
differ in reflection history the launch angle is bisected. Reason: the multipath estimator
needs the arrival pattern at thousands of candidate depths, and one fan gives all of them
with times good to nanoseconds in isovelocity water. The elevation stage can use the same
code (`elevation.ray_method: arcs`), but **the stepped ray remains its default**, because
unifying outright would move existing depths by up to 2 m. Multipath depth through rays is
`multipath_depth.rays: true`. Tests: `test_raytrace_cpp.py`, `test_refraction_rays.py`
(against an independent midpoint integration and against Bellhop).

### D-113 A depth is not searched where no direct and surface path exist; the reason is stated

With rays, a candidate depth that has no direct or no once-surface-reflected path on
either leg has no pattern and is not searched; a detection with no such depth reports
`no_direct_path` or `no_surface_path` instead of a depth. Reason: the flat-boundary pattern
fitted 384 m with a 1500 m sigma at 12 km in a water where no surface path exists. A depth
found while part of the column is in shadow is flagged (`multipath_note: shadow`, with the
ambiguity flag and the shadowed interval), because a target there is heard by other paths.
`multipath_depth.withhold_flagged: true` turns any flagged depth into no depth, with the
note as the reason and the fitted value kept as `multipath_candidate_m`.

### D-114 The coverage picture: its statistic, detection from a table, no budget defaults

`chain.coverage` (one C++ class for both engines; `python/gpusonar/coverage.py`,
`tools/coverage_map.py`) reports, per target class, the greatest range of detection, the
detection probability at listed ranges, where the direct path ends, blind intervals and
nulls, and the own depth that would gain most.

- **Statistic.** The direct and once-surface-reflected paths add as pressures (their
  interference is deterministic); everything that touched the bottom, or the surface
  twice, and what roughness removes from the coherent path add as power; all through the
  compressed pulse, peak over delay. This equals `lloyd.py`'s peak figure in open water
  (0.13 dB at most) and is cautious in ducts, where bottom paths are limited to three
  bounces.
- **Detection from a table** made at the first ping by `live.detection_probability`. No
  threshold is typed, and the C++ needs no chi-square code.
- **The budget numbers have no defaults in the chain.** Levels come from
  `buildspec.budget`; the member says `error` when anything is missing.

Not in it: change of the water or bottom with range, a rough bottom, caustic phase,
diffraction into shadows, reverberation as a limit on detection, the target's aspect.

### D-115 The profile ensemble is a set of hypotheses with a least and a greatest answer

`elevation.profile.ensemble: {gradients_per_s, extent_m, sigma_members}` continues the
unsampled water with each configured gradient over each configured extent and reports the
span of the answers (`depth_ensemble_m`, `depth_multipath_ensemble_m`,
`max_range_ensemble_m`). No probability is claimed, and there are no defaults for the
gradients and extents: they are statements about a site. Reason: a single tilt about the
vehicle cannot stand for a thermocline the vehicle has not seen. **It does not solve the
problem it was built for:** at 3 km the span contained the truth in no more cases than two
sigma did (`RESULTS.md`, "Depth by rung").

### D-116 The elevation search is limited to where rays can arrive from

`elevation.sector_from_rays: {bounces, margin_deg, max_depth_m}`: each elevation of the
search sector is traced back through the profile the chain holds and searched only if a
ray arrives from it. Reason: the three-row static array (rows at -0.75, -0.2, +0.75 m) has
far lobes only 4.2 dB down; in the deepening-thermocline scenario several arrivals shared
a range cell and a lobe at -31 degrees beat the true arrivals on every ping, the bearing
correction moved the detection by 1.1 degrees and no track was confirmed. The alternative,
a prior on the lobe, would need a water depth; this removes what cannot arrive and needs
none. `sector_from_depth` with the water depth given also fixes that case; raising
`ambiguity_db` only flags it.

### D-117 The link budget's threshold is the chain's own detector's

The budgets used a typed 13 dB at the summed map with a closed-form row gain on top. The
shipped detector needs 15.2 to 15.6 dB of one row's SNR for a detection probability of 0.5
on one ping, so every margin was 2.1 to 2.7 dB kind. The budget now states one row's SNR
and asks the detector what it needs (`buildspec.detection`, from
`live.budget_threshold_db` on the build's template record and deployment settings). The
probability is a spec field, `operation.detection_probability`, default 0.5; 0.9 is quoted
beside it. Rejected: a new typed number (it would drift again), and keeping a row gain in
the budget (the detector's figure already has the rows, so it would count twice).
Consequences: one long row at 197 dB has -1.0 dB at 12 km (detection probability 0.24),
two rows +1.5 dB (0.89), three +3.0 dB (0.99); 32 hydrophones in one row need 198 dB, not
195 dB. Measured against the chain, the budget is still 0.45 dB kind at 12 km (sweep taper
with the absorption slope 0.26 dB, noise slope 0.13 dB, row gain across the band 0.06 dB);
these are stated and not added. `SonarEquation.from_sonar` now takes the array's gain
against isotropic noise, which closes the limit noted in D-103. The general table of
`DESIGN.md` Sec. 5 for the imaging sonars keeps a 13 dB line, now named
`REFERENCE_SNR_DB` and described as a reference line, not a detector's threshold.
Tests: `test_buildspec.py`, `test_design.py`, `test_lloyd.py`. The margins and
probabilities above moved by about 0.1 dB when the detector's tests were recounted (D-134).

### D-118 Stacked-row deployment files search only where rays can arrive from (owner approved)

The owner approved the three deployment defaults D-118 to D-120 ("Sure I accept your
recommendation"). `elevation.sector_from_rays: {}` is written by the generator for every
build of two or more rows and is in the hand-written rehearsal file. Reason: D-116's case.
With the key the three-row static array in the deepening thermocline reads +1.6 to +2.0
degrees, bearing error -0.08 degrees, and tracks from ping 3, where it read -31 degrees
and tracked nothing. Fjord, front and rehearsal cases are unchanged to the printed digits;
per-ping time is unchanged. **Known cost:** at 12 km (`popoto_12km`, Bellhop) more pings
are flagged elevation-ambiguous (two rows 3 of 6 becomes 5 of 6, three rows 1 of 6 becomes
3 of 6) and the two-row track depth moved from 2.1 m to 34.5 m for a target at 4 m; the
stated depth sigma there is about 2600 m in both states, so neither depth carries
information. **It does not fix** the two-row face self-test (`STATUS.md`, open items; D-133 does).

### D-119 Multipath depth is deployed through rays, with flagged depths withheld (owner approved)

Any build that enables `multipath_depth` gets `rays: true, withhold_flagged: true`
(`buildspec.MULTIPATH_DEPTH_DEPLOYED`), with the spec's own keys laid over. No shipped
build enables it, so this lives in the generator only. Reason: the flat pattern returned
the water depth (30.3 ± 3.5 m) for a 15 m target in 30 m of water, a wrong depth with a
small sigma. Consequence: in that case no depth is reported at all (reason `window`); why
the ray mode answers `window` in 30 m of water has not been investigated. (It since has:
D-136. The answer is now `unresolvable`.)

### D-120 Coverage is on for the face and vehicle files; water depth and bottom are required (owner approved)

A build spec names its site (`operation.coverage.environment`); the generator takes water
depth, bottom and wind from it and refuses a site lacking any. A chain given `coverage`
without `water_depth_m` or `bottom` does not start, in Python or in C++
(`coverage_site_missing`). Reason: without them the picture is the deep-water one, and it
said a ship that the chain detects at 12 km in 100 m over sand could not be seen.
`water_depth_m` is a fallback for pings that carry no depth; what the ping carries wins
(the report says `water_depth_from: site` when the fallback was used). The shipped site is
illustrative (`shelf_100m_sand`: 100 m over sand, 5 m/s wind) and flagged `unconfirmed`:
the real site must replace it before a deployment. Coverage is on in the three face specs
and the hand-written `popoto_vehicle` and `popoto_elevation_rehearsal` files; the test AUV
and the static builds have none (no listed environment is 30 m deep). Cost: about 0.6 s on
the first ping, then nothing until it refreshes. Tests: `test_deployment_defaults.py`,
`test_coverage.py`.

### D-121 Addendum to D-102 and D-103: the loosened limit in the thinned `fls_170k_traffic` test

After source levels became RMS and the baseband noise took its spectrum, the anchored
boat's track speed in the thinned test form of `fls_170k_traffic` read 1.9 m/s where it had
read 0.10 m/s, and the test's limit was loosened from 0.6 to 2.5 m/s. The cause is not a
level effect alone: the 9 m hull gives two clusters 4.2 m apart and so two tracks; on the
second of four pings the weak cells between them cross the threshold and the two become
one detection at their centroid; on the third ping the two tracks hold each other's
cluster, and 4.2 m in 2.25 s reads 1.9 m/s. Either change alone (the 3 dB of level, or the
new noise) tips the cells between the clusters over the threshold. A fix is proposed and
**not applied**: merge detections closer than the tracker can keep apart before tracking
in `run_fls_transit`, as the long-range chain does (D-95). It changes detection and track
counts in every FLS scene and needs the full gallery re-run against `gallery.json`. The
loosened limit stands until then and is the owner's call. (Measured since on the six full
scenes and left off: D-135.)

## Choices made between D-82 and D-105 that were recorded only in hand-back reports

These were made while D-83 to D-105 were being written and reached the code and the
other documents, but not this file. They are recorded here from the reports of the work
packages, in the order of the work. Where a later decision changed one, that is said.

### D-122 Device stages are templates over an executor; reverberation and beamforming on the device

- Device stages that are sequences of whole-array steps (the batched reverberation, the
  far-field and MVDR beamformers) are templates over an executor
  (`sonar/dsp/exec_host.hpp`, `exec_cuda.hpp`), so the device formulation is tested on the
  host; the reference host code stays untouched and bit-identical.
- Reverberation on CUDA runs from the random deviates on; NumPy still draws them (D-61),
  now on threads by aligning jump-ahead chunks on the true sequence, bit for bit.
- MVDR on the device accumulates covariances in double precision (the reference uses
  single); a covariance that is not positive definite sends the whole call to the host.

### D-123 One-way sources, near-specular removal, noise statistics, the bubble-loss law

- **One-way sources (`sources:`).** Pulses use the echo kernel through a constant 1 m
  first leg that moves with the source: exact, no C++ change, CUDA for free. Continuous
  signals (noise, tonals, files) use a NumPy streamer on both engines. A passive capture
  is `pings.transmit: false`. The relay point that stood in for a one-way source in
  `sst_asiaex` (D-78) is retired.
- **`exclude_specular` is opt-in and bistatic only.** Double counting exists only when the
  transmitter's pulse is also a one-way source over time-spread bounces.
- **Noise statistics.** K-distributed noise per sample as TR 0702 has it, and as a compound
  texture with a correlation time, common across hydrophones by default because that
  preserves the coherence matrix. Alpha-stable noise is white, its level defined by scale.
- **Bubble loss.** `apl_1994` is adopted on the owner's ruling as the recommended law
  (the formulation as arlpy and UnderwaterAcoustics.jl implement it, with their
  attribution); the `hall_novarini` laws are kept as physical models and the difference is
  recorded as model disagreement.

### D-124 Noise fitted to a recording; validation against measured channels

- **Fitted noise** uses the closed-form spectral factor of the Welch cross-spectral matrix
  rather than an optimiser, so the estimator's variance is known. Heavy tails are a
  texture on white innovations filtered in the time domain, so spectra and coherence do
  not depend on the tail. Signals are excluded by rule (long or periodic events, lines).
- **Channel validation** fits one free parameter (the wind) to one measured number per
  recording and tunes no option; the comparison is made through identical estimators. The
  data sets state no sea state, so every sea parameter is an assumption.

### D-125 The projector's direct sound and receiver overload in the simulator

- The direct sound is a one-way source at the projector's true position, so near-field
  geometry per hydrophone, clocks, sampled waveforms and both engines come from the tested
  kernel path. An omnidirectional projector is the pessimistic choice until a baffle
  pattern is measured.
- The overload model is behavioural and has **no hardware defaults**: two limiting stages
  separated by the coupling capacitor, because which stage clips first depends on where
  the gain sits and the capacitor is what turns a clipped burst into a long offset.
- The front end is run from the transmit instant, not from the opening of the window.
- Truth about overload stays out of the record; a reported gain switch uses the existing
  gain-ramp fields. The schema is unchanged.

### D-126 The surface image in the link budget, own depth, target classes

- The surface-reflected path is a factor on the free-field budget that depends on both
  depths, range, band, roughness and bubbles, not a fixed number of dB; detection uses the
  peak of the compressed coherent echo (`lloyd.py`).
- The rms wave height is its own input; the wind law is only its default.
- The depth planner recommends on the worst margin over a target class's depths at the low
  end of its target strength.
- Target classes and environments are planning inputs. They change no generated sonar or
  deployment file; each is marked assumed or illustrative, and the validator warns until
  confirmed. Every target strength in the repository is a placeholder.
- The platform's depth limit is 500 m (the owner); recording raw samples and a GPU for
  stacked rows are the defaults in the build specs (the owner's decisions).

### D-127 Build specs, row modules and the staged Popoto builds

- Stage configurations are generated from build specs; hand edits to generated files are
  not kept. One source per build, checked by a test (`--check-shipped`).
- `unconfirmed` flags accumulate through `base:`; only `confirmed` removes one.
- A row module is a concept of the spec: the long row for the static array and the face,
  the test AUV's short row the exception; "a smaller array" on the face means fewer rows.
- The static array's rows sit at the face's heights (outer rows ±0.75 m, middle row at
  -0.20 m, the lowest wideband repeat for the 1.5 m baseline). This supersedes the first
  study's ±0.90 m and -0.35 m, which were for another geometry. No frame dimensions were
  given; matching the face makes the frame a rehearsal of the face.
- A static frame without a tilt sensor carries `roll_sigma_deg: 1.0`, a placeholder that
  makes the reported sigma own up to an unknown tilt.
- The projector is placed on the face at [4.0, 0, 0], a placeholder.
- The receive window is stated per sonar file from the pulse's blind range.
- Self-test tolerances: 3 m in range in free field, 6 m with boundaries; 0.7 degrees of
  elevation with boundaries. The `popoto_face_2x32` rehearsal self-test is left failing
  rather than tuned (`STATUS.md`, open items).
- The prototype keeps 4 to 6 kHz with a 0.1 s pulse: the same band and chain as the full
  face; a 1 s pulse would hide everything inside 750 m.

### D-128 Stacked rows: how elevation is estimated and reported

- A repeat of the elevation pattern within 1 dB of the best is never trusted, and further
  repeats are weighed against noise. With two rows the chain therefore reports a list of
  possible elevations (the rows 1.5 m apart repeat every 11.5 degrees) unless water depth
  rules the others out.
- The possible elevations go to the track; tracks do not start on a guess.
- An ambiguous elevation never moves a detection: the bearing correction is applied only
  to unambiguous elevations.
- Attitude is taken at the middle of the received pulse.
- Refraction is a sigma term kept out of the track filter: it is systematic, so averaging
  pings must not shrink it.

### D-129 Stage 2 of the Popoto chain: decimation, the sound-speed ladder, blanking, navigation

- The synchronisation reference lies above the acoustic band, so when the chain decimates
  first only the stretches holding the two reference bursts are kept raw and the
  correction is applied at baseband. The default decimation factor is 17; 25 costs 0.2
  degrees of bearing.
- The sound-speed ladder (`elevation.profile`: supplied, built, family, local, prior) keeps
  its state in the record's existing fields and `extra_json`, so a replay equals the live
  run with no schema change.
- The profile's error is treated as one tilt about the vehicle, capped by the gradient
  prior, with the depth sigma as the linear sensitivity to it. The first form (the spread
  of two fully tilted rays) under-reported once rays bounced (167 m of error against a 5 m
  sigma) and was replaced.
- There is no default for the sound-speed sensor's accuracy: it is a datasheet number, and
  the `built` rung refuses to work without it.
- A board timed from one reference burst is flagged and the ping is not `ok`: its drift is
  unknown.
- Blanking crops the map rather than zeroing it: zeros would bias the CFAR's training.

### D-130 Passive tonal analysis (Goertzel lines and DEMON)

- Tonal analysis is a separate stream analyser hooked into the passive report, not part of
  the passive bearing estimate: it needs its own time scale (1 s blocks, 8 s segments
  against 0.26 s frames) and state across stretches. Frames without a result are
  byte-identical to before.
- A Goertzel line on beams is a Goertzel filter on each hydrophone followed by narrowband
  phase steering. Frequency comes from a two-bin Hann interpolation in one pass (1.7 to
  1.8 times the Cramér-Rao bound measured); phase is reported at the middle of the block.
- The DEMON envelope is the band power of short Hann blocks in the frequency domain, where
  steering is cheap. Rows are steered separately and their powers added. A line is
  attributed to the bearing where it is strongest.
- The threshold follows from a false-alarm rate (default 1e-6, because one result has
  about 50,000 cells). Shaft rate is the highest fundamental explaining at least 0.9 of
  the best score; the blade count is a heuristic and flagged as one.
- There is no CUDA path. The formulation was read from arlpy and SignalAnalysis.jl and
  re-implemented (`OPEN_SOURCE.md`).

### D-131 Hard-coded assumptions: what a forgotten value does

Following `docs/HARDCODED_AUDIT.md`:

- Facts with no physical default (sound speed, boresight, reflection coefficients,
  sediment, bounce count, solver scales) are NaN or unset in the C++ structs and rejected
  at entry points. A forgotten value is an error, not a silent 1500 m/s or 6 dB a bounce.
- Fixed capacities either raise or report (absorption table coverage, arrival passes, gain
  clamp, image-series cut). A caller who states arrival passes gets an error; default
  passes are raised with a warning.
- Sizes the chain derives from the sonar are a standing value with a floor from the sonar
  (`libs/live` `derive.hpp`).
- A report too long for a datagram is split into complete lines, each marked
  `"part": [k, n]`.
- The sample word length of the board stream stays configuration until a change of the
  block header is approved (an open question for the owner).
- A track is called multipath only against a track already classed as an object.
- The chain's 2 m range cell of mean power, kept at the time of this audit, was
  superseded by the multiscale detector (D-93).

### D-132 Sediment table and literature models

- Reflection-only sediment rows use values printed in an open MIT package
  (UnderwaterAcoustics.jl) with that package's attribution to TR 9407; the handbook itself
  stays unread. Rows keep the source's units and reference water and the loader forms the
  ratios. Backscatter models refuse rows without roughness and volume parameters rather
  than borrow them from another sediment.
- Scattering parameters are the published grain-size defaults, applied at the grain size
  an open thesis table gives each name: class defaults, not site values, each parameter
  with its provenance. `medium_sand` keeps its source's volume parameter (D-90).
- Rock and pebbles get a Lambert parameter from measurements, not a forced fluid fit
  (D-88). The sandy-gravel sound speed is corrected on three independent prints against
  one.
- `sea_state` is resolved once in `run_scenario`; whole states only; a conflict with
  `wind_speed` is an error. Thorp absorption is TR 0702's two-term form and warns above
  100 kHz.
- The bubble-layer surface backscatter model takes its optical depth from our loss laws
  and omits the small-scale roughness term, because those two pieces of the published
  model are not printed anywhere open; it is documented as a lower bound between 30 and 70
  degrees (D-87).
- `small_scale_kh = 0.5` is now cited to a published criterion rather than "our choice".
- No default was changed on literature evidence alone; the candidates are in
  `REFERENCES.md` Sec. 11 for the owner.

## Six fixes after the first whole-suite run

Numbers and tables are in `RESULTS.md`, "Six fixes after the first whole-suite run";
derivations in `PHYSICS_NOTES.md` Sec. 29.

### D-133 The depth sigma of stacked rows says when boundary echoes share the range cell

An echo that goes out by one path and back by another has the same length as its mirror,
so the two add as pressures and the rows read the mean of the two elevations. Direct with
surface reads the sea surface; direct with bottom reads the sea bed, whatever the target's
depth. The two-row face at 3 km in 100 m of water read 98.5, 97.0 and 11.0 m for a target
at 60 m with a stated sigma of 11 to 15 m. `elevation.multipath_sigma: {bounces,
max_depth_m}` is opt-in in the chain and on in every stacked-row deployment file (the
generator writes it; the hand-written `popoto_elevation_rehearsal` has it). For every
target depth the water allows, the chain's ray fan gives the echoes, the clump the stage's
own first-arrival rule takes, and how well that depth explains the reading. The spread of
those depths about the reported one, less what the angle's sigma already holds, goes into
`depth_sigma_m` of the detection and of the track (`depth_unresolved_sigma_m`), and
`depth_multipath_limited` is true where that term is the larger part. One C++ function
(`libs/live` `multipath_depth_limit`) serves both engines. The same three pings now state
25.0, 24.4 and 41.0 m (1.54, 1.52 and 1.19 sigma of error). The generator's self-test
judge changed with it: it looks at every ping, asks for the truth within two stated sigma
and for the multipath term to stay bounded (by the water column with boundaries; by
3 m + 0.3 % of range in free field). All six builds pass 7 of 7 on both engines.

Rejected:

- Correcting the depth. A reading at a boundary holds no target depth to correct.
- A fixed allowance in the self-test (0.7 degrees stood there). It passed or failed by
  luck.
- Feeding the term to the track filter. The same boundary is read ping after ping, so
  averaging must not shrink it (as for refraction, D-128).

Limits:

- Both boundaries are perfect mirrors. A soft bottom or a rough surface weakens the pairs,
  and the term is then too wide.
- Too wide by three to five times for shallow targets (10 and 30 m) at 1 km with two rows:
  22 to 38 m stated against errors of 0.4 to 6.3 m.
- Without a water depth there is no bottom in the picture; the report says so
  (`depth_unresolved_assumed`).
- The `shallow` self-test of the two-row builds passes with a 118 m sigma that comes from
  the attitude-gap term (the scenario has no attitude history), which was there before.
- Not checked against Bellhop in refracting water. At 12 km the depth sigma is still 2960
  to 3010 m (D-118's case).

Tests: `test_multipath_sigma.py`, `tests/cpp/test_elevation.cpp`.

### D-134 The detector counts its tests as measured

The thresholds follow from a false-alarm budget per ping (D-93), so the number of
independent tests on the map must be known. It was one flat search factor of 2.5, measured
with one row. Measured on 400 noise-only pings per row count, each test alone, that factor
ran 0.86, 1.16 and 1.24 times the budget for one, two and three rows (pooled over budgets
of 3 to 0.1 a ping), and every row count ran further over as the budget tightened (three
rows: 1.08 at 10 a ping, 1.60 at 0.1). The count now grows with the threshold: in
proportion to u = rows x looks x threshold for the single sample, and by Rice's rate along
bearing for the cell mean. One constant per test and row count is measured:
`kSearchSample` = 0.191, 0.208, 0.192 and `kSearchMean` = 0.515, 0.532, 0.598 for one, two
and three rows (`tools/detector_search_factor.py` regenerates and verifies them). The
looks of a cell mean are taken from the replica the chain compresses with: 5.48 in a 2 m
cell, not 5.80 (5.42 measured on the map). The whole detector then gave 0.79 to 0.96 of
the budget on 300 other pings per row count at 3, 1 and 0.3 a ping.

Consequences: the one-row budget SNR each build needs rose by 0.11 to 0.15 dB (about
0.02 dB of it the looks). At 12 km the margins are -1.1 / +1.4 / +2.9 dB for one, two and
three rows (they were -1.0 / +1.5 / +3.0, D-117) and the detection probabilities 0.22 /
0.87 / 0.99 (were 0.24 / 0.89 / 0.99). Reach moves by 0.1 km at most. No deployment file
changed. The thresholds are in `buildspec.detection`, so the Python engine, the budget and
`detection_probability` follow the same plan.

Rejected: keeping the flat factor with a constant per row count. It would still run over
as the budget tightens. `cfar.search_factor: 2.5` restores the old count.

Limits:

- Nothing is measured at the default budget (one false detection in a thousand pings; it
  would take about 100 000 pings). The thresholds there rest on the growth law beyond the
  measured range.
- The three-row cell-mean test does not follow the law: its constant rises from 0.55 to
  0.73 as the budget tightens from 10 to 0.1 a ping. The cause is not found. At the
  default budget that test could run about 1.5 times over its share.
- The constants were measured on one map only (long-range array, bearings every 0.5
  degrees, 2 m cells of 8 samples, Hann window, 15 m merge). Every chain using
  `multiscale` now takes them; another map needs the tool run again.
- Thresholds still assume Gaussian noise.

Tests: `test_detector.py`, `tests/cpp/test_detector.cpp`, `test_buildspec.py`.

### D-135 Addendum to D-121: the FLS merge was measured on six full scenes and is left off

The fix D-121 proposed exists: `detect.group_flagged` is the multiscale detector's
grouping rule as one function, and `detect.cluster(merge=...)` uses it with distances from
the chain's own plan (1.9 m in range, 0.73 degrees). It is behind
`gallery.FLS_MERGE = False`. In the thinned `fls_170k_traffic` test it cures the anchored
boat (1.92 m/s becomes 0.29 m/s). On the six full scenes (CPU, one simulation each):

| Scene | Tracks: `gallery.json` / head / merged | Named objects |
| --- | --- | --- |
| `fls_120k_transit` | 7 / 7 / 6 | pinnacle 0.62 / 0.53 / 0.03 m/s; others unchanged |
| `fls_170k_transit` | 4 / 4 / 3 | anchored boat static / uncertain / static (0.03 / 0.14 / 0.02 m/s) |
| `fls_330k_transit` | 2 / 4 / 3 | anchored boat not tracked / 1.68 m / 2.29 m; pinnacle not tracked in all three |
| `fls_170k_traffic` | 11 / 13 / 8 | anchored boat 0.16 / 0.18 / 0.18 m/s; false movers 3 / 2 / 0 |
| `fls_120k_cluttered_seabed` | 65 / 64 / 36 | line tracks 15 / 16 / 0; false movers 4 / 4 / 7; anchored boat 0.02 / 0.03 / 0.32 m/s |
| `fls_120k_rough_surface` | 57 / 56 / 52 | boat under way 4.72 / 4.72 / 8.08 m/s (true 5.0), mover / mover / uncertain |

The merge harms two of the six: the cluttered seabed loses every line track and gains
false movers, and under the rough surface the boat under way reads 8.08 m/s and is no
longer classed a mover. The thinned `fls_120k_transit` test also fails with it
(pinnacle 1.6 m/s against a 1.2 limit). So it is left off and the loosened 2.5 m/s limit
of D-121 stays. A new test runs the thinned traffic scene with the switch on and holds the
anchored boat under 0.6 m/s. Turning it on, with those losses, is the owner's call.

### D-136 Addendum to D-113 and D-119: `unresolvable` through rays; the `window` answer was a NaN

D-119 left open why the ray mode answered `window` for a 15 m target in 30 m of water. It
was a bug, and behind it a real limit. The ray pattern's depth grid is laid in whole
carrier fringes and ran past the sea bed; the last depth has no path, its surface delay is
NaN, and that delay sized the beam window. The window now takes the longest finite delay.
With that fixed the stage fitted the 0.6 m of water left and gave 29.8 m for the 15 m
target. In that trial the surface echo separates only from 29.3 m down, the sea bed is at
30 m, and the depth step the delay can resolve is 9.8 m. So through rays a searched band
thinner than one resolvable depth step answers `multipath_reason: unresolvable`, with
`multipath_searched_m` clipped to where paths exist ([29.3, 29.9] m here). The answer is
the same in both waters and with or without a bottom model. The flat pattern is unchanged.
Rejected: reporting the fitted 29.8 m with a flag; a band that thin cannot tell one depth
from another by delay. Limit: for that geometry no depth can be measured by timing at all;
the range and the two depths set it, not the estimator. Test: `test_refraction_rays.py`;
`docs/figures/ssp_range_study.md` was regenerated and only those two rows changed.

### D-137 The trial, 12 km and environment scenarios keep arrivals by energy (owner approved; closes D-105's open question for these)

`popoto_face_2row_shallow`, `popoto_static_trial_2row`, `popoto_testauv_trial_1row`,
`popoto_12km`, `_2row`, `_3row` and the four `popoto_env_*` no longer state `max_order` or
`max_arrivals`; the generator's trials do not either. The `ssp_*`, fjord and remaining
scenarios keep their stated counts. Consequences accepted with it: the three image-source
trials, `popoto_12km*` and the gravel scenario stop at the cap with its warning; a
detection is 58 to 142 m long and `range_m` is its energy-weighted mean, so the generator's
`shallow` and `rehearsal` self-tests check the first arrival; two rows 1.5 m apart at
2.6 km no longer give one elevation, because later bounces arrive from up to 17°, and
`popoto_static_2x32` fails its self-test on that. The failure is left standing as a known
wrong rather than removed by stating one bounce again. `test_multipath_sigma` states
`max_order: 1` itself: its subject is the three first-order paths.

### D-138 Enclosed water is a named option of the build spec: `operation.site_kind: enclosed` writes `min_snr_db: 15` (owner approved)

There is no fjord or harbour deployment file, so the setting is a key of the generator and
open-water builds do not move. `min_snr_db: 15` over the short CFAR window because, on the
fjord scenarios as shipped, both keep the target on 4 pings of 4 with one, two and three
rows, and the first leaves 1.5 / 1.5 / 0.2 other detections per ping and no track off the
target against 6.0 / 0.5 / 1.8 and two. Known cost: under the own-ping what-if the short
window finds a 10 dB target that this setting loses (three rows at 150 m; one row at 1 km
on 3 of 4 pings), and three rows confirm one ping later. The short window stays available
through `operation.chain`. `elevation.ssp` from a cast is site data: a warning asks for it,
nothing is generated.

Note after D-141: with the tracker's most-probable-first association the three-row track
confirms at ping 3 under `min_snr_db: 15` as under the other settings; the cost of one
ping stated above no longer holds (`docs/figures/fjord_rehearsal.md` at `e10b17f`).

### D-139 `elevation.ray_method: arcs` is the default (owner approved; supersedes the last sentence of D-112)

Both engines trace the stacked rows' depth with the chain's one ray calculation; `steps`
remains by name. Measured on the same records: 0.003 m in the generator self-tests, at
most 0.6 m in the shipped scenarios (12 km), no test expectation moved. The 2 m of D-112
was an estimate for the deployed gradient; in stronger downward refraction at 12 km the
two differ by up to 10 m, and the stepped ray leaves an exactly level ray unbent. The
library functions keep their own defaults.

### D-140 Scenario sound speed follows the water (owner approved; D-101, first clause only)

44 of the 47 scenarios take Mackenzie's or Marczak's speed for their temperature and
salinity (1496.8, 1489.8, 1482.4 m/s); `tank_20c_1480` is removed. Three keep a typed
speed because it is the intended quantity, each saying why in its `overrides`:
`popoto_elevation_rehearsal` (1490), `sst_pursuit` and `sst_asiaex` (1500, D-78). The
speed is the surface value (reference depth 0), as the one rule gives it. The other kinds
of D-101 (flat surface, unmentioned bottom, two sands) are unchanged. The generator's own
trial scenarios and the link budget still use the build's nominal 1500 m/s.

### D-141 The tracker shares detections out most probable pair first (owner approved; both engines)

`gpusonar.detect.Tracker` and `libs/live` `Tracker` now take the (track, detection) pairs
inside the gate in order of the Gaussian negative log-likelihood d² + ln|S|, confirmed
tracks before tentative ones, each track and each detection used once; ties go to the
earlier track, then the earlier detection, the same in both engines. Before, each track in
turn took its nearest gated detection by d², so a track earlier in the list could take a
detection that fitted a later one better. The old rule stays by name:
`tracker.association: in_order` (`tracker_association` for `sonar_live`).

Why: in `fls_170k_traffic` new tracks climbing the crossing boat's wake (detections fixed in
the water 5 to 10 m apart; a new track's speed sigma of 6 m/s lets it reach the next one)
took the hull's detection from the hull's own track. The boat was right on 2 of 7 seeds;
with this rule on 7 of 7 (RESULTS, "Why the tracker loses the crossing boat"). d² alone was
tried and rejected: a track with a wide covariance looks close to everything and took the
head-on boat's detection on one seed; ln|S| charges it for its width.

Known cost, measured on saved detections of full-size scenes before the change: in
`fls_120k_rough_surface` more tracks among the wave glints are called movers (22 to 34 over
three seeds) and the boat is right on 1 of 3 seeds (2 of 3 before). In the cluttered
seabed movers that are not a boat fall from 17 to 2 over three seeds; the three transit
scenes are unchanged but for one anchored boat (`static` to `uncertain`, one seed). The
12 km ship's misclassification is not changed by any rule tried: over six seeds its own
track is a `mover` on 2 with either rule, mostly classed as an echo of a stronger track (at
the gallery's seed it is now `uncertain`; open item 44). Rejected: an optimal one-to-one assignment (Hungarian), which kept
the rough surface even but added false movers and misclassed anchored boats in the
crossing-boat scene.

### D-142 The shallow trials keep bounces until the next is 20 dB down: cap 16; the three-row trial too (owner approved)

At the default cap of 8 bounces the energy rule (D-105, D-137) stopped while the next order
was still 4.1, 4.7 and 18.0 dB below the strongest arrival in `popoto_face_2row_shallow`,
`popoto_static_trial_2row` and `popoto_testauv_trial_1row`. These three scenarios and the
generator's own trials (`buildspec.scenario_configs`) now set `propagation.max_order_cap:
16`: they keep 14, 12 and 9 bounces and no series is cut. The default cap stays 8 for every
other scenario. Cost: the two-row static trial takes 26 s a ping to simulate (13 at cap 8).
`popoto_static_trial_3row` no longer states `max_order: 1`: the energy rule keeps 5 bounces
there, under the cap, with the same detection on 4 of 4 pings and a depth of 16.9 m (17.2
with one bounce).

What moved with it:
- `popoto_static_2x32` passes its `shallow` and `rehearsal` self-tests again: its depths
  are within two of a stated sigma of about 47 m. That sigma is wide, half the water column:
  honest, not informative. `popoto_face_2x32`'s `rehearsal` failed instead, on the judge's
  angle rule (D-143). With D-143 all six builds pass 42 of 42 on both engines.
- The two-row static trial no longer gives candidates: one elevation, -3.42 degrees, read
  as a sea-bed bounce and folded to 13 to 14 m for a target at 60 m, sigma 123 m.
- `popoto_face_2row_shallow` reads 73.7 to 92.6 m for 60 m. With the multipath term of the
  sigma (D-133, on in every stacked-row deployment file) it states 18.7 to 20.0 m (within
  1.63 sigma); `test_elevation` now runs that case with the term on.
- In the two-row static trial the decimated and full-rate chains differ by 0.12 to 0.13
  degrees and up to 6 m on the bounce reading (a tenth of the stated sigma); for detections
  not on the direct path `test_decimate` now allows a fifth of the stated elevation sigma
  and a tenth of the depth sigma. Direct-path detections keep the fixed tolerances.

### D-143 The self-test judges an ambiguous elevation by depth, as it judges any other (owner approved)

`tools/build_from_spec.py` `_check_depth` asked, for an ambiguous detection, for the direct
path's elevation within two sigma of one candidate ANGLE, and for an unambiguous one for the
true depth within two sigma of the DEPTH. A candidate angle is an arrival angle, and with
boundaries an arrival may be a bounce whose fold is the right depth: the face two-row
rehearsal failed with a reported depth of 35 m +- 42 m for 60 m because its angle, -3.9
degrees, is a bounce's and not the direct path's. Now each candidate angle is folded into a
depth as the chain folds the reported one (the straight line down the detection's range from
the ownship's depth, `elevation.fold_depth`), the chosen candidate keeps the chain's own
depth, and the truth must be within two stated sigma of one. Recorded with it: each
candidate's depth, the old angle error, and how far the recomputation is from the chain's
depth for the chosen candidate: at most 1.6 m over the 54 ambiguous detections of the six
builds (the array's origin is not the ownship's depth), 0.005 m in free field. In the face
two-row `shallow` and `rehearsal` runs one candidate folds to 60.7 and 64.0 m for a target
at 60 m: the chain offered the right depth, and the angle rule failed it. Limit: in water
with a profile the alternatives are folded along straight lines, not traced rays.

### D-144 The TR 0702 comparison names SST's Lorentzian Doppler spectrum

`tools/sst_validation.py` runs the ASIAEX reproduction with TR 0702's own rules named, so
that later changes of the suite's defaults do not move it; its comment listed the Doppler
spectrum among them, but only `decay_time: rule_of_thumb` was named. From D-84 on, the
ASIAEX runs took the Gaussian default, and the matrix moved without anyone choosing it: the
surface's ping-to-ping variation 1.88 to 2.85 dB, the sum-against-convolve figures -2.3 /
+3.7 to -2.8 / +3.0 dB. Found by bisecting the file's shift between `5956b10` and `e10b17f`
(two runs on one tree agree in every number, so each commit's output is fixed): ASIAEX
moved at `710b7ef` (D-84); Pursuit moved at `99665ef`, source levels as RMS (D-102), 3.01 dB
in every level with echo or reverberation in it, as that decision intended. The runs now
name `doppler_spectrum: lorentzian` (SST's AR(1)), as they name the decay rule; the two
runs that compare the spectra still name their own. With it every ASIAEX value equals the
`5956b10` run. The suite's default stays Gaussian (D-84).

### D-145 `sonar_live` refuses an option it cannot read

The program's key=value options (`--options`, `options.cpp`) were read with `std::atof`
and `std::atoi`, so a value the reader did not understand became 0 without a word: the
one-row builds' `decimate={}` switched decimation off while the chain file asked for it
(API audit A1), and a list or a `True` in a profile key did the same (A2). The readers in
`libs/live/include/sonar/live/option_value.hpp` now throw `std::invalid_argument` naming the
key. Booleans accept `0`, `1`, `true`, `false`, `True` and `False`; `window` accepts `hann`
and `rect`; `look`, `backend` (`auto|host|cuda`), `blanking_mode`, `beamformer`, `doa` and
`passive_method` are checked against their choices in both engines (`live.CHOICES`). The
Python side writes the program's options from one function, `live.program_options`, used
by `tools/sonar_live.py` and `tools/build_from_spec.py`, and a test round-trips every key
through both routes. An options file written before this with `window=none` or a profile
boolean the reader did not know now stops the program instead of running a different chain.
The program's own default backend stays `host`; whether it should become `auto`, as the
Python chain's is, is an open owner question (STATUS).
*Later (pointer added 2026-11-24): superseded by D-147, the program starts from `auto`.*

### D-146 Simulation inputs that were assumed silently are now stated or refused (API audit)

- **Bellhop's seabed** has no default: an `eigenray` environment with no `sediment` and not
  all three bottom numbers is an error, as for `image_source` (D-104). All 21 shipped eigenray
  scenarios already describe their bottom; nothing moved.
- **`ReverbParams::sound_speed`** defaults to NaN, which the existing check refuses.
- **Brackish water** (0 < S < 5 PSU) takes a linear blend in salinity between Marczak's pure
  water (with Mackenzie's linear depth term) at S = 0 and Mackenzie at 5 PSU: our own mix, no
  new coefficient. The earlier mix stepped down by 0.81, 1.31 and 1.67 m/s at 5 PSU (4, 10,
  20 C). No shipped configuration has 0 < S < 5; S = 0 and S >= 5 are unchanged.
- **`lloyd.bellhop_transfer`** references spherical spreading to the slant range, as its
  docstring and `one_way` do (it used the horizontal range). The depth planner's two-way peak
  factor moves by at most 0.20 dB at 300 m, 0.018 dB at 1 km and 1e-4 dB at 12 km
  (`slope_1000m_silt_profile`, own depths 5/20/50 m); at 60 m range with a target 30 m down the
  old error against the image model was up to 3.7 dB one way. The committed plan files were
  not regenerated.
- **The boards' 100 Hz AC coupling** is hardware description (`hardware.boards.highpass_hz`),
  not a receive key: simulating it needs `receive.overload`, which also turns on clipping.
  In the band it changes every channel's phase alike by under 1.5 degrees and level by under
  0.003 dB. `receive.highpass_corner_hz` and `receive.bandpass_hz` (read by nothing) are refused.
- **An unknown `receive.reference.mode`** is refused (only `sync_chirp` exists).
- **Python's report line writes a non-finite number as `null`**, as `libs/live` does.
- **The platform self-noise generators were removed** as unused (audit G4).

### D-147 `auto` is the default everywhere; `cuda` never falls back (owner's ruling)

The owner ruled on the two questions D-145 left (2026-11-21). `build/bin/sonar_live` now
starts from `backend` auto (`live::program_defaults`), as the Python chain and the deployment
files do: CUDA when the build has it and a device is usable, else the host. And a chain or
pipeline that asks for `cuda` where no device is usable stops at construction with an error
naming the backend (`LiveChain` in both engines, `pipeline.run`), where the pipeline used to
run on the host instead; `auto` is the choice that falls back. `ChainOptions{}` in the library
keeps `Backend::Host`. Tests: `Options.TheProgramStartsFromAutoAndAChainRefusesCudaWithoutADevice`,
`test_live_cpp.py::test_backend_cuda_without_a_device_is_an_error_and_auto_falls_back`.

### D-148 Motion estimation: the DVL altitude is referred to the seabed under the ping; delay backends

Eq. 7's altitude was used as if the seabed were at z = 0 while ping 0 was anchored at
`floor_z - altitude`, so any `seafloor_z` other than 0, including `auto` when it resolves away
from 0, put a step between ping 0 and the rest (flat seabed 5 m down with `seafloor_z: 5`:
heave RMS 4.79 m; known undulating bathymetry: sway RMS 62 mm). The altitude now enters as
`alt - seabed_z`, seabed_z being `seafloor_z` or the bathymetry under the ping (in pass two at
the coarse pass's sway). Ours: a DVL measures altitude above the local seabed. The default
(`seafloor_z: 0`, no bathymetry) is arithmetically unchanged; `auto` and non-zero datums move,
to a common heave offset equal to the datum's error (STATUS, motion limits).
`MotionOptions.delay_backend`: `numpy` (default and reference), `host` or `cuda` (an error
without a device, as D-147); one kernel pair for host and device, sums in double. The lambda
sweep was run (`tools/motion_study.py sweep`); defaults kept (D-16).

### D-149 PyRAM and Bellhop3D as test oracles; out-of-plane propagation quantified, not modelled

PyRAM 1.3.0 (BSD-3; numba BSD-2, llvmlite BSD-2 and Apache-2.0 with the LLVM exception) and
Bellhop3D (GPL-3, the Acoustics Toolbox pinned at 8b4682b, run as a separate process) are test
oracles only; the library never imports either. The ray tier is compared with the parabolic
equation in 500 m range windows (point-by-point interference does not compare: p90 12 dB). Over
a 180 m seamount in 200 m of water at 8 kHz, Bellhop3D's 3-D mode minus its N×2-D mode is +3.0
to +3.6 dB over the flanks and -1.1 to -1.2 dB beside them; our N×2-D tier is 3.8 dB too loud
over the flanks (converged beams). The test holds that gap as a characterisation, to become an
agreement bound if 3-D propagation is added. Our tier and Bellhop3D's N×2-D differ by up to
1.0 dB even over a level bed (cause not found; issue filed). The oracle gate knows `pyram` and
`GPUSONAR_BELLHOP3D`, matched as whole words; a Bellhop3D build failure only warns.

### D-150 The multiscale detector's statistic runs on the chain's backend

`multiscale_detect` is split into `multiscale_cells` (background, strongest sample, flags) and
`multiscale_group` (sidelobe dropping and grouping, on the host). With `Backend::Cuda` the cells
run as three kernels (`detector_kernel.hpp`, `detector_cuda.cu`) doing the host loops'
operations in the same order, compiled with `--fmad=false` so no product is contracted; the host
loops are unchanged and the reference. Ours. Agreement: identical flags, strongest samples and
detections on eight C++ maps (one takes the slope clamp at both ends) and on the benchmark's
live pings; background within 4.3e-16 relative. **This changes a default result in its last
bits on a GPU machine**: the default backend is `auto`, so there the background, and the
`snr_db` computed from it, now come from the device (as CFAR's already did). A NaN in the first
or last two training lengths of a bearing is not held to the host (the two sorts order it
differently); the chain's maps are powers of finite samples. The Python engine keeps its NumPy
statistic. L4 time per ping, as deployed, 1 / 2 / 3 rows: 16.0 / 29.5 / 40.6 ms before,
12.7 / 25.9 / 37.5 ms after (`docs/benchmarks/live_chain_ablation_l4_b2a918e.json`), still 2.8
to 3.6 ms over `cell_statistic: mean`: the map returns to the host and is uploaded again.

### D-151 Holoscan: the conditioning operator reuses the chain's gain step; the emulated board spreads its catch-up

The tvg step's body moved unchanged from `pipeline._stage` into `recon.gain.apply_tvg_step` /
`tvg_step_parameters`; the pipeline and the Holoscan `ConditionOp` both call it. `ConditionOp`'s gain
stays off unless asked: `tvg: true` (the chain's default step, absorption from the record) or a mapping
of the step's keys; its older `tvg_db_per_decade` keeps the old law of D-91 (`absorption: none`), bit
for bit. The Sensor Bridge emulated board (`HsbBoardEmulator`) sends frames it owes after a stall no
faster than two a period (`catch_up: spread`, default; `burst`, the old behaviour): every lost
datagram in the loopback had been refused by the receiver's full buffer (kernel drop count equal to the
packet-number gaps in every lossy run). Ours. `sonar_live_holoscan`'s chain operators start from
`program_defaults` as `sonar_live` does (D-147); only `test_holoscan_cpp.py` on a GPU shows the
difference.

### D-152 Classification may use the detections, through opt-in rules with no default

Tracks keep each detection's cell count in both engines. `classify.detections` (default null):
`min_mover_cells`: a mover whose median detection extent is below it becomes `uncertain`;
`row_snr_db` (gallery only, through `classify_tracks(linear=...)`; refused by the live chain,
which never runs the row test): a track counts in a mover's row only if its mean SNR is at most
that far below the mover's. Ours, thresholds included. They stay off: on one seed of the full
gallery, `min_mover_cells` 3 to 5 removes the rough-surface glint movers (13 to 0 at 5) but the
real boat there is 5 to 7 cells and turns `uncertain` from 6; `row_snr_db` 3 or 6 returns the
cluttered scene's boat from `line` to `mover`; nothing changes in the Popoto scenes (the 12 km
ship's `multipath` comes from echo grouping). The thresholds were seen on these scenes and are not
validated. The PICMUS evaluation (`benchmarks.py`) is a port of the challenge's MATLAB code
(Bernard and Rodriguez-Molares, 2016; no licence file in the archive; the site states its datasets
and code are free of use, cite the IUS 2016 paper; read 2026-11-21).

### D-153 The Sensor Bridge loopback test runs at real time (owner's ruling)

D-79 slowed the loopback test to half real time because real time lost a burst per run. The cause
was measured since (D-151: datagrams refused by the receiver's full socket buffer while the board sent
its owed frames back to back after a stall) and the board now spreads its catch-up; at real time 20
runs with spread lost nothing, 3 of 14 with the old burst did. The owner ruled (2026-11-21) that the
test runs at pace 1.0 (`GPUSONAR_HSB_PACE` still overrides). The test's assertions are unchanged:
no frame lost or damaged, reports equal to the direct chain's. The margin is thin: the container's
`rmem_max` (212,992 B) is about 20 ms of stream, and a halved buffer still loses frames, so a run
on a loaded machine can fail; a failure is reported, not retried. First run at pace 1.0: L4, job
6b012ef14fdc5ab3c55c698a, the Sensor Bridge and Holoscan selection 22 passed, 0 failed. Deployed hosts should follow
Sensor Bridge's own setup advice (`DEPLOYMENT.md`).

### D-152 addendum (2026-11-21): the detection rules over ten seeds; they stay off

`tools/classify_rules_eval.py`, 12 gallery scenes with movers, 10 seeds each (the scene's seed + 0..9),
full size on an L4 (job 6b0130d0efaf5e14f4b46b34; `docs/benchmarks/classify_rules_eval_190f8ce.json`);
seed 0 reproduces the committed gallery. With the rules off there are 160 false movers over the 120
captures. `min_mover_cells` 4 is the largest value that loses no real target in any capture and removes
86 of them (8.6 +- 2.5 a seed), nearly all in `fls_120k_rough_surface`; it has no margin: the
rough-surface boat's median extent is exactly 4.0 cells in 3 of the 9 captures where it is a mover,
and 5 loses it in 3 captures, 6 in 4, 8 in 9. Glint and boat extents overlap (glints up to 6.5 cells,
the traffic scene's closing boat down to 6), and the values hold only for these maps' cells. `row_snr_db`
6 turns the cluttered scene's boat from `line` into `mover` on all 4 seeds where it was a line, with no
loss and no new false mover; 3 adds 2 false movers. No Popoto scene changes except `deep_silt` at 8
cells. Kept off by default. Candidate, for the owner: `row_snr_db: 6` in the gallery's FLS transit
classification (it would change the committed gallery).

### D-154 The gallery's FLS classification uses `row_snr_db: 6` (owner's ruling)

After the ten-seed evaluation (D-152 addendum) the owner ruled (2026-11-21) that the gallery's FLS
transit classification passes `detections={"row_snr_db": 6}` (`tools/gallery.py`): a track counts in a
mover's row only if its mean SNR is at most 6 dB below the mover's. Over 120 captures it lost no target,
added no false mover and made the cluttered scene's boat a mover on the 4 seeds where it had been a
line. The live chain is unchanged (it refuses the key, D-152); the Popoto scenes do not run the row test.
`tools/classify_rules_eval.py` keeps the gallery's own rules apart from its grid, whose first cell stays
"off". The committed gallery was regenerated with it.

### D-155 Echo grouping on both sides in range, transitively; the object is the earliest established arrival; the gallery scores the object (owner's ruling)

Issue sonar-ygc. In `popoto_12km` one ship gives 11 to 13 confirmed tracks: arrivals by longer
paths in 100 m of water, from the true range to about 130 m beyond, nearly all at the ship's
velocity. The old grouping made the strongest track the object and grouped only weaker tracks
farther out; the strongest arrival lies 34 to 93 m beyond the true range on all ten seeds, so the
earlier arrivals were never grouped (3 to 6 movers or uncertain tracks a ship), and the gallery
scored whichever track was nearest the truth (3 of 10 seeds right).
- **Grouping (ours), both engines identically** (`classify.py`, `classify.cpp`): anchors strongest
  first by mean SNR, each an established track (mover, static, line) not yet grouped and with a
  finite range, bearing and velocity. A group grows transitively: every ungrouped track of any kind
  within `max_excess_range` in range on either side, the bearing gate and the velocity gate of any
  member joins, until none does. Its object is the established member nearest in range (ties: lowest
  id), with its own class and fit; the others are its multipath.
- **Why:** the strongest arrival in a shallow channel need not be the direct one; the earliest
  established arrival is nearest the true range; a short uncertain track is not made the object
  because its fit is poor.
- **Cost:** objects at one bearing and velocity are merged in either direction, and a chain of tracks
  can link objects farther apart than one gate (not yet measured on a two-object scene). The
  object's cross-range error at 12 km is up to 117 m (range error -27 to -0.3 m).
- **Scoring (ours):** `tools/gallery.py` scores a true object by the nearest established track within
  the gate, else the nearest track of any kind within it; the per-target position and velocity come
  from that same track; `tools/classify_rules_eval.py` calls the gallery's functions.
- **Measured** (re-score of the 120 saved captures, no simulation): popoto_12km ship one mover on
  10 of 10 seeds (was a mover on 3; mover tracks 40 -> 10, uncertain 11 -> 0 over the seeds); speed
  4.8 to 7.4 m/s (true 6.0); FLS anchored boats and the pinnacle gain 5 correct by the scoring;
  deep_silt false movers 14 -> 12; no target lost and no false mover added in any scene. The rules
  evaluation moves (`classify_rules_eval_a479519.json`: `min_mover_cells` 5 loses no target).

### D-156 Each detection at its reflection time, transmit + R/c (owner's choice, issue sonar-960)

A monostatic echo shows its target where it was when the sound reached it, at transmit + R/c, not
at transmit. At 12 km that is 8 s, 48 m of a 6 m/s ship's travel; stamped at transmit, the ship's
track ran about 47 m ahead along its course on every seed (the sonar-960 diagnosis).
- **Time (ours, geometry):** R is the detection's one-way range, (two-way time)/2 · c, and c is the
  record's sound speed after `sound_speed`, the same c the beamformers turn time into range with
  (`beamform.cpp`, `chain.cpp`, `live.py`). Both engines set `Detection.time` = t + R/c (Python
  None, C++ NaN or any non-finite value: the ping's time, bit-identical to before; golden
  `tests/python/data/tracker_before_d156.json` from both trackers at 1be89d4). Every shipped sonar
  is quasi-monostatic (projector within 4 m of the array); a bistatic receiver needs its own
  geometry. The platform's own motion during the round trip is not modelled: detections are placed
  from the array's pose at transmit (issue sonar-2hz).
- **Tracker, both engines** (`detect.Tracker`, `tracker.cpp`): each track is predicted to each
  detection's own time for the gate and the d² + ln|S| cost (D-141), takes it there, and a new
  track is born there; history holds those times. A track that takes nothing counts a miss and is
  predicted to the ping time, or held where it is when it already holds at a later time (a ping
  interval shorter than R/c), so a missed track never moves back. Prediction to an earlier time
  still grows the uncertainty: Q = G Gᵀ q² is positive semi-definite for either sign of dt, and the
  depth's random walk grows by q_d²|dt| (ours). Not handled: a track that takes a nearer, earlier
  detection within a ping still steps back in time.
- **Report:** x, y and range are the fitted line at the report's `time_ns`
  (`classify_tracks(time=)`, `ClassifyOptions::time`); a track under three points is moved there by
  its Kalman velocity. Echo grouping (D-155) then compares positions at one time.
- **Gallery and `classify_rules_eval`:** each track's state is compared with the truth at the
  state's time (its last history point); classes at the last ping with the truth there.
- **Measured** (`classify_rules_eval_c148f56.json` against `_a479519.json`, ten seeds, the
  gallery's own pipeline): no target class changes; false movers 158 -> 152. popoto_12km report
  cross-range error, median -54.6 -> -32.0 m, |range| 11.8 -> 4.4 m; nine seeds shift by +43 to
  +59 m (v·R/c), seed 56 is worse (+9 -> -91 m) by track hopping between arrivals (sonar-960, open).
  fls_170k_traffic `boat_closing` 3.2 -> 4.6 m: its detections sit about 4.6 m behind it in range,
  which transmit stamping had hidden by 1.45 m (sonar-9r2; D-169: they are on its propeller, not a lag). flank_patrol (seeds 17-26, host):
  submarine cross-range -0.3 m (was +5.6), report error 29.5 m (29.3) dominated by a +29 m range
  bias (sonar-a98); no class or count changes.
- **The live chain at 12 km** (`popoto_vehicle` chain, seeds 51-60, host; Python, C++ and
  `sonar_live` agree to 5e-6 m): cross-range error +26.1 m mean (was -23.1 m), range +49.5 m
  (unchanged), total 56.2 m (55.1 m). The timing is right (each seed moves by v·R/c, the last
  history point is at t + R/c in both engines); the old -48 m error had been hiding a +26 m
  cross-range bias. Kept because a correct clock is needed to find that bias (sonar-2hz).
- The committed gallery predates D-156 and was not regenerated.

### D-157 Fit the measured positions; leading-edge range opt-in; score extended objects from their outline (owner's choice, issue sonar-2hz)

The sonar-2hz diagnosis split the live chain's 12 km error (+26 m cross-range, +50 m range) into
three parts. A lag: the report's line went through the tracker's filtered history, and a track is
born at zero velocity with a 1 deg bearing sigma (210 m at 12 km), so a noise-free crossing came
out 38.7 m behind the ship. A glint: coherent echoes of the 5-point ship side-on at the last ping
(-46 m, about +41 m along track in the measured position of seed 51). And a range bias: the
energy-weighted centroid of an echo spread over about 120 m by multipath (+49.5 m; its leading
edge -0.8 m). All three changes are ours.
- **A (default, both engines):** each history row also carries the measured position of the
  detection the track took there (`Track.measured`, `HistoryPoint::mx, my`; NaN on a miss), and
  `classify_tracks` fits its line through those, the rows of misses left out. Under three measured
  rows: the short-track rule, from the last filtered point moved by the Kalman velocity. A track
  that records no measurements (built by hand, a capture from before) is fitted through its history
  as before. Student's t degrees of freedom count the fitted points; `min_points` and `seen` still
  count pings. With no time given, the position is the line's at the last history row (the last
  ping), also when the track missed it (review fix). Noise-free 12 km: 38.7 -> 0.0 m.
  Thresholds kept: FLS target fit_rms at most 1.3 m against 2.5 m; Popoto targets at most 53 m
  against 60 m (a 7 m margin; the derived limit is from range resolution alone, while the
  measured scatter includes cross-range R·sigma_bearing: sonar-4jm). Golden: the tracker
  scenario's default classes in `tests/python/data/classes_d157.json` (both engines);
  `tracker_before_d156.json` keeps holding the history fit (`fit_measured=False`, measured left out).
- **B (opt-in):** `cfar.range_estimate: centroid | first` (`multiscale` only; anything else, or
  `first` with another cell statistic, is refused in both engines). `first` takes the nearest
  sample of the detection's group above the single-sample threshold, else the nearest cell's
  centre. Default `centroid`, byte-identical (seed 51 detections of both engines against b5d75b8).
  The threshold is the first scale's; with `integrate_cells` not starting at 1 it is not the
  single-sample test (as the existing arrival code; sonar-d99). Corrected in D-158: the first
  scale is always the single sample.
- **C (tools):** `tools/gallery.py` and `tools/classify_rules_eval.py` score an extended true
  object (several points, a hull plan, a box) by the distance to the convex hull of its footprint
  (Andrew's monotone chain), and gate tracked/untracked and false movers by that distance;
  `centroid_error_m` is kept. On these captures C changed no gate decision.
- **Measured, live chain at 12 km** (`popoto_vehicle`, seeds 51-60, host, mean; Python and C++
  within 5e-6 m): range +49.5 -> +50.6 (A) -> +0.9 m (A + B `first`); cross-range +26.0 -> -23.2 ->
  -23.3 m; distance to the hull 49.4 -> 50.7 -> 1.0 m; speed 6.1 -> 6.4 -> 6.3 m/s (true 6.0).
  Worse with A: the line now follows the glinting last measurement, so the error moves from 26 m
  behind the ship to 23 m ahead (a robust fit: sonar-ljh).
- **Measured, gallery pipeline** (`docs/benchmarks/classify_rules_eval_afe0806.json`, ten seeds of
  twelve scenes, L4, job 6b02383cefaf5e14f4b6fe35; the same simulations as c148f56): targets in
  their class 195 -> 194, false movers with the rules off 152 -> 94 (rough_surface 118 -> 78,
  cluttered 7 -> 2, traffic 10 -> 4, winter_sand 3 -> 0), all from A. Lost: `fls_120k_cluttered_seabed`
  seed 69, `boat_under_way` mover -> line with the rules off (a mover with `row_snr_db 6`).
  Scored from the centre the Popoto scenes move 15-20 m ahead (popoto_12km median 35.4 -> 52.7 m;
  seeds 56 and 57, -119 and -103 m cross-range, are the track hopping of sonar-960); from the hull
  the popoto_12km median is 8.2 m. `flank_patrol` (seeds 17-26): submarine 29.5 -> 28.0 m, speed
  2.98 -> 3.01 m/s (true 3.0), one false mover (seed 4) where there were none.
- The committed gallery predates D-157 and was not regenerated; whether `first` becomes the
  deployment default is open (sonar-1p2).

### D-158 The detector's first scale is checked to be the single sample; pings whose navigation sent no velocity are counted (issues sonar-d99, sonar-8is)

- **d99 (a check, outputs unchanged, both engines):** the leading edge (`cfar.range_estimate:
  first`), the arrivals and the flag test read the plan's first scale as the single-sample test.
  `plan_detector` always puts that test first (cells 0), whatever `cfar.integrate_cells` holds:
  `[2, 4]` leaves out the cell mean, not the sample. The D-157 B sentence that said otherwise was
  wrong; no shipped or derived plan was affected (no file in `configs/` sets `integrate_cells`).
  A plan built by hand with a mean first, or a second sample scale, is now refused
  (`check_plan`, `DetectorPlan::sample_factor`; `detect.check_plan`).
- **8is (a counter, both engines):** the backprojection follows the array's motion over the round
  trip by the pose's velocity (sonar-2hz diagnosis: -24 m at 12 km without it). A navigation fix
  that sent no `vx, vy, vz`, or a non-finite one, was taken as 0 silently (a NaN went into the
  pose). Now its missing or non-finite components are 0 and the fix is marked; a ping whose pose
  came from a marked fix (either of the two interpolated between, or the one held) is counted in
  `pings_without_velocity`. It is cumulative like `pings_without_nav`, on a report line only once
  above 0, with one warning. The count is kept where each program applies navigation
  (`live::NavCounts`: `LiveChain`, `apps/sonar_live` on its UDP path, the Holoscan `UdpSourceOp`),
  and the pose and its flag come from one snapshot of the fixes. A counter, not a refusal: a
  refused ping loses detections that a stale velocity would only bias. No record schema change.
  A static pose is at rest unless it says otherwise. Lines with velocity are unchanged.
- **Validated:** CPU tests of both packages; GPU suite on 42d4539, HF job 6b026f2defaf5e14f4b79e70,
  1686 passed, 92 skipped, Holoscan operators compiled (they cannot be built in the CPU container).

### D-159 Bearing sigma kept; an opt-in outlier test per axis of the line of sight; the fit limit grows with range (issues sonar-ym1, sonar-ljh, sonar-4jm; owner's choices)

With the defaults the gallery and the live chain give the outputs of 688271f: the gallery re-score
of the saved captures reproduces `classify_rules_eval_afe0806.json` (194 targets in their class,
94 false movers), and the live chain's popoto_12km report (seeds 51-60, both engines, both range
estimates) is unchanged. The along-track error the issues named is not reduced: on those seeds
the cross-range mean stays -23.3 m with `first`, a glint spread over several pings.
- **ym1 (kept):** the derived `tracker.bearing_sigma` stays max(1 deg, beam/16) (Popoto: 1 deg,
  floor 0.22 deg), although a point target's bearing scatters 0.03-0.05 deg at 12 km in
  simulation. Since D-157 the report's line goes through the detections, not the filtered state:
  1, 0.25 and 0.05 deg give one report to 1e-6 m (a guard test,
  `test_the_12km_report_does_not_depend_on_the_trackers_bearing_sigma`). A ship's own detections
  scatter 0.1-0.3 deg side-on, and the sigma still sets the gate (D-99).
- **ljh (opt-in, both engines, ours):** `classify.outlier_alpha`, default 0 (off). When above 0,
  through the measured positions at most one point is left out of a track's line. The residuals
  are rotated once onto range and cross-range about the sonar (the ownship in the live chain;
  `classify_tracks(origin=)`, else the multipath origin, else the frame's origin), each axis with
  its own variance. The externally studentized residual, Student's t with n - 3 degrees of
  freedom, is tested two-sided at alpha / (n x axes tested); an axis is skipped while the other's
  scatter times the line of sight's turn exceeds half its own; four points at least; never on
  filtered history. The t tail is Abramowitz and Stegun 26.7.3/26.7.4, the same finite sums in
  both engines (against scipy within 1e-9 relative or 1e-15 absolute). Clean tracks lose a point
  at alpha within Monte Carlo error for range : cross-range 1:1 to 1:50 and 10:1, n = 4-20.
  The first version (one pooled variance, F(2, 2n - 6)) dropped 4-13 % at 0.01 under the
  sonar's unequal errors (review); its live-chain gain (-23.3 -> -13.4 m) came from that.
  Measured on at 0.01, worst first: no point left out on popoto_12km seeds 51-60 in the live chain
  (six points give 3 degrees of freedom: a glint needs |t| > 13.7); gallery
  (`classify_rules_eval_e0d9665.json`, HF job 6b0289c7efaf5e14f4b7e3f7) false movers 94 -> 95
  (popoto_env_deep_silt_1000m seed 55, through the echo-grouping velocity gate), popoto_12km mean
  speed error 0.773 -> 0.863 m/s; targets 194 unchanged, all-mover speed error 0.209 -> 0.181
  m/s, distance to the hull 8.06 -> 7.78 m. Parity: no decision differs in 1600 fuzzed tracks.
- **4jm (on in the live chain, derived, ours):** a track's fit limit is at least
  `classify.max_fit_rms_bearing_deg` x its range from the sonar, derived as four bearing floors
  over sqrt 2 (Popoto 0.63 deg: 132 m at 12 km, 221 m at 20 km; 0 when the beam is unknown).
  No measured scene changes; on seeded six-ping crossings at 20 km with 0.2 deg of bearing error
  it stops the fixed 60 m limit from making 6 % of ships uncertain. The gallery and
  `classify_rules_eval` keep their typed limits. False movers under the grown limit are not
  measured: no long-range clutter scene exists. A negative, infinite or NaN value is refused by
  every entry of both engines and by `sonar_live`. The 688271f live-chain outputs need
  `max_fit_rms_bearing_deg: 0`.
- **Validated:** CPU tests of the affected files on the merged tree (246 passed); GPU suite on
  ff0769a's tree, HF job 6b029f6a4fdc5ab3c55fd2fc, 1705 passed, 92 skipped, Holoscan operators compiled.

### D-160 The leading edge is the live chain's default range, made consistent (owner's choice, issue sonar-1p2)

- **Default:** `cfar.range_estimate: first` with `cell_statistic: multiscale`, in both engines
  (live.py `chain_range_estimate`; `ChainOptions::range_estimate`, an optional the parser sets).
  Not given with `mean` or `peak` the range stays the centroid; `first` given with them is
  refused; `centroid` is always accepted. The detector's own default stays `centroid`, so direct
  detector calls and the gallery tools (`detect.cfar` + `detect.cluster`) are unchanged.
- **Stages kept on the centroid** whichever range is reported: the elevation search, SBL's range
  gates (grouping and centre), `multipath_depth`, blanking and the reference-burst filter. Each
  multiscale detection carries `Detection.centroid_range` (both engines, also through SBL;
  `live_multiscale_detect` returns it). Without this the elevation search, which runs from the
  detection's range to 1.5 cells beyond, stopped before the echo's peak: on popoto_static_free a
  10 m target read at 209 m, and with `doa: sbl` five detections came out of three targets.
- **One range per placed detection:** with `first`, a detection the elevation places (stacked
  rows, `correct_bearing`, not ambiguous) is placed, reported and timed (D-156) at the elevation's
  own range. Moving its position to the leading edge put tracks 1.9-3.7 m short; keeping the
  centroid for stacked rows lost the 12 km gain. With `centroid` given the output is that before
  D-160, byte for byte where checked (`tests/python/data/centroid_d158.json`, from 42d4539),
  within 1e-6 m across machines.
- **Owner's evidence (live chain, centroid -> first):** popoto_12km (chain `popoto_vehicle`) ship
  range +49.5 -> +0.9 m; flank_patrol submarine +36.8 -> -4.2 m, wreck +33.4 -> +0.6 m; FLS boats
  1.5-1.8 m nearer, to their near face; FLS false movers 461 -> 411; no class changes.
  Three stacked rows at 12 km (seeds 51-60, both engines alike): ship track +37.3 -> +2.7 m, its
  last detection +43.5 -> +0.6 m. popoto_static_free range_m: 0.00 / +0.01 / +0.01 m (centroid
  +0.05 / +0.03 / -0.02 m).
- **Worse:** a detection no elevation places (one row, or an ambiguous elevation) reads short for
  a strong point echo: -1.0 m at 0 dB, -2.0 m 30 dB up, -1.9 to -3.6 m at 0.4-0.5 km, -2.1 m at 1 km (the centroid
  within 0.05 m). The ship's depth at 12 km with three rows is 8-35 m (true 4 m) with either
  estimate (ambiguous elevation). The blanking and burst-filter change is untested (at most one
  cell).
- **Sensor Bridge loopback:** the real-time loopback test lost one frame in 4 of 25 runs on
  leading-edge trees against 0 of 28 on centroid trees (one-sided p about 0.04). In the study jobs
  6b03468b8d268c0fbd228881 and 6b03468c130a00b5aa80fa64 a halved socket buffer lost frames alike
  with either estimate (9 and 13 frames in 4 runs each), every loss 12-18 frames after a ping
  completed, and keeping the chain off the receiver's core did not help (24 against 22 frames). The
  test sits at the edge of the 212,992-byte buffer whatever the estimate; the receiver's stall
  during ping processing is investigated separately. The core-reservation option was not adopted.
- **Validated:** merged tree 02593ef: local tests of the touched files 307 passed; GPU suite, HF job
  6b02c630efaf5e14f4b86a50, 1713 passed, 92 skipped, 1 failed (the loopback above).

### D-161 Flank stream receive: blocks placed by their sample counter; pings lost whole counted; repeats told from restarts (owner's request, issues sonar-ilb, sonar-070)

Before, a frame arriving after a later-numbered one opened a gap and its ping was dropped although
every frame arrived; a ping whose frames were all lost was not counted in `dropped_pings`; and
without frame numbers a repeated frame was taken for a restart and dropped a complete ping. Both
assemblers (Python `flank_stream.PingAssembler`, also under `FlankUdpSource` and the Sensor Bridge
`FrameReceiver`; C++ `live::PingAssembler`, under `sonar_live` and the Holoscan `UdpSourceOp`)
follow the same rules, all ours:
- **Placement:** blocks go to their place by the sample counter. An earlier ping stays open beside
  a later one only for the reorder grace, `reorder_blocks` = 2 blocks; at most 2 are open, so at
  most 2 raw windows (about 2 x 116 MB with the default flank_32 window; one before). A ping
  transmitted before the previous window closes gives that one up at once, as before: in-order
  streams, overlapping ones included, give the records and counts they gave before. Records come
  out in transmit order; a complete ping waits only while an earlier one is undecided, at most 2
  blocks (12.5 ms with 400-sample Sensor Bridge frames); in order nothing waits.
- **Holes** are given up (`lost_samples`) 2 blocks after them, or with the ping whose window they
  cut. `finish()` gives up what is still awaited at the end of a stream; pings a stopped caller
  discards are counted in `pings_discarded_at_stop`. Rows held behind holes on the decimating
  path are at most `max_held_bytes` = 64 MiB in all; past it the earliest open ping is given up.
- **Pings lost whole:** a jump of d in the header's ping index counts d - 1 dropped pings when the
  transmits are d median intervals apart (last 5 intervals, within half an interval, and beyond
  d - 1 of the longest recent interval by half an interval); before an interval is known nothing
  is counted. Known failure: a sender that skips indices over a gap near d intervals is counted as
  a loss (4 of 4000 fuzzed streams made to renumber).
- **Repeats and restarts:** a block that does not fit the stream is held. A restart needs two
  contiguous blocks of another capture start, or of the same one lying a restart span before the
  first sample heard, or 8 consistent blocks in a row; a stale copy (samples already received)
  meets none of these except as a burst of 8 while the stream is silent. A capture left is never
  returned to. Without frame numbers `frames_lost` stays 0.
- **Counters:** `frames_lost` (numbered frames given up), `frames_late`, `frames_too_late` (once
  per frame), `frames_repeated` (duplicates only), `trouble_frames` (when a loss is given up),
  `pings_discarded_at_stop`, `open_peak`, `held_peak_bytes`. pings + dropped_pings +
  pings_discarded_at_stop = pings heard; no ping is both emitted and dropped.
- **Refused:** a block naming a transmit at or beyond its own end, and sample indices at or above
  2^62 (Python `BlockError`, C++ `bad_blocks`).
- **Reviewed twice** (independent reviewers with cross-engine fuzz: 1000-4000 streams per script,
  hostile headers, a comparison with the code before over 600 in-order streams, 310 of them
  overlapping, records identical). **Validated:** local receive tests 119 passed; GPU suite on the
  merged tree, HF job 6b036ba48d268c0fbd22d9fe, 1808 passed, 92 skipped, Holoscan operators
  compiled.

### D-162 Sensor Bridge receive: the chain's OpenMP below the core count (owner's choice, issue sonar-ihd)

With one OpenMP thread per vCPU, `flank.backprojection_map` kept Sensor Bridge's receiver thread
runnable but waiting 5-13 ms for a CPU on an 8-vCPU L4 host (4 cores x 2), and the 212,992-byte
socket buffer (about 20 ms of stream) overflowed: the occasional frame lost by the real-time
loopback test (D-160). Ruled out by measurement: the GIL (a probe thread took it without delay),
the engine (the C++ engine loses alike), hypervisor steal (0). Pinning the receiver alone, nice 10
for the chain and 7 threads did not help; Sensor Bridge 2.7.0 left its receiver thread unpinned.
- `receive --chain-omp-threads N` (opt-in; default unchanged) sets OpenMP's thread count on the
  thread that runs the chain, before its first ping; the receive application uses Holoscan's default
  single-thread scheduler, so that is every chain call. Reported as `chain_omp_threads`.
- The loopback test runs it at half the available CPUs, keeps the default buffer and every
  zero-loss assertion (no retries), and asserts the worst receive delay (same-host CLOCK_REALTIME,
  receive after the frame's last datagram minus its send) under 5 ms, with clock checks.
- Measured on L4: halved buffer loss in 9 of 15 runs -> 0 of 10, normal buffer 0 of 6; worst delay
  10-13 -> 1-2 ms; about +10 % chain time. Loopback alone 8 of 8 passed, worst 0.8-2.3 ms (HF job
  6b037e168d268c0fbd23006c); full GPU suite 1809 passed, 92 skipped (6b037e1b130a00b5aa817f04).
- Open: the receive tool still hangs to the test's timeout after heavy loss (quarter buffer, 5 of
  8 runs, job 6b0386c9130a00b5aa8190e2); filed separately.


### D-163 Sensor Bridge receive always stops (owner's go-ahead, issue sonar-dkb)

Since the first Sensor Bridge commit (4f080b1) `receive` hung, without statistics, when its stream
ended before `--pings` pings were reported: heavy loss drops a whole ping, so only the idle watch
could stop it, and its stop was never seen. Not introduced by D-161/D-162: the pre-session commit
111b7c0 hung in 2 of 7 quarter-buffer runs (HF job 6b039cea130a00b5aa81c058), and the stop logic
was unchanged from 111b7c0 to ec04d29. Cause: Sensor Bridge 2.7.0's receiver operator by default
waits on an asynchronous condition that only a new frame sets, and Holoscan's greedy scheduler
looks at it again only then, so a stop set once frames have stopped is never acted on (reproduced on
the CPU with a stand-in source; py-spy on the hung GPU runs showed the scheduler waiting).
- The receiver operator is built with `use_frame_ready_condition=False`; its compute waits at most
  1 s for a frame. Sensor Bridge logs `Ingress frame timeout; ignoring.` once at the end of a stream.
- `hsb.ReceiveStop`, applied by `HsbStopOp` on the scheduler's thread every 250 ms: stop once
  reports plus dropped pings reach `--pings` ("pings"), or after `--idle` s with no frame ("idle"),
  whatever was lost. The idle-watch thread is removed. Statistics gain `reports` and `stopped`.
- Fewer pings reported than asked: a `receive: stopped short (...)` line with the loss counts on
  stderr and exit status 3, so the loopback test fails with numbers instead of a timeout.
- A second fault the hang had masked: an idle stop after the board has gone fails in Sensor Bridge's
  `unconfigure()` (control-plane timeout) and its receiver thread cannot be joined. `receive` reports
  it as `shutdown_error`, prints the statistics and ends with `os._exit`; a clean shutdown would need
  a change inside Sensor Bridge. `serve --hold` keeps the emulated board up after its linger; the
  loopback test passes `--hold 30`.
- Measured on L4: quarter buffer 12 of 12 runs ended in 16 s; idle stops 4 of 4 at 53 s with
  statistics; loopback test 9 of 9, worst receive delay 0.82-2.08 ms; full GPU suite 1814 passed,
  92 skipped (6b03a7ae130a00b5aa81da2e).
- Open: on the GPU the idle path ran only in runs set up for it (every lossy run stopped on
  "pings"); `run_receiver` keeps each application object alive until the process ends (harmless for
  the tool, grows if called repeatedly in one process).

### D-164 Motion estimation: ping 0's height is an unknown tied to its DVL sample (owner's choice, issue sonar-jh8)

Holding ping 0's height at minus its single DVL altitude (D-15, D-31) turned one sample's noise
into a common heave offset, and about 0.3-0.4 times that offset went into sway through the 17-27°
line of sight. On seed 2 of the acceptance scenario a −2.54 mm first DVL sample gave 0.87 mm sway
RMS against the λ/20 = 0.375 mm bound; over ten seeds the old sway RMS correlates 0.98 with that
sample's error. Not the scene (no window rejected), not the lambdas, not the reference truth (the
simulator starts sway and heave at zero).
- Ping 0's sway stays held at its navigated value (DPC sees only relative sway); its height is now
  free and enters Eq. 7 like every other ping (`ceres::SubsetManifold`). The paper lists only
  velocities as unknowns and does not say how p_z,1(0) is fixed: the choice is ours.
  `MotionOptions.fix_first_height` (C++ `SolveOptions::fix_first_height`) keeps the old anchor;
  without a DVL term the height is held as before. Lambdas unchanged (D-16).
- What got worse: `minefield_sas` sway 0.081 -> 0.139 λ RMS (0.61 -> 1.05 mm), mostly a common
  offset (mean −0.50 -> −0.97 mm, spread 0.34 -> 0.39 mm; sharpness 319.8 -> 323.1, true motion
  318.0), heave 0.087 -> 0.101 λ; freed, its ping-0 height settles 1.76 mm off against a 0.34 mm
  DVL error, a heave tilt in its delays not yet explained. Acceptance seed 3 sway 0.059 -> 0.104
  mm, seed 10 heave 0.215 -> 0.383 mm, `seafloor_z=5` sway 0.071 -> 0.226 mm, all inside the
  bounds. `test_smoothness_term_reduces_noise_and_dvl_term_holds_altitude` keeps its 2 mm check
  with `fix_first_height` (1.999 mm) and checks the default against 2.5 mm (2.115 mm).
- Better: acceptance worst sway over seeds 1-10 0.87 -> 0.24 mm; `wreck_sas` sway 0.089 -> 0.045 λ
  (sharpness 274.4 -> 276.8, true 280.6); undulating seabed 0.342 -> 0.188 mm.
- The acceptance test runs seeds 1-8 with the same bounds; those bounds are ours (milestone 8):
  ASASIN Sec. II-D reports no sway or heave accuracy and no λ1, λ2 (arXiv:2101.05888v2, opened).
  Sweep after the fix, seeds 1-5: every seed inside both bounds at the defaults (worst 0.64 of the
  bound). Gallery figures and JSON for `wreck_sas` and `minefield_sas` were not regenerated.
- Validation: full GPU suite 1827 passed, 92 skipped (HF job 6b03bf4d8d268c0fbd238053, with D-165).

### D-165 No fitted velocity from echo delays alone (issue sonar-054)

With D-156 a point's time is transmit + R/c. Pings that carry one transmit time (a source without
timestamps) left only dR/c between a track's points, and the line through them measured range
against R/c: a radial speed of the order of c. With `cfar.range_estimate: centroid` the C++ chain
reported two standing targets at 118 and 48 km/s, 103 and 43 km out; the default leading edge
(D-160) hides the fault, since its range falls on the same sample every ping.
- Rule (ours, derived, both engines): each history row records its ping's time (`Track.pings`,
  `HistoryPoint::ping`); `classify_tracks` fits no velocity when the fitted points' transmit
  times span no more than their echo delays, ptp(T) <= ptp(t − T). The track is then treated as a
  short track: speed 0, never a mover. A real target's delays span ptp(R)/c <= (|dR/dt|/c)·ptp(T),
  below ptp(T) at any radial speed under c, so no threshold is needed. Rows without a ping time
  (hand-built tracks, `classify_rules_eval` captures) are not tested.
- Not chosen: refusing non-increasing transmit times; a timestamp-less source would give nothing.
- A timestamp-less source now gives tracks at the right ranges with speed 0, never a mover, no
  warning. Open: its Kalman tracker has no clock (dt ≈ 0), so a real mover may fall out of its gate.
- Measured: with increasing transmit times no class, position or velocity changes (bit for bit,
  `test_ping_times_change_no_fit_when_transmit_times_increase`); smallest margin ptp(T)/ptp(d) on
  six gallery quick scenes 167 (fls_120k_cluttered_seabed). Full GPU suite 1827 passed, 92 skipped
  (HF job 6b03bf4d8d268c0fbd238053).

### D-166 Both live-chain engines refuse an option key they do not read (issue sonar-6hf)

`core.LiveChainCpp` read its dict with `contains()` for known keys only, so a misspelt key (e.g.
`classify_min_mover_cell`) left its setting at the default without a word; only `multipath_depth`
and `coverage` refused unknown keys. The Python engine (`LiveChain`, `cpp_chain_options`) merged
the chain section over DEFAULTS and also kept extra keys in silence (`classify: {min_sped: 2}` ran
with min_speed 1).
- C++: the constructor accepts exactly the keys it reads (a `Keys` wrapper records each lookup),
  at the top and in `elevation`, `elevation.profile`, `decimate`, `blanking` and `sync`; any other
  key is a ValueError naming it and the nearest key it reads (edit distance within max(3, len/3),
  ours). Python: `check_keys` refuses keys outside DEFAULTS (plus `multipath_depth`, `coverage`)
  and outside each section, naming the nearest (difflib). The key=value path (options.cpp)
  already refused unknown keys, without a suggestion.
- What could get worse: a configuration outside the repo with a stale or misspelt key that ran on
  the default now stops at construction. Every `configs/deploy` chain section and every
  `configs/builds` chain_config passes both engines. No result moves.
- Not covered: the other dict-taking bindings (PingAssemblerCpp, SoundSpeedLadder, Navigation,
  WaterDepthTracker, live_resolve_defaults, live_detector_plan) still pass over unknown keys.
- Tests: `test_live_cpp.py::test_cpp_chain_refuses_an_unknown_key_*`,
  `::test_python_chain_refuses_an_unknown_key_by_name` (each fails on the old code). Full GPU suite
  1854 passed, 92 skipped (HF job 6b03d92e130a00b5aa825684, with D-167).

### D-167 The navigation unit is not the array: lever arm and mounting rotation (issue sonar-3rv)

The live chain took the navigation pose as the array's body origin; nothing held the surveyed
offset between them. Still open: the values (unconfirmed until surveyed, default zero), and the
array's velocity is not corrected (the nav datagram carries no attitude rates; error ω·|L|, e.g.
0.05 m/s at 3°/s and 1 m, about 0.8 m of array position over a 16 s round trip at 12 km).
- Keys `nav.lever_arm_m` (m, from the unit's reference point to the array's, the unit's body axes
  x forward, y starboard, z down) and `nav.mount_rpy_deg` (degrees, ASASIN Eq. 2 order, the array's
  axes against the unit's); `nav_lever_arm_m`, `nav_mount_rpy_deg` for `sonar_live`. Three finite
  numbers each or refused by name.
- Both engines move each fix at intake (`NavListener.push` / `nav_to_array`; C++
  `Navigation::push`): p_a = p_u + R_u·L, R_a = R_u·R_m, Euler angles read back from R_a (ours,
  derived; the same relations are standard in navigation texts, not opened: unconfirmed). The
  record's pose, attitude history and depth samples are then the array's, as RECORD_FORMAT's
  body-pose definition requires: no schema change. Replay applies no navigation, so the offset is
  applied once. Water depth from altitude and the sound-speed samples stay at the unit.
- Zero offsets are bit-for-bit the previous behaviour (test). In `configs/deploy/popoto_vehicle.yaml`
  the keys are present commented out (`test_hsb.py::test_tool_arguments` asserts that file's nav
  section); the generated deploy files (`tools/build_from_spec.py`) do not carry them yet.
- Tests: `tests/python/test_nav_lever_arm.py` (a hand-worked rolled case; roll 12°, pitch −7°, yaw
  125° against independent rotation matrices, engines agree to 1e-12 m; a chain run on
  `popoto_long_range_3row` fed the unit's fixes plus the offset reports the same detections as fed
  the array's: without it ownship is about 2.5 m off and depth reads 28.5 m for 48.0 m),
  `test_live_cpp.py::test_program_moves_navigation_to_the_array_by_the_lever_arm`. Full GPU suite
  1854 passed, 92 skipped (HF job 6b03d92e130a00b5aa825684).

### D-168 The detector's constants stand at the default budget, now measured (owner's choice, issue sonar-ydi; addendum to D-134)

D-134's thresholds at one false alarm per thousand pings were extrapolated, on one map. Now
counted on a GPU, 40 000 to 100 000 noise-only pings per map and row count, on D-134's map and a
second one (`flank_32` for one row: 15 samples a cell, 10.8 looks; the deployed static arrays over
75 m - 3 km for two and three rows): the whole detector gave 0.73 to 1.01 of its budget, pooled
0.82 (0.74-0.91) (RESULTS, "False alarms at the default budget, counted").
- What is off: the one-row intervals lie just under the budget (0.63-0.98, 0.60-0.95); the others
  contain it. The three-row static count stopped at 40 000 pings (job timeout), not split by test.
- Kept (owner's choice): under the budget means fewer false alarms; re-fitting the sample constant
  at 1e-3 (about 0.13 against 0.19-0.21) would lower its threshold 0.07-0.085 dB, Pd at most +0.027
  where it is 0.5; the constants depend on the budget, so a fit at 1e-3 would make budgets of 3 to
  0.1 run about 1.4 times over.
- Still open: the growth law in u counts too many single-sample tests at high thresholds, and the
  two-row cell mean runs 1.26 times its share, neither explained. The chain's 9 dB `min_snr_db`
  floor drops 33 % and 54 % of two- and three-row noise detections, so the shipped report runs well
  under budget for stacked rows.
- Tools: `detector_search_factor.py --at-default` (counts the shipped detector's output through the
  C++ chain), `--replay` (splits counted pings between the tests on the host);
  `GPUSONAR_JOB=falsealarm` in `tools/gpu_job.sh`. Test:
  `test_detector.py::test_the_default_budget_count_on_a_second_map_at_cpu_scale`. No default,
  figure or JSON changed. Full GPU suite 1855 passed, 92 skipped (HF job 6b0444b4a02d1cf5a7b80bb9).

### D-169 fls_170k_traffic boat_closing's 4.6 m is its propeller, not a lag (issue sonar-9r2; no change to the chain)

The boat's main detection on every ping is the propeller cluster under the stern (TS −5 dB at
0.45 L astern, `scene.py`); the bow points at the sonar, so it lies 5.3 m beyond amidships in range.
Measured with the gallery's FLS pipeline (seed 63, host): propeller alone +4.40 m, the same as the
full scene (+4.34 m); hull alone puts the track on the bow (−0.33 m); a stationary point target at
propeller depth +0.61 m. Breakdown of +4.34 m: propeller place +5.28 m, FLS slant range read as
horizontal +0.47 m (h²/2R, h = 19.4 m at 400 m) plus cell quantisation +0.14 m, LFM range-Doppler
coupling −1.49 m (up minus down chirp −2.62 m measured against 2·f0·v·T/B = 2.70 m expected),
tracker +0.07 m. Timing (D-156) is correct. Since D-157 the score is the distance from the hull
outline: 0.00 m on all ten seeds for every boat of fls_170k_traffic and for
fls_120k_cluttered_seabed's boat_under_way (from the centre 4.63 and 0.64 m; the latter is its
detection 0.5 m astern of amidships against truth D-156 moved 0.74 m ahead along the course).
- Not compensated (follow-ups): the LFM coupling, a closing target with the up-chirp reads
  f0·v·T/B nearer (fls_170k 0.17 s × v: 1.35 m at 8 m/s, 0.34 m on every static object ahead from
  ownship's 2 m/s; fls_120k 0.06 s × v); the FLS slant-range placement, +h²/2R.
- Test: `tests/python/test_lfm_range_doppler.py` holds the coupling at 2·f0·v·T/B within 5 % (ours,
  derived in its docstring). No default result moved; no HF job used.

### D-170 FLS classes: what remains of sonar-9z8 and sonar-9ir (no change to the chain)

Re-scored on 09583b6 from the 120 saved full-size L4 captures (simulated at e0d9665; nothing since
touches the gallery's simulation, detector or tracker): identical to
`classify_rules_eval_afe0806.json` in every grid cell. The planned GPU rerun did not run (L4
capacity; job 6b049e57ca8b51043c3a4932 cancelled before it started).
- sonar-9z8, gone since D-141: both boats of the thinned cluttered scene are right at 1496.8 and
  1500 m/s (seeds 61-67, CPU). A 0.2 % sound speed changes two scatterers' relative two-way phase
  by 2·f·d·Δ(1/c), one cycle at 3 m and 120 kHz, so each hull's interference is drawn afresh; with
  list-order association a wake track then took the hull's detections (reproduced with
  `association: in_order`). Guard: `test_realistic_scenes.py::test_cluttered_classes_do_not_turn_on_the_sound_speed`.
- sonar-9ir: the cluttered boat as `line` is gone in the gallery since D-154 (live chain never runs
  the corridor test); the `fls_170k_transit` anchored boat as `uncertain` is gone since D-155.
  Kept as known limits, no classifier change: `fls_120k_rough_surface` glint movers (78 over ten
  seeds, at the near edge of the surface footprint; fit RMS, extent, SNR, hits and speed/sigma
  each separate at most 57 of 78 without losing a boat track, a threshold fitted to one scene),
  and the winter-sand ship's speed (a stray detection 18-22 m beyond the hull at ping 1 on every
  seed; the opt-in per-axis outlier test of D-159 removes it, its default stays the owner's call).

### D-171 Motion estimation: range windows with no seabed at the footprint (owner chose both remedies, issue sonar-7r7)

The heave tilt left in `minefield_sas` after D-164 did not come from its objects or ripples:
removing the objects' echoes, flattening the ripples, or giving the footprints the relief each moved
ping 0's height by less than 0.07 mm (it stayed 1.76 mm off against a 0.34 mm DVL error). Both SAS
gallery scenes laid their seabed only to 48 m ground range (slant 50.3 m) while the receive window
reaches slant 53.6 m. The last of the eight delay windows (slant 52.6 m) was 38-39 dB below the
ping's median window, with no seabed at the broadside footprint Eq. 4 is evaluated at; without
receiver noise its coherence stayed 0.99. At the truth its delays read 0.94 of the model's plus
18.9 µm per ping pair, and as the window with the lowest depression angle it moved the split between
sway and heave (the DVL term resisted it, which is why a smaller λ2 made it worse).
- `MotionOptions.min_window_energy_db` (default −20; ours: windows with seabed in the main lobe were
  −9.4 to +18.3 dB of the median, a window past the seabed's edge −34 to −52 dB) drops a window that
  far below its ping pair's median energy. `None` restores the old behaviour.
- Both scenes' seabed now spans y 34-53 m, covering the window's ground extent of 35.0-51.5 m. That
  re-draws each scene (scatterers: `minefield_sas` 14,870 -> 20,558, `wreck_sas` 9,180 -> 13,555), so
  before and after are not paired.
- Full size (host): `minefield_sas` sway 0.140 -> 0.022 λ, heave 0.101 -> 0.044 λ, ping-0 height
  error −1.76 -> +0.03 mm. What got worse: `wreck_sas` sway 0.045 -> 0.053 λ (heave 0.083 -> 0.044 λ),
  the new draw's motion; the gate drops nothing in the new scenes. Object levels in the gallery rows
  move with the draw (`minefield_sas` 18.9 / 21.7 / 23.4 dB, `wreck_sas` 26.9 dB). On the old scenes
  the gate alone gave 0.021 / 0.019 λ. Acceptance seeds 1-10 identical, nothing dropped.
- `test_sas_scenes_with_unknown_motion`: "estimated sharpness > 2 × navigation-only" is replaced by
  "navigation < 0.6 × true and the estimate wins back more than 80 % of the loss" (still est > 0.8 ×
  true): in the new `minefield_sas` draw the true-motion image is only 1.91 × the navigation one.
- Not done: the gallery and the `minefield_sas` benchmark row are not regenerated; lambdas unchanged.
- Full GPU suite 1875 passed, 92 skipped (L4, HF job 6b04f660c97b8ed4fcd7093a, with D-172, D-173).

### D-172 Echo groups within one gate of their object; gallery places Popoto echoes from the array at reception (owner's choice, issues sonar-a5x, sonar-960)

sonar-960: the 12 km object's cross-range error was not the cone angle (bias for its arrival at
most 6.7 m at the first ping, under 0.01 m at the report; earliest arrival about 0.05° elevation),
nor heading (zero in the scene), nor the tracker (the filtered state errs as much as the measured
line). The gallery placed Popoto echoes from the array at transmit while the simulator receives
where the array is when the echo arrives: v·2R/c = 24 m behind at 12 km. `gallery.echo_position`
places them from the array at reception (ours, first order; |d|²/R 0.05 m at 12 km; motion during
the pulse, at most 0.75 m for a 1 s pulse, left out); tools only. Full size (a10g-small, job
6b04b891c97b8ed4fcd689d3, against the afe0806 L4 captures re-scored; track counts identical):
`popoto_12km` median to hull 8.2 -> 2.3 m, mean cross-range from centre −55.9 -> −31.8 m, worst
−118.8 -> −94.7 m; `popoto_far_mover` −16.3 -> −4.2 m; `gravel_30m` −15.8 -> −9.7 m. What got worse:
`winter_sand_200m` now reads ahead of the ship's centre, mean −7.0 -> +9.2 m, worst +24.5 m (distance
to the hull unchanged, 0.6-4 m). No class, track-count or false-mover change. What remains: low-SNR
and glinting end points (0.4-0.8° of bearing; 12 km seeds k5, k6 79-95 m), and a 33-41 m range error
in `popoto_far_mover` and `gravel_30m` this does not touch. Gallery not regenerated.

sonar-a5x: grown transitively (D-155), copies of the 12 km ship's arrivals up to 900 m apart in range
(gate 800 m) were one object on 10/10 seeds. Now a group member farther than `max_excess_range` from
the group's object in range is left out and free to form its own group (when it is the anchor, the
anchor groups again from what is left); both engines, parity-tested. Measured group extents are at
most 193 m. The new scene `popoto_12km_two_ships` (two ships 852 m apart at one bearing and velocity,
`classify_rules_eval.py` study scene, not in the gallery): ship B its own mover on 0/10 seeds before,
10/10 after, ship A unchanged, 0 false movers. The 120 gallery captures do not change.
`max_excess_range` is not scaled to water depth: excess grows with range, and the 1000 m scene has the
smallest groups (21 m). Two tests now expect the opposite result by design
(`test_a_group_grows_through_any_member_not_only_the_anchor` moved to a one-gate case;
`test_transitive_echo_grouping_matches_python`'s third arrival is its own object). Full GPU suite
1875 passed, 92 skipped (HF job 6b04f660c97b8ed4fcd7093a).

### D-173 Sensor Bridge end to end with `serve --scenario` and `engine: cpp`; thread count from the CPU quota (issue sonar-8rj)

`serve --scenario` and `receive` with the chain's C++ engine now have real-time loopback tests
through Sensor Bridge; each run's reports equal the direct chain of the same engine on the same pings
(largest difference 0.0). Nothing in either path needed a fix. `serve` simulates every ping before
announcing the board, so `receive --find-timeout` (opt-in; default 30 s as before) lets `receive`
wait longer. The loopback test's OpenMP count is half of min(affinity, cgroup CPU quota): on
quota-limited hosts the affinity lists the host's CPUs (a10g-small listed 16 or 96 for a 3- or 4-CPU
quota), and with 8 or 48 threads the receiver waited 44-184 ms behind CFS throttling (jobs
6b04b897c97b8ed4fcd689e1, 6b04b837c97b8ed4fcd68929); unchanged where there is no quota (L4: 4).
- Pace (ours, a10g-small, 160 runs, job 6b04bedcc97b8ed4fcd6969f): with the cap, 0 of 80 runs lost
  frames at pace 1.0 and 0.5, default and halved buffer, worst delay at most 2.4 ms; without it 2-20 of
  20 per configuration, worst 169 ms. Cost +40-57 % chain time with one thread there (+10 % on L4 with
  four). Pace 1.0 kept (D-153). Not repeated on L4 (no capacity at the time).
- Open: `serve` has no `--backend` (realistic scenarios simulate on the host).
- Full GPU suite 1875 passed, 92 skipped (L4, HF job 6b04f660c97b8ed4fcd7093a), including the new
  loopback tests and the quota-aware count on L4.

### D-174 Gallery regenerated with D-164, D-171 and D-172 (issue sonar-gae)

Full-size CUDA run on an NVIDIA L4 (HF job 6b05051fc97b8ed4fcd725d4, tree 77f5b98); it also takes in
D-156 and D-157, which the committed gallery predated too. Every change traces to one of those
decisions; detection and confirmed-track counts are unchanged in every scene, image-only scenes and
animations agree to at most 2 grey levels (except the two re-drawn SAS scenes), and no named object
changed class. The classes equal the gallery seed of `classify_rules_eval_afe0806.json`.
- Better: `fls_120k_rough_surface` movers off a moving object 13 -> 6, `fls_170k_traffic` 1 -> 0,
  `winter_sand_200m` false movers 1 -> 0 (ship 19.8 -> 2.7 m from its hull); `popoto_12km` ship 67.9 ->
  2.1 m from its hull; SAS sway `wreck_sas` 0.089 -> 0.053 λ, `minefield_sas` 0.081 -> 0.022 λ.
- Worse or not comparable: `popoto_12km` ship speed 6.05 -> 5.68 ± 0.50 m/s (true 6.0); the SAS
  sharpness numbers are new scene draws (estimate 0.978 and 0.991 of the true-motion image, against
  0.978 and 1.006 before).
- `tools/gpu_job.sh`'s gallery mode now uploads to `GPUSONAR_UPLOAD_PATH` (default `gallery`). No
  test expectation changed (`test_documented_full_size_classes` passes on the new `gallery.json`).
- Not done: the `minefield_sas` echo-synthesis benchmark row in RESULTS is of the old scene (14,870
  scatterers, now 20,558); `docs/SCENARIOS.md`'s gallery tables are stale.

### D-175 Tracker: earlier detections at the track's time; bounces never a group's object; pings without time counted (issues sonar-uc5, sonar-bal, sonar-4bs)

None of these moves a default result: with increasing transmit times and a ping interval longer than
R/c the tracker golden (D-156), the classes golden (D-157), the engine-parity tests and the C++
tracker, classify and report tests are unchanged. Both engines throughout.
- sonar-uc5: a detection timed before the time a track already holds is gated against and taken by
  it at that time (dt = 0; ours), not predicted back to, so a track's history stays in time order.
  One target cannot give such a detection when transmit times increase by more than the echo delays
  differ; it comes from another arrival or from pings without timestamps. Rejecting it would lose
  closing targets on such sources; reordering would be an out-of-sequence update. A held missed row
  was already left out of the fit (D-157; a test now holds it). Changed on purpose: the C++ test
  `Tracker.APredictionToAnEarlierDetectionStillGrowsTheDepthVariance` became
  `Tracker.AnEarlierDetectionIsTakenAtTheTracksTimeWithItsDepthVariance`; the illustration in D-165's
  test now reads 874 m/s from the old fit rule (asserted > 500, was > 1000; the guard is unchanged).
- sonar-bal: echo grouping is its own step (`group_echoes`), run after the elevation step finds the
  tracks outside the water; those are made multipath first, so a bounce is never an anchor or an
  object, joins a group as a member, and otherwise has no parent. No track's parent is multipath.
  With no bounce the grouping is unchanged. Not measured on gallery captures with `multipath` on.
- sonar-4bs: pings whose transmit time does not increase are counted (`pings_without_time`, a report
  member once above 0; one warning). `tracker.ping_interval_s` (opt-in, default off;
  `tracker_ping_interval_s`) advances the tracker's clock by that interval on such pings; no
  interval is assumed by default.
- Full GPU suite 1925 passed, 93 skipped (L4, HF job 6b0536c7ca8b51043c3b3ed9, with D-176 to D-178).

### D-176 Popoto gallery bearing: opt-in sub-grid estimate; the Kalman state's lag explained (issues sonar-4s2, sonar-i6i; no default change)

The gallery's Popoto bearing is the power-weighted mean over a hit's flagged cells; at low SNR that
set is a small, noise-cut part of a 5° beam on a 0.5° grid, worse than the strongest row's grid
bearing below 12.5 dB. `tools/gallery.py` `POPOTO_BEARING = "parabola"` (opt-in; `classify_rules_eval
--popoto-bearing`) takes the vertex of a parabola through the bearing profile in dB about the
strongest flagged row (ours, algebra in the docstring; the textbook reference could not be opened:
unconfirmed). Measured on `--profiles` captures (L4 job 6b0506e2c97b8ed4fcd729d6, six Popoto scenes ×
10 seeds, `tools/popoto_bearing_study.py`): per-hit RMS bearing error at 12 km 0.447 -> 0.297° at
9-11.5 dB, 0.138 -> 0.109° at 14-17 dB, no gain above 17 dB; 12 km report worst cross-range −94.7 ->
−50.2 m, speed error 0.77 -> 0.41 m/s. Kept opt-in because it is worse elsewhere: 12 km distance to
the hull median 2.3 -> 3.3 m, `gravel_30m` mean cross-range −9.7 -> −11.8 m, `far_mover` mean |cross|
5.3 -> 5.8 m. Not adopted (offline only): an SNR-weighted line fit (12 km worst −94.7 -> −68.1 m).
The repeatable per-ping bias of the five-point ship side-on (glint) is physics and is left.
sonar-i6i: on current code the Kalman state is not farther than the fitted line at 12 km (4/10
seeds). Where detections are clean it trails the report by the zero-velocity birth lag under the
gallery tracker's 1° bearing sigma (scatter is 0.05-0.3°): `winter_sand_200m` +15.4 to +17.0 m on
every seed against +15.7 m noise-free. The 12 km outliers (seeds 56, 57) are low-SNR end points both
estimates share; track hopping between arrivals 35-109 m beyond the ship on seeds 53, 56, 57, 59.
`classify_rules_eval --profiles` records each hit's bearing profiles for such studies.

### D-177 Every options dict refuses a key it does not read; index skips; failed stops; the beads hook in worktrees (issues sonar-1rf, sonar-2ht, sonar-qnk, sonar-dh6)

- sonar-1rf (extends D-166): PingAssemblerCpp, PingDecimator, StreamAssemblerCpp,
  live_decimate_plan/record, live_synchronise, NavigationCpp (with water_depth), WaterDepthTracker,
  SoundSpeedLadder, live_family_profile, live_water_profile, live_derive and live_detector_plan refuse
  an unread key, naming it and the nearest key read; `sonar_live` key=value says "unknown option K; did
  you mean J?" (`live::nearest_key`, the D-166 rule). Python: decimate, nav (`from_config` and
  `program_options` now refuse unknown nav keys), elevation.profile (with `built` and `supplied`),
  detector_plan and budget_threshold_db. A reference dict may carry the sonar file's mode, level and
  channels. water_depth reads `source` ("altitude" is from_altitude; the two must agree). What could get
  worse: an out-of-repo configuration with a stale nav, profile or water_depth key now stops; all 18
  in-repo deploy and build files pass. Not covered: the stages' own key=value readers (tonal_,
  multipath_depth_, coverage_).
- sonar-2ht (amends D-161): a ping-index jump of d counts d − 1 lost pings only if, besides the timing
  rule, a given-up or still-awaited sample span lies between the two transmits (a ping lost whole
  leaves samples missing; a renumbering leaves none; ours). Both engines. Fuzz, 4000 provoked streams:
  renumberings counted as loss 1268/1378 -> 0/1378; with an unrelated lost block 574/622 -> 162/622;
  losses missed 35/2000 -> 35/2000. Still miscounts a skip that also lost samples in that gap, and a
  real loss with jitter beyond half an interval.
- sonar-qnk: `run_receiver` keeps an application alive only after a failed stop (closes D-163's open).
- sonar-dh6: `.beads/hooks/pre-commit` exits before calling bd in a linked worktree
  (`BEADS_HOOK_IN_WORKTREE=1` overrides), so a worktree commit no longer exports into the main
  checkout's `issues.jsonl`; main-checkout commits are unchanged (tested with a stub bd).

### D-178 The gallery's range in the multipath scenes is the leading edge (owner's choice, issues sonar-ca3, sonar-a98)

The 33-41 m range error of `popoto_far_mover` and `popoto_env_gravel_30m`, and `flank_patrol`'s +29 m,
came from `detect.cluster`'s power-weighted centroid of one detection made of several surface and
bottom arrivals, the later ones stronger. Separated with one-ping host simulations (m from the
target's centre): direct path only far_mover +0.7, gravel −0.3, flank −1.3 (timing and placement
correct); point target with multipath gravel +34.1, flank +27.9, far_mover's echo split from +2.7 to
+100 m (strongest +41); LFM coupling f0·v·T/B 0.9-2.0 m (far_mover), 3.1-4.8 m (gravel), 0.85-1.2 m
(flank), nearer for these closing targets. `tools/gallery.py` (`RANGE_ESTIMATE`, `--range-estimate`)
and `tools/classify_rules_eval.py` now default to `first` (`Detection.first_range`; the live chain's
default since D-160); `centroid` stays selectable; a capture with no recorded estimate is read as
centroid and a default run refuses to reuse it. Distance to the hull, ten seeds (Popoto L4 job
6b05072eca8b51043c3b0d03 against the D-172 captures; flank on host from saved pings):
`far_mover` 36.9 -> 0.4 m; `flank_patrol` submarine 30.0 -> 1.1 m, wreck 6.9 -> 5.2 m, false movers 1
-> 0. What got worse: `gravel_30m` 34.3 m beyond -> 10.1 m short (the near end of a hull at about 15°
aspect read at the cluster's mean bearing, plus the closing up-chirp's coupling), mover -> uncertain
on seed 52; `winter_sand_200m` 0.8 -> 2.5 m; `popoto_12km` median 2.3 -> 3.4 m, seed 60 3.0 -> 65.4 m
(track hopping, −65 m cross-range); seeds 56 / 57 better (34.8 -> 5.4, 18.6 -> 7.0 m). `flank_patrol` is
now classified (`run_flank` -> `classify_and_score` through `flank_tracks_yx`: its tracker's (x, y)
was read as (y, x)) and `classify_rules_eval --scenes flank_patrol` scores it; its `small_vehicle` has
no detection near it on any ping. The committed gallery (D-174, centroid) is regenerated separately.
Full GPU suite 1925 passed, 93 skipped (HF job 6b0536c7ca8b51043c3b3ed9).

### D-179 Gallery regenerated with D-178 (issues sonar-27y, sonar-627)

Full-size CUDA run on an NVIDIA L4 (HF job 6b054768c97b8ed4fcd78015, tree 4a80876). Only
`flank_patrol` and the six Popoto scenes moved, each to the gallery seed of D-178's ten-seed run;
every other scene equals the D-174 gallery to GPU rounding (relative 4e-4 at most; detection, track
and class counts equal), so D-175 moved nothing in the gallery.
- Better: `far_mover` 34.2 -> 0.9 m from its hull (confirmed tracks 4 -> 3); `winter_sand_200m` speed
  5.24 -> 6.06 m/s (true 6.0); `flank_patrol` submarine 25.8 -> 1.0 m, wreck 5.8 -> 3.5 m.
- Worse: `gravel_30m` ship 33.7 m beyond -> 10.0 m short of its hull (speed 5.92 -> 5.79 m/s);
  `popoto_12km` 4.7 -> 9.8 m from the ship's centre (2.1 m from the hull in both).
- `flank_patrol` is now classified in the gallery: 1 mover (the submarine, 3.02 m/s), 2 static, 2
  uncertain; the small vehicle has still no detection.
- No test expectation changed. `docs/SCENARIOS.md`'s gallery tables, last updated at 1442de5, now quote
  this run.

### D-180 YAML numbers PyYAML reads as strings; sonar_live_hsb's navigation (issues sonar-iwq, sonar-plk)

PyYAML follows YAML 1.1, where a float needs a dot and a signed exponent, so `38.0e3` loads as a
string. Our loader (`config.load_yaml`) has always read it as a float, but plain `yaml.safe_load` (the
sediment table, the WAV and navigation adapters, any outside script) does not, and the C++ dictionary
readers then refuse it with a cast error: the rehearsal's sync reference failed in PingAssemblerCpp and
PingDecimator. Every shipped YAML now writes such numbers with a signed exponent (`38.0e+3`): 77 values
in 48 files, identical through our loader (checked file by file). `tests/python/test_yaml_numbers.py`
loads every tracked YAML with plain PyYAML and fails on any such string. `SedimentFit.yaml_row` writes
`1.0e-05` where `:g` gave `1e-05`. `tools/sonar_live_hsb.py receive` now builds its navigation with
`NavListener.from_config`, as `tools/sonar_live.py` does: water_depth, keep_s, surface_z,
profile_step_m and history_margin_s used to be dropped silently, and an unknown nav key is now refused
before the board is looked for (D-177). Open: five readers still use plain `yaml.safe_load` on
user-written files. Full GPU suite 2065 passed, 93 skipped (L4, HF job 6b054bd0c97b8ed4fcd78494).

### D-181 Opt-in LFM range-Doppler correction; the leading edge's costs explained (issues sonar-fyj, sonar-l4i, sonar-oqa; no default change)

sonar-fyj: `range_doppler: off | ownship | full` in both live-chain engines (`range_doppler=` for
`sonar_live`; `gallery --range-doppler`, `classify_rules_eval --range-doppler`), default off. Factor
f0·T/(f_stop − f_start), signed by the sweep, 0 for a non-LFM waveform (ours, derived from the LFM
matched filter in `tests/python/test_lfm_range_doppler.py`; textbook unconfirmed). `ownship` adds
factor × the ownship's closing speed along each detection's line of sight (mean of the transmit and
receive pose velocities) to its range, placement and time; `full` also moves each reported track by
factor × its own closing speed (from the fitted velocity). Engine parity within 0.05 m. Measured on
gallery captures (L4 job 6b0551faca8b51043c3b5e25, offline replay, `first` -> `first` + `full`, median
distance to the hull over 10 seeds): gravel 10.1 -> 7.5 m, flank submarine 1.1 -> 0.3 m, winter_sand
2.5 -> 2.1 m, 12 km 3.4 -> 2.7 m. What got worse: far_mover 0.4 -> 0.9 m; fls_170k_traffic boat_closing
4.6 -> 5.9 m from its centre (the coupling had hidden part of its propeller offset, D-169; hull 0.00 m
both). No class or false-mover change. Kept opt-in. The live chain is held by its tests only (not
measured on these scenes).

sonar-l4i (explained, no change): `gravel_30m`'s leading edge reads 10.3 m short at the last ping =
hull geometry 7.8 m (the near end's range read on the cluster's bearing, 43-51 m inside the near end)
+ LFM coupling 3.0 m + leading edge +0.4 m. Seed 52 loses its mover because the nearer leading edge
brings the ship's detection within the tracker's gate of a weak hit at the near end, 80 m along the
hull; the tracks swap on ping 2 (speed/sigma 6.9 < 7.66). `full` removes the coupling term only.

sonar-oqa (explained, no change): at 12 km the gallery tracker's 1° bearing sigma is 210 m across,
and about a dozen arrivals lie inside one 4σ gate, so association is decided by small likelihood
differences; the leading edge reshuffles which arrival each track takes (seed 60 3.0 -> 65.4 m, the
mover left with an 11.5 dB hit 1.1° off). Tighter bearing sigmas move the hop to other seeds (0.5°:
seed 60 65.4 -> 3.5 m but seed 53 1.1 -> 29.2 m). Open: association among many arrivals in one gate.
Full GPU suite 2078 passed, 93 skipped (L4, HF job 6b055ee4c97b8ed4fcd799db).

### D-182 The handoff documentation set; stale statements corrected (issue sonar-4mu; no chain change)

At the owner's request, `docs/handoff/` holds a set from which an engineer or a coding model can
continue the hardware and software work: `README.md` (entry point, reading orders, state, risks,
document map), `HARDWARE.md` (22 unknowns, block diagram, the assumptions register with status and
config key, the six builds, link budget, interfaces, a 19-step bring-up plan), `SOFTWARE.md`
(architecture, repository map, the two engines, every live-chain stage's defaults and their
decisions, tools, how to add things), `TESTS.md` (how to run, a map of 114 pytest files and 15
GoogleTest binaries, decision-to-test table, false failures, gaps), `SIMULATION.md` (what is not
modelled, pipeline, models table, 76 scenarios, validation), `EXAMPLES.md` with `examples/` (twelve
examples run on the CPU with commands, times, outputs and pictures), `OUTPUTS.md` (the gallery by
family and the headline results). `docs/HANDOFF.md` is refreshed to this tree; `CLAUDE.md` and
`AGENTS.md` point to the set.
- What writing it found, filed as issues: `sonar_live --emulate` lays out the receive window at the
  nominal sound speed and fails board synchronisation on the multi-board trial scenarios (sonar-8zj);
  one GSFB block carries every channel on one sample counter, so several boards' streams need a merge
  that is not designed; `transmit.time_ns` drifts with the board clock (ours, not tested); a grating
  lobe is in view beyond ±42° at 6 kHz on five of six builds; the Sensor Bridge loopback has only
  run at 32 channels and 64 kHz; function defaults (`detect.Tracker()`, `classify_tracks()`) differ
  from the chain's.
- Forty stale or contradictory statements in existing documents were corrected against the code,
  the configs and the JSON files (RESULTS, STATUS, DEPLOYMENT, SCENARIOS, ARCHITECTURE, REFERENCES,
  SIM_CALIBRATION, JETSON, HOLOSCAN_NOTES, the root README). Changed besides text: the architecture
  diagrams' table (`tools/architecture_diagrams.py`: device builds of the beamformers, motion delays,
  reverberation FFTs, the multiscale statistic; Sensor Bridge frames; echo grouping) with
  `tests/python/test_architecture.py`'s pin of CUDA rows 7 -> 10, because the table was wrong; the
  build diagrams (direct sound 3 dB low before, D-102); `popoto_parts`, `DESIGN.md` and
  `popoto_stage2.json` regenerated through their generators (the full-rate GPU ping is measured:
  0.479 / 1.028 / 1.495 s on an L4); the simulator-calibration self-check reads +5.9, −4.0, 38.6 dB,
  0.16 dB RMS (was +6.1, −4.0, 38.0, 0.17; the cause of the move not traced). New tools:
  `tools/handoff_hardware.py`, `tools/software_diagrams.py`.
- The last whole CPU suite on record is the tree of D-157 (157 of 157); the last full GPU suite is
  2078 passed, 93 skipped at D-181. Checked for this merge on the CPU: test_architecture,
  test_tools_help (78), test_yaml_numbers, test_hardcoded, test_design, test_buildspec, test_simcal.

### D-183 `source.sound_speed`: the receive window's layout speed in every receiver; `sonar_live` reports an error on its UDP path (issue sonar-8zj)

Found by running the examples of the handoff set. `tools/sonar_live.py <deploy> --emulate <scenario>`
received at the sonar's nominal 1500 m/s what the simulator made at the scenario's 1496.8 m/s
(D-140): `popoto_testauv_trial_1row` ended on its first ping with "sync: reference burst not found
on a board" (the tail burst 2.5 ms from where it is looked for, search 2 ms), and one board read
ranges 0.2 % long (1001.5 m for 999.5 m).
- The Python `flank_udp` source takes `source.sound_speed` (m/s; absent, None or 0: the sonar
  file's nominal speed, so existing deployments are bit-for-bit unchanged): the speed at which
  `receive.window.range` becomes the window's first sample and length. `--export-template` (the C++
  program's template) and `sonar_live_hsb.py receive` already read the same key; no change to
  `libs/live`, whose receiver takes the window from the template. `--emulate` sets it to the
  scenario's speed. Now: four pings synchronised (worst residual 0.43-0.47 µs), target at 498, 498,
  496, 496 m (direct chain 498, 498, 498, 496: the stream opens on a whole sample, up to half a
  sample from the simulated record); one board 999.5 m, equal to the direct chain within 0.05 m.
- Open (owner's question, kept as it was): the key both lays out the window and is the speed a
  record carries until navigation or `chain.sound_speed` replaces it. Not verified: `sonar_live_hsb.py
  receive` against `serve --scenario` of a multi-board scenario at a non-nominal speed.
- `apps/sonar_live.cpp`, UDP path: an exception from a ping (e.g. that sync failure) unwound past
  two joinable threads and the program died with "terminate called without an active exception"
  (exit −6), the reason lost. The receiver and navigation threads are now stopped and joined however
  the block is left; the program prints `sonar_live: <reason>` and exits 1 as on its other error
  paths. Not changed: the Holoscan `UdpSourceOp` joins in `stop()` only.
- `tools/build_diagrams.py`: the heading wraps inside the page and close row labels are spread with
  leader lines (the 2×24 example ran off the edge); no number changes.
- Tests: `tests/python/test_emulate.py` (15; five fail without the fix; the program case pins the
  message and exit status), `tests/python/test_build_diagrams_layout.py` (8). Full GPU suite 2102
  passed, 95 skipped (L4, HF job 6b060a4b1460ffe1d5e93111), which also covers the merge of D-182.

### D-184 One stream per capture board, merged in front of the assembler (issue sonar-6f9; no default result moves)

Until now every receiver took one whole-array stream, which no hardware in the plan produces: each
board (8 channels) streams on its own counter. What it costs, first:
- A wire-format change, ours: a per-board `GSFB` version 1 block sets flag bit 1 and carries its
  board id (u16) at header offset 44, both reserved until now. Whole-array blocks leave them zero
  and are read bit for bit as before; a whole-array receiver now refuses a per-board block by name
  instead of taking it for a block of the wrong channel count. **The owner accepted the field
  (2026-11-25).**
- A span lacking any board's part is left out and the ping it falls in is dropped (`missing: drop`,
  **the owner's choice of default**); `zero` keeps the ping with that board's channels silent. One
  dead board therefore costs every ping until `zero` is set.
- Boards are aligned by their capture starts (and their rates when configured, D-185); everything
  finer is the synchronisation stage's, which searches 2 ms. So the capture starts must agree with
  true time to about 0.8 ms (2 ms less 0.3 ms of simulated board offset and 0.9 ms of drift between
  two boards 50 ppm apart over the face's 17.9 s window). PTP time does that; NTP may not. Two
  free-running boards 50 ppm apart leave the search in 40 s unless their rates are configured.
  Unconfirmed: how the boards stamp their capture start.
- Not merged: `build/bin/sonar_live --listen` (passive), the Holoscan application and
  `sonar_live_hsb.py receive` refuse `stream_boards`; the relay `tools/board_merge.py` goes in
  front of them. Never run against a real board or a real network.

What was built: `BoardMerger` in both engines (`python/gpusonar/sources/board_merge.py`,
`libs/live` `board_merge.hpp/.cpp`, bound as `BoardMergerCpp`), in front of the unchanged
`PingAssembler`, so D-161 and D-177's rules for order, holes, lost pings and restarts hold as they
were. Keys `source.boards` (`stream_boards=1`, `stream_board_channels`, `stream_board_missing`,
`wait_s` 0.05 s, `reference`): the map defaults to the sonar file's `receive.boards` in order; the
merged stream runs on the reference board's counter, capture start, transmit marks and gain code.
Per-board statistics end a run (`sonar_live: boards {...}`). `tools/sonar_live.py --emulate
SCENARIO --per-board` rehearses it over UDP. Rules and statistics: `DEPLOYMENT.md`, "One stream
per board".
- Tests: `tests/python/test_board_streams.py` (23: merged pings equal the whole-array pings in both
  engines, loss, restarts, a silent board, a late start, refusals, the program and the relay).

### D-185 A board clock's rate: configured, estimated, never assumed (issue sonar-ubg; defaults unchanged)

`transmit.time_ns` is capture start plus the transmit's sample index over the nominal rate, so a
board clock fast by 25 ppm stamps a ping 90.0 ms late after an hour, and navigation interpolated
there gives the pose of 90 ms later: 0.18 m at 2 m/s, 0.18° at 2°/s of roll (both engines). What
is still wrong, first:
- Nothing is corrected by default (`clock_ppm` 0: the stamp as before, bit for bit). The rates are
  unconfirmed hardware facts and live in config.
- Ranges are not corrected: a board fast by e reads every range e × R long (25 ppm: 5 cm at 2 km,
  0.32 m at 12.7 km). The record's sample rate stays the nominal one.
- The estimate from arrival times takes the host clock's steering and any drift of the network's
  latency whole: a latency wandering 1 ms over 300 s biased it by 1.6 ppm (10 minutes of stream) to
  16 ppm (1 minute) without raising its sigma. It was tried on synthetic delays only (shaped like
  D-162's L4 loopback: within 0.013 ppm after 10 minutes, 0.4 ppm after 1), never on a real network.
  Its discipline is therefore opt-in.

What was built: `source.clock_ppm` (`stream_clock_ppm`) for a whole-array stream and
`source.boards.clock_ppm` (`stream_board_ppm`) per board: the stamp becomes capture start +
tx / (fs (1 + ppm × 1e-6)), and the merge of D-184 aligns boards with their rates (a rate known to
0.1 ppm holds two boards inside the 2 ms search for 5.6 hours). `source.clock: {}`
(`stream_clock_estimate=1`): `ClockEstimator`, per second of stream the earliest arrival, a
least-squares line over the last 600 s, reported at the end of a run (`ppm`, `sigma_ppm`, `span_s`,
`bins`); `discipline: true` stamps with it once it spans 300 s with sigma at most 0.5 ppm. The
report's `sync.drift_ppm` already gives each board against the first; the first board's own rate
needs a reference time (1PPS) or this estimate.
- Tests: `tests/python/test_board_clock.py` (20, both engines and the program).

### D-186 The Sensor Bridge loopback at Popoto rates loses frames; `UdpSourceOp` joins its threads in its destructor (issues sonar-x8v, sonar-ogg)

Every Sensor Bridge loopback so far ran `flank_32` (32 channels, 64 kHz, 49 Mbit/s). At a Popoto
build's 32, 64 and 96 channels and 102.4 kHz (78.6, 157.3, 235.9 Mbit/s) it is not lossless here:
- Real time, default socket buffer, C++ engine, 20 runs each on Hugging Face a10g-large (12-CPU
  quota, `rmem_max` 212,992 B; no L4 could be had for the study): frames were lost in 12, 20 and
  20 runs (median 1, 11 and 11.5 frames a run; at most 8, 57 and 66), and a ping was dropped in 2,
  1 and 3 runs. Python engine, 10 runs each: 6, 10 and 10 runs with loss; at 96 channels its
  receiver queue overflows (median 368 frames lost), so it is not real time there and is not tested.
- Every datagram lost was refused by the receiver's full socket buffer: the socket's kernel drop
  count equals `RcvbufErrors` in all 210 study runs. The chain keeps up (C++: 0.13, 1.40 and 2.17 s
  a ping, median, against the 3 s interval).
- Not cured by the largest buffer the container allows (425,984 B), by 200-sample frames (worse:
  every run lossy, up to 286 frames), or by the chain's OpenMP on 2 threads.
- The cause is not located. The loss is regular, about 3 frames a ping (16 of 20 C++ runs at 64 and
  96 channels lost 9 to 13 frames), which points at something the receiving side does once a ping.
  The emulated sender was itself 10 to 34 ms late at worst in every run.
- Untested: an L4 for the study; a host with Sensor Bridge's own setup (`net.core.rmem_max`
  31,326,208, an isolated receiver core), which these containers cannot set; the RoCE receiver; a
  full 12.7 km window (same stream rate, 17 s and about 1.7 M samples a channel per ping).
- On the same a10g-large machines 2 of 12 runs of the existing `flank_32` loopback tests missed
  their 5 ms worst-delay bound (6.7 and 6.3 ms); those tests are unchanged and L4 stays the
  reference flavour.

The new loopback test (`tests/python/test_hsb.py`, C++ at 1, 2 and 3 rows, Python at 1 and 2)
asserts what held in every run: losses only as counted refusals of the socket buffer, every ping
accounted for, no malformed frame, the chain inside the ping interval, and the reports and the
recording of the pings that came through equal to the direct chain's on the same pings. It then
**ends as an expected failure (xfail) carrying its numbers when frames were lost or the worst
receive delay reached 5 ms**; a run with neither passes. A CPU-only test checks that each trial is
framed at its build's rate (`docs/handoff/hardware_builds.json`) and assembles back. The study is
opt-in (`GPUSONAR_HSB_STUDY=1`). The receiver's queue is set to 4 s of stream in these tests; the
tool's default 256 frames is 1.0 s at these rates (kept; changing it is the owner's).

`UdpSourceOp` (sonar-ogg) now sets its stop flag and joins its two threads in its destructor as
well as in `stop()`, and `sonar_live_holoscan` returns 1 on an exception that is not a
`std::exception`. **No failure was shown before the fix**: with the old operator, Holoscan 4.6
deactivates the graph after another operator's exception, calls `stop()` on every operator, and
`run()` rethrows; the application exited 1 with its reason (also when an operator's `start()`
throws). The destructor guards a path no run produced. The new test
(`tests/python/test_holoscan_cpp.py`) pins the exit status and the reason and passed before the
fix too.
- Jobs (a10g-large): 6b063a21d0582cfbbefd60a3, 6b0641ce1460ffe1d5e98956 (study),
  6b064e8e1460ffe1d5e9a072 (3 repetitions; its recording comparison was then corrected to compare
  with the direct chain's recording, the two differing from the assembled samples by float32
  rounding with or without transport). 
- Full GPU suite on the merged tree of D-184 to D-186: 2150 passed, 97 skipped, 4 xfailed, none failed (L4, HF job 6b06553a1460ffe1d5e9aa10). The four
  expected failures are four of the five Popoto-rate loopback cases; one passed (no loss, worst
  delay under 5 ms). The job does not print which or their numbers (the run has no `-rx`), so the
  only L4 result is this count. The suite's two further skips are the opt-in study and one not
  looked up.

### D-187 The Popoto-rate loopback loses its frames at the start of the run, not once a ping; the receiver queue defaults to 4 s of stream (issues sonar-bep, sonar-us7; amends D-186)

D-186 said the loss was "regular, about 3 frames a ping". **That was wrong**: it divided a run's
loss by its four pings without locating it. Located now (`test_popoto_rate_loopback_study` prints
every missing frame's number and place, `POPOTO_LOSS`):
- In all 52 runs located (C++ engine, real time, default socket buffer; 34 on an L4, jobs
  6b0666a6d0582cfbbefda036 and 6b066a811460ffe1d5e9c92a with the queue at 4 s or 12 s; 18 on
  a10g-large, job 6b066a2cd0582cfbbefda4be) every frame that never arrived was among the first
  after the receiver connected: frames 1 to 5 on the L4 at 4 s of queue, 1 to 13 on a10g-large.
  Nothing was lost later, all four pings were reported in every run, and the worst receive delay
  after the start was 1.2 to 3.4 ms. The sender was not late there (0.07 ms, 3.9 ms apart).
- The start-up loss grows with the receiver queue (L4, 64 and 96 channels, 4 runs each): none
  with 64 or 256 frames, 2 to 5 frames with 1024, 14 to 22 with 3072. Likely cause, read in Sensor
  Bridge's source (the checkout here, `linux_receiver.cpp` `LinuxReceiver::run`; 2.7.0's was not
  re-read) and consistent with those counts but not proven by a trace: the receiver thread
  allocates the whole queue as pinned host memory and one CUDA event a frame before its first
  `recv`, while its socket is already open with a buffer of 1.1 to 3.3 frames.
- What stays wrong: the first 12 to 20 ms (L4) or 51 ms (a10g-large) of a stream are lost at these
  rates with the 4 s queue. A board must start streaming at least 0.25 s before the first transmit
  that matters; nothing enforces it. Sensor Bridge's receiver is not ours and is unchanged. Not
  explained: the earlier a10g-large runs that lost 49 to 66 frames or a ping (D-186), and whether
  a host with Sensor Bridge's socket setup loses the start too.

The receiver queue's default (the owner's ruling, 2026-11-25): `sonar_live_hsb.py receive --queue`
and `hsb.run_receiver(queue_frames=None)` hold `hsb.QUEUE_SECONDS` = 4 s of the stream, whatever its
rate (640 frames of 400 samples at 64 kHz, 1024 at 102.4 kHz), in place of 256 frames (1.6 s of
`flank_32`, 1.0 s at Popoto rates). `--queue N` still gives a count; the statistics name
`queue_frames`. What it costs: pinned memory (65,536 B a frame for `flank_32`, 131,072 B at 64 and
96 channels: 42 and 134 MB) and the start-up loss above, which the old 256 frames did not have.
What it buys: with 256 frames an L4 lost 73 to 81 frames a run at 64 channels and 588 to 618 at
96 to the queue itself (the chain takes 1.1 and 1.7 s a ping); they fell between these trials'
short windows, so no ping was lost and D-186's tests, which set 4 s, did not see it.

The test (`test_loopback_at_a_popoto_builds_channel_count_and_rate_gives_the_direct_chains_reports`)
now passes when every missing frame lies in the first 0.25 s of stream, every ping was reported and
the worst delay after that start is under 5 ms; any other loss still ends as an expected failure
with its numbers. The send log gained the send call's time per frame (`send_ms`).
- Tests: `tests/python/test_hsb_receive.py::test_the_receiver_queue_defaults_to_four_seconds_of_the_stream`,
  `tests/python/test_hsb.py::test_the_loss_location_places_missing_frames_in_the_pings_cycle`.
- Full GPU suite on this tree: running when this was written (L4, HF job 6b06728ad0582cfbbefdb5eb); the
  result is added below when it ends.
- Result of that job: **2155 passed, 97 skipped, 1 xfailed**, none failed (49 min 31 s). The one expected
  failure is one of the five Popoto loopback cases; the suite's log does not print which.
- What the remaining expected failure is (L4, HF job 6b0681011460ffe1d5e9f89f: the five cases run three
  times with the reasons printed). In each run four cases passed and one ended as an expected failure: 64
  channels with the Python engine once, 96 channels with the C++ engine twice. Each of the three lost 2
  frames (frames 1 and 2), every ping was reported, the worst delay was 1.3 to 1.9 ms, and **1 frame was
  counted damaged**: the test refuses any damaged frame. The 12 passing runs lost 0 frames at 32 channels
  and 2 to 4 at 64 and 96 (frames 1 to 4), none damaged. A damaged frame is one that arrived in part;
  the log does not give its index, so that it is the frame at the end of the start-up gap (frame 3) is
  likely, not shown. Not changed: the test still refuses it. Follow-up sonar-h0e.

### D-188 The receiver names its damaged frames; a frame damaged at the start of a Popoto-rate loopback is start-up loss (issue sonar-h0e; the owner: "accept it if the index confirms"; amends D-187)

What got worse first: the Popoto loopback test now passes in a run that D-187's test ended as an expected
failure (a frame that arrived in part, at the start). The receiver's statistics gained a key, so anything
that compares the statistics dictionary exactly must know it (two tests did; no tool or C++ code does).

- `FrameReceiver` (`python/gpusonar/sources/hsb.py`) reports `damaged_frames`: the frame numbers of the
  first 16 frames counted in `frames_damaged`. A damaged frame without a frame number is counted and not
  listed, so fewer listed than counted means some number is not known. Every other field is as it was.
- The index, measured (L4, HF job 6b0693c6d0582cfbbefdf55e: the five cases, eight runs each, 40 runs).
  One run had a damaged frame: 96 channels, C++ engine, frames 1 and 2 never arrived and **frame 3 was the
  damaged one**, the frame at the end of the start-up gap. That is one observation; D-187's three runs
  with a damaged frame did not log its number. The other 39 runs had none damaged: no loss in 15 of the 16
  runs at 32 channels (one lost frame 3), and frames 1 to 2, 3, 4 or (once) 7 missing at 64 and 96
  channels. All 40 passed.
- The test's rule (`popoto_start_only` in `tests/python/test_hsb.py`): something was lost; every frame
  that never arrived and every damaged frame lies in the first 0.25 s of stream; and every damaged
  frame's number is known. As before: every ping reported and the worst delay after the start under
  5 ms. A frame missing or damaged later, or a damaged frame whose number is not known, still ends as the
  expected failure, its message naming the damaged frames.
- Not changed: the chain, the record, any default result. Still open from D-187: the earlier a10g-large
  runs that lost 49 to 66 frames or a ping; nothing enforces the 0.25 s lead before the first transmit;
  a host with Sensor Bridge's own socket setup (sonar-02c).
- Tests: `tests/python/test_hsb.py::test_a_damaged_frames_number_is_reported_wherever_it_lies`,
  `::test_damaged_frames_keep_the_first_sixteen`,
  `::test_start_up_loss_is_accepted_only_where_every_index_confirms_it`.
- Full GPU suite on this tree: **2159 passed, 97 skipped**, none failed, no expected failure (NVIDIA L4,
  HF job 6b0699f4d0582cfbbefe0373, 49 min 44 s).

### D-189 A host check and a runbook for the Sensor Bridge measurement that needs a real host (issue sonar-02c, not closed by this)

What is still not measured: a Popoto build's stream through Sensor Bridge's socket receiver on a host set
up as its guide asks (`net.core.rmem_max = 31326208`, an isolated receiver core), and the RoCE receiver.
Nothing below measures either.

- Why not here, measured (NVIDIA L4 Hugging Face job 6b06b5ef1460ffe1d5ea6937): in the container
  `net.core.rmem_max` is 212,992 and `/proc/sys` is read-only; forcing a 31 MB socket buffer
  (`SO_RCVBUFFORCE`), a real-time priority (`SCHED_FIFO`) and a raw socket are each refused ("Operation
  not permitted"; effective capabilities `00000000a80405fb`, without CAP_NET_ADMIN, CAP_NET_RAW and
  CAP_SYS_NICE); no CPU is isolated; 8 CPUs by affinity and by quota.
- `tools/hsb_host_check.py` (new, read-only): reports `rmem_max` against the guide's value, the socket
  buffer the kernel grants and what it holds of each Popoto stream (payload bytes only: an upper bound),
  isolated CPUs and the receiver core, the CPUs for the chain (D-173's quota rule, now in this one
  place and used by `tests/python/test_hsb.py`), whether a real-time priority is possible (read, not
  tried), the raw socket, the imports, a CUDA device, RDMA devices. Exit status 0 ready, 1 not, 2 usage.
  It prints the guide's commands for what is missing and the commands that make the measurement.
- The opt-in study `test_popoto_rate_loopback_study` takes `GPUSONAR_HSB_STUDY_RECEIVER_CORE` (unset: as
  before). No default-run test and no default of the receive path changed.
- Runbook: `DEPLOYMENT.md`, "Checking a host and making the measurement".
- RoCE: untested; beyond a ConnectX card it needs a Sensor Bridge build with RoCE (`tools/build_hsb.sh`
  builds without) and new receiver code (`sources/hsb.py` uses `LinuxReceiverOp` only).
- Not verified: that `receive --receiver-core` can move the receiver onto an isolated core outside the
  process's default affinity (Linux behaviour, not tried: no isolated core here).
- Tests: `tests/python/test_hsb_host_check.py` (12).
- Full GPU suite on this tree: **2171 passed, 98 skipped**, none failed (NVIDIA L4, HF job
  6b06bbda1460ffe1d5ea75b5, 49 min 39 s): 12 more passed (the new file) and one more skipped (the new
  tool's `--help` test, which skips in the job's tarball with the other 58).

### D-190 The Python engine is real time at 96 channels through Sensor Bridge on an L4: the frame path decodes only what a window takes, and a board's rows are resampled together (issue sonar-h1f)

What got worse first: nothing in a result (every output is the same to the bit). The receive path's
assembler, which every live source uses, now keeps a block's bytes until a window takes rows from them
and decodes a block it holds as a possible restart when it holds it; a caller that changed a block's
bytes during `feed` would see it (none does).

- Where the time went (measured on this container's CPU, 96 channels, 115,328-byte frames, before):
  `FrameReceiver.take` 0.27 to 0.32 ms a frame on average, most of it decoding every 24-bit sample
  although about 83 % of the stream lies between receive windows; 67 to 80 ms on the frame that
  completes a ping (the record's array reordered); in the Python engine `sync.resample_boards` 2.0 s
  of 3.4 s a ping, row by row.
- Changed, outputs equal to the bit (samples, receiver statistics, record metadata, both engines'
  reports of the 96-channel trial, clean and with frames lost):
  `sources/flank_stream.py`: `decode_header` and `decode_rows` (`decode_block` is the two together);
  `PingAssembler` decodes only the rows a window takes, with an in-place 24-bit decoder, and builds the
  record's array channel-major in pieces. `sync.py`: a board's sinc weights once, its rows together
  (the Python engine only; `libs/` is untouched).
- Measured on an L4 through Sensor Bridge (HF job 6b06cd69b6a90bbe997d1b7a; four runs of the six cases
  with the change, then four with the two files as before, same job, 4 s queue):

  | Python engine, chain time a ping | before | after |
  |---|---|---|
  | 64 channels, after the first ping | 1.30 to 1.35 s | 0.27 to 0.30 s |
  | 64 channels, first ping | 1.79 to 1.82 s | 0.74 to 0.78 s |
  | 96 channels, after the first ping | 2.06 to 2.20 s | 0.42 to 0.47 s |
  | 96 channels, first ping | 2.53 to 2.56 s | 0.99 to 1.02 s |

  32 channels (one board, nothing to resample): 0.13 to 0.15 s both ways. The C++ engine is unchanged:
  0.10 to 0.12, 1.05 to 1.12 and 1.64 to 1.83 s. At 96 channels the Python engine lost its first 2 to 4
  frames and nothing after, before and after alike: **on an L4 it was already real time at 96 channels
  with D-187's 4 s queue** (2.1 s of the trial's 3 s); D-186's "not real time" was a10g-large (2.64 s
  a ping, median; 368 frames lost in the median run), which was not measured again. The frame path
  on that L4: 0.066 ms a frame on average, 31 ms on the frame that completes a ping.
- The (3, "python") Popoto loopback case is in the suite from here on.
- Seen in that job and not explained: in the first of the four runs with the change, the 32-channel
  C++ case lost nothing and had a worst receive delay of 6.4 ms (its first ping took 0.92 s against
  0.12 in every other run: the job's first use of the device), and the 64-channel Python case had frames
  1 and 2 missing, frame 3 damaged and a worst delay of 5.1 ms; both ended as expected failures on the
  5 ms bound. The other 22 runs with the change and all 24 without passed.
- Now visible, not looked into: the C++ engine takes 1.05 to 1.12 s a ping at 64 channels and 1.64 to
  1.83 s at 96, four times the Python engine after this change. Follow-up sonar-drc.
- New: `tools/hsb_frame_timing.py` (the frame path's cost, a profile, the one-thread schedule emulated
  with the real chain); the opt-in `test_hsb.py::test_frame_path_timing_study` (`GPUSONAR_HSB_TIMING=1`).
- Tests: `test_flank_stream.py` (the 24-bit decoder against the byte formula; assembled samples; a held
  block survives the caller reusing its buffer), `test_calibration_sync.py` (the resampler against the
  row-by-row arithmetic, to the bit), `test_hsb.py` (the queue emulation).
- Full GPU suite on this tree: **2180 passed, 100 skipped**, none failed, no expected failure (NVIDIA L4,
  HF job 6b06d64fd91f621b93079be0, 48 min 30 s): 9 more passed (8 new tests and the (3, python) case), 2
  more skipped (the opt-in timing study; the new tool's `--help` test in the job's tarball).

### D-191 The C++ engine resamples a board's rows together: 0.09 to 0.17 s a ping at 64 and 96 channels on an L4, from 1.05 to 1.83 s (issue sonar-drc)

What got worse first: nothing in a result (every output is the same to the bit). The stage holds up
to 8 rows of a board at once besides the record (before: one).

- What was slow: `resample` in `libs/live/src/sync.cpp`, which moves a board's samples onto the first
  board's timeline. It ran one channel at a time on one thread and computed a sine and a cosine for
  each of 16 taps at every sample of every channel, though a board's channels share their positions.
  On this container's CPU (host backend, 2 threads) it was 0.95 s of the chain's 1.55 s a ping at 64
  channels and 1.57 s of 2.30 s at 96. The Python engine had the same cost until D-190.
- Why one board was fast: the one-row deployment file decimates first (`decimate: {}`), so its boards
  are synchronised on the short burst stretches; the two- and three-row files have no decimate stage and
  every channel is resampled at the boards' rate.
- Not the cause: the backend. In the L4 loopback both engines ran on CUDA (`backend: auto`); board
  synchronisation is on the host in both and no CUDA source does it.
- Changed: `resample_rows` computes a board's weights once a block of 256 samples and applies them to
  up to 8 of its rows, the blocks in parallel (`parallel_for_host`). Each sample is the same double
  weight (sinc times window) times the same input, added in the same tap order.
- Equal to the bit, checked two ways: the old and the new build gave the same bytes for 44 arrays and 37
  facts (the synchronised samples, the sync facts, the board metadata, the C++ chain's reports, beam
  maps and conditioned samples; the 1-, 2- and 3-row Popoto trials, `flank_32` cut into 4 boards clean
  and with offsets and drifts, `popoto_4x8` at two seeds; 1, 2 and 4 threads);
  `Sync.BoardsResampledTogetherGiveTheRowByRowSamplesToTheBit` (`tests/cpp/test_stage2.cpp`) holds the
  stage against a copy of the old loop.
- Measured on an L4 (HF job 6b074405b6a90bbe997e6017, chain's OpenMP on 4 threads):

  | C++ engine, chain time a ping after the first | before (D-190's job) | after |
  |---|---|---|
  | 32 channels, through Sensor Bridge | 0.10 to 0.12 s | 0.11 to 0.12 s |
  | 64 channels, through Sensor Bridge | 1.05 to 1.12 s | 0.09 to 0.11 s |
  | 96 channels, through Sensor Bridge | 1.64 to 1.83 s | 0.13 to 0.17 s |

  The direct chain (`tools/live_chain_timing.py`, 3 repeats of 4 pings): 0.086 s a ping at 64
  channels, of which synchronisation 0.056 s, and 0.111 s at 96, of which 0.085 s; the Python engine
  0.27 and 0.40 s, of which 0.22 and 0.33 s. All 24 loopback runs of that job passed (the six cases,
  four times); the first run's 32-channel C++ case lost one frame at the start and had a worst receive
  delay of 10.8 ms before the start window's end, as D-190's first run did (the job's first use of the
  device).
- Not done, the owner's to decide if wanted: computing the weights from the fractional part alone would
  cut what is left of synchronisation several times (it is now most of the C++ chain's time at 64 and
  96 channels) and changes the last bits of the samples; not implemented, not measured. The burst
  estimates still run board by board on one thread (tens of milliseconds, by the agent's estimate).
- New: `tools/live_chain_timing.py` (chain and synchronisation time a ping, each engine).
- Full GPU suite on this tree: **2180 passed, 101 skipped**, none failed (NVIDIA L4, HF job
  6b07440eb6a90bbe997e6027, 46 min 47 s; one more skipped: the new tool's `--help` test in the job's
  tarball). The C++ suites ran with no failure listed; the job's log keeps only ctest's last five
  lines, which do not include its count.

### D-192 The board resampler's 16 weights from three trig calls a sample, in both engines (issue sonar-fy6; the owner: "do that and measure it")

What got worse first: nothing measured. The owner was told this would change the last bits of the
samples; on every case measured no float32 output changed. A sample could still round one step the
other way on other data.

- Changed: `resample_weights` (`libs/live/src/sync.cpp`, declared in `sync.hpp`) and `_sinc_weights`
  (`python/gpusonar/sync.py`) make a sample's weights from sin(pi f), cos(pi f / 8) and sin(pi f / 8)
  and a 16-entry table, f the fractional position: sin(pi (k - f)) = -(-1)^k sin(pi f) and
  cos(pi (k - f) / 8) = cos(pi k / 8) cos(pi f / 8) + sin(pi k / 8) sin(pi f / 8). The sine is taken
  from f - 1 above one half, or the tap next to the following sample loses its accuracy just below 1.
  `resample_rows` and the burst check in `estimate_from_bursts` use it. The binding
  `core.live_resample_weights` lets the two engines' weights be compared (equal on 1.2 million
  fractions).
- How far the outputs moved, old against new (this container's CPU; the 1-, 2- and 3-row Popoto trials,
  `flank_32` cut into 4 boards clean and with offsets and drifts, `popoto_4x8` at two seeds; both
  engines): the weights by at most 2.8e-16 against the formula; all 59,778,304 synchronised samples an
  engine equal to the bit, as are the sync facts, the 28 beam maps and every field of the 10 sets of
  reports. Before the rounding to float32 a sample moves by at most 5.9e-16 of the largest input, 0.0094
  of a float32 step. The two engines agree with each other exactly as before.
- Zero drift is not a special case: the fractional position is not constant in floating point (21 to 42
  values a window).
- Tests re-pinned, nothing loosened: `Sync.BoardsResampledTogetherGiveTheRowByRowSamplesToTheBit`
  (`tests/cpp/test_stage2.cpp`) and
  `test_resampling_the_boards_gives_the_row_by_row_resamplers_samples_to_the_bit`
  (`tests/python/test_calibration_sync.py`) compared against the per-tap formula to the bit (D-190,
  D-191). They now hold rows together against row by row with the new weights, to the bit, and against
  the formula within one float32 step plus 16 x 4.5e-16 x the largest input (measured: every one of the
  test's 216,400 samples equal). New: `Sync.ResampleWeightsFromThreeTrigCallsAreTheFormulasTapByTap`,
  `test_sinc_weights_from_three_trig_calls_are_the_formulas_tap_by_tap`,
  `test_whole_sample_positions_pass_the_samples_on_as_they_are`,
  `test_live_cpp.py::test_both_engines_resample_with_the_same_weights`.
- Measured on an L4 (HF job 6b076087b6a90bbe997e9449, 4 OpenMP threads; before: D-191's job):

  | chain time a ping after the first | C++ before | C++ after | Python before | Python after |
  |---|---|---|---|---|
  | 64 channels, direct chain (median) | 0.086 s | 0.054 s | 0.273 s | 0.180 s |
  | 96 channels, direct chain (median) | 0.111 s | 0.065 s | 0.395 s | 0.248 s |
  | 64 channels, through Sensor Bridge | 0.09 to 0.11 s | 0.06 to 0.07 s | 0.27 to 0.33 s | 0.18 to 0.22 s |
  | 96 channels, through Sensor Bridge | 0.13 to 0.17 s | 0.08 to 0.11 s | 0.42 to 0.49 s | 0.28 to 0.32 s |

  Synchronisation alone: C++ 0.056 to 0.027 s at 64 channels and 0.085 to 0.039 s at 96; Python 0.22 to
  0.12 s and 0.33 to 0.19 s.
- Seen again, not explained: in the first of that job's four loopback runs the 32-channel C++ case lost
  no frame and had a worst receive delay of 7.9 ms, and ended as an expected failure on the 5 ms bound;
  the other 23 runs passed. The same case in the first run of D-190's job (6.4 ms) and D-191's (10.8 ms,
  inside the start window). It is the first test of the job to use the device. Follow-up sonar-6rs.
- Full GPU suite on this tree: **2183 passed, 101 skipped**, none failed (NVIDIA L4, HF job
  6b07608ed91f621b930918b0, 46 min 30 s): 3 more passed, the new Python tests.

### D-193 The boundary models against Gallaudet 2001's printed values at 68 kHz; "a lower bound" no longer said of the surface model against data (issue sonar-5wa; amends the wording of D-87 and D-132; no model, default or value changed)

What is wrong first: at 68 kHz and 3 m/s wind `bubble_layer_surface` (with `apl_1994`) lies ABOVE the
dissertation's printed band means, by 3.4 dB at 44-66 degrees grazing (-38.1 against -41.5 dB) and by
13.0 dB at 27-38 degrees (-38.3 against -51.3 dB). D-87 and D-132 call the model a lower bound between
30 and 70 degrees because it omits a positive term; that is true of the omitted term and not of the
model against data at low wind, where its level follows our optical-depth law. It is not shown to be a
defect of the model: the printed values are raw sonar-equation numbers with up to -9 dB of
transmit-pattern bias, the averaging (dB or intensity) is not stated, the lower band includes bottom
reverberation, the site was dominated by the towing ship's wake, and the one wind reading was taken
4.5 hours before the runs.

- The document: Gallaudet, dissertation (2001), opened (the handle's host was refused by this
  session's network proxy; read from the hosting repository's own copy of the same handle); only values
  it prints as numbers were used (Ch. 6 Tables I and II, pp. 196-198; Table VI, p. 60; Ch. 5 Table III,
  p. 158). Its curves exist only as figures and bound nothing here. The download is not kept.
- Bottom (`fluid_sediment` with the dissertation's fitted inputs, Table VI): 0.2 dB below its printed
  mean at 48-66 degrees (-22.8 against -22.6 dB) and 3.9 dB above at 29-37 degrees (-24.8 against
  -28.7 dB), inside its own stated systematic error. All five roughness models give the same numbers
  to 0.001 dB: at this site the roughness term is 21 to 28 dB below the volume term away from the
  specular direction, so these data cannot tell the roughness models apart. No defect found.
- The site is a silt-sand mixture (grain size 6.36 phi, estimated from porosity), not the "silty clay"
  our documents said. Table VI's density, sound-speed and loss ratios equal
  `sediment_from_grain_size(6.36)` within 4e-5. Our class default `fine_silt` is 5.0 dB low at 48-66
  degrees and 0.8 dB low at 29-37: the volume parameter (0.001 in the class, 0.003 in the fit, worth
  4.8 dB).
- Surface, the other models at 3 m/s: `hall_novarini` extinction 12.6 and 22.2 dB above the printed
  means, `chapman_harris` 15.1 and 19.9 dB (outside its range, as it warns), `kraus_surface` 10.7 and
  15.8 dB. The four printed bubble-cloud peaks (-23.1 to -25.2 dB) are within 1 dB of our saturation
  level.
- Volume: printed values (-65.5 to -76.3 dB) and no model of ours predicts them; nothing compared.
- Not verified: how the band means were averaged and whether they were corrected for the transmit
  pattern; the grazing angles of the bubble clouds (figures only).
- Tests (characterisations: the bound is what we get plus a small margin; nothing tuned):
  `tests/python/test_gallaudet_boundaries.py` (10). CPU. The full GPU suite was not rerun for this
  (documents and one new CPU test file).

### D-194 Stale simulator statements corrected: the TR 0702 matrix, RESULTS, SCENARIOS, SIMULATION, REFERENCES (issues sonar-nyn, sonar-kob; no result changed)

What was wrong first: documents denied things that exist. Found by the audit of the M12 and SST
issues (2026-11-25).

- The TR 0702 matrix (`tools/sst_validation.py`, `docs/figures/sst_validation.md`): the Thorp row said
  "not implemented" (`acoustics.absorption_thorp`, key `absorption: thorp`, D-132, held by
  `test_medium_models.py::test_thorp_absorption`); two rows said near-specular removal is lacking
  (`exclude_specular`, D-123, opt-in, bistatic items only, held by `test_reverb_specular.py`). What
  still cannot be checked stays said: TR 0702's own near-specular surface model is McDaniel's, which
  we do not have. The matrix was regenerated (203 s on the CPU): it differs from the committed one in
  those three rows and in its run-time row only (not taken: the machine was loaded); the JSON is
  equal but for two timing keys.
- New test: `test_sst_validation.py::test_matrix_names_only_tests_that_exist_and_does_not_deny_features_we_have`
  (every test a row names exists; the rows above name their tests).
- `SCENARIOS.md`, the two SST scenarios, against the matrix and the scenario files: Pursuit's
  "25 Hz below the ridge" is 30 Hz (30.27); "figure from the first run" was stale (both figures were
  regenerated after D-80); `sst_asiaex` uses a one-way source (D-123), not a "-20 dB relay", and one
  hydrophone. The figures themselves were not looked at again.
- `RESULTS.md`: the 2026-11-16 passage listing a flat boundary, bubble absorption, a high-frequency
  surface model, a Jackson-type bottom, volume scattering, time spreading and the RAM cross-check as
  missing is kept and dated, with what superseded each (D-73, D-68/D-123, D-87/D-132,
  D-63/D-66/D-86/D-132, D-62 and later, D-149).
- `STATUS.md`: near-specular reverberation is not removed by default (it can be, D-123); the bistatic
  mode has run with Bellhop on a flat bed only with both platforms at one point.
- `handoff/SIMULATION.md`: the scattering-function generator is ours; TR 0702 Sec. 11.3's algorithm is
  not what it does (the matrix was right).
- `configs/sediments.yaml` and `REFERENCES.md`: the YAML's `status: confirmed` (the loader's word: read
  in the cited open document or computed by a relation it prints) is not REFERENCES' "confirmed"; the
  tabulated values are "via secondary" there. Said in both; no data changed. For the owner: renaming
  the YAML's status would need a loader and a test change.
- `REFERENCES.md` gains rows for BELLHOP3D and PyRAM (test oracles, D-149).
- The KRAKEN 0.21 dB, wherever quoted: Bellhop's own rays against KRAKEN, not our provider, and
  recorded, not asserted (the test bounds it at 0.5 dB, 0.1 dB from 2 km; the provider with 80 arrivals
  within 0.5 dB).
- A comment in `sim/timespread.py` named only `hall_novarini`; `apl_1994` is adopted (D-123).
- Checked on the CPU: `test_sst_validation.py` 24 passed, with the neighbouring files (the agent's
  run); the full GPU suite was not rerun for this.

### D-195 The decimate-first correction works by board, with the board resampler's weights (issue sonar-59u; the owner: "do the decimate-first resampler too"; no output changed)

What got worse first: the stage holds one copy of the moving boards' samples beside the record until it
ends (before: one channel a thread; at most about 2.4 MB on the cases measured). A channel whose board
has no clock entry is now refused (`std::invalid_argument`); before, the code read past the array's end.

- What it did: `correct_baseband` (`libs/live/src/decimate.cpp`), the complex resampler of the
  decimate-first stage (the path of deployment files with a `decimate:` stage, such as the one-row
  Popoto build), worked a channel a task and computed, for every sample of every channel, a sine and a
  cosine for each of 16 taps and the carrier turn. It uses the board resampler's own 16-tap Hann sinc
  (taps -7 to 8, the same 1e-12 rule), and a board's channels share their positions and their turn.
- Changed in two steps, measured apart: (a) the weights and the turn once a board and a block of 256
  samples for up to 8 of its channels, every (board, group, block) in one `parallel_for_host`; (b) the
  weights from `resample_weights` (D-192). The Python engine calls this C++ stage through the bindings
  (`core.live_decimate_record`, `core.PingDecimator`), so both engines change together; no CUDA source
  does this stage.
- Outputs, old against (a) and against (b), both engines, on the 1-, 2- and 3-row Popoto trials (4, 8
  and 12 boards) and test_decimate's fixtures and chains: the corrected baseband samples (6,032,512
  from the stage and 200,576 from the streaming receivers), the sync facts and board metadata, the beam
  maps and every field of every report are equal to the bit. Before the rounding to float32 step (b)
  moves a sample by at most 5.5e-16 of the largest input, 0.023 of a float32 step.
- Speed on this container's CPU, the correction alone, old to (b): 53 to 4.2 ms at 32 channels on one
  thread, 58 to 4.6 ms at 64, 91 to 7.3 ms at 96 (12 to 13 times); the whole chain gains 3 to 12 %,
  the correction having been a small share of a ping.
- Measured on an L4 (HF job 6b076cffb6a90bbe997ea8e8, 4 OpenMP threads, chain time a ping after the
  first, median): the one-row build, whose deployment decimates first, C++ 0.087 s (D-192's job) to
  0.048 s, of which the decimate stage 0.032 s; Python 0.124 to 0.084 s. With `--decimate` on the 2- and
  3-row trials: C++ 0.054 and 0.063 s (stage 0.028 and 0.044 s). All 18 loopback runs of that job passed.
- Tests: `Decimate.BoardsCorrectedTogetherGiveTheChannelByChannelSamplesToTheBit`
  (`tests/cpp/test_stage2.cpp`): to the bit against the channel-by-channel loop with the same weights,
  and against the per-tap formula within one float32 step plus 16 x 4.5e-16 x the largest input
  (measured: every one of its 37,170 parts equal); a board out of range is refused. No existing
  tolerance or pinned value changed. The exact pin does not tell the old weights from the new (they
  never differ in a float32 output here).
- `tools/live_chain_timing.py` gains the decimate stage's time and `--decimate`.
- Not run: `popoto_face_1row` (the same `decimate: {}` stage). The 12 km scenarios simulate baseband
  directly and never pass this stage.
- Full GPU suite on this tree: **2183 passed, 101 skipped**, none failed (NVIDIA L4, HF job
  6b076d06d91f621b93092f13, 46 min 40 s).

### D-196 The K shape of reverberation from a density of scatterers or patches per cell (issue sonar-6fc; the owner approved the opt-in input; no present result changed)

What got stricter first: a reverberation item's `statistics` mapping now refuses a key it does not read,
with the nearest known one (D-177's rule); before, such a key was ignored. No shipped file uses
reverberation `statistics`.

- What was wrong: the K shape was one number for every range and every sonar. Abraham and Lyons, IEEE J.
  Oceanic Eng. 27, 800 (2002) (opened, the owner's copy; not kept in the repository) ties it to the
  resolution cell: n scatterers of exponentially distributed amplitude in a cell give shape n / 2 (eq. 15);
  n patches of exponentially distributed area give shape n (eq. 19). So the shape is proportional to the
  cell's area.
- New, opt-in: `statistics: {model: k, scatterers_per_m2: N}` or `{model: k, patches_per_m2: N}` in place
  of `shape` (exactly one of the three). Each cell's shape is N x A / 2 or N x A, A the area of boundary
  in the cell as the generator lays it: azimuth width x slant range x slant-range extent, clipped at the
  array's height above a flat boundary (ours; the grazing angle cancels). No density has a default: the
  paper gives none for any seabed and says it must be estimated from data.
- Our `shape` is the paper's alpha: shown from its eqs. (7), (12)-(13) and (28) (moments of orders 1 to
  6 to 1e-12; the exceedance to 1e-5 down to a probability of 1e-8). Its worked example recomputes: at a
  false-alarm probability of 1e-6 the K threshold is 1.685 dB above Rayleigh's for alpha 10 (printed 1.7)
  and 2.694 dB for alpha 5 (printed 2.7).
- With `shape` given, the outputs are the same to the bit (12 outputs hashed before and after: both
  engines, monostatic and bistatic). The texture is drawn in Python and multiplies the deviates handed to
  the generator, so both engines and both backends take the same texture; no C++ or CUDA source changed.
  The device path was not run for this (no device here): the full GPU suite's result is added below.
- Limits: a cell with fewer than one scatterer or patch warns (the paper's fixed-count picture does not
  describe it); refraction and bounces do not enter the area; a bistatic item uses the range c t / 2 from
  the receiver as if monostatic (the true cell of separated platforms is larger); over relief the height
  is taken as 0.
- Still open (the paper says the second-order statistics are not established): correlation of the
  texture between cells and between pings; none was invented. And a density for any seabed.
- Tests: `tests/python/test_reverb_statistics.py` (19 new cases: the mapping, the worked example, the
  cell area, the statistics and the mean level in a ping, both engines, the keys, the unchanged fixed
  shape), `tests/python/test_dict_keys.py` (1). CPU: 51 passed in those two files.
