# Results

Measured by this repository's tests and tools. Accuracy numbers were produced on the
**CPU (host) backend**; the same tests pass on the CUDA backend (NVIDIA T4). Speed numbers
are in "Performance" and say which machine each came from.
Each row names what it is compared with. Where the brief asks for a comparison that
could not be made, that is stated.

## Performance

Source: `tools/benchmark.py`, run through `tools/gpu_job.sh` on rented x86_64 GPU machines
(Hugging Face Jobs, CUDA 12.6). Raw output: `docs/benchmarks/*.json`. Protocol as in the
paper: each dataset processed three times in a row, the last run reported. Times cover the
backprojection call only, including copying pings into the kernel's input and the
host-to-device transfer. They exclude simulation and conditioning, which run on the CPU.
**None of these is a DGX Spark or a Jetson.**

Datasets (simulated, fixed):

- **SAS:** `sas_auv`, 400 pings × 16 channels × 9,534 baseband samples; seafloor image
  2560 × 1200 pixels at 5 cm (128 m × 60 m); linear interpolation; each ping limited to
  the beam's λ/D extent. Collection time 80 s.
- **FLS:** `l12_5d`, both mux halves (2 × 64 channels); fan of 4000 ranges × 1602 bearings.
- **Flank:** `flank_32`, one ping of 32 channels × 171,072 baseband samples; map of
  16,000 ranges × 482 bearings out to 10 km. Ping cycle 14.3 s.

Two rounds were run. The first exposed that the kernel's float64 work dominated on most
GPUs; the kernel was then changed to reject out-of-beam (point, ping) pairs with a
float32 test before any float64 work and to skip float64 trigonometry for pings without
attitude rates. Images are unchanged (same tests, bit-identical culled and unculled).

Current kernel:

| GPU | SAS, no culling | SAS, culling b = 16 | SAS real-time factor (culled) | FLS fan (6.4 Mpixel) | Flank map (7.7 Mpixel) | Flank real-time factor |
| --- | --- | --- | --- | --- | --- | --- |
| Tesla T4 | 1.48 s | 1.26 s | 64× | 0.20 s | 0.27 s | 54× |
| L4 | 0.81 s | 0.68 s | 118× | 0.09 s | 0.16 s | 90× |
| A10G | 0.80 s | 0.69 s | 116× | 0.10 s | 0.17 s | 82× |
| A100 (80 GB) | 0.86 s | 0.68 s | 118× | 0.13 s | 0.15 s | 96× |

Before the change (`docs/benchmarks/before_pretest/`), for the SAS dataset:

| GPU | No culling | Culling b = 16 | Culling speedup |
| --- | --- | --- | --- |
| Tesla T4 | 13.68 s | 2.13 s | 6.4× |
| L4 | 7.46 s | 1.22 s | 6.1× |
| A10G | 6.55 s | 1.14 s | 5.7× |
| A100 (80 GB) | 1.05 s | 0.80 s | 1.3× |

- **Float64 was the bottleneck.** The T4, L4 and A10G run float64 at a small fraction of
  their float32 rate; the A100 does not, and was 6× faster than the A10G before the
  change although its nominal float32 throughput is lower. The float32 pre-test made the
  unculled kernel 9× faster on the T4 and 8× on the L4 and A10G.
- **Ray culling now adds 1.16× to 1.26×** on these GPUs, because the pre-test already
  skips most of what culling skips. The paper's "factor of two" was measured against a
  kernel without such a test; the first-round figures (5.7× to 6.4×) are the comparable
  ones here, and they depend strongly on the GPU's float64 speed. The culled image is
  bit-identical on every GPU; 11% of (ping, block) pairs are kept.
- **These one-shot SAS numbers are overhead-bound.** Three very different GPUs finish in
  0.68 s: what is left is mostly copying 0.5 GB of samples into the kernel's input and
  uploading it, which a one-shot call repeats every time. The session figures below
  remove that.
- Real-time factor is collection time over imaging time for this image size. It excludes
  conditioning (see "Conditioning on the device" below) and motion estimation.
- **FLS:** a 6.4-megapixel fan takes 0.09–0.20 s. The frame rate the rig can deliver is
  not known yet, so no real-time factor is claimed.

**Sessions: samples kept on the device** (`docs/benchmarks/session_*.json`, D-40). The
tables above are one-shot calls, which rebuild the kernel inputs and upload 0.5 GB of
samples every time. A session does both once (0.28 s on the A10G host, 0.37 s on the T4
host; 489 MB resident) and each call then moves only points and image. Third of three
runs, images bit-identical to the one-shot calls on every row:

| GPU, dataset | One-shot | Session | Session, culled | Real-time factor (session, culled) | Two GPUs (session, culled) | Scaling on two |
| --- | --- | --- | --- | --- | --- | --- |
| Tesla T4, SAS 2560 × 1200 at 5 cm | 1.53 s | 1.04 s | 0.80 s | 100× | — | — |
| L4, SAS 2560 × 1200 at 5 cm | 0.81 s | 0.49 s | 0.36 s | 225× | — | — |
| A10G, SAS 2560 × 1200 at 5 cm | 0.83 s | 0.44 s | 0.33 s | 243× | 0.19 s | 1.71× |
| A100 (80 GB), SAS 2560 × 1200 at 5 cm | 0.70 s | 0.23 s | 0.19 s | 412× | — | — |
| Tesla T4, SAS 6400 × 3000 at 2 cm | 6.61 s | 6.15 s | 4.61 s | 17× | — | — |
| L4, SAS 6400 × 3000 at 2 cm | 3.36 s | 3.03 s | 2.18 s | 37× | — | — |
| A10G, SAS 6400 × 3000 at 2 cm | 3.09 s | 2.71 s | 1.99 s | 40× | 1.15 s | 1.73× |
| A100 (80 GB), SAS 6400 × 3000 at 2 cm | 2.02 s | 1.38 s | 1.19 s | 67× | — | — |
| Tesla T4, FLS fan 4000 × 1602 | 0.20 s | 0.19 s | — | — | — | — |
| A10G, FLS fan 4000 × 1602 | 0.094 s | 0.085 s | — | — | — | — |
| Tesla T4, flank map 16000 × 482 | 0.18 s | 0.15 s | — | 94× (no culling) | — | — |
| A10G, flank map 16000 × 482 | 0.18 s | 0.073 s | — | 197× (no culling) | — | — |

- **The overhead was the per-call upload.** With it gone the 5 cm image takes 1.04 s on
  the T4, 0.49 s on the L4, 0.44 s on the A10G and 0.23 s on the A100: the GPUs separate
  again.
- **The benchmark is no longer overhead-bound at 2 cm.** The A10G images 4.5 × 10¹⁰
  pixel-records per second at both pixel sizes (4.46 and 4.53), so the 2 cm figure
  measures the kernel.
- **Multi-GPU:** two A10G give 1.71× to 1.73× over one, with an identical image (the
  one-shot figure was 1.13× to 1.20×, bounded by each device's upload). What is left of
  the gap to 2× is not broken down; point and image transfers run per call on each device.
- Two-GPU scaling was measured on one GPU type only (A10G).

**Conditioning on the device** (`docs/benchmarks/conditioning_*.json`, D-42). Raw
real-passband SAS pings (16 channels × 95,334 samples at 1 MHz) to compressed complex
baseband: conversion with decimation, then pulse compression. Per ping, third of three
runs, against the 0.2 s ping interval:

| Machine | Host (CPU threads) | Device (cuFFT) | Speed-up | Real-time factor on the device | Largest difference from the host result |
| --- | --- | --- | --- | --- | --- |
| Tesla T4 job (4 vCPU) | 217 ms (0.92× real time) | 6.2 ms | 35× | 32× | 2.5 × 10⁻⁷ of the peak |
| A10G job (4 vCPU) | 165 ms (1.2× real time) | 3.4 ms | 48× | 58× | 2.1 × 10⁻⁷ of the peak |

- **Conditioning, not backprojection, was the limit on a whole SAS chain.** On the CPU it
  barely keeps up with the sonar (and falls behind on the T4 machine), while the kernel
  images at 100× real time or more. On the device, conditioning plus the session
  backprojection (0.33 s per 400 pings, culled, A10G) comes to about 4 ms per ping, some
  50× real time. Motion estimation is still CPU and is not in that figure.
- The first device call of a run includes making the FFT plans (24 to 34 ms per ping in
  the first run above); plans are cached per size and device.
- Samples go to the device and back for each stage; they are not yet handed to the
  backprojection session without returning to host memory.
- Whitening, median TVG, calibration and DRC are still host only.

**Power-law fit** (the paper's Fig. 9 model, `y = a·x^b`, runtime against single-precision
GFLOPS), four GPUs, SAS dataset:

| Runs | Fit | R² in log space |
| --- | --- | --- |
| Sessions, 2 cm image, no culling (compute-bound) | y ≈ 783 · x^−0.57 | 0.34 |
| Sessions, 2 cm image, culled | y ≈ 779 · x^−0.59 | 0.44 |
| Sessions, 5 cm image, no culling | y ≈ 188 · x^−0.60 | 0.38 |
| One-shot calls, 5 cm, no culling (overhead-bound) | y ≈ 88 · x^−0.46 | 0.94 |
| One-shot calls, 5 cm, culled (overhead-bound) | y ≈ 74 · x^−0.46 | 0.87 |
| First round, no culling | y ≈ 644 · x^−0.49 | 0.08 |
| Paper (its own data, seven GPUs) | y ≈ 485307 · x^−0.755 | not stated |

- The GFLOPS values used (T4 8,141; L4 30,290; A10G 31,240; A100 19,490) are nominal
  float32 figures **recalled from vendor specifications and not verified here**; the paper
  took its values from Wikipedia, which the build container cannot reach.
- Only the exponent is comparable with the paper; its prefactor belongs to its dataset.
  On the compute-bound session timings ours is −0.57 to −0.60 against the paper's −0.755.
- **The fit is poor (R² 0.34 to 0.44) and should not be used to predict a new GPU.** The
  A100 is twice as fast as the T4-L4-A10G trend puts a card of its nominal float32 rate
  (without it the exponent stays at −0.58 to −0.61 but the prefactor moves by 40%). The
  kernel's float64 reference-delay solve favours cards with fast float64, which a
  single-precision GFLOPS axis cannot represent; the unverified GFLOPS figures may also
  be wrong. The high R² of the one-shot rows was three cards sitting on the same overhead
  floor, not a good model.

CPU reference point: the build container's 4 cores run the culling test set at about
1e8 pixel-records per second; the GPUs above reach 1e10 to 3e10 on the SAS dataset.

## FLS: resolution on simulated point targets

Source: `tests/python/test_fls.py`. Free-field point target, no noise, 12-bit ADC,
sound speed 1480 m/s, uniform aperture weighting, sinc interpolation, pulse compressed.
Widths are −3 dB. Definitions: `DECISIONS.md` D-26.

| Sonar | Elements | Range | Range width | Theory for the pulse | Nominal c/2B | Beam width | Theory 0.886·λ/L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `l12_5d`, two-ping full aperture | 128 | 1.5 m | 278.8 µm | 260.8 µm (+6.9%) | 296.0 µm (−5.8%) | 0.3886° | 0.3913° (−0.7%) |
| `l12_5d`, one mux half | 64 | 1.5 m | 278.8 µm | 260.8 µm (+6.9%) | 296.0 µm (−5.8%) | 0.7589° | 0.7826° (−3.0%) |
| `p4_2v` | 64 | 7.0 m | 347.1 µm | 323.9 µm (+7.2%) | 367.6 µm (−5.6%) | 1.4186° | 1.4386° (−1.4%) |

All are inside the 10% acceptance band. The range width sits about 7% above the ideal
Gaussian figure because baseband conversion keeps 1.5× the −6 dB bandwidth and tapers the
band edges.

Grating lobes of the linear probe (`l12_5d`, pitch 0.3 mm against a 0.296 mm wavelength),
target at 20° bearing:

| Quantity | Value |
| --- | --- |
| Largest steering angle with no grating lobe in visible space | 0° (none) |
| Predicted lobe bearing at centre frequency | −40.1° |
| Predicted lobe level, narrowband, element pattern only | −4.2 dB |
| Measured lobe bearing in the image | −38.7° |
| Measured lobe peak level in the image | −33.2 dB |

The measured level is far below the narrowband figure because the pulse is two cycles
long: away from the true bearing only a few elements overlap in time, so the lobe is
smeared instead of focused. A longer or narrower-band pulse would raise it towards the
narrowband value.

## FLS: PICMUS plane-wave benchmark

Source: `python/gpusonar/benchmarks.py`, `tests/python/test_picmus.py`.
Data: PICMUS simulation, resolution phantom, IQ, 128-element L11-4v at 5.208 MHz,
from the Hugging Face mirror `zeahub/picmus` (a format conversion of the original).
Metric: mean −6 dB width of the envelope over the eleven point targets, as in the
challenge. Sinc interpolation, uniform weighting, no post-processing.

| Plane waves | Receive aperture | Axial FWHM | Lateral FWHM |
| --- | --- | --- | --- |
| 1 | full | 0.395 mm | 0.324 mm |
| 11 | full | 0.395 mm | 0.297 mm |
| 75 | full | 0.393 mm | 0.300 mm |
| 1 | F-number 1.75 | 0.409 mm | 0.706 mm |
| 11 | F-number 1.75 | 0.409 mm | 0.500 mm |
| 75 | F-number 1.75 | 0.408 mm | 0.519 mm |

- **Comparison with published numbers: not done.** The brief asks for it, but the
  challenge paper and site could not be fetched from the build container (publisher and
  host blocked), so no published figure is quoted here. The F-number 1.75 rows use the
  receive aperture the challenge's reference beamformer is commonly run with, so they are
  the rows to compare once the paper is at hand.
- What was checked instead: the kernel matches an independent float64 delay-and-sum on
  this data to 2e-4 relative; all eleven targets are found on the phantom's 5 mm lattice;
  axial width is the same for 1 and 75 plane waves; compounding does not blur.
- Not yet measured: contrast and CNR on the cyst phantom, and the experimental and
  in-vivo sets.
- The targets image 0.2 mm deeper than the nominal lattice (10.2, 15.2, … mm). The
  offset is uniform; it is a property of the data's time origin and was left alone.

## SAS with known navigation

Source: `tests/python/test_sas.py`, scenarios `stripmap_points` and `rail_points`. CPU.
Widths are −3 dB. Navigation is the simulator's truth.

Stripmap: `sas_auv` (200 kHz, 40 kHz LFM, 16 receivers at 4 cm, element length D = 4 cm),
15 m altitude, 1.6 m/s, 90 pings at 0.32 m, each ping processed over the beam's λ/D extent.

| Target ground range | Along-track width | Against D/2 = 20.00 mm | Along-track position error |
| --- | --- | --- | --- |
| 38.0 m | 20.49 mm | +2.5% | −2.0 mm |
| 41.2 m | 20.74 mm | +3.7% | −2.0 mm |
| 45.0 m | 20.51 mm | +2.6% | −2.0 mm |

- Ground-range width at 41.2 m: 18.84 mm against 17.68 mm (0.886·c/2B projected onto the
  seafloor), +6.6%.
- The width does not change with range, as D/2 predicts. A real aperture of the same
  0.64 m array would resolve about 10 cm at this range.
- Processing the whole main lobe (null to null) instead gives 15.6 mm, finer than D/2.
  D/2 is the figure for a processed extent of λ/D; that extent is the default.
- The constant −2 mm along-track offset is a tenth of a resolution cell. It is consistent
  with range-Doppler coupling of the LFM pulse on a moving platform, which nothing
  compensates yet; that attribution has not been tested by removing the motion.

Rail: `sas_rail` (P4-2v probe, 2.72 MHz, 64 elements), 151 stop-and-go positions at 2 mm
(0.30 m of travel), target at 2.0 m.

| Quantity | Measured | Theory |
| --- | --- | --- |
| Along-track width | 1.579 mm | 1.558 mm = 0.886·λR/(2L), L = 0.3095 m (+1.3%) |

On a rail the element beam (about 110°) is far wider than the angle the rail subtends,
so resolution is limited by the rail length, not by D/2 (which would be 0.125 mm and
would need 5.7 m of rail at this range).

## Flank array: beamforming and the sonar equation

Source: `tests/python/test_flank.py`. CPU. Free field (`free_field` tier), one ping.

Far-field beamformer against backprojection, `flank_32`, target at 5 km, bearings 0°, 17°
and −35°: the two complex maps differ by less than 2% of the peak over a ±6° by ±1 m
patch. At 8 m (inside the 96 m Fraunhofer distance) the far-field beamformer keeps 49% of
the peak and its beam is 13.6° wide against 3.2° for backprojection.

Beam width (−3 dB) and ambiguities, uniform weighting:

| Array | Measured width | Theory | Highest lobe outside the main lobe, measured (wideband pulse) | Predicted, single frequency | Predicted, band average |
| --- | --- | --- | --- | --- | --- |
| `flank_32`, 32 × 9.4 cm, 6–10 kHz | 3.14° | 3.16° (0.886·λ/L) | −17.6 dB | −13.2 dB (a sidelobe) | −15.1 dB |
| `popoto_8` dense, 8 × 2.7 cm, 20–35 kHz | 12.83° | 12.82° | −17.4 dB | −12.8 dB (a sidelobe) | −14.9 dB |
| `popoto_8` sparse, 8 elements over 3 m | 0.70° | 0.73° (array factor at 27.5 kHz) | −10.2 dB | 0.0 dB (a full ambiguity) | −8.3 dB |

- `flank_32` and the dense Popoto layout have no grating lobes at broadside; what is
  listed is the ordinary first sidelobe.
- The sparse layout buys an 18× narrower beam with ambiguities: at any single frequency
  some other bearing responds at full height. Over the 20–35 kHz band those lobes fall at
  different bearings and average down to about −8 dB (−10 dB measured with the simulated
  pulse). `flank.ambiguity_report` produces this for any layout and band.

Sonar equation against simulation, `flank_32`, 200 dB source, 1 s LFM, 10 dB target,
white noise at 50 dB re 1 µPa²/Hz, 32 hydrophones:

| Range | Predicted output SNR | Simulated output SNR (mean of two noise draws) |
| --- | --- | --- |
| 1 km | 53.7 dB | 52.4 dB |
| 5 km | 20.5 dB | 18.7 dB |
| 10 km | 1.9 dB | 4.2 dB |

- All within the 3 dB acceptance band. Predicted detection range at a 13 dB threshold:
  6.7 km.
- At 10 km the simulated figure reads high because the peak is picked from a noisy
  envelope at low SNR.
- This is spherical spreading plus absorption in white noise. Multipath, reverberation
  and ambient noise spectra are milestone 12, and will change these numbers.

## Flank array: adaptive beamforming, detection, tracking

Source: `tests/python/test_detect.py`. CPU. `flank_32`, 0.1 s pulse, targets near 2 km.

| Check | Result |
| --- | --- |
| MVDR against conventional, white noise only (target SNR) | 17.3 dB against 19.7 dB: 2.4 dB worse |
| MVDR against conventional, with a broadband jammer 30 dB above ambient at −20° | 19.4 dB against 5.5 dB: 13.9 dB better |
| MVDR output looking straight at the jammer | within 1.5 dB of conventional (nothing to reject) |
| CA-CFAR false-alarm rate on noise with a 20 dB level ramp, design 1e-2 and 1e-3 | within 25% of design |
| Split-window normaliser on a 30 dB level ramp | background flat within 0.5 dB |
| Detection of a 10 dB target at 2040 m, 12°, in noise | range within 1 m, bearing within 1°, SNR above 20 dB |
| Tracking a target crossing at 5 m/s over 20 pings, two injected false alarms per ping | position within 10 m at 1.9 km, velocity within 1 m/s, at most one spurious confirmed track |

- MVDR's 2.4 dB loss in white noise is the price of estimating a 32 × 32 covariance per
  sub-band from one ping. It pays off only when there is something directional to reject.
- The MVDR here rejects interference that extends in range (jammers, directional noise,
  reverberation). It does not separate two point targets at the same range: the cell under
  test is kept out of the covariance so that the echo is not cancelled.
- The last pulse length of a compressed record has only a partial replica overlap. Beam
  maps mark it (`BeamMap.n_valid`); running CFAR over it produces false alarms.
- A target's own sidelobes cross a CFAR threshold at the same range and other bearings.
  The tracking test keeps the strongest detection per ping; a sidelobe-aware clustering
  step is not written.

## Flank array: calibration and multi-board synchronisation

Source: `tests/python/test_calibration_sync.py`. CPU.

Calibration, `flank_32`, five known sources at 1 km from −40° to +40°. Injected: gain
1 dB RMS, phase 20° RMS, element position 5 mm RMS in the array plane.

| Quantity | Worst error over 32 channels |
| --- | --- |
| Gain | 0.007 dB |
| Phase | 0.015° |
| Position | 0.018 mm |

Applying the estimate restores the beam peak to within 1% and removes most of the raised
far-sidelobe floor. Noise-free; errors with noise have not been characterised. The array's
mean gain, mean phase and mean position offset cannot be observed and are set to 1, 0, 0.

Synchronisation, `popoto_4x8` (four boards, 32 hydrophones, 20–35 kHz). Injected clock
offsets of 310, −170 and 52 µs and drifts of 12, −8 and 25 ppm relative to the first
board, plus 20 ns of jitter. Shared reference: two out-of-band chirp bursts per ping.

| Quantity | Result | Requirement |
| --- | --- | --- |
| Offset estimate, worst error | 0.05 µs | — |
| Drift estimate, worst error | 0.00014 ppm | — |
| Misalignment left after resampling | 0.50 µs | 1.43 µs (a twentieth of a period at 35 kHz) |
| Beam peak, unsynchronised, against a single-clock array | −8.2 dB | — |
| Beam peak, synchronised | −0.003 dB | — |
| Largest difference on any lobe above −20 dB | 0.18 dB | 1 dB |

- The 0.50 µs residual is dominated by the resampler acting on the reference bursts
  themselves, which sit at 38–48 kHz, close to the 51.2 kHz Nyquist limit. In the acoustic
  band the alignment is better than that, as the beam pattern shows.
- The stage raises `SyncError` and refuses to pass the record on when the residual exceeds
  the tolerance (tested by giving it the wrong reference description).
- The 1PPS and direct-path methods recover offsets in their tests (to 1 ns and 0.5 µs);
  they do not measure drift within a ping.

## Propagation, reverberation and noise (milestone 12, in progress)

Source: `tests/python/test_propagation.py` (needs Bellhop, an external GPL-3 program run
as a separate process), `test_reverberation.py`, `test_noise.py`. CPU.

| Check | Result | Tolerance asked |
| --- | --- | --- |
| Image-source provider against Bellhop, Pekeris waveguide (100 m, 1600 m/s bottom), 1 kHz: coherent transmission loss, 0.1–3 km | 0.06 dB median, 0.4 dB worst | "within stated tolerances": 0.3 dB median, 1 dB worst |
| Same, incoherent transmission loss, 1 and 8 kHz | 0.1 dB median, 0.7 dB worst | 1 dB |
| Same, delays of the first four arrivals | within 2 µs | 2 µs |
| Bellhop provider lookup between grid nodes against a direct Bellhop run, 8 kHz | delays within 20 µs | 20 µs |
| Echo of a target at 1 km through the `eigenray` tier against the `image_source` tier | envelope correlation above 0.9, energy within 1 dB | — |
| Bottom reverberation level against time, Lambert seabed, against the analytic prediction | within 2 dB in every range bin, mean within 1 dB | 2 dB |
| Scattering-function reverberation level against the same analytic prediction (6 pings averaged) | within 2 dB in every range bin, mean within 1 dB | 2 dB |
| Scattering-function against point-scatterer reverberation, same seabed | within 2 dB in every range block | — |
| Coherence of generated reverberation between hydrophones 1 and 2 spacings apart | J0(kd) within 0.1 | — |
| Surface reverberation level, Chapman–Harris at 8 m/s, against S(θ) × area / r⁴ (`test_surface.py`) | within 2 dB in every range bin | — |
| Chapman–Harris formula against its second published form (wind in m/s, constant 107) | within 0.3 dB | — |
| Reverberation envelope statistics | Rayleigh: mean within 3%, scintillation index 1 ± 0.25 | — |
| Clutter with patch texture (shape 1) | scintillation index 2.0–2.6 against 1.2 without | K-distributed "can be made" |
| Generated ambient noise against the requested spectrum, 0.5–30 kHz | within 1 dB | — |
| Coherence between hydrophones in isotropic noise against sin(kd)/(kd) | within 0.12 | — |

- Bellhop writes bounce phases for the opposite time convention; the reader conjugates
  them. Before that the coherent comparison was off by 4 dB median.
- The Munk-profile test checks Bellhop's own output for sound-channel behaviour. No
  provider of ours claims a refracting profile except the Bellhop one, so there is nothing
  independent to compare there.
- Range-dependent bathymetry (`OceanEnvironment(bathymetry=...)`, passed to Bellhop as a
  bathymetry file): a ridge rising from 200 m to 60 m at 4 km leaves the loss unchanged
  before it (within 0.3 dB), gives no arrivals inside it, and adds more than 1.5 dB behind
  it at 8 kHz (measured 2 to 10 dB). The model is 2-D: one depth profile on every bearing.
  Reverberation still assumes a flat boundary, so there is no ridge echo yet.
- Not done: bubble absorption and a high-frequency surface model (Chapman–Harris is a
  400–6400 Hz fit), a Jackson-type bottom model, volume scattering, time spreading, the
  RAM cross-check. TR 0702 has no fully specified worked example to
  reproduce (`PHYSICS_NOTES.md` section 9).
- The ambient-noise coefficients (four-term fit to the Wenz curves) were checked against
  two independent copies of the formulas, not against the original book.

## Rays against normal modes at 8 kHz

Source: `tests/python/test_kraken_crosscheck.py`. Bellhop and KRAKEN, both run as separate
programs. Isovelocity 100 m of water over a fluid bottom (1600 m/s, 1.8 g/cm³, 0.5 dB/λ),
source at 30 m, receiver at 50 m, 8 kHz, 1 to 10 km. Incoherent transmission loss (arrivals
summed in power against modes summed in power; the coherent field at 8 kHz changes over
centimetres and is not a useful comparison).

| Check | Result |
| --- | --- |
| Rays minus modes, 1–10 km | at most 0.21 dB |
| Rays minus modes, 2–10 km | below 0.1 dB (−0.01 dB at 10 km, where the loss is 66.2 dB) |
| Eigenray provider with 80 arrivals kept, against modes | within 0.5 dB at every range |
| Eigenray provider with 8 arrivals kept, at 10 km | 4.9 dB too much loss one way |

- **A fault this check found.** The long-range scenarios first shipped with 8 arrivals per
  leg. That keeps the strongest paths, so the peak of the compressed echo was right, but
  it dropped two thirds of the one-way energy at 10 km, and with it most of the
  reverberation. The scenarios now keep 24 to 80 arrivals (D-39) and the table below was
  re-measured.
- RAM (parabolic equation) was not run: PyRAM is the permissive candidate and is not
  installed here.

## Long-range scenarios at the eigenray tier

Source: `tests/python/test_long_range.py`, scenarios `flank_target_{1,5,10}km` and
`popoto_trial_{100,1000,2000}m`. Bellhop arrivals (16 to 80 per leg, so up to 6400 two-way
paths per target), isovelocity water over a flat fluid bottom (1600 m/s, 1.8 g/cm³,
0.5 dB/λ), flat sea surface, Ainslie–McColm absorption at 12 °C, ambient noise with
isotropic-field coherence across the array, +10 dB target, one ping, conventional
beamforming. All add bottom reverberation by the scattering-function method
(Lambert, −27 dB). Output SNR is the peak of the
compressed echo over the mean background on the same beam. "Spherical estimate" is the
sonar-equation tool with the same source, noise and target, noise only. CPU.

| Scenario | Water depth | Output SNR | Same, without reverberation | Spherical estimate | Difference |
| --- | --- | --- | --- | --- | --- |
| `flank_32`, 200 dB source, 1 s LFM, noise 45 dB: 1 km | 100 m | 28.1 dB | 57.9 dB | 58.8 dB | −30.7 dB |
| 5 km | 100 m | 24.1 dB | 29.6 dB | 25.7 dB | −1.6 dB |
| 10 km | 100 m | 18.6 dB | 18.1 dB | 7.2 dB | +11.4 dB |
| `popoto_8`, 190 dB source, 0.5 s LFM, noise 38 dB: 100 m | 30 m | 53.6 dB | 83.1 dB | 86.9 dB | −33.3 dB |
| 1 km | 30 m | 26.2 dB | 39.3 dB | 36.6 dB | −10.4 dB |
| 2 km | 30 m | 21.6 dB | 22.1 dB | 13.1 dB | +8.5 dB |

- At long range the shallow-water duct keeps more energy than spherical spreading
  assumes, so the noise-only estimate is pessimistic there by about 10 dB in this
  environment. At short range multipath interference can cost a few dB (−3.8 dB at 100 m
  without reverberation).
- The Popoto trial in 30 m of water is reverberation-limited at 100 m and 1 km (30 and
  13 dB below the figure without reverberation) and noise-limited at 2 km.
- With this seabed and source the flank array is reverberation-limited at 1 km (30 dB
  below the noise-only figure), about evenly limited at 5 km (reverberation costs 5.5 dB)
  and noise-limited at 10 km (the 0.5 dB change there is within one noise draw). The
  balance depends directly on the assumed −27 dB Lambert parameter and 45 dB noise level,
  neither measured.
- The SNR here is a peak. Multipath spreads the echo over tens of milliseconds, so the
  energy gain of the duct is larger than the peak gain shown.
- **Sensitivity to the sea surface** (flank scenarios, 5 m/s wind, not shipped as
  scenarios). Adding Chapman–Harris surface reverberation: 26.4, 22.7 and 19.4 dB at 1, 5
  and 10 km (a loss of 1.7 and 1.4 dB at 1 and 5 km; +0.8 dB at 10 km, within a noise
  draw). Adding the rough-surface reflection loss as well: 36.4, 28.3 and 18.4 dB. The
  SNR *rises* at 1 and 5 km because the coherent-loss model strips surface-bounce paths,
  and those carry most of the reverberation while the target's direct path is untouched.
  That model overstates the loss (D-38), so this is a bound on the effect, not an estimate.
- **This is not a detection-range prediction.** The shipped scenarios have bottom
  reverberation only, with a projector taken as omnidirectional, and no
  surface roughness loss, no platform self-noise, one noise draw, and an idealised flat
  bottom. The plan's own caveat applies: each of those can move the result by 10 dB.
- The SNR is the peak within 60 ms after the first arrival over the mean noise power on
  the same beam. Multipath spreads the echo, so its energy is higher than its peak shows.

## 3D near-field imaging of a buried object

Source: `tests/python/test_volume.py`. CPU. A `subbottom` sonar (20 kHz, 20 kHz
bandwidth, one projector, four receivers) rastered over a 1.8 m square at 7.5 cm steps
(625 pings), 1.5 m above the seabed. Water 1500 m/s, sediment 1700 m/s. A 0.5 m cylinder
(a line of scatterers on its axis) buried 0.30 m. Navigation is the truth. The volume is
built as horizontal layers at 5 mm depth steps.

| Quantity | Refraction on | Refraction off |
| --- | --- | --- |
| Depth of the brightest voxel | 0.300 m | 0.265 m |
| Error against the true 0.30 m | 0.000 m | −0.035 m (predicted at nadir: −0.035 m) |
| Peak level | reference | −0.75 dB |
| Level in the depth slice at the true depth | reference | −15.0 dB |

- "Defocuses with refraction off": mostly it is misplaced. The peak drops by only
  0.75 dB because this aperture subtends at most about 27° at the object, so the delay
  error is nearly the same for every ping and reads as a depth shift. A wider aperture
  or a larger sound-speed contrast would blur it more.
- The travel-time routine matches a brute-force least-time search to 1e-10 and satisfies
  Snell's law at the crossing point to 1e-7.
- Not modelled: transmission loss and attenuation in the sediment, a rough or sloping
  interface, shear waves, and the cylinder's own scattering (it is a line of points).
- Multi-layer GeoTIFF output of the volume is not written yet.

## Motion estimation

Source: `tests/python/test_motion.py`. CPU. Scenario: `sas_auv` at 200 kHz (wavelength
7.5 mm), 40 pings at 0.24 m (so 4 of 16 phase centres overlap between pings), 15 m
altitude, speckle seafloor of 25 scatterers/m² plus one bright point, short Gaussian
pulse. True trajectory: smooth random sway of 10 mm RMS (39 mm peak, five wavelengths)
and heave of 4 mm RMS. Navigation reports a straight level line; the DVL altitude has
1 mm noise. 1248 delay measurements, none rejected.

| Quantity | Result | Stated bound |
| --- | --- | --- |
| Sway error, RMS | 0.109 mm = λ/69 | λ/20 |
| Heave error, RMS | 0.328 mm = λ/23 | λ/4 |
| Sway error after the coarse pass only | 0.295 mm | — |
| Point-response peak, navigation only, against the true-motion image | −10.01 dB | — |
| Point-response peak, after motion estimation | +0.03 dB | within 1 dB |

- Both Ceres passes converged (8 and 7 iterations).
- Heave is the weaker estimate, as expected: grazing angles here span only 17° to 27°, so
  the delays say much more about sway than heave, and the DVL term carries the altitude.
- Solver checks on synthetic delays: exact delays are inverted to 10 µm; with twelve
  20 µs outliers the Huber loss keeps sway inside a wavelength (3.8 mm) where the squared
  loss gives 198 mm; correlation weighting cuts the error by more than 3× when a fifth of
  the measurements are poor.
- Not measured: image contrast loss (only the point-response peak), behaviour with
  attitude errors, and anything on real data.

## SAS on an extended scene with unknown motion (`wreck_sas`)

Source: scenario `wreck_sas`; `tests/python/test_scene_objects.py` (the end-to-end test
runs only where there is a CUDA device; the figures below are from a CPU run of the same
steps). A 3 × 1 × 1 m box on a seafloor of 7500 Lambert scatterers, 80 pings at 0.24 m,
15 m altitude, sway 10 mm RMS (27 mm peak, 3.6 wavelengths) and heave 4 mm RMS that
navigation does not report. Chain: simulate, pulse compress, displaced-phase-centre
motion estimation, backprojection at 2.5 cm.

![wreck_sas with navigation only, estimated motion and true motion](figures/wreck_sas.png)

| Quantity | Navigation only | Estimated motion | True motion |
| --- | --- | --- | --- |
| Sway error, RMS | 1.34 λ | 0.089 λ | — |
| Heave error, RMS | — | 0.098 λ | — |
| Image sharpness (Σ I² · N / (Σ I)²) | 69 | 272 | 279 |
| Seafloor over shadow behind the box | 13.7 dB | 14.7 dB | 14.8 dB |
| Box face over seafloor | 20.4 dB | 20.3 dB | 20.2 dB |

- With the estimated track the image is within 3% of the true-motion sharpness; with
  navigation alone it is four times less sharp. The shadow contrast barely depends on
  focus because the shadow is much larger than the blur.
- The bright band at 40.6 to 41 m is the box's top and near face laid over towards the
  sonar (it is 1 m above the image plane), and it smears along track for the same reason.
- The seafloor is 25 scatterers per square metre, far fewer than one per resolution cell,
  so it shows as separate points, not developed speckle. Synthesis took 245 s on four CPU
  cores; motion estimation 1 s (2528 delay measurements).

## Kernel accuracy

Source: `tests/python/test_backproject.py`.

| Check | Result | Requirement |
| --- | --- | --- |
| Carrier phase error, 32-element flank array, 10 kHz, targets 0.2–10 km, platform moving and yawing | below 0.05° | below 1° |
| Carrier phase error, 128-element probe, 5 MHz, 0.75–3 m, both mux halves | below 0.05° | below 1° |
| Kernel against independent float64 reference, nearest / linear | 2e-5 relative | — |
| Kernel against independent float64 reference, sinc | 2e-4 relative | — |
| Same with a platform moving at 3 m/s, weights on | 2e-4 relative | — |

The phase errors are the worst case over random targets, against long-double arithmetic.
The tests assert 0.05°; the requirement is 1°.

## Conditioning, pulse compression and post-processing

Source: `tests/python/test_conditioning.py`. CPU.

| Check | Measured | Against |
| --- | --- | --- |
| LFM compression gain, B = 4 kHz, T = 0.25 s | 29.99 dB | 10·log10(BT) = 30.00 dB |
| LFM compression gain, B = 4 kHz, T = 0.05 s | 23.00 dB | 23.01 dB |
| LFM compression gain, B = 15 kHz, T = 0.02 s | 24.76 dB | 24.77 dB |
| Unweighted LFM peak sidelobe | −13.3 dB ± 1 | −13.3 dB (sinc) |
| Hamming-weighted replica peak sidelobe | below −30 dB | — |
| Whitening (Eq. 9): in-band spread of a 12 dB tilt + 6 dB ripple, after | below 1 dB | flat |
| Median TVG (Eq. 10): intensity spread over a 30 dB decay, after | below 1 dB | flat |
| Median TVG with a 60 dB target on 6 of 64 channels: change in gain | below 3 dB | a mean-based gain would notch by more than 40 dB |
| DRC (Eq. 11–12): output median for targets 0.1, 0.2, 0.5 | within 0.001 | the target |
| Calibration: residual after removing injected gain, phase, delay | below 1e-4 | exact |
| Doppler bank, 1 s LFM, target closing at 6 m/s | 6 m/s ± 1 (1 m/s grid) | truth; the zero-Doppler replica loses more than half the peak |

The compression gain is the output SNR over the input SNR measured in the signal band.

## Ray culling

Source: `tests/python/test_culling.py::test_report_cpu_speedup`. **CPU backend, 4 cores.**
Stripmap geometry: `sas_auv`, 200 pings over 64 m of track, 16 receivers each, image
400 × 240 points, linear interpolation, field of view 1.5 × FWHM. Best of three runs.
Without culling: 1.78 s.

| Block size b | Speedup | (ping, block) pairs kept |
| --- | --- | --- |
| 4 | 1.60× | 19% |
| 8 | 1.68× | 21% |
| 16 | 1.62× | 25% |
| 32 | 1.61× | 33% |

- The image with culling on is bit-identical to the image without it, in every test.
- The paper reports "a factor of two" for its GPU implementation on its own data. These
  figures are CPU and a different geometry, so they are not a like-for-like comparison.
  The gain depends on how much of the image each ping can see: a 15 m collection where
  each ping covered most of the image gave only 1.1×.
- These CPU figures predate the kernel's float32 pre-test. With it, the same unculled
  run takes 1.11 s and culling adds about 1.1×. GPU figures are in "Performance".

## Simulator

Source: `tests/python/test_sim.py`.

| Check | Result |
| --- | --- |
| Echo delay, Gaussian pulse, three probes | within 0.01 carrier period |
| Echo amplitude (spreading, absorption, both directivities) | within 0.5% |
| Echo delay, LFM by matched filter, four sonars | within 0.02 / bandwidth |
| Flank echo level at 1 km against SL − 2TL + TS + sensitivity | within 0.7 dB (absorption varies across the sweep) |
| Time-domain synthesis against frequency-domain oracle | 5e-3 relative (Gaussian truncation) |
| Arrival time with a 30 m/s receiver | within 0.2 µs of the implicit solution |
| Doppler shift, 5 m/s closing target, 8 kHz | within 0.2% of 53.5 Hz |

## Not yet measured

Speed on the DGX Spark and Jetson; multi-GPU scaling on a dataset large enough to show it.
