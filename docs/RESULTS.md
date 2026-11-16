# Results

Measured by this repository's tests and tools. Every number here was produced on the
**CPU (host) backend** in a container with no GPU; nothing in this file is a GPU result.
Each row names what it is compared with. Where the brief asks for a comparison that
could not be made, that is stated.

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
- GPU speedup: not measured (no device).

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

Motion-estimation accuracy, detection range, and every speed figure.
