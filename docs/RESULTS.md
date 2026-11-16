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

SAS resolution, motion-estimation accuracy, detection range, and every speed figure.
