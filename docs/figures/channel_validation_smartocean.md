# SFI Smart Ocean dataset, 4-8 kHz: measured against our propagation and time spreading

Data: SFI Smart Ocean dataset (van Walree, Otnes, Tomasi, Henriksen, Danre, Bergh, IEEE DataPort 2025, doi:10.21227/3aa6-4k33, CC BY 4.0). Made by `tools/channel_smartocean.py`; nothing of the dataset is in this repository.

160 recordings (80 cycles, two hydrophones). Values are medians over cycles with the 16 % and 84 % points in brackets.

## Arrivals: straight rays, Bellhop with the measured profiles, and the measurement

Stated: positions, depths, the two casts. From the chart: seabed along the paths. Assumed: the seabed half-space (1600 m/s, 1.8, 0.5 dB/wavelength), a 1.5 m tripod at R1. Fitted: nothing.

| | straight rays | Bellhop, isovelocity | Bellhop, cast 1 (17:05) | Bellhop, cast 2 (07:07) | measured |
| --- | --- | --- | --- | --- | --- |
| R1 (274 m): surface reflection after direct, ms | 4.14 | 4.14 | 4.53 | 4.45 | 4.313 (4.282 to 4.41) |
| R1: grazing angle at the surface, degrees | 13.0 | 13.0 | 14.7 | 14.7 | not measured |
| R1: surface reflection re direct, dB | -0.2 | -0.2 | 0.3 | -0.1 | -1.5 (-2.1 to -0.97) |
| R2 (1020 m), Bellhop isovelocity: arrivals within 8 dB, first 6 ms | | | | | +0.00 ms 0 dB S0B0; +0.24 ms -0 dB S0B1; +0.37 ms -0 dB S1B0 |
| R2 (1020 m), Bellhop cast1: arrivals within 8 dB, first 6 ms | | | | | +0.00 ms 0 dB S1B2; +2.15 ms -5 dB S1B1 |
| R2 (1020 m), Bellhop cast2: arrivals within 8 dB, first 6 ms | | | | | +0.00 ms -3 dB S1B1 |
| R2: the two strong arrivals: spacing ms; second re first dB | 0.37 (direct, surface) | | | | 1.84 (1.68 to 1.91); 1.6 (-5.5 to 8) |

First arrival at R2 minus first arrival at R1 (recorder time stamps): 0.49574 (0.49562 to 0.49589) s.

## The surface reflection at R1 (the one arrival that is a single surface bounce and stands alone)

| Quantity | 4 Hz chirp soundings, 16 s, all cycles | least squares on the 60 s sequence, 4 Hz, cycles 14-80 | 40 Hz chirp soundings, 10 s, cycles 14-80 |
| --- | --- | --- | --- |
| soundings' fluctuation floor, dB re direct | -38.4 (-39.7 to -37) | -32.9 (-33.2 to -32.3) | -24.4 (-25.8 to -23.1) |
| coherent share | 0.9873 (0.9777 to 0.9921) | 0.9877 (0.9788 to 0.9916) | 0.9904 (0.975 to 0.9954) |
| phase variance, rad^2 (below 1 Hz, white part removed) | 0.00836 (0.00526 to 0.0166) | 0.0086 (0.00585 to 0.014) | 0.00913 (0.00482 to 0.0212) |
| relative amplitude variance (same) | 0.0037 (0.0019 to 0.0078) | 0.00433 (0.00294 to 0.0073) | 0.00662 (0.00363 to 0.0108) |
| scintillation index of the arrival's energy | 0.0146 (0.00736 to 0.0309) | 0.015 (0.0101 to 0.0266) | 0.0132 (0.00481 to 0.0311) |
| rms width of the fluctuation's spectrum, Hz | 0.335 (0.284 to 0.395) | 0.265 (0.201 to 0.337) | 0.551 (0.434 to 0.658) |
| shape w90/w50 (Gaussian 2.44) | 1.47 (1.29 to 1.89) | 1.43 (1.21 to 1.81) | 2.35 (1.73 to 3.06) |
| phase variance, upper over lower half band; in brackets what this estimator reads for a pure change of path length (1 for none) | 1.57 +- 0.07 (1.42) | 1.26 +- 0.04 (1.11) | 1.61 +- 0.12 (1.42) |

Coherent share of the arrival and its tail together (0.5 ms before to 0.9 ms behind): 0.9688. The fit below splits the fluctuating energy in two:

| Part | share of the arrival's energy | rms frequency, Hz | as an rms surface height at 14.7 degrees, cm |
| --- | --- | --- | --- |
| unspread (the steady profile's own shape) | 0.0106 | 0.36 | 0.80 |
| tail | 0.0217 | 0.63 | 1.14 |

The sender hangs from a floating platform: 1.3 cm rms of heave would do to the peak what the unspread part does. A single wave of the unspread part's height and frequency has a mean-square slope of 1.7e-05: no tail from it under either rule.

A Pierson-Moskowitz sea of the tail's height is a 1.5 m/s wind; that wind's laws give an rms frequency of 1.32 Hz (measured behind the arrival: 0.63) and a Cox-Munk mean-square slope of 0.0105.

### The tail behind it

Fluctuating intensity around the arrival, mean of 80 cycles, fitted as (steady profile) x a + (pulse convolved with a tail of mean excess delay T) x b:

| Tail shape | window | T, ms | tail's share of the fluctuating energy | misfit (without a tail) |
| --- | --- | --- | --- | --- |
| exponential | 3 ms behind | 0.96 | 0.59 | 0.86 (23.8) |
| facet shape | 3 ms behind | 0.90 | 0.73 | 0.60 (23.8) |
| exponential | 0.9 ms behind | 0.50 | 0.67 | 0.44 (8.2) |
| facet shape | 0.9 ms behind | 1.08 | 0.69 | 0.58 (8.2) |

Four quarters of the cycles, 0.9 ms window: exponential T = 0.27, 0.63, 0.53, 0.63 ms; facet shape T = 0.24, 1.74, 1.22, 1.96 ms.

The two rules for this bounce (t1 t2 / (t1 + t2) = 40.7 ms) with that slope: facet 0.45 ms, rule of thumb 0.85 ms. Our model with that surface (height and vertical speed of the tail part: two fitted numbers; slope from the wind law), sampled like the soundings and passed through the same fit:

| | T read with an exponential, 0.9 ms window | with the facet shape, 0.9 ms | with an exponential, 3 ms | mean excess delay put in |
| --- | --- | --- | --- | --- |
| model, facet | 0.67 | 2.21 | 0.44 | 0.45 |
| model, rule of thumb | 4.00 | 4.00 | 0.96 | 0.85 |
| measured | 0.50 | 1.08 | 0.96 | |

Read the other way, the measured T is a mean-square slope of 0.0221 (facet) or 0.0118 (rule of thumb) from the exponential 3ms fit; 0.0115 (facet) or 0.0061 (rule of thumb) from the exponential 0.9ms fit.

| Fluctuation over time (floor removed) | rms frequency at the peak, Hz | 0.4-0.9 ms behind, Hz | share above 0.5 Hz at the peak | behind |
| --- | --- | --- | --- | --- |
| 4 Hz chirp soundings | 0.36 | 0.63 | 0.13 | 0.74 |
| 40 Hz chirp soundings | 0.49 | 0.56 | 0.15 | 0.52 |
| least squares, 60 s | 0.37 | 0.55 | 0.06 | 0.47 |

## R2 (1.02 km): the second of the two strong arrivals against the first

| Quantity | 4 Hz chirp soundings | least squares, 60 s |
| --- | --- | --- |
| coherent share | 0.986 (0.9649 to 0.9952) | 0.9822 (0.9502 to 0.9911) |
| phase variance, rad^2 | 0.00645 (0.0027 to 0.0196) | 0.00897 (0.00407 to 0.0195) |
| relative amplitude variance | 0.00827 (0.00302 to 0.0196) | 0.00972 (0.0047 to 0.0267) |
| rms width, Hz | 0.276 (0.174 to 0.354) | 0.225 (0 to 0.277) |
| soundings' floor, dB re the first | -30 (-33.1 to -23.5) | -26.3 (-30.6 to -21.6) |

Same transmissions at both hydrophones (67 cycles, 60 s each): correlation of the first arrivals' phase (line removed) between R1 and R2 -0.7 (-0.83 to -0.44); of the second-against-first phase 0.014 (-0.12 to 0.16) (largest over +-2 s of lag, absolute: 0.22 (0.16 to 0.31); the same between different cycles: 0.18 (0.13 to 0.26)).

## Delay spread against range

| | R1, 274 m | R2, 1020 m |
| --- | --- | --- |
| 90 % of the energy within, ms | 165 (159 to 175) | 275 (220 to 327) |
| 99 % within, ms | 465 (456 to 479) | 636 (601 to 666) |
| 99.9 % within, ms | 866 (852 to 882) | 1.01e+03 (977 to 1.04e+03) |
| rms delay spread, ms | 100 (98.6 to 103) | 145 (134 to 153) |
| share in the first 10 ms | 0.64 (0.62 to 0.67) | 0.41 (0.29 to 0.53) |

## Levels

| | R1 | R2 |
| --- | --- | --- |
| first strong arrival, dB re 1 uPa rms | 135.5 | 110.5 |
| loss from the stated source level (183 measured, 181 expected), dB | 47.5, 45.5 | 72.5, 70.5 |
| Bellhop, strongest arrival, casts 1 and 2 (plus 0.39 dB/km), dB | 48.8, 48.7 | 58.2, 57.1 |
| spherical spreading to the slant range, dB | 48.8 | 60.2 |
| whole response re the first strong arrival, dB | 10.1 | 14.3 |

## Noise (before each cycle's first transmission)

| | R1 | R2 |
| --- | --- | --- |
| signal-free time, s | 222 | 219 |
| 4-8 kHz spectrum level, dB re 1 uPa^2/Hz | 62.9 (62 to 63.6) | 43.9 (41.7 to 48.5) |
| slope over 4-8 kHz, dB/octave (simulator's wind noise: -5.2) | 0.8 | -3.5 |
| wind speed whose Wenz-type curve has that level, m/s (one fitted number) | above 15 | 1.9 |
| var(log envelope) (complex Gaussian: 0.411) | 0.407 (0.398 to 0.417) | 0.425 (0.418 to 0.441) |
| envelope q99.95 / q50 (complex Gaussian: 3.31) | 3.22 (3.12 to 3.36) | 5.97 (3.95 to 9.52) |
| tools/fit_noise.py family | gaussian 0.00 | k 3.49 |

Coherence between the two hydrophones (about 1.2 km apart, 179 s in common): mean magnitude 0.007, largest 0.047; the estimator's floor for independent channels is 0.007.

Figures: `channel_validation_smartocean_arrivals.png`, `channel_validation_smartocean_surface.png`, `channel_validation_smartocean_spread_noise.png`.
