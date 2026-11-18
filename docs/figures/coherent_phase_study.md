# The wandering coherent surface arrival: long pulses and ping-to-ping phase at 5 kHz

Made by `tools/coherent_phase_study.py` from the phase law of `time_spread.coherent_phase` (sim/timespread.py item 8). The echo that goes out and comes back by the surface (twice the one-way phase). Waves with a Fresnel angle above 1 rad take no part (they scatter; the arrival is then steady and the row reads zero): the last column is the two-way Gamma^2 that height has as scattering roughness instead. "Best shift": after the best constant frequency shift per pulse (what a chirp absorbs as a range offset): the curvature the simulator's linear-within-a-ping law leaves out.

## measured fjord path, 274 m (20 m to 43 m); grazing 12.9 degrees

| Sea | Fresnel angle, rad | height taking part | rms phase, rad (amplitude) | loss of a 0.1 s pulse: CW / best shift, dB | loss of a 1 s pulse: CW / best shift, dB | rms Doppler over 1 s, Hz | ping to ping, 5 s / 20 s: rms phase change, rad | four pings added coherently, 5 s / 20 s: loss, dB | two-way Gamma^2 if it scatters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fjord, calm night: 0.8 cm rms, 5 s; waves along the path | 0.751 | 0.78 | 0.103 (0.041) | 0.00 / 0.00 | 0.01 / 0.00 | 0.018 | 0.06 / 0.14 | 0.01 / 0.04 | 0.0113 |
| fjord, calm night: 0.8 cm rms, 5 s; waves across the path | 0.0377 | 1.00 | 0.150 (0.003) | 0.00 / 0.00 | 0.01 / 0.00 | 0.028 | 0.09 / 0.21 | 0.03 / 0.09 | 0.0113 |
| swell 5 cm rms, 8 s; waves along the path | 0.115 | 1.00 | 0.926 (0.060) | 0.00 / 0.00 | 0.18 / 0.00 | 0.113 | 1.69 / 1.47 | 2.69 / 3.14 | 0.441 |
| swell 5 cm rms, 8 s; waves across the path | 0.00575 | 1.00 | 0.934 (0.003) | 0.00 / 0.00 | 0.19 / 0.00 | 0.114 | 1.70 / 1.49 | 2.72 / 3.19 | 0.441 |
| swell 25 cm rms, 10 s; waves along the path | 0.0469 | 1.00 | 4.618 (0.122) | 0.03 / 0.00 | 2.02 / 0.02 | 0.456 | 9.12 / 4.90 | 5.72 / 6.30 | 11 |
| swell 25 cm rms, 10 s; waves across the path | 0.00236 | 1.00 | 4.625 (0.006) | 0.03 / 0.00 | 2.03 / 0.02 | 0.456 | 9.13 / 4.91 | 5.72 / 6.49 | 11 |
| sea state 2 (4.4 m/s): all 10 cm rms as large-scale, 2.3 s; waves along the path | 17 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 1.89 |
| sea state 2 (4.4 m/s): all 10 cm rms as large-scale, 2.3 s; waves across the path | 0.856 | 0.66 | 1.199 (0.438) | 0.04 / 0.01 | 2.03 / 0.74 | 0.363 | 1.04 / 1.67 | 2.65 / 3.53 | 1.89 |
| sea state 3 (6.9 m/s): all 25 cm rms as large-scale, 3.6 s; waves along the path | 2.82 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 11.4 |
| sea state 3 (6.9 m/s): all 25 cm rms as large-scale, 3.6 s; waves across the path | 0.141 | 1.00 | 4.709 (0.353) | 0.24 / 0.00 | 5.71 / 1.27 | 1.147 | 8.23 / 6.72 | 5.95 / 6.56 | 11.4 |
| sea state 4 (9.8 m/s): all 51 cm rms as large-scale, 5.1 s; waves along the path | 0.693 | 0.84 | 7.050 (0.601) | 0.25 / 0.00 | 5.65 / 1.57 | 1.241 | 4.34 / 9.60 | 6.35 / 5.52 | 46.5 |
| sea state 4 (9.8 m/s): all 51 cm rms as large-scale, 5.1 s; waves across the path | 0.0348 | 1.00 | 9.631 (0.186) | 0.47 / 0.00 | 7.18 / 1.12 | 1.772 | 5.72 / 13.36 | 5.92 / 5.38 | 46.5 |

## test AUV trial, 500 m (10 m to 15 m); grazing 2.9 degrees

| Sea | Fresnel angle, rad | height taking part | rms phase, rad (amplitude) | loss of a 0.1 s pulse: CW / best shift, dB | loss of a 1 s pulse: CW / best shift, dB | rms Doppler over 1 s, Hz | ping to ping, 5 s / 20 s: rms phase change, rad | four pings added coherently, 5 s / 20 s: loss, dB | two-way Gamma^2 if it scatters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fjord, calm night: 0.8 cm rms, 5 s; waves along the path | 29.8 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 0.00056 |
| fjord, calm night: 0.8 cm rms, 5 s; waves across the path | 0.0743 | 1.00 | 0.033 (0.001) | 0.00 / 0.00 | 0.00 / 0.00 | 0.006 | 0.02 / 0.05 | 0.00 / 0.00 | 0.00056 |
| swell 5 cm rms, 8 s; waves along the path | 4.55 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 0.0219 |
| swell 5 cm rms, 8 s; waves across the path | 0.0113 | 1.00 | 0.208 (0.001) | 0.00 / 0.00 | 0.01 / 0.00 | 0.025 | 0.38 / 0.33 | 0.18 / 0.20 | 0.0219 |
| swell 25 cm rms, 10 s; waves along the path | 1.86 | 0.06 | 0.183 (0.094) | 0.00 / 0.00 | 0.00 / 0.00 | 0.015 | 0.35 / 0.34 | 0.17 / 0.17 | 0.547 |
| swell 25 cm rms, 10 s; waves across the path | 0.00465 | 1.00 | 1.031 (0.003) | 0.00 / 0.00 | 0.15 / 0.00 | 0.102 | 2.04 / 1.09 | 2.98 / 2.69 | 0.547 |
| sea state 2 (4.4 m/s): all 10 cm rms as large-scale, 2.3 s; waves along the path | 677 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 0.0939 |
| sea state 2 (4.4 m/s): all 10 cm rms as large-scale, 2.3 s; waves across the path | 1.69 | 0.12 | 0.102 (0.057) | 0.00 / 0.00 | 0.02 / 0.01 | 0.030 | 0.12 / 0.15 | 0.04 / 0.05 | 0.0939 |
| sea state 3 (6.9 m/s): all 25 cm rms as large-scale, 3.6 s; waves along the path | 112 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 0.568 |
| sea state 3 (6.9 m/s): all 25 cm rms as large-scale, 3.6 s; waves across the path | 0.279 | 1.00 | 1.011 (0.161) | 0.01 / 0.00 | 0.91 / 0.09 | 0.245 | 1.76 / 1.44 | 2.99 / 2.10 | 0.568 |
| sea state 4 (9.8 m/s): all 51 cm rms as large-scale, 5.1 s; waves along the path | 27.5 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 2.31 |
| sea state 4 (9.8 m/s): all 51 cm rms as large-scale, 5.1 s; waves across the path | 0.0686 | 1.00 | 2.142 (0.083) | 0.03 / 0.00 | 1.75 / 0.07 | 0.394 | 1.27 / 2.97 | 3.82 / 5.03 | 2.31 |

## static array trial, 2.6 km (30 m to 60 m); grazing 2.0 degrees

| Sea | Fresnel angle, rad | height taking part | rms phase, rad (amplitude) | loss of a 0.1 s pulse: CW / best shift, dB | loss of a 1 s pulse: CW / best shift, dB | rms Doppler over 1 s, Hz | ping to ping, 5 s / 20 s: rms phase change, rad | four pings added coherently, 5 s / 20 s: loss, dB | two-way Gamma^2 if it scatters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fjord, calm night: 0.8 cm rms, 5 s; waves along the path | 299 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 0.000269 |
| fjord, calm night: 0.8 cm rms, 5 s; waves across the path | 0.358 | 1.00 | 0.021 (0.004) | 0.00 / 0.00 | 0.00 / 0.00 | 0.004 | 0.01 / 0.03 | 0.00 / 0.00 | 0.000269 |
| swell 5 cm rms, 8 s; waves along the path | 45.6 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 0.0105 |
| swell 5 cm rms, 8 s; waves across the path | 0.0546 | 1.00 | 0.144 (0.004) | 0.00 / 0.00 | 0.00 / 0.00 | 0.018 | 0.26 / 0.23 | 0.08 / 0.10 | 0.0105 |
| swell 25 cm rms, 10 s; waves along the path | 18.7 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 0.262 |
| swell 25 cm rms, 10 s; waves across the path | 0.0224 | 1.00 | 0.714 (0.009) | 0.00 / 0.00 | 0.07 / 0.00 | 0.070 | 1.41 / 0.76 | 1.93 / 1.44 | 0.262 |
| sea state 2 (4.4 m/s): all 10 cm rms as large-scale, 2.3 s; waves along the path | 6.79e+03 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 0.0451 |
| sea state 2 (4.4 m/s): all 10 cm rms as large-scale, 2.3 s; waves across the path | 8.12 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 0.0451 |
| sea state 3 (6.9 m/s): all 25 cm rms as large-scale, 3.6 s; waves along the path | 1.12e+03 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 0.273 |
| sea state 3 (6.9 m/s): all 25 cm rms as large-scale, 3.6 s; waves across the path | 1.34 | 0.25 | 0.257 (0.130) | 0.00 / 0.00 | 0.06 / 0.02 | 0.056 | 0.31 / 0.33 | 0.29 / 0.19 | 0.273 |
| sea state 4 (9.8 m/s): all 51 cm rms as large-scale, 5.1 s; waves along the path | 276 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 1.11 |
| sea state 4 (9.8 m/s): all 51 cm rms as large-scale, 5.1 s; waves across the path | 0.33 | 1.00 | 1.389 (0.258) | 0.01 / 0.00 | 0.88 / 0.07 | 0.253 | 0.82 / 1.92 | 2.09 / 4.21 | 1.11 |

## long range, 12 km (10 m to 4 m); grazing 0.1 degrees

| Sea | Fresnel angle, rad | height taking part | rms phase, rad (amplitude) | loss of a 0.1 s pulse: CW / best shift, dB | loss of a 1 s pulse: CW / best shift, dB | rms Doppler over 1 s, Hz | ping to ping, 5 s / 20 s: rms phase change, rad | four pings added coherently, 5 s / 20 s: loss, dB | two-way Gamma^2 if it scatters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fjord, calm night: 0.8 cm rms, 5 s; waves along the path | 1.11e+06 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 3.06e-07 |
| fjord, calm night: 0.8 cm rms, 5 s; waves across the path | 1.51 | 0.16 | 0.000 (0.000) | 0.00 / 0.00 | 0.00 / 0.00 | 0.000 | 0.00 / 0.00 | 0.00 / 0.00 | 3.06e-07 |
| swell 5 cm rms, 8 s; waves along the path | 1.7e+05 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 1.19e-05 |
| swell 5 cm rms, 8 s; waves across the path | 0.231 | 1.00 | 0.005 (0.001) | 0.00 / 0.00 | 0.00 / 0.00 | 0.001 | 0.01 / 0.01 | 0.00 / 0.00 | 1.19e-05 |
| swell 25 cm rms, 10 s; waves along the path | 6.96e+04 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 0.000299 |
| swell 25 cm rms, 10 s; waves across the path | 0.0947 | 1.00 | 0.024 (0.001) | 0.00 / 0.00 | 0.00 / 0.00 | 0.002 | 0.05 / 0.03 | 0.00 / 0.00 | 0.000299 |
| sea state 2 (4.4 m/s): all 10 cm rms as large-scale, 2.3 s; waves along the path | 2.53e+07 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 5.13e-05 |
| sea state 2 (4.4 m/s): all 10 cm rms as large-scale, 2.3 s; waves across the path | 34.4 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 5.13e-05 |
| sea state 3 (6.9 m/s): all 25 cm rms as large-scale, 3.6 s; waves along the path | 4.18e+06 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 0.00031 |
| sea state 3 (6.9 m/s): all 25 cm rms as large-scale, 3.6 s; waves across the path | 5.69 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 0.00031 |
| sea state 4 (9.8 m/s): all 51 cm rms as large-scale, 5.1 s; waves along the path | 1.03e+06 | 0.00 | 0.000 (0.000) | -0.00 / -0.00 | -0.00 / -0.00 | 0.000 | 0.00 / 0.00 | -0.00 / -0.00 | 0.00126 |
| sea state 4 (9.8 m/s): all 51 cm rms as large-scale, 5.1 s; waves across the path | 1.4 | 0.22 | 0.016 (0.009) | 0.00 / 0.00 | 0.00 / 0.00 | 0.003 | 0.02 / 0.03 | 0.00 / 0.00 | 0.00126 |

