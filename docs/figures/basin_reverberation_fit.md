# Late reverberation of a fjord: the decay law fitted to the measured energy-time curves

Data: SFI Smart Ocean dataset (van Walree, Otnes, Tomasi, Henriksen, Danre, Bergh, IEEE DataPort 2025, doi:10.21227/3aa6-4k33, CC BY 4.0); soundings by `tools/channel_smartocean.py`. One fjord, one night, 4-8 kHz.

Made by `tools/basin_reverberation_fit.py`. The curve fitted is the share of the received energy still to come, against delay after the first arrival (median of the cycles; 0.745 s chirps, 5 kHz of band). Fits in dB, every 5 ms, from the onset to the floor named.

| Hydrophone | fitted from, down to | to delay, s | exponential (2 numbers): rms / max misfit, dB; T, s | power law (3): rms / max; t0, s; exponent | two exponentials (4): rms / max; T1, T2, s; weight |
| --- | --- | --- | --- | --- | --- |
| R1, 274 m | 10 ms, -30 dB | 0.86 | 0.59 / 1.33; 0.145 | 0.18 / 0.47; 1.1; 11.4 | 0.17 / 0.44; 0.104, 0.202; 0.83 |
| R1, 274 m | 10 ms, -40 dB | 1.25 | 0.81 / 2.14; 0.158 | 0.33 / 0.79; 1.53; 14.4 | 0.24 / 0.51; 0.0919, 0.183; 0.73 |
| R1, 274 m | 20 ms, -30 dB | 0.86 | 0.58 / 1.39; 0.145 | 0.18 / 0.48; 1.1; 11.3 | 0.18 / 0.45; 0.104, 0.202; 0.82 |
| R1, 274 m | 20 ms, -40 dB | 1.25 | 0.79 / 2.21; 0.159 | 0.33 / 0.79; 1.55; 14.4 | 0.24 / 0.53; 0.0914, 0.183; 0.72 |
| R1, 274 m | 50 ms, -30 dB | 0.86 | 0.53 / 1.18; 0.147 | 0.18 / 0.49; 1.12; 11.2 | 0.17 / 0.41; 0.105, 0.204; 0.81 |
| R1, 274 m | 50 ms, -40 dB | 1.25 | 0.73 / 1.54; 0.160 | 0.33 / 0.76; 1.65; 14.8 | 0.24 / 0.57; 0.182, 0.0906; 0.31 |
| R2, 1020 m | 10 ms, -30 dB | 1.01 | 0.55 / 1.57; 0.160 | 0.55 / 1.57; 3.27e+04; 2.04e+05 | 0.55 / 1.59; 0.0101, 0.161; 0.15 |
| R2, 1020 m | 10 ms, -40 dB | 1.41 | 0.49 / 1.59; 0.162 | 0.49 / 1.60; 68; 425 | 0.49 / 1.61; 0.0119, 0.162; 0.18 |
| R2, 1020 m | 20 ms, -30 dB | 1.01 | 0.55 / 1.58; 0.161 | 0.55 / 1.58; 2.6e+04; 1.62e+05 | 0.55 / 1.58; 0.161, 0.161; 0.00 |
| R2, 1020 m | 20 ms, -40 dB | 1.41 | 0.49 / 1.60; 0.162 | 0.49 / 1.61; 101; 628 | 0.49 / 1.61; 0.0101, 0.162; 0.12 |
| R2, 1020 m | 50 ms, -30 dB | 1.01 | 0.56 / 1.59; 0.161 | 0.56 / 1.59; 1.5e+04; 9.33e+04 | 0.56 / 1.59; 8.97e-05, 0.161; 0.00 |
| R2, 1020 m | 50 ms, -40 dB | 1.41 | 0.49 / 1.61; 0.162 | 0.49 / 1.62; 254; 1.57e+03 | 0.49 / 1.61; 3.52e+05, 0.162; 0.00 |

The exponential from 10 ms to -30 dB, as `basin_reverberation` takes it, and what it then reads against the measurement (the modelled arrivals as one arrival at delay 0):

| | R1, 274 m | R2, 1020 m |
| --- | --- | --- |
| `energy_db` (tail over the rest) | -4.2 | +0.2 |
| `onset_s` | 0.010 | 0.010 |
| `decay_s` | 0.145 | 0.160 |
| tail's share of all the energy: fitted exponential (measured later than 10 ms) | 0.273 (0.358) | 0.513 (0.594) |
| 90 % of the energy within, ms: model (measured) | 156 (165) | 272 (275) |
| 99 % of the energy within, ms: model (measured) | 489 (465) | 642 (636) |
| 99.9 % of the energy within, ms: model (measured) | 823 (866) | 1011 (1012) |
| rms delay spread, ms: model (measured) | 102 (100) | 143 (145) |
| tail's energy re the source's at 1 m, dB (source level 183 measured / 181 expected) | -43.1 / -41.1 | -61.1 / -59.1 |

Share of the energy in a band of delays that is the same on five chirps 3 s apart (noise removed; median of the cycles, 16 to 84 %). The source hung from a floating platform; the hydrophones stood on the seabed.

| Delay, ms | R1, 274 m | R2, 1020 m |
| --- | --- | --- |
| 0-10 | 0.97 (0.93 to 0.98) | 0.92 (0.88 to 0.96) |
| 10-50 | 0.73 (0.67 to 0.81) | 0.66 (0.55 to 0.72) |
| 50-100 | 0.70 (0.61 to 0.80) | 0.57 (0.49 to 0.65) |
| 100-200 | 0.70 (0.62 to 0.80) | 0.46 (0.41 to 0.56) |
| 200-300 | 0.68 (0.60 to 0.78) | 0.38 (0.30 to 0.47) |
| 300-500 | 0.64 (0.56 to 0.75) | 0.49 (0.43 to 0.59) |
| 500-800 | 0.57 (0.48 to 0.69) | 0.47 (0.40 to 0.55) |
| 800-1200 | 0.51 (0.42 to 0.62) | 0.34 (0.29 to 0.45) |

Figure: `basin_reverberation_fit.png`.
