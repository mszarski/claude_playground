# Detection with the long-range array: how much the decay-time rule matters

Made by `tools/decay_time_sensitivity.py`: each scenario without time spreading and with a wind-roughened surface under each rule. One row of 32 hydrophones, C++ chain; pings, Bellhop arrivals kept and taps as named in the first column (the same in every variant of a row). SNR: the ship's detection over the local background, median over the pings it was detected on; in brackets each seed of the scattered taps.

| Scenario | wind, m/s | no time spreading | facet | rule of thumb | facet, multi_bounce sum | rule of thumb minus facet, dB | spread between seeds, dB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| popoto_12km, image-source paths (6 pings, 6 taps) | 5 | 13.5 dB; 3/6, track 5 | 13.5 dB (13.5, 13.6); 2/6, track none; 2/6, track none | 13.5 dB (13.5, 13.5); 3/6, track 5; 3/6, track 5 | 13.5 dB (13.5, 13.6); 2/6, track none; 2/6, track none | -0.1 | 0.1 |
| popoto_12km, Bellhop (3 pings, 12 strongest arrivals in every variant, 3 taps) | 5 | 15.1 dB; 3/3, track 3 | 15.0 dB (15.0, 15.0); 3/3, track 3; 3/3, track 3 | 14.8 dB (14.9, 14.7); 3/3, track 3; 3/3, track 3 | - | -0.2 | 0.2 |
| popoto_env_gravel_30m, Bellhop (3 pings, 12 strongest arrivals in every variant, 3 taps) | 5 | 32.5 dB; 3/3, track 3 | 32.8 dB (33.1, 32.4); 3/3, track 3; 3/3, track 3 | 32.4 dB (32.4, 32.4); 3/3, track 3; 3/3, track 3 | - | -0.4 | 0.7 |
| popoto_env_summer_mud_60m, Bellhop (3 pings, 12 strongest arrivals in every variant, 3 taps) | 5 | not detected; 0/3, track none | not detected (-, -); 0/3, track none; 0/3, track none | not detected (-, -); 0/3, track none; 0/3, track none | - | - | - |
| popoto_env_winter_sand_200m, Bellhop (3 pings, 12 strongest arrivals in every variant, 3 taps) | 5 | 22.9 dB; 3/3, track 3 | 23.3 dB (22.9, 23.8); 3/3, track 3; 3/3, track 3 | 23.0 dB (23.2, 22.8); 3/3, track 3; 3/3, track 3 | - | -0.3 | 0.9 |

Without noise, reverberation or a detector: one point target, image-source paths with two bounces, one hydrophone after pulse compression. The strongest sample of the echo, re a flat surface:

| Geometry | wind, m/s | pings | facet: mean (median), ping-to-ping sd, dB | rule of thumb | rule of thumb minus facet, dB | its standard error, dB | echo energy, facet / rule of thumb, dB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flank_sst_shallow geometry (1 km) | 6 | 24 | +0.15 (-0.02), 1.02 | +0.07 (-0.60), 1.04 | -0.08 | 0.30 | +0.72 / +0.65 |
| popoto_12km geometry (12 km) | 5 | 24 | -0.04 (-0.04), 0.01 | -0.04 (-0.04), 0.00 | -0.00 | 0.00 | -0.00 / -0.01 |
