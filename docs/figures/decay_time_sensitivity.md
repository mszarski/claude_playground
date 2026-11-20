# Detection with the long-range array: how much the decay-time rule matters

Made by `tools/decay_time_sensitivity.py`: each scenario without time spreading and with a wind-roughened surface under each rule. One row of 32 hydrophones, C++ chain; pings, Bellhop arrivals kept and taps as named in the first column (the same in every variant of a row). SNR: the ship's detection over the local background, median over the pings it was detected on; in brackets each seed of the scattered taps.

| Scenario | wind, m/s | no time spreading | facet | rule of thumb | facet, multi_bounce sum | rule of thumb minus facet, dB | spread between seeds, dB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| popoto_12km, image-source paths (6 pings, 6 taps) | 5 | 17.7 dB; 6/6, track 3 | 17.6 dB (17.5, 17.7); 6/6, track 3; 6/6, track 3 | 17.8 dB (17.8, 17.7); 6/6, track 3; 6/6, track 3 | 17.5 dB; 6/6, track 3 | +0.2 | 0.1 |
| popoto_12km, Bellhop (3 pings, 12 strongest arrivals in every variant, 3 taps) | 5 | 20.2 dB; 3/3, track 3 | 19.7 dB (19.7, 19.8); 3/3, track 3; 3/3, track 3 | 19.4 dB (19.5, 19.4); 3/3, track 3; 3/3, track 3 | - | -0.3 | 0.1 |
| popoto_env_gravel_30m, Bellhop (3 pings, 12 strongest arrivals in every variant, 3 taps) | 5 | 34.3 dB; 3/3, track 3 | 35.3 dB (35.7, 34.9); 3/3, track 3; 3/3, track 3 | 34.9 dB (34.5, 35.4); 3/3, track 3; 3/3, track 3 | - | -0.4 | 0.8 |
| popoto_env_summer_mud_60m, Bellhop (3 pings, 12 strongest arrivals in every variant, 3 taps) | 5 | not detected; 0/3, track none | not detected (-, -); 0/3, track none; 0/3, track none | not detected (-, -); 0/3, track none; 0/3, track none | - | - | - |
| popoto_env_winter_sand_200m, Bellhop (3 pings, 12 strongest arrivals in every variant, 3 taps) | 5 | 30.4 dB; 3/3, track 3 | 30.7 dB (30.6, 30.8); 3/3, track 3; 3/3, track 3 | 29.8 dB (29.7, 29.9); 3/3, track 3; 3/3, track 3 | - | -0.9 | 0.2 |

Without noise, reverberation or a detector: one point target, image-source paths with two bounces, one hydrophone after pulse compression. The strongest sample of the echo, re a flat surface:

| Geometry | wind, m/s | pings | facet: mean (median), ping-to-ping sd, dB | rule of thumb | rule of thumb minus facet, dB | its standard error, dB | echo energy, facet / rule of thumb, dB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flank_sst_shallow geometry (1 km) | 6 | 24 | +0.15 (-0.02), 1.02 | +0.07 (-0.60), 1.04 | -0.08 | 0.30 | +0.72 / +0.65 |
| popoto_12km geometry (12 km) | 5 | 24 | -0.04 (-0.04), 0.01 | -0.04 (-0.04), 0.00 | -0.00 | 0.00 | -0.00 / -0.01 |
