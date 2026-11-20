# A trial in fjord-type water against the open-water rehearsal

Made by `tools/fjord_rehearsal.py` at commit e10b17f: each variant simulated (C++ synthesis, Bellhop for the fjord's water) and passed through the C++ chain of the build's deployment file as shipped. Four pings each. SNR: the target's detection over the chain's local background, median of the pings it was detected on (the strongest whose ranges, from its first arrival to the end of what it spans, reach from 30 m before the target to 60 m behind it: the detector reports the echo and the tail behind it as one detection at their energy-weighted range). Range error: that range / its first arrival, against the target's. Other detections per ping: behind the target on its bearing (within 3 degrees, up to 400 m later) / anywhere else. The fjord's profile, section and seabed are ILLUSTRATIVE (modelled on one site); the late tail's numbers are one fjord, one night, 4-8 kHz; the own-ping level is a what-if that nothing measured.

## Test AUV, one row of 32 (`popoto_testauv_trial_1row` against `fjord_testauv_trial_1row`)

Target of 10 dB at 500 m, 15 m deep; array at 10 m.

| Variant | detected, pings | SNR, dB | range error, m: detection / its first arrival | other detections per ping: behind / elsewhere | strongest behind / elsewhere, dB | track on the target confirmed at ping | tracks not on the target (last ping) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| open water | 4/4 | 58.0 | +10.2 / -0.4 | 0.0 / 0.0 | - / - | 3 | 0 |
| profile, section | 4/4 | 49.5 | +24.0 / -1.7 | 0.0 / 0.0 | - / - | 3 | 0 |
| + late tail | 4/4 | 33.5 | +31.3 / -0.7 | 0.0 / 5.5 | - / 17.1 | 3 | 0 |
| + own ping, what-if | 0/4 | not detected | - / - | 0.0 / 0.0 | - / - | none | 0 |
| fjord as shipped | 4/4 | 33.8 | +29.3 / -0.7 | 0.0 / 2.8 | - / 17.1 | 3 | 0 |
| + machinery noise | 4/4 | 32.2 | +30.5 / -0.7 | 0.0 / 7.2 | - / 20.7 | 3 | 0 |
| open water @ 300 m | 4/4 | 63.0 | +8.0 / -1.1 | 0.0 / 0.0 | - / - | 3 | 0 |
| profile, section @ 300 m | 4/4 | 61.5 | +6.9 / -0.2 | 0.0 / 0.0 | - / - | 3 | 0 |
| + late tail @ 300 m | 4/4 | 46.5 | +14.5 / -0.2 | 0.0 / 2.2 | - / 22.5 | 3 | 0 |
| + own ping, what-if @ 300 m | 0/4 | not detected | - / - | 0.0 / 0.0 | - / - | none | 0 |
| open water @ 1000 m | 4/4 | 47.5 | +14.2 / -1.1 | 0.0 / 0.0 | - / - | 3 | 0 |
| profile, section @ 1000 m | 4/4 | 39.5 | +20.7 / +0.6 | 0.0 / 0.0 | - / - | 3 | 0 |
| + late tail @ 1000 m | 4/4 | 28.4 | +27.3 / +0.6 | 0.2 / 5.2 | 13.5 / 16.7 | 3 | 1 |
| + own ping, what-if @ 1000 m | 0/4 | not detected | - / - | 0.0 / 0.0 | - / - | none | 0 |
| open water @ 2000 m | 4/4 | 38.5 | +11.2 / -0.7 | 0.0 / 0.0 | - / - | 3 | 0 |
| profile, section @ 2000 m | 4/4 | 37.6 | +18.2 / +2.9 | 0.0 / 0.0 | - / - | 3 | 0 |
| + late tail @ 2000 m | 4/4 | 32.4 | +24.1 / +2.9 | 0.0 / 5.5 | - / 13.5 | 3 | 0 |
| + own ping, what-if @ 2000 m | 4/4 | 32.0 | +23.8 / +2.9 | 0.0 / 3.0 | - / 12.4 | 3 | 0 |

The same records through the chain with other settings:

| Variant | chain setting | detected, pings | SNR, dB | other detections per ping: behind / elsewhere | track on the target confirmed at ping | tracks not on the target (last ping) |
| --- | --- | --- | --- | --- | --- | --- |
| + own ping, what-if | min_snr_db 15 | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if | cfar guard 30 m, train 100 m | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if | min_snr_db 15, cfar guard 30 m, train 100 m | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if | min_snr_db 15, confirm on 4 of 4 | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| fjord as shipped | min_snr_db 15 | 4/4 | 33.8 | 0.0 / 1.5 | 3 | 0 |
| fjord as shipped | cfar guard 30 m, train 100 m | 4/4 | 27.5 | 0.0 / 6.0 | 3 | 2 |
| fjord as shipped | min_snr_db 15, cfar guard 30 m, train 100 m | 4/4 | 27.5 | 0.0 / 0.0 | 3 | 0 |
| fjord as shipped | min_snr_db 15, confirm on 4 of 4 | 4/4 | 33.8 | 0.0 / 1.5 | 4 | 0 |
| + own ping, what-if @ 300 m | min_snr_db 15 | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if @ 300 m | cfar guard 30 m, train 100 m | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if @ 300 m | min_snr_db 15, cfar guard 30 m, train 100 m | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if @ 300 m | min_snr_db 15, confirm on 4 of 4 | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if @ 1000 m | min_snr_db 15 | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if @ 1000 m | cfar guard 30 m, train 100 m | 3/4 | 13.7 | 0.0 / 0.0 | 4 | 0 |
| + own ping, what-if @ 1000 m | min_snr_db 15, cfar guard 30 m, train 100 m | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if @ 1000 m | min_snr_db 15, confirm on 4 of 4 | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if @ 2000 m | min_snr_db 15 | 4/4 | 32.0 | 0.0 / 0.0 | 3 | 0 |
| + own ping, what-if @ 2000 m | cfar guard 30 m, train 100 m | 4/4 | 26.4 | 0.0 / 0.5 | 3 | 0 |
| + own ping, what-if @ 2000 m | min_snr_db 15, cfar guard 30 m, train 100 m | 4/4 | 26.4 | 0.0 / 0.0 | 3 | 0 |
| + own ping, what-if @ 2000 m | min_snr_db 15, confirm on 4 of 4 | 4/4 | 32.0 | 0.0 / 0.0 | 4 | 0 |

## Static array, two rows (`popoto_static_trial_2row` against `fjord_static_trial_2row`)

Target of 15 dB at 2600 m, 60 m deep; array at 30 m.

| Variant | detected, pings | SNR, dB | range error, m: detection / its first arrival | other detections per ping: behind / elsewhere | strongest behind / elsewhere, dB | track on the target confirmed at ping | tracks not on the target (last ping) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| open water | 4/4 | 38.9 | +36.2 / +0.3 | 0.0 / 0.0 | - / - | 3 | 0 |
| profile, section | 4/4 | 43.9 | +41.5 / +6.3 | 0.0 / 0.0 | - / - | 3 | 0 |
| + late tail | 4/4 | 43.6 | +63.6 / +8.3 | 0.0 / 1.8 | - / 26.3 | 3 | 0 |
| + own ping, what-if | 4/4 | 43.5 | +63.6 / +8.3 | 0.0 / 1.8 | - / 26.0 | 3 | 0 |
| fjord as shipped | 4/4 | 50.0 | +63.1 / +8.3 | 0.0 / 1.5 | - / 32.5 | 3 | 0 |
| + machinery noise | 4/4 | 32.1 | +60.9 / +8.3 | 0.0 / 2.0 | - / 16.6 | 3 | 0 |

The same records through the chain with other settings:

| Variant | chain setting | detected, pings | SNR, dB | other detections per ping: behind / elsewhere | track on the target confirmed at ping | tracks not on the target (last ping) |
| --- | --- | --- | --- | --- | --- | --- |
| + own ping, what-if | min_snr_db 15 | 4/4 | 43.5 | 0.0 / 1.5 | 3 | 0 |
| + own ping, what-if | cfar guard 30 m, train 100 m | 4/4 | 15.2 | 0.0 / 0.0 | 3 | 0 |
| + own ping, what-if | min_snr_db 15, cfar guard 30 m, train 100 m | 4/4 | 15.2 | 0.0 / 0.0 | 3 | 0 |
| + own ping, what-if | min_snr_db 15, confirm on 4 of 4 | 4/4 | 43.5 | 0.0 / 1.5 | 4 | 0 |
| fjord as shipped | min_snr_db 15 | 4/4 | 50.0 | 0.0 / 1.5 | 3 | 0 |
| fjord as shipped | cfar guard 30 m, train 100 m | 4/4 | 15.2 | 0.0 / 0.5 | 3 | 0 |
| fjord as shipped | min_snr_db 15, cfar guard 30 m, train 100 m | 4/4 | 15.2 | 0.0 / 0.0 | 3 | 0 |
| fjord as shipped | min_snr_db 15, confirm on 4 of 4 | 4/4 | 50.0 | 0.0 / 1.5 | 4 | 0 |

Elevation and depth of the target's detection (median of the pings; in brackets every ping):

| Variant | sound speed known to the chain | elevation, degrees (true) | pings with several candidates | depth, m (true) | stated one sigma, m | profile rung used |
| --- | --- | --- | --- | --- | --- | --- |
| open water | as deployed | -3.39 (-0.66) [-3.42, -3.44, -3.35, -3.36] | 0/4 | 36.2 (60.0) [34.4, 33.3, 38.6, 38.0] | 123.3, 123.3, 123.3, 123.3 | None |
| profile, section | as deployed | +1.77 (-0.66) [+1.73, +1.77, +1.76, +1.79] | 0/4 | 45.7 (60.0) [47.0, 45.5, 46.0, 44.8] | 124.1, 124.5, 124.5, 124.4 | None |
| profile, section | profile given | +1.77 (-0.66) [+1.73, +1.77, +1.76, +1.79] | 0/4 | 21.1 (60.0) [21.5, 21.0, 21.2, 20.9] | 123.8, 124.2, 124.2, 124.2 | None |
| + late tail | as deployed | -2.38 (-0.66) [-2.38, -1.30, -2.38] | 2/4 | 41.0 (60.0) [41.1, 30.3, 41.0] | 127.2, 125.7, -, 127.2 | None |
| + late tail | profile given | -2.38 (-0.66) [-2.38, -1.30, -2.38] | 2/4 | 39.8 (60.0) [39.8, 39.0, 39.9] | 126.0, 125.5, -, 126.0 | None |
| + own ping, what-if | as deployed | -2.38 (-0.66) [-2.38, -1.29, -2.38] | 2/4 | 41.1 (60.0) [41.1, 31.2, 41.1] | 127.2, 125.7, -, 127.2 | None |
| + own ping, what-if | profile given | -2.38 (-0.66) [-2.38, -1.29, -2.38] | 2/4 | 39.8 (60.0) [39.8, 39.0, 39.8] | 126.0, 125.5, -, 126.0 | None |
| fjord as shipped | as deployed | -2.38 (-0.66) [-2.38, +1.60, -2.38] | 2/4 | 41.2 (60.0) [41.2, 49.2, 41.2] | 127.2, 125.4, -, 127.2 | None |
| fjord as shipped | profile given | -2.38 (-0.66) [-2.38, +1.60, -2.38] | 2/4 | 39.6 (60.0) [39.6, 23.1, 39.6] | 126.0, 125.3, -, 126.0 | None |
| + machinery noise | as deployed | +0.01 (-0.66) [-2.38, +2.40, +2.40, -2.38] | 2/4 | 29.4 (60.0) [41.3, 17.4, 17.5, 41.2] | 127.2, 125.5, 125.1, 127.2 | None |
| + machinery noise | profile given | +0.01 (-0.66) [-2.38, +2.40, +2.40, -2.38] | 2/4 | 24.8 (60.0) [39.5, 9.8, 10.0, 39.6] | 126.0, 126.2, 125.8, 126.0 | None |

## Static array, three rows (`popoto_static_trial_3row` against `fjord_static_trial_3row`)

Target of 10 dB at 150 m, 18 m deep; array at 10 m.

| Variant | detected, pings | SNR, dB | range error, m: detection / its first arrival | other detections per ping: behind / elsewhere | strongest behind / elsewhere, dB | track on the target confirmed at ping | tracks not on the target (last ping) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| open water | 4/4 | 76.8 | +5.7 / -1.4 | 0.0 / 0.0 | - / - | 3 | 0 |
| profile, section | 4/4 | 71.6 | +6.5 / -1.4 | 0.0 / 0.0 | - / - | 3 | 0 |
| + late tail | 4/4 | 29.1 | +14.0 / -1.4 | 0.0 / 4.0 | - / 15.2 | 3 | 0 |
| + own ping, what-if | 0/4 | not detected | - / - | 0.0 / 2.2 | - / 18.4 | 4 | 0 |
| fjord as shipped | 4/4 | 29.1 | +14.0 / -1.4 | 0.0 / 4.0 | - / 15.1 | 3 | 0 |
| + machinery noise | 4/4 | 29.2 | +14.0 / -1.4 | 0.0 / 3.8 | - / 22.3 | 3 | 0 |

The same records through the chain with other settings:

| Variant | chain setting | detected, pings | SNR, dB | other detections per ping: behind / elsewhere | track on the target confirmed at ping | tracks not on the target (last ping) |
| --- | --- | --- | --- | --- | --- | --- |
| + own ping, what-if | min_snr_db 15 | 0/4 | not detected | 0.0 / 1.0 | 4 | 0 |
| + own ping, what-if | cfar guard 30 m, train 100 m | 4/4 | 10.6 | 0.0 / 0.0 | 3 | 0 |
| + own ping, what-if | min_snr_db 15, cfar guard 30 m, train 100 m | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if | min_snr_db 15, confirm on 4 of 4 | 0/4 | not detected | 0.0 / 1.0 | none | 0 |
| fjord as shipped | min_snr_db 15 | 4/4 | 29.1 | 0.0 / 0.2 | 3 | 0 |
| fjord as shipped | cfar guard 30 m, train 100 m | 4/4 | 36.4 | 0.0 / 1.8 | 3 | 0 |
| fjord as shipped | min_snr_db 15, cfar guard 30 m, train 100 m | 4/4 | 36.4 | 0.0 / 1.8 | 3 | 0 |
| fjord as shipped | min_snr_db 15, confirm on 4 of 4 | 4/4 | 29.1 | 0.0 / 0.2 | 4 | 0 |

Elevation and depth of the target's detection (median of the pings; in brackets every ping):

| Variant | sound speed known to the chain | elevation, degrees (true) | pings with several candidates | depth, m (true) | stated one sigma, m | profile rung used |
| --- | --- | --- | --- | --- | --- | --- |
| open water | as deployed | -2.66 (-3.03) [-2.66, -2.66, -2.66, -2.66] | 0/4 | 16.9 (18.0) [16.9, 16.9, 16.9, 16.9] | 2.6, 2.6, 2.6, 2.6 | None |
| profile, section | as deployed | -4.38 (-3.03) [-4.37, -4.37, -4.38, -4.38] | 0/4 | 21.4 (18.0) [21.4, 21.4, 21.4, 21.4] | 2.7, 2.7, 2.7, 2.7 | None |
| profile, section | profile given | -4.38 (-3.03) [-4.37, -4.37, -4.38, -4.38] | 0/4 | 18.1 (18.0) [18.1, 18.1, 18.1, 18.1] | 2.6, 2.6, 2.6, 2.6 | None |
| + late tail | as deployed | -4.47 (-3.03) [-4.49, -4.51, -4.41, -4.44] | 0/4 | 21.6 (18.0) [21.7, 21.7, 21.5, 21.6] | 2.7, 2.7, 2.7, 2.7 | None |
| + late tail | profile given | -4.47 (-3.03) [-4.49, -4.51, -4.41, -4.44] | 0/4 | 18.4 (18.0) [18.5, 18.5, 18.2, 18.3] | 2.6, 2.6, 2.6, 2.6 | None |
| fjord as shipped | as deployed | -4.46 (-3.03) [-4.46, -4.51, -4.41, -4.45] | 0/4 | 21.6 (18.0) [21.6, 21.7, 21.5, 21.6] | 2.7, 2.7, 2.7, 2.7 | None |
| fjord as shipped | profile given | -4.46 (-3.03) [-4.46, -4.51, -4.41, -4.45] | 0/4 | 18.4 (18.0) [18.4, 18.5, 18.2, 18.4] | 2.6, 2.6, 2.6, 2.6 | None |
| + machinery noise | as deployed | -4.39 (-3.03) [-4.93, -4.50, -4.28, -4.04] | 0/4 | 21.4 (18.0) [22.8, 21.7, 21.1, 20.5] | 2.7, 2.7, 2.7, 2.7 | None |
| + machinery noise | profile given | -4.39 (-3.03) [-4.93, -4.50, -4.28, -4.04] | 0/4 | 18.2 (18.0) [19.8, 18.5, 17.8, 17.0] | 2.7, 2.6, 2.6, 2.6 | None |

