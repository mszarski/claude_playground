# A trial in fjord-type water against the open-water rehearsal

Made by `tools/fjord_rehearsal.py` at commit f2bf420: each variant simulated (C++ synthesis, Bellhop for the fjord's water) and passed through the C++ chain of the build's deployment file as shipped. Four pings each. SNR: the target's detection over the chain's local background, median of the pings it was detected on (the strongest from 30 m before to 60 m behind it). Other detections per ping: behind the target on its bearing (within 3 degrees, 60 to 400 m later) / anywhere else. The fjord's profile, section and seabed are ILLUSTRATIVE (modelled on one site); the late tail's numbers are one fjord, one night, 4-8 kHz; the own-ping level is a what-if that nothing measured.

## Test AUV, one row of 32 (`popoto_testauv_trial_1row` against `fjord_testauv_trial_1row`)

Target of 10 dB at 500 m, 15 m deep; array at 10 m.

| Variant | detected, pings | SNR, dB | range error, m | other detections per ping: behind / elsewhere | strongest behind / elsewhere, dB | track on the target confirmed at ping | tracks not on the target (last ping) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| open water | 4/4 | 47.5 | +0.7 | 0.0 / 0.0 | - / - | 3 | 0 |
| profile, section | 4/4 | 39.4 | +19.4 | 2.5 / 1.8 | 32.8 / 21.8 | 3 | 2 |
| + late tail | 4/4 | 26.3 | +20.0 | 3.8 / 20.2 | 20.3 / 13.7 | 3 | 16 |
| + own ping, what-if | 0/4 | not detected | - | 0.0 / 53.8 | - / 14.4 | none | 46 |
| fjord as shipped | 4/4 | 26.5 | +20.3 | 3.0 / 46.5 | 16.1 / 15.1 | 3 | 32 |
| + machinery noise | 4/4 | 24.0 | +20.3 | 3.5 / 0.8 | 18.5 / 17.2 | 3 | 1 |
| open water @ 300 m | 4/4 | 55.8 | +0.9 | 0.0 / 2.0 | - / 15.4 | 3 | 0 |
| profile, section @ 300 m | 4/4 | 52.8 | +5.8 | 1.5 / 3.2 | 34.5 / 27.4 | 3 | 2 |
| + late tail @ 300 m | 4/4 | 29.9 | +1.8 | 5.5 / 31.8 | 16.2 / 13.6 | 3 | 21 |
| + own ping, what-if @ 300 m | 0/4 | not detected | - | 0.0 / 56.2 | - / 14.5 | none | 54 |
| open water @ 1000 m | 4/4 | 32.3 | +0.5 | 0.0 / 0.0 | - / - | 3 | 0 |
| profile, section @ 1000 m | 4/4 | 31.0 | +20.6 | 0.0 / 0.0 | - / - | 3 | 0 |
| + late tail @ 1000 m | 4/4 | 22.5 | +19.3 | 6.0 / 0.2 | 14.9 / 10.7 | 3 | 5 |
| + own ping, what-if @ 1000 m | 0/4 | not detected | - | 0.0 / 53.2 | - / 14.4 | none | 51 |
| open water @ 2000 m | 4/4 | 16.3 | -0.1 | 0.0 / 0.0 | - / - | 3 | 0 |
| profile, section @ 2000 m | 4/4 | 28.6 | +4.9 | 0.0 / 0.0 | - / - | 3 | 0 |
| + late tail @ 2000 m | 4/4 | 24.4 | +5.0 | 0.5 / 0.0 | 11.7 / - | 3 | 0 |
| + own ping, what-if @ 2000 m | 4/4 | 24.2 | +5.0 | 0.5 / 0.0 | 11.4 / - | 3 | 0 |

The same records through the chain with other settings:

| Variant | chain setting | detected, pings | SNR, dB | other detections per ping: behind / elsewhere | track on the target confirmed at ping | tracks not on the target (last ping) |
| --- | --- | --- | --- | --- | --- | --- |
| + own ping, what-if | min_snr_db 15 | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if | cfar guard 30 m, train 100 m | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if | min_snr_db 15, cfar guard 30 m, train 100 m | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if | min_snr_db 15, confirm on 4 of 4 | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| fjord as shipped | min_snr_db 15 | 4/4 | 26.5 | 0.8 / 0.5 | 3 | 0 |
| fjord as shipped | cfar guard 30 m, train 100 m | 4/4 | 20.7 | 0.5 / 1.0 | 3 | 0 |
| fjord as shipped | min_snr_db 15, cfar guard 30 m, train 100 m | 4/4 | 20.7 | 0.0 / 0.0 | 3 | 0 |
| fjord as shipped | min_snr_db 15, confirm on 4 of 4 | 4/4 | 26.5 | 0.8 / 0.5 | 4 | 0 |
| + own ping, what-if @ 300 m | min_snr_db 15 | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if @ 300 m | cfar guard 30 m, train 100 m | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if @ 300 m | min_snr_db 15, cfar guard 30 m, train 100 m | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if @ 300 m | min_snr_db 15, confirm on 4 of 4 | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if @ 1000 m | min_snr_db 15 | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if @ 1000 m | cfar guard 30 m, train 100 m | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if @ 1000 m | min_snr_db 15, cfar guard 30 m, train 100 m | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if @ 1000 m | min_snr_db 15, confirm on 4 of 4 | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if @ 2000 m | min_snr_db 15 | 4/4 | 24.2 | 0.0 / 0.0 | 3 | 0 |
| + own ping, what-if @ 2000 m | cfar guard 30 m, train 100 m | 4/4 | 19.4 | 0.0 / 0.0 | 3 | 0 |
| + own ping, what-if @ 2000 m | min_snr_db 15, cfar guard 30 m, train 100 m | 4/4 | 19.4 | 0.0 / 0.0 | 3 | 0 |
| + own ping, what-if @ 2000 m | min_snr_db 15, confirm on 4 of 4 | 4/4 | 24.2 | 0.0 / 0.0 | 4 | 0 |

## Static array, two rows (`popoto_static_trial_2row` against `fjord_static_trial_2row`)

Target of 15 dB at 2600 m, 60 m deep; array at 30 m.

| Variant | detected, pings | SNR, dB | range error, m | other detections per ping: behind / elsewhere | strongest behind / elsewhere, dB | track on the target confirmed at ping | tracks not on the target (last ping) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| open water | 4/4 | 32.3 | +1.0 | 0.0 / 0.0 | - / - | 3 | 0 |
| profile, section | 4/4 | 34.9 | +40.9 | 1.0 / 0.0 | 24.9 / - | 3 | 1 |
| + late tail | 4/4 | 15.0 | +28.0 | 1.0 / 2.8 | 34.9 / 13.5 | 3 | 1 |
| + own ping, what-if | 4/4 | 15.0 | +28.0 | 1.0 / 3.2 | 34.8 / 13.4 | 3 | 1 |
| fjord as shipped | 4/4 | 15.1 | +28.0 | 1.0 / 0.2 | 41.3 / 11.4 | 3 | 1 |
| + machinery noise | 4/4 | 18.4 | +35.8 | 1.0 / 0.0 | 23.4 / - | 3 | 1 |

The same records through the chain with other settings:

| Variant | chain setting | detected, pings | SNR, dB | other detections per ping: behind / elsewhere | track on the target confirmed at ping | tracks not on the target (last ping) |
| --- | --- | --- | --- | --- | --- | --- |
| + own ping, what-if | min_snr_db 15 | 2/4 | 15.4 | 1.0 / 0.0 | none | 1 |
| + own ping, what-if | cfar guard 30 m, train 100 m | 4/4 | 12.3 | 0.2 / 0.0 | 3 | 0 |
| + own ping, what-if | min_snr_db 15, cfar guard 30 m, train 100 m | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if | min_snr_db 15, confirm on 4 of 4 | 2/4 | 15.4 | 1.0 / 0.0 | none | 1 |
| fjord as shipped | min_snr_db 15 | 2/4 | 15.4 | 1.0 / 0.0 | none | 1 |
| fjord as shipped | cfar guard 30 m, train 100 m | 4/4 | 12.4 | 0.0 / 0.0 | 3 | 0 |
| fjord as shipped | min_snr_db 15, cfar guard 30 m, train 100 m | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| fjord as shipped | min_snr_db 15, confirm on 4 of 4 | 2/4 | 15.4 | 1.0 / 0.0 | none | 1 |

Elevation and depth of the target's detection (median of the pings; in brackets every ping):

| Variant | sound speed known to the chain | elevation, degrees (true) | pings with several candidates | depth, m (true) | stated one sigma, m | profile rung used |
| --- | --- | --- | --- | --- | --- | --- |
| open water | as deployed | -0.57 (-0.66) [-0.58, -0.49, -0.56, -0.62] | 0/4 | 17.4 (60.0) [17.8, 14.0, 17.0, 19.7] | 121.5, 121.5, 121.5, 121.5 | None |
| profile, section | as deployed | +1.59 (-0.66) [+1.84, +1.56, +1.54, +1.62] | 0/4 | 47.3 (60.0) [42.8, 47.5, 47.1, 49.3] | 124.1, 123.7, 123.7, 123.7 | None |
| profile, section | profile given | +1.59 (-0.66) [+1.84, +1.56, +1.54, +1.62] | 0/4 | 23.3 (60.0) [20.4, 23.7, 23.9, 22.9] | 124.1, 123.7, 123.7, 123.7 | None |
| + late tail | as deployed | +0.89 (-0.66) [-1.34, +2.40, +2.40, -0.61] | 0/4 | 18.3 (60.0) [24.7, 17.9, 17.6, 18.7] | 125.4, 124.7, 124.7, 125.3 | None |
| + late tail | profile given | +0.89 (-0.66) [-1.34, +2.40, +2.40, -0.61] | 0/4 | 24.3 (60.0) [39.5, 10.6, 10.0, 38.0] | 125.4, 124.7, 124.7, 125.3 | None |
| + own ping, what-if | as deployed | +0.89 (-0.66) [-1.34, +2.40, +2.40, -0.61] | 0/4 | 18.2 (60.0) [24.7, 17.8, 17.6, 18.6] | 125.4, 124.7, 124.7, 125.3 | None |
| + own ping, what-if | profile given | +0.89 (-0.66) [-1.34, +2.40, +2.40, -0.61] | 0/4 | 24.2 (60.0) [39.5, 10.5, 10.0, 38.0] | 125.4, 124.7, 124.7, 125.3 | None |
| fjord as shipped | as deployed | +0.88 (-0.66) [-1.37, +2.40, +2.40, -0.64] | 0/4 | 18.7 (60.0) [21.8, 17.7, 17.6, 19.8] | 125.4, 124.7, 124.7, 125.3 | None |
| fjord as shipped | profile given | +0.88 (-0.66) [-1.37, +2.40, +2.40, -0.64] | 0/4 | 24.1 (60.0) [39.6, 10.2, 10.0, 38.0] | 125.4, 124.7, 124.7, 125.3 | None |
| + machinery noise | as deployed | -0.89 (-0.66) [-1.36, +2.40, -1.77, -0.42] | 0/4 | 14.3 (60.0) [22.0, 17.3, 11.2, 9.8] | 125.7, 125.6, 125.4, 125.0 | None |
| + machinery noise | profile given | -0.89 (-0.66) [-1.36, +2.40, -1.77, -0.42] | 0/4 | 38.7 (60.0) [39.2, 9.7, 43.0, 38.2] | 125.7, 125.6, 125.4, 125.0 | None |

## Static array, three rows (`popoto_static_trial_3row` against `fjord_static_trial_3row`)

Target of 10 dB at 150 m, 18 m deep; array at 10 m.

| Variant | detected, pings | SNR, dB | range error, m | other detections per ping: behind / elsewhere | strongest behind / elsewhere, dB | track on the target confirmed at ping | tracks not on the target (last ping) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| open water | 4/4 | 68.6 | +1.7 | 0.0 / 1.5 | - / 15.5 | 3 | 0 |
| profile, section | 4/4 | 64.0 | +7.0 | 0.0 / 7.0 | - / 22.0 | 3 | 1 |
| + late tail | 4/4 | 24.6 | +1.3 | 0.2 / 3.0 | 11.1 / 12.9 | 3 | 1 |
| + own ping, what-if | 4/4 | 14.0 | +1.1 | 0.0 / 0.0 | - / - | 3 | 0 |
| fjord as shipped | 4/4 | 24.6 | +1.3 | 0.2 / 3.0 | 11.1 / 12.9 | 3 | 1 |
| + machinery noise | 4/4 | 24.7 | +1.3 | 0.5 / 1.0 | 18.0 / 13.0 | 3 | 0 |

The same records through the chain with other settings:

| Variant | chain setting | detected, pings | SNR, dB | other detections per ping: behind / elsewhere | track on the target confirmed at ping | tracks not on the target (last ping) |
| --- | --- | --- | --- | --- | --- | --- |
| + own ping, what-if | min_snr_db 15 | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if | cfar guard 30 m, train 100 m | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if | min_snr_db 15, cfar guard 30 m, train 100 m | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| + own ping, what-if | min_snr_db 15, confirm on 4 of 4 | 0/4 | not detected | 0.0 / 0.0 | none | 0 |
| fjord as shipped | min_snr_db 15 | 4/4 | 24.6 | 0.0 / 0.0 | 3 | 0 |
| fjord as shipped | cfar guard 30 m, train 100 m | 4/4 | 21.9 | 0.8 / 0.0 | 3 | 0 |
| fjord as shipped | min_snr_db 15, cfar guard 30 m, train 100 m | 4/4 | 21.9 | 0.2 / 0.0 | 3 | 0 |
| fjord as shipped | min_snr_db 15, confirm on 4 of 4 | 4/4 | 24.6 | 0.0 / 0.0 | 4 | 0 |

Elevation and depth of the target's detection (median of the pings; in brackets every ping):

| Variant | sound speed known to the chain | elevation, degrees (true) | pings with several candidates | depth, m (true) | stated one sigma, m | profile rung used |
| --- | --- | --- | --- | --- | --- | --- |
| open water | as deployed | -2.78 (-3.03) [-2.78, -2.78, -2.78, -2.78] | 0/4 | 17.2 (18.0) [17.2, 17.2, 17.2, 17.2] | 2.6, 2.6, 2.6, 2.6 | None |
| profile, section | as deployed | -4.54 (-3.03) [-4.54, -4.54, -4.54, -4.54] | 0/4 | 21.8 (18.0) [21.8, 21.8, 21.8, 21.8] | 2.6, 2.6, 2.6, 2.6 | None |
| profile, section | profile given | -4.54 (-3.03) [-4.54, -4.54, -4.54, -4.54] | 0/4 | 18.6 (18.0) [18.6, 18.6, 18.6, 18.6] | 2.6, 2.6, 2.6, 2.6 | None |
| + late tail | as deployed | -4.34 (-3.03) [-4.35, -4.34, -4.35, -4.34] | 0/4 | 21.3 (18.0) [21.3, 21.3, 21.3, 21.3] | 2.6, 2.6, 2.6, 2.6 | None |
| + late tail | profile given | -4.34 (-3.03) [-4.35, -4.34, -4.35, -4.34] | 0/4 | 18.0 (18.0) [18.0, 18.0, 18.0, 18.0] | 2.6, 2.6, 2.6, 2.6 | None |
| + own ping, what-if | as deployed | -3.98 (-3.03) [-4.14, -3.82, -4.23, -3.69] | 0/4 | 20.4 (18.0) [20.8, 19.9, 21.0, 19.6] | 2.7, 2.7, 2.7, 2.7 | None |
| + own ping, what-if | profile given | -3.98 (-3.03) [-4.14, -3.82, -4.23, -3.69] | 0/4 | 16.8 (18.0) [17.4, 16.3, 17.7, 15.9] | 2.7, 2.7, 2.7, 2.7 | None |
| fjord as shipped | as deployed | -4.34 (-3.03) [-4.35, -4.34, -4.34, -4.34] | 0/4 | 21.3 (18.0) [21.3, 21.3, 21.3, 21.3] | 2.6, 2.6, 2.6, 2.6 | None |
| fjord as shipped | profile given | -4.34 (-3.03) [-4.35, -4.34, -4.34, -4.34] | 0/4 | 18.0 (18.0) [18.0, 18.0, 18.0, 18.0] | 2.6, 2.6, 2.6, 2.6 | None |
| + machinery noise | as deployed | -4.24 (-3.03) [-4.84, -4.28, -4.20, -4.01] | 0/4 | 21.0 (18.0) [22.6, 21.1, 20.9, 20.4] | 2.6, 2.6, 2.6, 2.6 | None |
| + machinery noise | profile given | -4.24 (-3.03) [-4.84, -4.28, -4.20, -4.01] | 0/4 | 17.7 (18.0) [19.6, 17.8, 17.6, 16.9] | 2.6, 2.6, 2.6, 2.6 | None |

