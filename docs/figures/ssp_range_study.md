# Water that changes with range against a chain that knows the local profile

Made by `tools/ssp_range_study.py` at commit e10b17f. Each scenario is simulated twice (C++ synthesis; Bellhop's eigenrays): with its water as written (`environment.profiles_by_range`: Bellhop's 2-D sound-speed field) and with the profile at the sonar at every range. Same seed, target and noise. The records go through the C++ chain of the build's deployment file as shipped, and again with the local profile given to the elevation stage (stacked rows) and depth from multipath timing switched on as a generated deployment file holds it (`rays: true`, `withhold_flagged: true`: a depth with a note is not reported, its note is the reason). The chain is never told that the water changes. Four pings each: a look at what happens, not statistics. The profiles are ILLUSTRATIVE: nobody measured them. Range is horizontal distance from the sonar and the section is the same on every bearing.

Numbers are the median of the pings the target was detected on; in square brackets every ping. Range error: the detection's range minus the straight-line distance to the target, for the detection's strongest sample / for its first arrival. Elevation: positive up; `straight` is the straight line to the target, which is not where the sound arrives from in this water (see the eigenrays).

## Static array, two rows: a front 1.2 to 1.4 km out, target at 2.6 km (`ssp_front_static_trial_2row`, chain of `configs/deploy/popoto_static_2row.yaml`)

Target of 15 dB, 60 m deep, 2600 m away horizontally; array at 30 m; the record's sound speed is 1488.75 m/s (the local profile at the array) in both waters.

| Water | Chain | detected, pings | SNR, dB | range error, m: strongest / first arrival | bearing error, degrees [every ping] | track on the target confirmed at ping | tracks elsewhere, last ping: range, m (kind) | other detections per ping |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| changes with range | as deployed | 4/4 | 45.7 | +48.6 / +21.5 | -0.07 [-0.09, -0.07, -0.07, -0.07] | 3 | none | 0.0 |
| changes with range | local profile, multipath depth | 4/4 | 45.7 | +48.6 / +21.5 | -0.07 [-0.09, -0.07, -0.07, -0.07] | 3 | none | 0.0 |
| local everywhere | as deployed | 4/4 | 45.5 | +39.3 / +10.2 | -0.08 [-0.08, -0.08, -0.09, -0.08] | 3 | none | 0.0 |
| local everywhere | local profile, multipath depth | 4/4 | 45.5 | +39.3 / +10.2 | -0.08 [-0.08, -0.08, -0.09, -0.08] | 3 | none | 0.0 |

Depth of the target's detection:

| Water | Chain | elevation, degrees (straight) [every ping] | pings flagged ambiguous | depth from the rows, m (true) [every ping] | its stated one sigma, m | depth from multipath timing, m: [every ping]; track | multipath stage's reason or note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| changes with range | as deployed | -4.22 (-0.66) [-4.36, -4.12, -4.29, -4.15] | 1/4 | 12.8 (60.0) [20.3, 7.4, 16.6, 9.1] | 125.0, 125.0, 125.0, 125.0 | [-, -, -, -]; - | - |
| changes with range | local profile, multipath depth | -4.22 (-0.66) [-4.36, -4.12, -4.29, -4.15] | 1/4 | 83.1 (60.0) [79.8, 85.5, 81.4, 84.8] | 125.0, 125.0, 125.0, 125.0 | [-, -, -, -]; - | merged |
| local everywhere | as deployed | +3.65 (-0.66) [+3.65, +3.65, +3.65, +3.65] | 4/4 | 75.1 (60.0) [75.1, 75.1, 75.1, 75.0] | 124.1, 124.1, 124.1, 124.1 | [-, -, -, -]; - | - |
| local everywhere | local profile, multipath depth | +3.65 (-0.66) [+3.65, +3.65, +3.65, +3.65] | 4/4 | 80.4 (60.0) [80.4, 80.4, 80.4, 80.5] | 124.1, 124.1, 124.1, 124.1 | [-, -, -, -]; - | merged |

The simulated sound: Bellhop's arrivals from the array to the target (one way, first ping), the six strongest of those kept (the scenario's `max_arrivals`). Range excess: the arrival's delay times the record's sound speed, minus the straight-line distance.

| Water | level re the strongest, dB | delay, ms | range excess, m | angle at the array, degrees (up positive) | surface / bottom reflections |
| --- | --- | --- | --- | --- | --- |
| changes with range | 0.0 | 1761.61 | +22.1 | +4.19 | 1 / 1 |
| changes with range | -1.1 | 1761.37 | +21.7 | -3.27 | 1 / 1 |
| changes with range | -4.3 | 1827.34 | +119.9 | +16.48 | 4 / 4 |
| changes with range | -4.7 | 1761.32 | +21.6 | -7.71 | 1 / 1 |
| changes with range | -4.9 | 1785.89 | +58.2 | -10.86 | 2 / 3 |
| changes with range | -5.1 | 1803.75 | +84.8 | -13.35 | 3 / 3 |
| local everywhere | 0.0 | 1754.07 | +10.8 | +2.14 | 0 / 1 |
| local everywhere | -0.8 | 1754.05 | +10.8 | -1.44 | 0 / 1 |
| local everywhere | -4.0 | 1758.82 | +17.9 | -5.19 | 0 / 2 |
| local everywhere | -8.0 | 1775.65 | +43.0 | +11.14 | 3 / 2 |
| local everywhere | -8.1 | 1769.76 | +34.2 | -9.77 | 2 / 2 |
| local everywhere | -8.4 | 1780.10 | +49.6 | -11.19 | 2 / 3 |

## Static array, three rows: the mixed layer deepens from 15 m to 45 m over 3 km, target at 2.6 km (`ssp_thermocline_static_trial_3row`, chain of `configs/deploy/popoto_static_3row.yaml`)

Target of 15 dB, 60 m deep, 2600 m away horizontally; array at 30 m; the record's sound speed is 1488.75 m/s (the local profile at the array) in both waters.

| Water | Chain | detected, pings | SNR, dB | range error, m: strongest / first arrival | bearing error, degrees [every ping] | track on the target confirmed at ping | tracks elsewhere, last ping: range, m (kind) | other detections per ping |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| changes with range | as deployed | 4/4 | 42.6 | +32.9 / +14.2 | -0.08 [-0.08, -0.08, -0.08, -0.09] | 3 | none | 0.0 |
| changes with range | local profile, multipath depth | 4/4 | 42.6 | +32.9 / +14.2 | -0.08 [-0.08, -0.08, -0.08, -0.09] | 3 | none | 0.0 |
| local everywhere | as deployed | 4/4 | 43.7 | +40.9 / +10.2 | -0.06 [-0.06, -0.06, -0.06, -0.06] | 3 | none | 0.0 |
| local everywhere | local profile, multipath depth | 4/4 | 43.7 | +40.9 / +10.2 | -0.06 [-0.06, -0.06, -0.06, -0.06] | 3 | none | 0.0 |

Depth of the target's detection:

| Water | Chain | elevation, degrees (straight) [every ping] | pings flagged ambiguous | depth from the rows, m (true) [every ping] | its stated one sigma, m | depth from multipath timing, m: [every ping]; track | multipath stage's reason or note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| changes with range | as deployed | +1.67 (-0.66) [+1.64, +1.70, +1.96, +1.59] | 0/4 | 50.8 (60.0) [49.8, 51.8, 60.0, 48.3] | 123.5, 123.5, 123.5, 123.5 | [-, -, -, -]; - | - |
| changes with range | local profile, multipath depth | +1.67 (-0.66) [+1.64, +1.70, +1.96, +1.59] | 0/4 | 55.0 (60.0) [54.6, 55.3, 58.1, 54.2] | 123.5, 123.5, 123.5, 123.5 | [-, -, -, -]; - | merged |
| local everywhere | as deployed | -5.36 (-0.66) [-5.30, -5.54, -5.35, -5.37] | 0/4 | 67.7 (60.0) [65.1, 76.0, 67.3, 68.1] | 124.1, 124.1, 124.1, 124.1 | [-, -, -, -]; - | - |
| local everywhere | local profile, multipath depth | -5.36 (-0.66) [-5.30, -5.54, -5.35, -5.37] | 0/4 | 56.9 (60.0) [58.3, 52.4, 57.1, 56.6] | 124.1, 124.1, 124.1, 124.1 | [-, -, -, -]; - | merged |

The simulated sound: Bellhop's arrivals from the array to the target (one way, first ping), the six strongest of those kept (the scenario's `max_arrivals`). Range excess: the arrival's delay times the record's sound speed, minus the straight-line distance.

| Water | level re the strongest, dB | delay, ms | range excess, m | angle at the array, degrees (up positive) | surface / bottom reflections |
| --- | --- | --- | --- | --- | --- |
| changes with range | 0.0 | 1756.83 | +14.8 | +2.96 | 0 / 2 |
| changes with range | -1.6 | 1756.72 | +14.7 | -2.81 | 0 / 2 |
| changes with range | -8.6 | 1803.35 | +84.1 | +15.06 | 4 / 3 |
| changes with range | -8.9 | 1794.25 | +70.6 | -13.81 | 3 / 3 |
| changes with range | -9.1 | 1760.94 | +21.0 | +8.73 | 2 / 2 |
| changes with range | -9.5 | 1830.60 | +124.7 | -17.81 | 4 / 4 |
| local everywhere | 0.0 | 1754.08 | +10.8 | +2.21 | 0 / 1 |
| local everywhere | -0.9 | 1754.05 | +10.7 | -1.50 | 0 / 1 |
| local everywhere | -3.8 | 1758.82 | +17.8 | -5.21 | 0 / 2 |
| local everywhere | -7.9 | 1775.66 | +42.9 | +11.15 | 3 / 2 |
| local everywhere | -8.0 | 1769.75 | +34.1 | -9.78 | 2 / 2 |
| local everywhere | -8.3 | 1780.09 | +49.5 | -11.20 | 2 / 3 |

## Test AUV, one row: a front 200 to 300 m out, target at 500 m (`ssp_front_testauv_trial_1row`, chain of `configs/deploy/popoto_testauv_1row.yaml`)

Target of 10 dB, 15 m deep, 500 m away horizontally; array at 10 m; the record's sound speed is 1497.5 m/s (the local profile at the array) in both waters.

| Water | Chain | detected, pings | SNR, dB | range error, m: strongest / first arrival | bearing error, degrees [every ping] | track on the target confirmed at ping | tracks elsewhere, last ping: range, m (kind) | other detections per ping |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| changes with range | as deployed | 4/4 | 57.1 | +13.5 / +7.0 | -0.18 [-0.20, -0.23, -0.16, -0.17] | 3 | none | 0.0 |
| changes with range | local profile, multipath depth | 4/4 | 57.1 | +13.5 / +7.0 | -0.18 [-0.20, -0.23, -0.16, -0.17] | 3 | none | 0.0 |
| local everywhere | as deployed | 4/4 | 51.5 | +16.0 / +7.8 | -0.32 [-0.35, -0.34, -0.30, -0.29] | 3 | none | 0.0 |
| local everywhere | local profile, multipath depth | 4/4 | 51.5 | +16.0 / +7.8 | -0.32 [-0.35, -0.34, -0.30, -0.29] | 3 | none | 0.0 |

Depth of the target's detection:

| Water | Chain | elevation, degrees (straight) [every ping] | pings flagged ambiguous | depth from the rows, m (true) [every ping] | its stated one sigma, m | depth from multipath timing, m: [every ping]; track | multipath stage's reason or note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| changes with range | as deployed | - (-0.57) [-, -, -, -] | 0/4 | - (15.0) [-, -, -, -] | -, -, -, - | [-, -, -, -]; - | - |
| changes with range | local profile, multipath depth | - (-0.57) [-, -, -, -] | 0/4 | - (15.0) [-, -, -, -] | -, -, -, - | [-, -, -, -]; - | unresolvable |
| local everywhere | as deployed | - (-0.57) [-, -, -, -] | 0/4 | - (15.0) [-, -, -, -] | -, -, -, - | [-, -, -, -]; - | - |
| local everywhere | local profile, multipath depth | - (-0.57) [-, -, -, -] | 0/4 | - (15.0) [-, -, -, -] | -, -, -, - | [-, -, -, -]; - | unresolvable |

The simulated sound: Bellhop's arrivals from the array to the target (one way, first ping), the six strongest of those kept (the scenario's `max_arrivals`). Range excess: the arrival's delay times the record's sound speed, minus the straight-line distance.

| Water | level re the strongest, dB | delay, ms | range excess, m | angle at the array, degrees (up positive) | surface / bottom reflections |
| --- | --- | --- | --- | --- | --- |
| changes with range | 0.0 | 339.39 | +8.6 | -5.09 | 1 / 1 |
| changes with range | -3.1 | 344.94 | +16.9 | +12.62 | 2 / 2 |
| changes with range | -3.5 | 340.86 | +10.8 | +8.62 | 2 / 1 |
| changes with range | -5.2 | 342.34 | +13.0 | -10.80 | 1 / 2 |
| changes with range | -7.6 | 337.71 | +6.1 | +6.15 | 1 / 1 |
| changes with range | -7.7 | 346.58 | +19.3 | -13.54 | 2 / 2 |
| local everywhere | 0.0 | 340.64 | +10.5 | -9.98 | 1 / 2 |
| local everywhere | -0.1 | 337.14 | +5.2 | -6.70 | 1 / 1 |
| local everywhere | -0.3 | 343.13 | +14.2 | +12.44 | 2 / 2 |
| local everywhere | -0.5 | 344.69 | +16.5 | -13.54 | 2 / 2 |
| local everywhere | -1.7 | 354.32 | +30.9 | +19.01 | 3 / 3 |
| local everywhere | -1.7 | 350.29 | +24.9 | -16.82 | 2 / 3 |

