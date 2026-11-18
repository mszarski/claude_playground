# All UACR sites against our time spreading

Data: Underwater Acoustic Channel Repository v1.0, doi:10.5281/zenodo.21287414 (CC BY 4.0); Li et al., arXiv:2609.03207.
No site states wind, sea state, a sound-speed profile or a seabed type.

## 1. What each site allows

| Site | Where, when | Carrier, band | Geometry as stated | Estimator | Protocol | Why |
| --- | --- | --- | --- | --- | --- | --- |
| purple | Martha's Vineyard (SPACE'08), Oct 2008 | 12.5 kHz, 9.2-15.8 kHz | moored; source 4 m above seabed, arrays' top element 3.3 m above it, water 15 m; 60 m, 200 m, 1 km | 10 (RLS, stated) | full | geometry stated per file; arrivals resolved at 60 and 200 m; RLS tracker 10.4 Hz |
| green | Norwegian fjord (Norway'24), Nov 2024 | 6 kHz, 4.5 kHz about 6 kHz | moored; source 20 m, hydrophone 43 m, water 60 m, 270 m | 0.05 (LMS, input power assumed) | none | geometry stated, but the estimator (LMS, 3600 taps, step 1/14400) follows only about 0.05 Hz and snapshots are 4.5 Hz apart: every arrival comes out steady. Nothing about spreading can be read from it |
| black | Mariana Trench, Oct 2024 | 18 kHz, 12.5 kHz about 18 kHz | vertical link: source 8718 m deep, 8-element array 6 m under a ship | 2.8 (LMS, input power assumed) | none | the delay window is 7 ms; the surface reflection above the array would come 8 ms after the direct path (2 x 6 m / c). Only the direct path is in the file |
| red | Singapore, Nov 2024 | 25 kHz, 20-30 kHz | drifting boat; source about 6 m, top hydrophone 4.6 m, water 8-20 m and not uniform, range 100-400 m (not per file) | 2 (LMS, input power assumed) | model-free | range and water depth are not given per file: arrivals cannot be assigned without fitting the geometry |
| yellow | Kauai (KAM'11), Jul 2011 | 13 kHz, 6.25 kHz about 13 kHz | moored; source and array at mid depth in 100 m; 3 km and 7 km | 5.3 (LMS, input power assumed) | model-free | geometry stated, but the arrivals are not where straight rays put them (no surface arrival 1.1 ms after the first at 3 km): the water is not isovelocity and no profile is given |
| blue | North Atlantic (MACE'10), Jun 2010 | 13 kHz, 10.5-15.5 kHz | source towed at up to 1.5 m/s, 30-60 m deep, 3-7 km from a moored array at 50 m in 100 m (none of it per file) | 7.8 (RLS, stated) | model-free | range, source depth and speed are not given per file |
| pink | off Kochi, Japan, Jul 2022 | 6 kHz, 4 kHz about 6 kHz | source 176 m, array 146-172 m, about 200 m of water over flat and sloping seabed, 14 km |  | not run | 14 km with no sound-speed profile and no bathymetry; 1.5 GB per file. Not downloaded |
| brown | North Pacific (ATOC), Nov 1994 | 0.075 kHz, 37.5 Hz about 75 Hz | source 652 m, array 900-1600 m, 3250 km |  | not applicable | deep refracted paths that never touch the surface in the way our boundary model describes; nothing to compare |

Fitted per Purple recording (one number each, the wind speed, from the Doppler width of the first surface arrival at 60 m): recording 1: 3.96 m/s, recording 2: 3.70 m/s, recording 3: 9.14 m/s.

## 2. Dependencies across the Purple arrivals

26 surface arrivals that were not used for a fit (grazing 1.3 to 41.5 degrees, one to two surface bounces).

| Ingredient | Measured / predicted | |
| --- | --- | --- |
| Doppler rms width: 2 f v sin(g) sqrt(bounces) / c / sqrt(1 - exp(-Gamma^2)) | median 0.89, scatter x1.13 (n=24) | closed form, fitted wind |
| ... grazing under 12 deg | median 0.88, scatter x1.10 (n=13) | |
| ... grazing over 12 deg | median 0.90, scatter x1.15 (n=11) | |
| ... 1 surface bounce(s) | median 0.91, scatter x1.13 (n=20) | |
| ... 2 surface bounce(s) | median 0.82, scatter x1.06 (n=4) | |
| coherent share exp(-Gamma^2): rms height that the measured share implies / Pierson-Moskowitz height | median 1.40, scatter x1.28 (n=10) | arrivals with a measurable coherent part |
| ... at 200 m | median 1.36, scatter x1.06 (n=6) | |
| ... at 1000 m | median 1.85, scatter x1.32 (n=4) | |
| decay time, facet rule (model A through the estimators) | median 0.99, scatter x5.05 (n=23) | |
| decay time, rule of thumb (model C) | median 0.47, scatter x2.98 (n=19) | |
| ... grazing under 12 deg (Gamma^2 of order 1): facet; rule of thumb | median 0.59, scatter x7.25 (n=10) | median 0.28, scatter x2.29 (n=6) |
| ... grazing over 12 deg (Gamma^2 over 7): facet; rule of thumb | median 1.28, scatter x2.11 (n=13) | median 0.75, scatter x3.00 (n=13) |
| scintillation index, fully rough arrivals with 1 bounce(s) in all | measured median 1.43, range 0.87 to 4.81 (n=4) | model 0.68; K shape for the median: 4.6 |
| scintillation index, fully rough arrivals with 2 bounce(s) in all | measured median 1.31, range 0.83 to 3.07 (n=7) | model 0.61; K shape for the median: 6.5 |
| scintillation index, fully rough arrivals with 3 bounce(s) in all | measured median 1.72, range 0.81 to 3.07 (n=6) | model 0.63; K shape for the median: 2.8 |
| Doppler shape w90/w50 (Gaussian 2.44; Lorentzian about 4 on this axis) | median 2.53, range 1.76 to 3.08 (n=27) | |

## 3. Scaling with frequency inside the band (Purple)

Upper against lower half of the band; a width proportional to frequency gives the frequency ratio.

| File | Range (m) | Arrival | frequency ratio | Doppler width ratio | coherent share, lower | upper |
| --- | --- | --- | --- | --- | --- | --- |
| purple_1 | 60 | S | 1.216 | 1.179 +-0.061 | 0.008 | 0.013 |
| purple_1 | 60 | SB | 1.216 | 1.215 +-0.065 | 0.006 | 0.006 |
| purple_1 | 60 | BS | 1.216 | 1.310 +-0.050 | 0.005 | 0.006 |
| purple_1 | 60 | BSB | 1.216 | 1.358 +-0.071 | 0.003 | 0.010 |
| purple_3 | 200 | S | 1.216 | 1.054 +-0.064 | 0.151 | 0.108 |
| purple_3 | 200 | SB+BS | 1.216 | 1.074 +-0.013 | 0.083 | 0.032 |
| purple_3 | 200 | BSB | 1.216 | 1.191 +-0.039 | 0.020 | 0.014 |
| purple_6 | 60 | S | 1.216 | 1.243 +-0.031 | 0.019 | 0.008 |
| purple_6 | 60 | SB | 1.216 | 1.233 +-0.073 | 0.009 | 0.006 |
| purple_6 | 60 | BS | 1.216 | 1.227 +-0.041 | 0.004 | 0.005 |
| purple_6 | 60 | BSB | 1.216 | 1.267 +-0.047 | 0.005 | 0.004 |
| purple_8 | 200 | S | 1.216 | 1.027 +-0.047 | 0.269 | 0.249 |
| purple_8 | 200 | SB+BS | 1.216 | 1.098 +-0.011 | 0.224 | 0.103 |
| purple_8 | 200 | BSB | 1.216 | 1.139 +-0.079 | 0.032 | 0.021 |
| purple_11 | 60 | S | 1.216 | 1.236 +-0.037 | 0.002 | 0.004 |
| purple_11 | 60 | SB | 1.216 | 1.158 +-0.043 | 0.004 | 0.003 |
| purple_11 | 60 | BS | 1.216 | 1.228 +-0.036 | 0.003 | 0.002 |
| purple_11 | 60 | BSB | 1.216 | 1.174 +-0.061 | 0.002 | 0.002 |
| purple_13 | 200 | S | 1.216 | 1.204 +-0.062 | 0.025 | 0.023 |
| purple_13 | 200 | SB+BS | 1.216 | 1.125 +-0.008 | 0.071 | 0.022 |
| purple_13 | 200 | BSB | 1.216 | 1.151 +-0.064 | 0.007 | 0.017 |

Doppler width ratio over 21 arrivals: mean 1.185, standard deviation 0.080, standard error 0.017; frequency ratio 1.216; no dependence on frequency would give 1.

## 4. Model-free numbers at the other sites

Every arrival within 18 dB of the strongest, up to four receivers; the strongest arrival is the phase reference and is left out.
These do not test a prediction (the geometry is not known); the tracker bandwidth limits the widths that can be seen.

| Site | File | Carrier | Tracker (Hz) | Arrivals | Doppler rms (Hz), median and range | Shape w90/w50, median and range | Scintillation index, median and range | Coherent share, median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| green | green | 6 kHz | 0.05 (LMS, input power assumed) | 28 | 0.00 (0.00-0.00) | 0.00 (0.00-1.22) | 0.00 (0.00-0.00) | 1.013 |
| black | black | 18 kHz | 2.8 (LMS, input power assumed) | 4 | 1.28 (0.91-1.87) | 4.60 (3.25-5.25) | 0.00 (0.00-0.00) | 1.000 |
| red | red_1 | 25 kHz | 2 (LMS, input power assumed) | 18 | 0.63 (0.43-0.92) | 2.49 (1.50-3.50) | 0.59 (0.18-0.89) | 0.093 |
| red | red_2 | 25 kHz | 2 (LMS, input power assumed) | 15 | 0.81 (0.46-1.28) | 2.31 (1.62-3.10) | 0.73 (0.18-1.02) | 0.013 |
| yellow | yellow_1 | 13 kHz | 5.3 (LMS, input power assumed) | none besides the reference | | | | |
| yellow | yellow_4 | 13 kHz | 5.3 (LMS, input power assumed) | 19 | 0.59 (0.39-1.15) | 2.34 (2.00-3.96) | 0.21 (0.08-0.41) | 0.495 |
| blue | blue_1 | 13 kHz | 7.8 (RLS, stated) | 22 | 0.41 (0.35-0.56) | 1.77 (1.64-2.50) | 0.25 (0.06-0.55) | 0.145 |
