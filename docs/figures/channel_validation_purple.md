# Time spreading against measured channels (UACR Purple, SPACE'08)

A: gaussian Doppler spectrum, facet decay time, bounces convolved. C: gaussian, rule_of_thumb, convolve (the suite's defaults).
B: lorentzian, rule_of_thumb, sum (SST-style). One free parameter per recording (wind speed), fitted to the row marked FIT;
every other row is a prediction. +- on the measured column: standard error from the four quarters of the record; on the model
columns: spread of one record from seed to seed. `in`/`OUT`: within twice the larger of the two, plus the resolution.
D: A with the rms height fitted too (to the coherent share of S at 200 m): two fitted numbers. E: D with a K-type tap texture
fitted to the scintillation index of S at 60 m: three fitted numbers. Rows a number was fitted to are marked FIT.

## purple_1: 60 m, wind 3.96 m/s (fitted here)
rms height 0.084 m, mean-square slope 0.0233, rms vertical speed 0.256 m/s; estimator bandwidth 10.4 Hz; floor -35.3 dB per delay bin re the whole channel; 4 receivers, 53 s; D, E: rms height 0.116 m; E: tap_shape 8.47

| arrival | quantity | measured | A | | C | | B | | D | | E | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D+B | delay after direct (ms) | 0.000 +-0.000 | 0.000 +-0.000 | in |  |  |  |  |  |  |  |  |
| D+B | level re whole channel (dB) | -5.3 +-0.0 | -6.3 +-0.0 | OUT |  |  |  |  |  |  |  |  |
| D+B | the same with bubble loss | -5.3 +-0.0 | -6.3 | OUT |  |  |  |  |  |  |  |  |
| D+B | coherent share, at the peak | 1.002 +-0.000 | 1.001 +-0.000 | in | 1.001 +-0.000 | in |  |  | 1.001 +-0.000 | in |  |  |
| D+B | scintillation index | 0.00 +-0.00 | 0.00 +-0.00 | in | 0.00 +-0.00 | in |  |  | 0.00 +-0.00 | in | 0.00 +-0.00 | in |
| S | delay after direct (ms) | 2.995 +-0.021 | 2.918 +-0.000 | in |  |  |  |  |  |  |  |  |
| S | level re whole channel (dB) | -9.5 +-0.1 | -10.4 +-0.1 | OUT |  |  |  |  |  |  |  |  |
| S | the same with bubble loss | -9.5 +-0.1 | -10.3 | OUT |  |  |  |  |  |  |  |  |
| S | coherent share, arrival and tail | 0.012 +-0.005 | 0.002 +-0.000 | in | 0.006 +-0.003 | in |  |  | 0.002 +-0.000 | in |  |  |
| S | coherent share, at the peak | 0.005 +-0.007 | 0.000 +-0.000 | in | 0.001 +-0.001 | in |  |  | 0.000 +-0.000 | in |  |  |
| S | scintillation index | 0.87 +-0.15 | 0.67 +-0.02 | in | 0.53 +-0.02 | in |  |  | 0.67 +-0.02 | in | 0.88 +-0.19 | fit |
| S | Doppler rms width (Hz) FIT | 1.53 +-0.17 | 1.54 +-0.10 | fit |  |  | 2.63 +-0.53 | fit | 1.54 +-0.10 | fit |  |  |
| S | Doppler 50 % half width (Hz) | 1.00 +-0.08 | 0.95 +-0.09 | in |  |  | 1.28 +-0.19 | in |  |  |  |  |
| S | Doppler shape w90/w50 | 2.53 +-0.15 | 2.71 +-0.14 | in |  |  | 4.34 +-0.32 | OUT |  |  |  |  |
| S | coherence time (ms) | 143 +-13 | 152 +-2 | in |  |  | 99 +-11 | OUT |  |  |  |  |
| S | tail decay time (ms) | 0.267 +-0.003 | 0.264 +-0.003 | in | 0.561 +-0.001 | OUT | 0.548 +-0.014 | OUT | 0.264 +-0.003 | in |  |  |
| S | tail mean excess delay (ms) | 0.200 +-0.004 | 0.246 +-0.005 | in | 0.354 +-0.000 | OUT | 0.354 +-0.001 | OUT |  |  |  |  |
| S | closed forms | | grazing 21.0 deg, Gamma^2 9.94, coherent 0.000 | | Doppler rms 1.53 Hz | | Lorentzian half width 1.53 Hz | |
| SB | delay after direct (ms) | 4.608 +-0.000 | 4.531 +-0.000 | in |  |  |  |  |  |  |  |  |
| SB | level re whole channel (dB) | -14.7 +-0.1 | -11.6 +-0.1 | OUT |  |  |  |  |  |  |  |  |
| SB | the same with bubble loss | -14.7 +-0.1 | -11.7 | OUT |  |  |  |  |  |  |  |  |
| SB | coherent share, arrival and tail | 0.002 +-0.002 | 0.004 +-0.003 | in | 0.003 +-0.002 | in |  |  | 0.004 +-0.003 | in |  |  |
| SB | coherent share, at the peak | 0.001 +-0.004 | 0.004 +-0.005 | in | 0.003 +-0.004 | in |  |  | 0.004 +-0.005 | in |  |  |
| SB | scintillation index | 1.14 +-0.16 | 0.66 +-0.07 | OUT | 0.62 +-0.05 | OUT |  |  | 0.66 +-0.07 | OUT | 0.86 +-0.01 | in |
| SB | Doppler rms width (Hz) | 1.98 +-0.10 | 1.84 +-0.16 | in |  |  | 3.00 +-0.25 | OUT | 1.84 +-0.16 | in |  |  |
| SB | Doppler 50 % half width (Hz) | 1.28 +-0.08 | 1.23 +-0.12 | in |  |  | 1.61 +-0.13 | OUT |  |  |  |  |
| SB | Doppler shape w90/w50 | 2.49 +-0.16 | 2.39 +-0.11 | in |  |  | 4.03 +-0.01 | OUT |  |  |  |  |
| SB | coherence time (ms) | 116 +-18 | 125 +-2 | in |  |  | 94 +-10 | in |  |  |  |  |
| SB | tail decay time (ms) | 0.481 +-0.031 | 0.353 +-0.001 | OUT | 0.617 +-0.027 | OUT | 0.511 +-0.024 | in | 0.353 +-0.001 | OUT |  |  |
| SB | tail mean excess delay (ms) | 0.163 +-0.006 | 0.134 +-0.001 | in | 0.165 +-0.001 | in | 0.161 +-0.000 | in |  |  |  |  |
| SB | closed forms | | grazing 25.7 deg, Gamma^2 14.55, coherent 0.000 | | Doppler rms 1.85 Hz | | Lorentzian half width 1.85 Hz | |
| BS | delay after direct (ms) | 4.992 +-0.052 | 5.107 +-0.000 | in |  |  |  |  |  |  |  |  |
| BS | level re whole channel (dB) | -15.9 +-0.2 | -11.2 +-0.2 | OUT |  |  |  |  |  |  |  |  |
| BS | the same with bubble loss | -15.9 +-0.2 | -11.1 | OUT |  |  |  |  |  |  |  |  |
| BS | coherent share, arrival and tail | 0.006 +-0.002 | 0.004 +-0.005 | in | 0.008 +-0.010 | in |  |  | 0.004 +-0.005 | in |  |  |
| BS | coherent share, at the peak | 0.002 +-0.001 | 0.004 +-0.004 | in | 0.005 +-0.006 | in |  |  | 0.004 +-0.004 | in |  |  |
| BS | scintillation index | 1.14 +-0.18 | 0.61 +-0.00 | OUT | 0.61 +-0.01 | OUT |  |  | 0.61 +-0.00 | OUT | 0.71 +-0.08 | OUT |
| BS | Doppler rms width (Hz) | 2.17 +-0.10 | 1.81 +-0.02 | OUT |  |  | 2.55 +-0.08 | OUT | 1.81 +-0.02 | OUT |  |  |
| BS | Doppler 50 % half width (Hz) | 1.40 +-0.09 | 1.26 +-0.02 | in |  |  | 1.40 +-0.00 | in |  |  |  |  |
| BS | Doppler shape w90/w50 | 2.50 +-0.09 | 2.34 +-0.08 | in |  |  | 3.95 +-0.19 | OUT |  |  |  |  |
| BS | coherence time (ms) | 108 +-7 | 122 +-2 | in |  |  | 109 +-3 | in |  |  |  |  |
| BS | tail decay time (ms) | 0.426 +-0.019 | 0.269 +-0.001 | OUT | 0.505 +-0.026 | in | 0.496 +-0.015 | in | 0.269 +-0.001 | OUT |  |  |
| BS | tail mean excess delay (ms) | 0.311 +-0.015 | 0.243 +-0.005 | in | 0.346 +-0.007 | in | 0.344 +-0.002 | in |  |  |  |  |
| BS | closed forms | | grazing 27.4 deg, Gamma^2 16.32, coherent 0.000 | | Doppler rms 1.96 Hz | | Lorentzian half width 1.96 Hz | |
| BSB | delay after direct (ms) | 7.219 +-0.000 | 7.104 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| BSB | level re whole channel (dB) | -22.8 +-0.4 | -20.2 +-0.1 | OUT |  |  |  |  |  |  |  |  |
| BSB | the same with bubble loss | -22.8 +-0.4 | -20.3 | OUT |  |  |  |  |  |  |  |  |
| BSB | coherent share, arrival and tail | 0.011 +-0.001 | 0.003 +-0.003 | in | 0.002 +-0.001 | in |  |  | 0.003 +-0.003 | in |  |  |
| BSB | coherent share, at the peak | 0.004 +-0.001 | 0.001 +-0.000 | in | 0.001 +-0.001 | in |  |  | 0.001 +-0.000 | in |  |  |
| BSB | scintillation index | 1.30 +-0.07 | 0.67 +-0.00 | OUT | 0.60 +-0.08 | OUT |  |  | 0.67 +-0.00 | OUT | 0.87 +-0.01 | OUT |
| BSB | Doppler rms width (Hz) | 2.07 +-0.12 | 2.29 +-0.02 | in |  |  | 3.31 +-0.22 | OUT | 2.29 +-0.02 | in |  |  |
| BSB | Doppler 50 % half width (Hz) | 1.39 +-0.13 | 1.49 +-0.05 | in |  |  | 1.78 +-0.00 | OUT |  |  |  |  |
| BSB | Doppler shape w90/w50 | 2.47 +-0.08 | 2.57 +-0.01 | in |  |  | 4.47 +-0.99 | in |  |  |  |  |
| BSB | coherence time (ms) | 105 +-6 | 100 +-6 | in |  |  | 82 +-8 | in |  |  |  |  |
| BSB | tail decay time (ms) | 0.289 +-0.009 | 0.403 +-0.002 | OUT | 0.629 +-0.037 | OUT | 0.611 +-0.008 | OUT | 0.403 +-0.002 | OUT |  |  |
| BSB | tail mean excess delay (ms) | 0.248 +-0.010 | 0.288 +-0.002 | in | 0.374 +-0.013 | OUT | 0.375 +-0.003 | OUT |  |  |  |  |
| BSB | closed forms | | grazing 31.6 deg, Gamma^2 21.19, coherent 0.000 | | Doppler rms 2.23 Hz | | Lorentzian half width 2.23 Hz | |

## purple_3: 200 m, wind 3.96 m/s (from the 60 m file of the same recording)
rms height 0.084 m, mean-square slope 0.0233, rms vertical speed 0.256 m/s; estimator bandwidth 10.4 Hz; floor -32.3 dB per delay bin re the whole channel; 4 receivers, 53 s; D, E: rms height 0.116 m; E: tap_shape 8.47

| arrival | quantity | measured | A | | C | | B | | D | | E | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D+B | delay after direct (ms) | 0.000 +-0.000 | 0.000 +-0.000 | in |  |  |  |  |  |  |  |  |
| D+B | level re whole channel (dB) | -7.6 +-0.0 | -9.1 +-0.0 | OUT |  |  |  |  |  |  |  |  |
| D+B | the same with bubble loss | -7.6 +-0.0 | -9.0 | OUT |  |  |  |  |  |  |  |  |
| D+B | coherent share, at the peak | 1.008 +-0.092 | 1.002 +-0.000 | in | 1.004 +-0.000 | in |  |  | 1.002 +-0.000 | in |  |  |
| D+B | scintillation index | 0.00 +-0.00 | 0.00 +-0.00 | in | 0.00 +-0.00 | in |  |  | 0.00 +-0.00 | in | 0.00 +-0.00 | in |
| S | delay after direct (ms) | 1.037 +-0.000 | 0.806 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| S | level re whole channel (dB) | -9.4 +-0.2 | -11.5 +-0.2 | OUT |  |  |  |  |  |  |  |  |
| S | the same with bubble loss | -9.4 +-0.2 | -11.7 | OUT |  |  |  |  |  |  |  |  |
| S | coherent share, arrival and tail | 0.136 +-0.044 | 0.606 +-0.009 | OUT | 0.791 +-0.008 | OUT |  |  | 0.205 +-0.023 | fit |  |  |
| S | coherent share, at the peak | 0.156 +-0.042 | 0.852 +-0.013 | OUT | 0.992 +-0.000 | OUT |  |  | 0.307 +-0.032 | fit |  |  |
| S | scintillation index | 0.67 +-0.07 | 0.11 +-0.00 | OUT | 0.00 +-0.00 | OUT |  |  | 0.43 +-0.03 | OUT | 0.60 +-0.08 | in |
| S | Doppler rms width (Hz) | 0.54 +-0.02 | 0.65 +-0.11 | in |  |  | 1.29 +-0.16 | OUT | 0.59 +-0.08 | in |  |  |
| S | Doppler 50 % half width (Hz) | 0.38 +-0.00 | 0.45 +-0.06 | in |  |  | 0.95 +-0.01 | OUT |  |  |  |  |
| S | Doppler shape w90/w50 | 2.12 +-0.08 | 2.32 +-0.13 | in |  |  | 12.05 +-0.01 | OUT |  |  |  |  |
| S | coherence time (ms) | 481 +-39 | 352 +-22 | OUT |  |  | 12 +-0 | OUT |  |  |  |  |
| S | tail decay time (ms) | 0.255 +-0.018 | 0.838 +-0.187 | OUT | - | - | - | - | 1.144 +-0.387 | OUT |  |  |
| S | tail mean excess delay (ms) | 0.121 +-0.004 | 0.206 +-0.004 | OUT | 0.273 +-0.006 | OUT | 0.278 +-0.053 | OUT |  |  |  |  |
| S | closed forms | | grazing 6.7 deg, Gamma^2 1.05, coherent 0.351 | | Doppler rms 0.62 Hz | | Lorentzian half width 0.50 Hz | |
| SB+BS | delay after direct (ms) | 1.459 +-0.017 | 1.382 +-0.000 | in |  |  |  |  |  |  |  |  |
| SB+BS | level re whole channel (dB) | -9.4 +-0.2 | -11.0 +-0.0 | OUT |  |  |  |  |  |  |  |  |
| SB+BS | the same with bubble loss | -9.4 +-0.2 | -10.9 | OUT |  |  |  |  |  |  |  |  |
| SB+BS | coherent share, arrival and tail | 0.042 +-0.025 | 0.206 +-0.034 | OUT | 0.339 +-0.009 | OUT |  |  | 0.035 +-0.004 | in |  |  |
| SB+BS | coherent share, at the peak | 0.045 +-0.035 | 0.437 +-0.031 | OUT | 0.683 +-0.000 | OUT |  |  | 0.056 +-0.004 | in |  |  |
| SB+BS | scintillation index | 0.73 +-0.08 | 0.40 +-0.00 | OUT | 0.35 +-0.05 | OUT |  |  | 0.54 +-0.02 | in | 0.58 +-0.02 | in |
| SB+BS | Doppler rms width (Hz) | 0.66 +-0.03 | 0.70 +-0.00 | in |  |  | 1.01 +-0.04 | OUT | 0.65 +-0.02 | in |  |  |
| SB+BS | Doppler 50 % half width (Hz) | 0.38 +-0.01 | 0.48 +-0.00 | OUT |  |  | 0.53 +-0.03 | OUT |  |  |  |  |
| SB+BS | Doppler shape w90/w50 | 2.72 +-0.18 | 2.32 +-0.04 | in |  |  | 5.85 +-2.54 | in |  |  |  |  |
| SB+BS | coherence time (ms) | 397 +-16 | 321 +-12 | OUT |  |  | 317 +-79 | in |  |  |  |  |
| SB+BS | tail decay time (ms) | 0.838 +-0.437 | 0.738 +-0.029 | in | 3.001 | OUT | 8.841 | OUT | 0.639 +-0.017 | in |  |  |
| SB+BS | tail mean excess delay (ms) | 0.132 +-0.005 | 0.317 +-0.011 | OUT | 0.370 +-0.003 | OUT | 0.355 +-0.013 | OUT |  |  |  |  |
| SB+BS | closed forms | | grazing 8.1 deg, Gamma^2 1.54, coherent 0.214 | | Doppler rms 0.68 Hz | | Lorentzian half width 0.60 Hz | |
| BSB | delay after direct (ms) | 1.843 +-0.017 | 2.266 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| BSB | level re whole channel (dB) | -12.7 +-0.3 | -11.6 +-0.1 | in |  |  |  |  |  |  |  |  |
| BSB | the same with bubble loss | -12.7 +-0.3 | -11.7 | in |  |  |  |  |  |  |  |  |
| BSB | coherent share, arrival and tail | 0.023 +-0.009 | 0.029 +-0.003 | in | 0.055 +-0.001 | in |  |  | 0.009 +-0.001 | in |  |  |
| BSB | coherent share, at the peak | 0.014 +-0.013 | 0.063 +-0.006 | OUT | 0.225 +-0.010 | OUT |  |  | 0.010 +-0.004 | in |  |  |
| BSB | scintillation index | 0.66 +-0.02 | 0.56 +-0.01 | OUT | 0.61 +-0.04 | in |  |  | 0.60 +-0.06 | in | 0.63 +-0.05 | in |
| BSB | Doppler rms width (Hz) | 0.81 +-0.04 | 0.78 +-0.03 | in |  |  | 1.18 +-0.20 | in | 0.73 +-0.03 | in |  |  |
| BSB | Doppler 50 % half width (Hz) | 0.48 +-0.04 | 0.51 +-0.00 | in |  |  | 0.66 +-0.10 | in |  |  |  |  |
| BSB | Doppler shape w90/w50 | 2.67 +-0.16 | 2.39 +-0.20 | in |  |  | 3.98 +-0.44 | OUT |  |  |  |  |
| BSB | coherence time (ms) | 297 +-9 | 289 +-15 | in |  |  | 251 +-3 | OUT |  |  |  |  |
| BSB | tail decay time (ms) | 0.651 +-0.045 | 0.844 +-0.058 | OUT | 3.873 +-2.411 | in | 2.217 +-0.238 | OUT | 0.879 +-0.053 | OUT |  |  |
| BSB | tail mean excess delay (ms) | 0.369 +-0.005 | 0.397 +-0.006 | in | 0.498 +-0.012 | OUT | 0.494 +-0.005 | OUT |  |  |  |  |
| BSB | closed forms | | grazing 10.4 deg, Gamma^2 2.49, coherent 0.083 | | Doppler rms 0.80 Hz | | Lorentzian half width 0.77 Hz | |
| SBS | delay after direct (ms) | 5.069 +-0.000 | 4.800 +-0.163 | in |  |  |  |  |  |  |  |  |
| SBS | level re whole channel (dB) | -18.8 +-0.4 | -16.8 +-0.7 | OUT |  |  |  |  |  |  |  |  |
| SBS | the same with bubble loss | -18.8 +-0.4 | -16.3 | OUT |  |  |  |  |  |  |  |  |
| SBS | coherent share, arrival and tail | 0.006 +-0.003 | 0.002 +-0.002 | in | 0.005 +-0.001 | in |  |  | 0.003 +-0.002 | in |  |  |
| SBS | coherent share, at the peak | 0.011 +-0.008 | 0.001 +-0.001 | in | 0.006 +-0.001 | in |  |  | 0.001 +-0.001 | in |  |  |
| SBS | scintillation index | 1.75 +-0.19 | 0.56 +-0.08 | OUT | 0.75 +-0.01 | OUT |  |  | 0.54 +-0.05 | OUT | 0.69 +-0.04 | OUT |
| SBS | Doppler rms width (Hz) | 1.27 +-0.02 | 1.34 +-0.10 | in |  |  | 1.88 +-0.26 | OUT | 1.46 +-0.04 | OUT |  |  |
| SBS | Doppler 50 % half width (Hz) | 0.86 +-0.05 | 0.81 +-0.11 | in |  |  | 0.97 +-0.09 | in |  |  |  |  |
| SBS | Doppler shape w90/w50 | 2.40 +-0.12 | 2.93 +-0.41 | in |  |  | 4.83 +-1.45 | in |  |  |  |  |
| SBS | coherence time (ms) | 178 +-5 | 159 +-5 | OUT |  |  | 149 +-22 | in |  |  |  |  |
| SBS | tail decay time (ms) | 1.082 +-0.194 | 1.562 +-0.402 | in | 3.857 +-1.133 | OUT | 1.481 +-0.038 | in | 1.345 +-0.117 | in |  |  |
| SBS | tail mean excess delay (ms) | 0.461 +-0.006 | 0.468 +-0.017 | in | 0.521 +-0.005 | OUT | 0.451 +-0.003 | in |  |  |  |  |

## purple_5: 1000 m, wind 3.96 m/s (from the 60 m file of the same recording)
rms height 0.084 m, mean-square slope 0.0233, rms vertical speed 0.256 m/s; estimator bandwidth 10.4 Hz; floor -33.2 dB per delay bin re the whole channel; 4 receivers, 53 s; D, E: rms height 0.116 m; E: tap_shape 8.47

| arrival | quantity | measured | A | | C | | B | | D | | E | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D+B+S+SB+BS+BSB | delay after direct (ms) | 0.000 +-0.000 | 0.000 +-0.000 | in |  |  |  |  |  |  |  |  |
| D+B+S+SB+BS+BSB | level re whole channel (dB) | -9.7 +-0.1 | -6.0 +-0.1 | OUT |  |  |  |  |  |  |  |  |
| D+B+S+SB+BS+BSB | the same with bubble loss | -9.7 +-0.1 | -5.9 | OUT |  |  |  |  |  |  |  |  |
| D+B+S+SB+BS+BSB | coherent share, arrival and tail | 0.720 +-0.016 | 1.073 +-0.000 | OUT | 1.068 +-0.001 | OUT |  |  | 1.086 +-0.003 | OUT |  |  |
| D+B+S+SB+BS+BSB | coherent share, at the peak | 0.874 +-0.020 | 1.042 +-0.000 | OUT | 1.034 +-0.000 | OUT |  |  | 1.050 +-0.001 | OUT |  |  |
| D+B+S+SB+BS+BSB | scintillation index | 0.00 +-0.01 | 0.00 +-0.00 | in | 0.00 +-0.00 | in |  |  | 0.00 +-0.00 | in | 0.00 +-0.00 | in |
| D+B+S+SB+BS+BSB | Doppler rms width (Hz) | 0.40 +-0.01 | 0.00 +-0.00 | OUT |  |  | 0.00 +-0.00 | OUT | 0.00 +-0.00 | OUT |  |  |
| D+B+S+SB+BS+BSB | Doppler 50 % half width (Hz) | 0.31 +-0.00 | 0.00 +-0.00 | OUT |  |  | 0.00 +-0.00 | OUT |  |  |  |  |
| D+B+S+SB+BS+BSB | Doppler shape w90/w50 | 1.83 +-0.03 | 0.00 +-0.00 | OUT |  |  | 0.00 +-0.00 | OUT |  |  |  |  |
| D+B+S+SB+BS+BSB | coherence time (ms) | 843 +-60 | 10 +-0 | OUT |  |  | 10 +-0 | OUT |  |  |  |  |
| D+B+S+SB+BS+BSB | tail decay time (ms) | 0.755 +-0.115 | - | - | - | - | - | - | - | - |  |  |
| D+B+S+SB+BS+BSB | tail mean excess delay (ms) | 0.497 +-0.020 | 0.000 +-0.000 | OUT | 0.000 +-0.000 | OUT | 0.000 +-0.000 | OUT |  |  |  |  |
| D+B+S+SB+BS+BSB | closed forms | | grazing 1.3 deg, Gamma^2 0.04, coherent 0.958 | | Doppler rms 0.49 Hz | | Lorentzian half width 0.10 Hz | |
| SBS+SBSB+BSBS+BSBSB | delay after direct (ms) | 1.190 +-0.041 | 0.922 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| SBS+SBSB+BSBS+BSBSB | level re whole channel (dB) | -11.4 +-0.4 | -9.6 +-0.1 | OUT |  |  |  |  |  |  |  |  |
| SBS+SBSB+BSBS+BSBSB | the same with bubble loss | -11.4 +-0.4 | -9.6 | OUT |  |  |  |  |  |  |  |  |
| SBS+SBSB+BSBS+BSBSB | coherent share, arrival and tail | 0.259 +-0.022 | 0.941 +-0.016 | OUT | 1.136 +-0.002 | OUT |  |  | 0.791 +-0.017 | OUT |  |  |
| SBS+SBSB+BSBS+BSBSB | coherent share, at the peak | 0.371 +-0.020 | 1.065 +-0.001 | OUT | 1.068 +-0.000 | OUT |  |  | 1.094 +-0.003 | OUT |  |  |
| SBS+SBSB+BSBS+BSBSB | scintillation index | 0.59 +-0.04 | 0.00 +-0.00 | OUT | 0.00 +-0.00 | OUT |  |  | 0.00 +-0.00 | OUT | 0.00 +-0.00 | OUT |
| SBS+SBSB+BSBS+BSBSB | Doppler rms width (Hz) | 0.53 +-0.01 | 0.00 +-0.00 | OUT |  |  | 0.00 +-0.00 | OUT | 0.00 +-0.00 | OUT |  |  |
| SBS+SBSB+BSBS+BSBSB | Doppler 50 % half width (Hz) | 0.37 +-0.01 | 0.37 +-0.02 | in |  |  | 0.34 +-0.03 | in |  |  |  |  |
| SBS+SBSB+BSBS+BSBSB | Doppler shape w90/w50 | 2.19 +-0.06 | 1.97 +-0.04 | in |  |  | 1.62 +-0.03 | OUT |  |  |  |  |
| SBS+SBSB+BSBS+BSBSB | coherence time (ms) | 598 +-14 | 146 +-29 | OUT |  |  | 13 +-0 | OUT |  |  |  |  |
| SBS+SBSB+BSBS+BSBSB | tail decay time (ms) | 0.668 +-0.046 | - | - | - | - | - | - | - | - |  |  |
| SBS+SBSB+BSBS+BSBSB | tail mean excess delay (ms) | 0.389 +-0.030 | 0.657 +-0.012 | OUT | 0.924 +-0.012 | OUT | 0.000 +-0.000 | OUT |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | delay after direct (ms) | 2.573 +-0.064 | 2.592 +-0.081 | in |  |  |  |  |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | level re whole channel (dB) | -16.9 +-0.3 | -15.2 +-0.2 | OUT |  |  |  |  |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | the same with bubble loss | -16.9 +-0.3 | -15.1 | OUT |  |  |  |  |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | coherent share, arrival and tail | 0.042 +-0.014 | 0.666 +-0.089 | OUT | 1.124 +-0.006 | OUT |  |  | 0.262 +-0.013 | OUT |  |  |
| SBSBS+SBSBSB+BSBSBS | coherent share, at the peak | 0.071 +-0.026 | 0.740 +-0.110 | OUT | 1.078 +-0.008 | OUT |  |  | 0.302 +-0.026 | OUT |  |  |
| SBSBS+SBSBSB+BSBSBS | scintillation index | 0.62 +-0.06 | 0.40 +-0.16 | in | 0.15 +-0.02 | OUT |  |  | 1.24 +-0.19 | OUT | 1.47 +-0.27 | OUT |
| SBSBS+SBSBSB+BSBSBS | Doppler rms width (Hz) | 0.79 +-0.03 | 0.58 +-0.01 | OUT |  |  | 0.70 +-0.10 | in | 0.44 +-0.03 | OUT |  |  |
| SBSBS+SBSBSB+BSBSBS | Doppler 50 % half width (Hz) | 0.46 +-0.02 | 0.42 +-0.06 | in |  |  | 0.39 +-0.01 | in |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | Doppler shape w90/w50 | 2.82 +-0.06 | 2.23 +-0.15 | OUT |  |  | 3.82 +-0.28 | OUT |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | coherence time (ms) | 373 +-22 | 342 +-29 | in |  |  | 326 +-98 | in |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | tail decay time (ms) | 0.783 +-1.553 | 2.350 | in | - | - | - | - | 1.258 +-0.269 | in |  |  |
| SBSBS+SBSBSB+BSBSBS | tail mean excess delay (ms) | 0.225 +-0.045 | 0.217 +-0.020 | in | 0.303 +-0.001 | in | 0.139 +-0.007 | in |  |  |  |  |
| BSBSBSB | delay after direct (ms) | 2.995 +-0.078 | 3.168 +-0.081 | in |  |  |  |  |  |  |  |  |
| BSBSBSB | level re whole channel (dB) | -16.4 +-0.2 | -15.8 +-0.6 | in |  |  |  |  |  |  |  |  |
| BSBSBSB | the same with bubble loss | -16.4 +-0.2 | -16.3 | in |  |  |  |  |  |  |  |  |
| BSBSBSB | coherent share, arrival and tail | 0.030 +-0.009 | 0.061 +-0.006 | in | 0.110 +-0.021 | OUT |  |  | 0.029 +-0.002 | in |  |  |
| BSBSBSB | coherent share, at the peak | 0.040 +-0.010 | 0.065 +-0.023 | in | 0.061 +-0.040 | in |  |  | 0.052 +-0.009 | in |  |  |
| BSBSBSB | scintillation index | 0.71 +-0.03 | 1.23 +-0.14 | OUT | 1.73 +-0.68 | in |  |  | 1.00 +-0.25 | in | 1.20 +-0.30 | in |
| BSBSBSB | Doppler rms width (Hz) | 0.79 +-0.02 | 0.70 +-0.05 | in |  |  | 0.88 +-0.02 | in | 0.52 +-0.04 | OUT |  |  |
| BSBSBSB | Doppler 50 % half width (Hz) | 0.47 +-0.03 | 0.54 +-0.04 | in |  |  | 0.57 +-0.09 | in |  |  |  |  |
| BSBSBSB | Doppler shape w90/w50 | 2.73 +-0.12 | 2.05 +-0.06 | OUT |  |  | 3.19 +-0.45 | in |  |  |  |  |
| BSBSBSB | coherence time (ms) | 381 +-19 | 321 +-46 | in |  |  | 277 +-50 | in |  |  |  |  |
| BSBSBSB | tail decay time (ms) | 10.829 +-0.146 | 0.574 +-0.645 | OUT | - | - | - | - | 2.647 +-1.479 | OUT |  |  |
| BSBSBSB | tail mean excess delay (ms) | 0.379 +-0.024 | 0.315 +-0.030 | in | 0.380 +-0.016 | in | 0.330 +-0.084 | in |  |  |  |  |
| SBSBSBS | delay after direct (ms) | 4.109 +-0.014 | 4.262 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBS | level re whole channel (dB) | -19.0 +-0.1 | -17.4 +-0.3 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBS | the same with bubble loss | -19.0 +-0.1 | -17.7 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBS | coherent share, arrival and tail | 0.017 +-0.008 | 0.119 +-0.049 | in | 0.080 +-0.022 | in |  |  | 0.018 +-0.006 | in |  |  |
| SBSBSBS | coherent share, at the peak | 0.018 +-0.001 | 0.073 +-0.030 | in | 0.020 +-0.009 | in |  |  | 0.011 +-0.001 | in |  |  |
| SBSBSBS | scintillation index | 0.60 +-0.03 | 1.19 +-0.03 | OUT | 1.41 +-0.20 | OUT |  |  | 1.25 +-0.30 | in | 1.45 +-0.18 | OUT |
| SBSBSBS | Doppler rms width (Hz) | 1.12 +-0.06 | 0.69 +-0.05 | OUT |  |  | 1.00 +-0.15 | in | 0.59 +-0.04 | OUT |  |  |
| SBSBSBS | Doppler 50 % half width (Hz) | 0.69 +-0.03 | 0.49 +-0.02 | OUT |  |  | 0.58 +-0.07 | in |  |  |  |  |
| SBSBSBS | Doppler shape w90/w50 | 2.68 +-0.08 | 2.43 +-0.28 | in |  |  | 3.86 +-0.27 | OUT |  |  |  |  |
| SBSBSBS | coherence time (ms) | 222 +-10 | 292 +-12 | OUT |  |  | 248 +-4 | in |  |  |  |  |
| SBSBSBS | tail decay time (ms) | 1.110 +-0.408 | 0.289 +-0.150 | in | 2.255 | OUT | - | - | 0.083 +-0.007 | OUT |  |  |
| SBSBSBS | tail mean excess delay (ms) | 0.143 +-0.013 | 0.089 +-0.001 | in | 0.047 +-0.003 | OUT | 0.041 +-0.006 | OUT |  |  |  |  |
| SBSBSBSB+BSBSBSBS | delay after direct (ms) | 4.608 +-0.040 | 4.915 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSB+BSBSBSBS | level re whole channel (dB) | -21.4 +-0.2 | -16.4 +-0.0 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSB+BSBSBSBS | the same with bubble loss | -21.4 +-0.2 | -16.5 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSB+BSBSBSBS | coherent share, arrival and tail | 0.027 +-0.009 | 0.076 +-0.022 | in | 0.144 +-0.033 | OUT |  |  | 0.025 +-0.012 | in |  |  |
| SBSBSBSB+BSBSBSBS | coherent share, at the peak | 0.033 +-0.017 | 0.044 +-0.035 | in | 0.077 +-0.020 | in |  |  | 0.011 +-0.008 | in |  |  |
| SBSBSBSB+BSBSBSBS | scintillation index | 0.62 +-0.01 | 0.96 +-0.23 | in | 1.88 +-0.49 | OUT |  |  | 1.16 +-0.09 | OUT | 1.37 +-0.16 | OUT |
| SBSBSBSB+BSBSBSBS | Doppler rms width (Hz) | 1.22 +-0.04 | 0.74 +-0.00 | OUT |  |  | 1.03 +-0.19 | in | 0.59 +-0.02 | OUT |  |  |
| SBSBSBSB+BSBSBSBS | Doppler 50 % half width (Hz) | 0.73 +-0.03 | 0.51 +-0.00 | OUT |  |  | 0.68 +-0.01 | in |  |  |  |  |
| SBSBSBSB+BSBSBSBS | Doppler shape w90/w50 | 2.80 +-0.10 | 2.47 +-0.05 | in |  |  | 2.97 +-0.64 | in |  |  |  |  |
| SBSBSBSB+BSBSBSBS | coherence time (ms) | 197 +-7 | 298 +-11 | OUT |  |  | 237 +-8 | OUT |  |  |  |  |
| SBSBSBSB+BSBSBSBS | tail decay time (ms) | 1.825 +-0.847 | 0.149 +-0.028 | in | - | - | 0.633 +-0.423 | in | - | - |  |  |
| SBSBSBSB+BSBSBSBS | tail mean excess delay (ms) | 0.233 +-0.007 | 0.092 +-0.010 | OUT | 0.100 +-0.007 | OUT | 0.102 +-0.010 | OUT |  |  |  |  |
| BSBSBSBSB | delay after direct (ms) | 5.376 +-0.021 | 5.472 +-0.027 | in |  |  |  |  |  |  |  |  |
| BSBSBSBSB | level re whole channel (dB) | -21.4 +-0.2 | -18.4 +-1.1 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSBSB | the same with bubble loss | -21.4 +-0.2 | -17.8 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSBSB | coherent share, arrival and tail | 0.014 +-0.006 | 0.015 +-0.008 | in | 0.036 +-0.019 | in |  |  | 0.024 +-0.007 | in |  |  |
| BSBSBSBSB | coherent share, at the peak | 0.012 +-0.014 | 0.023 +-0.019 | in | 0.093 +-0.040 | in |  |  | 0.014 +-0.001 | in |  |  |
| BSBSBSBSB | scintillation index | 0.81 +-0.02 | 1.36 +-0.26 | in | 2.11 +-0.43 | OUT |  |  | 1.07 +-0.01 | OUT | 1.14 +-0.03 | OUT |
| BSBSBSBSB | Doppler rms width (Hz) | 1.18 +-0.04 | 0.72 +-0.01 | OUT |  |  | 0.98 +-0.06 | OUT | 0.62 +-0.00 | OUT |  |  |
| BSBSBSBSB | Doppler 50 % half width (Hz) | 0.67 +-0.03 | 0.52 +-0.02 | OUT |  |  | 0.54 +-0.04 | in |  |  |  |  |
| BSBSBSBSB | Doppler shape w90/w50 | 3.01 +-0.11 | 2.34 +-0.02 | OUT |  |  | 4.39 +-0.45 | OUT |  |  |  |  |
| BSBSBSBSB | coherence time (ms) | 201 +-4 | 264 +-1 | OUT |  |  | 217 +-3 | in |  |  |  |  |
| BSBSBSBSB | tail decay time (ms) | 11.736 +-0.555 | 1.895 | OUT | 0.686 | OUT | - | - | 0.468 +-0.281 | OUT |  |  |
| BSBSBSBSB | tail mean excess delay (ms) | 0.507 +-0.016 | 0.461 +-0.002 | in | 0.505 +-0.018 | in | 0.400 +-0.009 | OUT |  |  |  |  |
| SBSBSBSBS | delay after direct (ms) | 6.643 +-0.017 | 6.758 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSBS | level re whole channel (dB) | -21.5 +-0.3 | -17.6 +-0.2 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSBS | the same with bubble loss | -21.5 +-0.3 | -17.6 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSBS | coherent share, arrival and tail | 0.016 +-0.004 | 0.018 +-0.002 | in | 0.016 +-0.005 | in |  |  | 0.024 +-0.018 | in |  |  |
| SBSBSBSBS | coherent share, at the peak | 0.014 +-0.005 | 0.015 +-0.002 | in | 0.013 +-0.011 | in |  |  | 0.024 +-0.021 | in |  |  |
| SBSBSBSBS | scintillation index | 0.69 +-0.01 | 1.61 +-0.34 | OUT | 1.42 +-0.68 | in |  |  | 1.19 +-0.36 | in | 1.21 +-0.34 | in |
| SBSBSBSBS | Doppler rms width (Hz) | 1.40 +-0.03 | 0.85 +-0.05 | OUT |  |  | 1.22 +-0.34 | in | 0.67 +-0.00 | OUT |  |  |
| SBSBSBSBS | Doppler 50 % half width (Hz) | 0.81 +-0.03 | 0.53 +-0.00 | OUT |  |  | 0.76 +-0.17 | in |  |  |  |  |
| SBSBSBSBS | Doppler shape w90/w50 | 2.82 +-0.06 | 2.72 +-0.24 | in |  |  | 4.46 +-1.40 | in |  |  |  |  |
| SBSBSBSBS | coherence time (ms) | 168 +-2 | 246 +-13 | OUT |  |  | 207 +-25 | in |  |  |  |  |
| SBSBSBSBS | tail decay time (ms) | 0.865 +-0.402 | 0.204 +-0.035 | in | 0.125 +-0.030 | in | 0.236 +-0.145 | in | - | - |  |  |
| SBSBSBSBS | tail mean excess delay (ms) | 0.140 +-0.001 | 0.088 +-0.027 | in | 0.124 +-0.013 | in | 0.118 +-0.002 | in |  |  |  |  |
| SBSBSBSBSB | delay after direct (ms) | 7.142 +-0.008 | 7.200 +-0.027 | in |  |  |  |  |  |  |  |  |
| SBSBSBSBSB | level re whole channel (dB) | -23.2 +-0.2 | -18.0 +-0.3 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSBSB | the same with bubble loss | -23.2 +-0.2 | -17.9 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSBSB | coherent share, arrival and tail | 0.007 +-0.004 | 0.028 +-0.003 | in | 0.022 +-0.005 | in |  |  | 0.028 +-0.016 | in |  |  |
| SBSBSBSBSB | coherent share, at the peak | 0.010 +-0.006 | 0.023 +-0.006 | in | 0.006 +-0.000 | in |  |  | 0.019 +-0.008 | in |  |  |
| SBSBSBSBSB | scintillation index | 0.64 +-0.03 | 1.15 +-0.06 | OUT | 1.55 +-0.49 | in |  |  | 1.93 +-0.24 | OUT | 1.84 +-0.17 | OUT |
| SBSBSBSBSB | Doppler rms width (Hz) | 1.55 +-0.02 | 0.92 +-0.06 | OUT |  |  | 1.15 +-0.21 | in | 0.70 +-0.01 | OUT |  |  |
| SBSBSBSBSB | Doppler 50 % half width (Hz) | 0.98 +-0.05 | 0.54 +-0.04 | OUT |  |  | 0.63 +-0.12 | OUT |  |  |  |  |
| SBSBSBSBSB | Doppler shape w90/w50 | 2.59 +-0.10 | 3.01 +-0.43 | in |  |  | 4.42 +-0.82 | in |  |  |  |  |
| SBSBSBSBSB | coherence time (ms) | 138 +-3 | 228 +-5 | OUT |  |  | 206 +-23 | OUT |  |  |  |  |
| SBSBSBSBSB | tail decay time (ms) | 1.359 +-0.643 | 0.430 +-0.047 | in | - | - | 0.427 +-0.063 | in | - | - |  |  |
| SBSBSBSBSB | tail mean excess delay (ms) | 0.144 +-0.002 | 0.103 +-0.007 | in | 0.088 +-0.004 | OUT | 0.152 +-0.009 | in |  |  |  |  |
| BSBSBSBSBS | delay after direct (ms) | 7.488 +-0.008 | 7.603 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSBSBS | level re whole channel (dB) | -24.3 +-0.3 | -19.5 +-0.2 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSBSBS | the same with bubble loss | -24.3 +-0.3 | -19.7 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSBSBS | coherent share, arrival and tail | 0.008 +-0.005 | 0.037 +-0.002 | in | 0.013 +-0.001 | in |  |  | 0.034 +-0.002 | in |  |  |
| BSBSBSBSBS | coherent share, at the peak | 0.004 +-0.004 | 0.023 +-0.006 | in | 0.004 +-0.001 | in |  |  | 0.025 +-0.019 | in |  |  |
| BSBSBSBSBS | scintillation index | 0.83 +-0.02 | 1.53 +-0.03 | OUT | 1.48 +-0.16 | OUT |  |  | 1.93 +-0.15 | OUT | 1.76 +-0.07 | OUT |
| BSBSBSBSBS | Doppler rms width (Hz) | 1.64 +-0.03 | 0.79 +-0.06 | OUT |  |  | 1.32 +-0.36 | in | 0.73 +-0.02 | OUT |  |  |
| BSBSBSBSBS | Doppler 50 % half width (Hz) | 1.12 +-0.01 | 0.53 +-0.07 | OUT |  |  | 0.73 +-0.18 | in |  |  |  |  |
| BSBSBSBSBS | Doppler shape w90/w50 | 2.42 +-0.08 | 2.61 +-0.02 | in |  |  | 4.48 +-0.62 | OUT |  |  |  |  |
| BSBSBSBSBS | coherence time (ms) | 130 +-1 | 248 +-14 | OUT |  |  | 209 +-62 | in |  |  |  |  |
| BSBSBSBSBS | tail decay time (ms) | 1.212 +-0.065 | 0.463 | OUT | - | - | 0.427 +-0.063 | OUT | 0.730 +-0.177 | OUT |  |  |
| BSBSBSBSBS | tail mean excess delay (ms) | 0.142 +-0.011 | 0.181 +-0.007 | in | 0.017 +-0.001 | OUT | 0.070 +-0.026 | in |  |  |  |  |

## purple_6: 60 m, wind 3.70 m/s (fitted here)
rms height 0.073 m, mean-square slope 0.0220, rms vertical speed 0.239 m/s; estimator bandwidth 10.4 Hz; floor -34.6 dB per delay bin re the whole channel; 4 receivers, 53 s; D, E: rms height 0.102 m; E: tap_shape 1.31

| arrival | quantity | measured | A | | C | | B | | D | | E | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D+B | delay after direct (ms) | 0.000 +-0.000 | 0.000 +-0.000 | in |  |  |  |  |  |  |  |  |
| D+B | level re whole channel (dB) | -7.1 +-0.2 | -6.3 +-0.0 | in |  |  |  |  |  |  |  |  |
| D+B | the same with bubble loss | -7.1 +-0.2 | -6.3 | in |  |  |  |  |  |  |  |  |
| D+B | coherent share, at the peak | 0.573 +-0.085 | 1.001 +-0.000 | OUT | 1.001 +-0.000 | OUT |  |  | 1.001 +-0.000 | OUT |  |  |
| D+B | scintillation index | 0.54 +-0.01 | 0.00 +-0.00 | OUT | 0.00 +-0.00 | OUT |  |  | 0.00 +-0.00 | OUT | 0.00 +-0.00 | OUT |
| S | delay after direct (ms) | 3.072 +-0.014 | 2.918 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| S | level re whole channel (dB) | -11.1 +-0.1 | -10.2 +-0.2 | in |  |  |  |  |  |  |  |  |
| S | the same with bubble loss | -11.1 +-0.1 | -10.1 | OUT |  |  |  |  |  |  |  |  |
| S | coherent share, arrival and tail | 0.014 +-0.012 | 0.002 +-0.001 | in | 0.005 +-0.002 | in |  |  | 0.002 +-0.001 | in |  |  |
| S | coherent share, at the peak | 0.029 +-0.021 | 0.001 +-0.001 | in | 0.001 +-0.000 | in |  |  | 0.000 +-0.000 | in |  |  |
| S | scintillation index | 1.97 +-0.10 | 0.69 +-0.04 | OUT | 0.58 +-0.04 | OUT |  |  | 0.68 +-0.04 | OUT | 1.98 +-1.05 | fit |
| S | Doppler rms width (Hz) FIT | 1.43 +-0.12 | 1.43 +-0.05 | fit |  |  | 2.51 +-0.47 | fit | 1.43 +-0.05 | fit |  |  |
| S | Doppler 50 % half width (Hz) | 1.01 +-0.10 | 0.88 +-0.02 | in |  |  | 1.25 +-0.19 | in |  |  |  |  |
| S | Doppler shape w90/w50 | 2.30 +-0.13 | 2.71 +-0.01 | in |  |  | 4.24 +-0.19 | OUT |  |  |  |  |
| S | coherence time (ms) | 172 +-17 | 161 +-0 | in |  |  | 111 +-8 | OUT |  |  |  |  |
| S | tail decay time (ms) | 0.415 +-0.006 | 0.222 +-0.002 | OUT | 0.543 +-0.014 | OUT | 0.519 +-0.018 | OUT | 0.222 +-0.002 | OUT |  |  |
| S | tail mean excess delay (ms) | 0.335 +-0.006 | 0.232 +-0.007 | OUT | 0.347 +-0.002 | in | 0.342 +-0.003 | in |  |  |  |  |
| S | closed forms | | grazing 21.0 deg, Gamma^2 7.61, coherent 0.000 | | Doppler rms 1.43 Hz | | Lorentzian half width 1.43 Hz | |
| SB | delay after direct (ms) | 4.646 +-0.016 | 4.531 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| SB | level re whole channel (dB) | -15.8 +-0.4 | -11.5 +-0.1 | OUT |  |  |  |  |  |  |  |  |
| SB | the same with bubble loss | -15.8 +-0.4 | -11.6 | OUT |  |  |  |  |  |  |  |  |
| SB | coherent share, arrival and tail | 0.006 +-0.004 | 0.005 +-0.005 | in | 0.004 +-0.003 | in |  |  | 0.005 +-0.005 | in |  |  |
| SB | coherent share, at the peak | 0.004 +-0.004 | 0.004 +-0.005 | in | 0.005 +-0.005 | in |  |  | 0.004 +-0.005 | in |  |  |
| SB | scintillation index | 1.85 +-0.18 | 0.65 +-0.02 | OUT | 0.63 +-0.07 | OUT |  |  | 0.65 +-0.02 | OUT | 1.86 +-0.30 | in |
| SB | Doppler rms width (Hz) | 1.54 +-0.07 | 1.73 +-0.12 | in |  |  | 2.70 +-0.38 | OUT | 1.73 +-0.12 | in |  |  |
| SB | Doppler 50 % half width (Hz) | 0.89 +-0.03 | 1.16 +-0.02 | OUT |  |  | 1.46 +-0.27 | in |  |  |  |  |
| SB | Doppler shape w90/w50 | 3.08 +-0.08 | 2.39 +-0.08 | OUT |  |  | 4.22 +-0.40 | OUT |  |  |  |  |
| SB | coherence time (ms) | 152 +-7 | 133 +-3 | in |  |  | 102 +-15 | OUT |  |  |  |  |
| SB | tail decay time (ms) | 0.830 +-0.067 | 0.344 +-0.003 | OUT | 0.534 +-0.011 | OUT | 0.505 +-0.064 | OUT | 0.344 +-0.003 | OUT |  |  |
| SB | tail mean excess delay (ms) | 0.164 +-0.006 | 0.133 +-0.000 | in | 0.162 +-0.001 | in | 0.160 +-0.002 | in |  |  |  |  |
| SB | closed forms | | grazing 25.7 deg, Gamma^2 11.14, coherent 0.000 | | Doppler rms 1.73 Hz | | Lorentzian half width 1.73 Hz | |
| BS | delay after direct (ms) | 4.992 +-0.042 | 5.107 +-0.000 | in |  |  |  |  |  |  |  |  |
| BS | level re whole channel (dB) | -15.6 +-0.3 | -11.1 +-0.0 | OUT |  |  |  |  |  |  |  |  |
| BS | the same with bubble loss | -15.6 +-0.3 | -11.1 | OUT |  |  |  |  |  |  |  |  |
| BS | coherent share, arrival and tail | 0.005 +-0.005 | 0.005 +-0.005 | in | 0.006 +-0.007 | in |  |  | 0.005 +-0.005 | in |  |  |
| BS | coherent share, at the peak | 0.006 +-0.006 | 0.004 +-0.004 | in | 0.003 +-0.003 | in |  |  | 0.004 +-0.004 | in |  |  |
| BS | scintillation index | 1.31 +-0.15 | 0.59 +-0.01 | OUT | 0.64 +-0.02 | OUT |  |  | 0.59 +-0.01 | OUT | 1.27 +-0.53 | in |
| BS | Doppler rms width (Hz) | 2.01 +-0.06 | 1.68 +-0.00 | OUT |  |  | 2.46 +-0.28 | in | 1.68 +-0.00 | OUT |  |  |
| BS | Doppler 50 % half width (Hz) | 1.34 +-0.05 | 1.16 +-0.01 | OUT |  |  | 1.31 +-0.13 | in |  |  |  |  |
| BS | Doppler shape w90/w50 | 2.65 +-0.13 | 2.32 +-0.11 | in |  |  | 4.14 +-0.08 | OUT |  |  |  |  |
| BS | coherence time (ms) | 121 +-5 | 129 +-1 | in |  |  | 116 +-11 | in |  |  |  |  |
| BS | tail decay time (ms) | 0.741 +-0.114 | 0.216 +-0.000 | OUT | 0.468 +-0.005 | OUT | 0.455 +-0.009 | OUT | 0.216 +-0.000 | OUT |  |  |
| BS | tail mean excess delay (ms) | 0.397 +-0.013 | 0.228 +-0.001 | OUT | 0.334 +-0.006 | in | 0.329 +-0.002 | OUT |  |  |  |  |
| BS | closed forms | | grazing 27.4 deg, Gamma^2 12.50, coherent 0.000 | | Doppler rms 1.83 Hz | | Lorentzian half width 1.83 Hz | |
| BSB | delay after direct (ms) | 7.142 +-0.028 | 7.104 +-0.000 | in |  |  |  |  |  |  |  |  |
| BSB | level re whole channel (dB) | -25.7 +-0.2 | -20.1 +-0.1 | OUT |  |  |  |  |  |  |  |  |
| BSB | the same with bubble loss | -25.7 +-0.2 | -20.2 | OUT |  |  |  |  |  |  |  |  |
| BSB | coherent share, arrival and tail | 0.009 +-0.003 | 0.004 +-0.004 | in | 0.002 +-0.002 | in |  |  | 0.004 +-0.004 | in |  |  |
| BSB | coherent share, at the peak | 0.004 +-0.002 | 0.001 +-0.001 | in | 0.001 +-0.001 | in |  |  | 0.001 +-0.001 | in |  |  |
| BSB | scintillation index | 1.68 +-0.23 | 0.66 +-0.02 | OUT | 0.57 +-0.02 | OUT |  |  | 0.66 +-0.02 | OUT | 1.70 +-0.10 | in |
| BSB | Doppler rms width (Hz) | 1.99 +-0.10 | 2.14 +-0.00 | in |  |  | 3.03 +-0.11 | OUT | 2.14 +-0.00 | in |  |  |
| BSB | Doppler 50 % half width (Hz) | 1.20 +-0.08 | 1.38 +-0.03 | in |  |  | 1.64 +-0.01 | OUT |  |  |  |  |
| BSB | Doppler shape w90/w50 | 2.84 +-0.17 | 2.59 +-0.05 | in |  |  | 4.47 +-0.36 | OUT |  |  |  |  |
| BSB | coherence time (ms) | 108 +-10 | 104 +-5 | in |  |  | 83 +-3 | in |  |  |  |  |
| BSB | tail decay time (ms) | 0.451 +-0.010 | 0.357 +-0.006 | OUT | 0.580 +-0.017 | OUT | 0.569 +-0.026 | OUT | 0.357 +-0.006 | OUT |  |  |
| BSB | tail mean excess delay (ms) | 0.381 +-0.006 | 0.275 +-0.001 | OUT | 0.363 +-0.010 | in | 0.366 +-0.000 | in |  |  |  |  |
| BSB | closed forms | | grazing 31.6 deg, Gamma^2 16.23, coherent 0.000 | | Doppler rms 2.09 Hz | | Lorentzian half width 2.09 Hz | |

## purple_8: 200 m, wind 3.70 m/s (from the 60 m file of the same recording)
rms height 0.073 m, mean-square slope 0.0220, rms vertical speed 0.239 m/s; estimator bandwidth 10.4 Hz; floor -32.2 dB per delay bin re the whole channel; 4 receivers, 53 s; D, E: rms height 0.102 m; E: tap_shape 1.31

| arrival | quantity | measured | A | | C | | B | | D | | E | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D+B | delay after direct (ms) | 0.000 +-0.000 | 0.000 +-0.000 | in |  |  |  |  |  |  |  |  |
| D+B | level re whole channel (dB) | -5.8 +-0.1 | -9.1 +-0.0 | OUT |  |  |  |  |  |  |  |  |
| D+B | the same with bubble loss | -5.8 +-0.1 | -9.1 | OUT |  |  |  |  |  |  |  |  |
| D+B | coherent share, at the peak | 1.003 +-0.001 | 1.001 +-0.000 | in | 1.004 +-0.000 | in |  |  | 1.002 +-0.000 | in |  |  |
| D+B | scintillation index | 0.00 +-0.00 | 0.00 +-0.00 | in | 0.00 +-0.00 | in |  |  | 0.00 +-0.00 | in | 0.00 +-0.00 | in |
| S | delay after direct (ms) | 1.075 +-0.010 | 0.806 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| S | level re whole channel (dB) | -9.8 +-0.3 | -10.6 +-0.1 | in |  |  |  |  |  |  |  |  |
| S | the same with bubble loss | -9.8 +-0.3 | -10.7 | in |  |  |  |  |  |  |  |  |
| S | coherent share, arrival and tail | 0.211 +-0.026 | 0.677 +-0.011 | OUT | 0.854 +-0.015 | OUT |  |  | 0.392 +-0.005 | fit |  |  |
| S | coherent share, at the peak | 0.229 +-0.029 | 0.886 +-0.006 | OUT | 0.992 +-0.000 | OUT |  |  | 0.582 +-0.010 | fit |  |  |
| S | scintillation index | 0.76 +-0.07 | 0.08 +-0.01 | OUT | 0.00 +-0.00 | OUT |  |  | 0.24 +-0.00 | OUT | 0.74 +-0.14 | in |
| S | Doppler rms width (Hz) | 0.55 +-0.03 | 0.66 +-0.10 | in |  |  | 2.36 +-0.34 | OUT | 0.57 +-0.09 | in |  |  |
| S | Doppler 50 % half width (Hz) | 0.38 +-0.00 | 0.47 +-0.05 | in |  |  | 1.37 +-0.04 | OUT |  |  |  |  |
| S | Doppler shape w90/w50 | 2.20 +-0.10 | 2.36 +-0.21 | in |  |  | 11.93 +-1.71 | OUT |  |  |  |  |
| S | coherence time (ms) | 518 +-35 | 332 +-22 | OUT |  |  | 11 +-0 | OUT |  |  |  |  |
| S | tail decay time (ms) | 1.681 +-0.128 | 0.668 +-0.129 | OUT | - | - | - | - | 1.230 +-0.509 | in |  |  |
| S | tail mean excess delay (ms) | 0.125 +-0.001 | 0.192 +-0.026 | in | 0.249 +-0.027 | OUT | 0.245 +-0.022 | OUT |  |  |  |  |
| S | closed forms | | grazing 6.7 deg, Gamma^2 0.80, coherent 0.449 | | Doppler rms 0.63 Hz | | Lorentzian half width 0.46 Hz | |
| SB+BS | delay after direct (ms) | 1.498 +-0.008 | 1.325 +-0.027 | OUT |  |  |  |  |  |  |  |  |
| SB+BS | level re whole channel (dB) | -10.4 +-0.3 | -10.5 +-0.5 | in |  |  |  |  |  |  |  |  |
| SB+BS | the same with bubble loss | -10.4 +-0.3 | -10.9 | in |  |  |  |  |  |  |  |  |
| SB+BS | coherent share, arrival and tail | 0.118 +-0.021 | 0.344 +-0.013 | OUT | 0.483 +-0.040 | OUT |  |  | 0.061 +-0.037 | in |  |  |
| SB+BS | coherent share, at the peak | 0.158 +-0.031 | 0.637 +-0.011 | OUT | 0.757 +-0.043 | OUT |  |  | 0.110 +-0.101 | in |  |  |
| SB+BS | scintillation index | 0.53 +-0.04 | 0.36 +-0.06 | OUT | 0.26 +-0.03 | OUT |  |  | 0.63 +-0.04 | in | 0.79 +-0.12 | in |
| SB+BS | Doppler rms width (Hz) | 0.65 +-0.02 | 0.71 +-0.02 | in |  |  | 1.08 +-0.18 | OUT | 0.64 +-0.01 | in |  |  |
| SB+BS | Doppler 50 % half width (Hz) | 0.38 +-0.02 | 0.48 +-0.05 | in |  |  | 0.60 +-0.11 | in |  |  |  |  |
| SB+BS | Doppler shape w90/w50 | 2.71 +-0.10 | 2.40 +-0.19 | in |  |  | 4.91 +-1.32 | in |  |  |  |  |
| SB+BS | coherence time (ms) | 466 +-40 | 324 +-9 | OUT |  |  | 280 +-23 | OUT |  |  |  |  |
| SB+BS | tail decay time (ms) | 0.834 +-0.251 | 1.531 +-1.052 | in | 2.435 | OUT | 11.273 +-6.133 | in | 0.792 +-0.416 | in |  |  |
| SB+BS | tail mean excess delay (ms) | 0.143 +-0.001 | 0.323 +-0.006 | OUT | 0.327 +-0.059 | OUT | 0.346 +-0.029 | OUT |  |  |  |  |
| SB+BS | closed forms | | grazing 8.1 deg, Gamma^2 1.18, coherent 0.307 | | Doppler rms 0.68 Hz | | Lorentzian half width 0.56 Hz | |
| BSB | delay after direct (ms) | 1.843 +-0.000 | 2.208 +-0.027 | OUT |  |  |  |  |  |  |  |  |
| BSB | level re whole channel (dB) | -10.6 +-0.3 | -11.9 +-0.2 | OUT |  |  |  |  |  |  |  |  |
| BSB | the same with bubble loss | -10.6 +-0.3 | -12.0 | OUT |  |  |  |  |  |  |  |  |
| BSB | coherent share, arrival and tail | 0.055 +-0.025 | 0.070 +-0.036 | in | 0.103 +-0.022 | in |  |  | 0.009 +-0.000 | in |  |  |
| BSB | coherent share, at the peak | 0.049 +-0.025 | 0.196 +-0.127 | in | 0.372 +-0.034 | OUT |  |  | 0.010 +-0.002 | in |  |  |
| BSB | scintillation index | 0.59 +-0.05 | 0.49 +-0.01 | in | 0.51 +-0.01 | in |  |  | 0.53 +-0.02 | in | 0.99 +-0.06 | OUT |
| BSB | Doppler rms width (Hz) | 0.73 +-0.04 | 0.77 +-0.01 | in |  |  | 1.21 +-0.12 | OUT | 0.71 +-0.06 | in |  |  |
| BSB | Doppler 50 % half width (Hz) | 0.44 +-0.03 | 0.50 +-0.01 | in |  |  | 0.66 +-0.01 | OUT |  |  |  |  |
| BSB | Doppler shape w90/w50 | 2.77 +-0.07 | 2.40 +-0.04 | OUT |  |  | 4.13 +-0.42 | OUT |  |  |  |  |
| BSB | coherence time (ms) | 360 +-18 | 290 +-7 | OUT |  |  | 275 +-18 | OUT |  |  |  |  |
| BSB | tail decay time (ms) | 0.649 +-0.010 | 0.793 +-0.007 | OUT | 2.462 +-0.167 | OUT | 2.134 +-0.199 | OUT | 0.844 +-0.002 | OUT |  |  |
| BSB | tail mean excess delay (ms) | 0.365 +-0.006 | 0.405 +-0.014 | in | 0.504 +-0.004 | OUT | 0.500 +-0.002 | OUT |  |  |  |  |
| BSB | closed forms | | grazing 10.4 deg, Gamma^2 1.91, coherent 0.148 | | Doppler rms 0.78 Hz | | Lorentzian half width 0.72 Hz | |
| SBS | delay after direct (ms) | 5.069 +-0.000 | 4.800 +-0.109 | in |  |  |  |  |  |  |  |  |
| SBS | level re whole channel (dB) | -20.9 +-1.1 | -16.3 +-0.4 | OUT |  |  |  |  |  |  |  |  |
| SBS | the same with bubble loss | -20.9 +-1.1 | -16.1 | OUT |  |  |  |  |  |  |  |  |
| SBS | coherent share, arrival and tail | 0.006 +-0.001 | 0.003 +-0.002 | in | 0.005 +-0.003 | in |  |  | 0.003 +-0.002 | in |  |  |
| SBS | coherent share, at the peak | 0.004 +-0.014 | 0.001 +-0.000 | in | 0.005 +-0.002 | in |  |  | 0.001 +-0.000 | in |  |  |
| SBS | scintillation index | 3.07 +-0.28 | 0.55 +-0.02 | OUT | 0.73 +-0.05 | OUT |  |  | 0.54 +-0.05 | OUT | 1.32 +-0.44 | OUT |
| SBS | Doppler rms width (Hz) | 1.08 +-0.05 | 1.34 +-0.01 | OUT |  |  | 1.74 +-0.18 | OUT | 1.35 +-0.00 | OUT |  |  |
| SBS | Doppler 50 % half width (Hz) | 0.64 +-0.04 | 0.78 +-0.04 | in |  |  | 0.91 +-0.02 | OUT |  |  |  |  |
| SBS | Doppler shape w90/w50 | 2.67 +-0.18 | 2.94 +-0.30 | in |  |  | 5.44 +-1.80 | in |  |  |  |  |
| SBS | coherence time (ms) | 194 +-10 | 163 +-2 | OUT |  |  | 152 +-20 | in |  |  |  |  |
| SBS | tail decay time (ms) | 0.977 +-0.148 | 1.350 +-0.242 | in | 6.805 +-0.738 | OUT | 2.026 +-0.034 | OUT | 1.365 +-0.217 | in |  |  |
| SBS | tail mean excess delay (ms) | 0.470 +-0.011 | 0.458 +-0.013 | in | 0.518 +-0.002 | in | 0.464 +-0.000 | in |  |  |  |  |

## purple_10: 1000 m, wind 3.70 m/s (from the 60 m file of the same recording)
rms height 0.073 m, mean-square slope 0.0220, rms vertical speed 0.239 m/s; estimator bandwidth 10.4 Hz; floor -34.5 dB per delay bin re the whole channel; 4 receivers, 53 s; D, E: rms height 0.102 m; E: tap_shape 1.31

| arrival | quantity | measured | A | | C | | B | | D | | E | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D+B+S+SB+BS+BSB | delay after direct (ms) | 0.000 +-0.000 | 0.000 +-0.000 | in |  |  |  |  |  |  |  |  |
| D+B+S+SB+BS+BSB | level re whole channel (dB) | -5.6 +-0.2 | -6.4 +-0.0 | in |  |  |  |  |  |  |  |  |
| D+B+S+SB+BS+BSB | the same with bubble loss | -5.6 +-0.2 | -6.3 | in |  |  |  |  |  |  |  |  |
| D+B+S+SB+BS+BSB | coherent share, arrival and tail | 0.839 +-0.016 | 1.067 +-0.001 | OUT | 1.066 +-0.001 | OUT |  |  | 1.079 +-0.000 | OUT |  |  |
| D+B+S+SB+BS+BSB | coherent share, at the peak | 0.931 +-0.007 | 1.038 +-0.000 | OUT | 1.033 +-0.000 | OUT |  |  | 1.047 +-0.001 | OUT |  |  |
| D+B+S+SB+BS+BSB | scintillation index | 0.11 +-0.01 | 0.00 +-0.00 | OUT | 0.00 +-0.00 | OUT |  |  | 0.00 +-0.00 | OUT | 0.00 +-0.00 | OUT |
| D+B+S+SB+BS+BSB | Doppler rms width (Hz) | 0.36 +-0.01 | 0.00 +-0.00 | OUT |  |  | 0.00 +-0.00 | OUT | 0.00 +-0.00 | OUT |  |  |
| D+B+S+SB+BS+BSB | Doppler 50 % half width (Hz) | 0.29 +-0.00 | 0.00 +-0.00 | OUT |  |  | 0.00 +-0.00 | OUT |  |  |  |  |
| D+B+S+SB+BS+BSB | Doppler shape w90/w50 | 1.76 +-0.07 | 0.00 +-0.00 | OUT |  |  | 0.00 +-0.00 | OUT |  |  |  |  |
| D+B+S+SB+BS+BSB | coherence time (ms) | 1350 +-90 | 10 +-0 | OUT |  |  | 10 +-0 | OUT |  |  |  |  |
| D+B+S+SB+BS+BSB | tail decay time (ms) | 0.483 +-0.147 | - | - | - | - | - | - | - | - |  |  |
| D+B+S+SB+BS+BSB | tail mean excess delay (ms) | 0.387 +-0.023 | 0.000 +-0.000 | OUT | 0.000 +-0.000 | OUT | 0.000 +-0.000 | OUT |  |  |  |  |
| D+B+S+SB+BS+BSB | closed forms | | grazing 1.3 deg, Gamma^2 0.03, coherent 0.968 | | Doppler rms 0.52 Hz | | Lorentzian half width 0.09 Hz | |
| SBS+SBSB+BSBS+BSBSB | delay after direct (ms) | 1.037 +-0.016 | 0.922 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| SBS+SBSB+BSBS+BSBSB | level re whole channel (dB) | -10.6 +-0.3 | -9.4 +-0.1 | in |  |  |  |  |  |  |  |  |
| SBS+SBSB+BSBS+BSBSB | the same with bubble loss | -10.6 +-0.3 | -9.5 | in |  |  |  |  |  |  |  |  |
| SBS+SBSB+BSBS+BSBSB | coherent share, arrival and tail | 0.382 +-0.043 | 0.990 +-0.000 | OUT | 1.118 +-0.004 | OUT |  |  | 0.856 +-0.003 | OUT |  |  |
| SBS+SBSB+BSBS+BSBSB | coherent share, at the peak | 0.549 +-0.083 | 1.053 +-0.001 | OUT | 1.056 +-0.000 | OUT |  |  | 1.061 +-0.001 | OUT |  |  |
| SBS+SBSB+BSBS+BSBSB | scintillation index | 0.39 +-0.03 | 0.00 +-0.00 | OUT | 0.00 +-0.00 | OUT |  |  | 0.00 +-0.00 | OUT | 0.00 +-0.00 | OUT |
| SBS+SBSB+BSBS+BSBSB | Doppler rms width (Hz) | 0.48 +-0.02 | 0.07 +-0.10 | OUT |  |  | 0.00 +-0.00 | OUT | 0.03 +-0.05 | OUT |  |  |
| SBS+SBSB+BSBS+BSBSB | Doppler 50 % half width (Hz) | 0.33 +-0.01 | 0.38 +-0.00 | in |  |  | 0.33 +-0.01 | in |  |  |  |  |
| SBS+SBSB+BSBS+BSBSB | Doppler shape w90/w50 | 1.99 +-0.06 | 2.05 +-0.01 | in |  |  | 1.86 +-0.49 | in |  |  |  |  |
| SBS+SBSB+BSBS+BSBSB | coherence time (ms) | 809 +-85 | 253 +-6 | OUT |  |  | 52 +-25 | OUT |  |  |  |  |
| SBS+SBSB+BSBS+BSBSB | tail decay time (ms) | 0.532 +-0.036 | - | - | - | - | - | - | - | - |  |  |
| SBS+SBSB+BSBS+BSBSB | tail mean excess delay (ms) | 0.350 +-0.014 | 0.639 +-0.026 | OUT | 0.838 +-0.007 | OUT | 0.000 +-0.000 | OUT |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | delay after direct (ms) | 2.650 +-0.055 | 2.534 +-0.000 | in |  |  |  |  |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | level re whole channel (dB) | -17.9 +-0.2 | -14.7 +-0.6 | OUT |  |  |  |  |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | the same with bubble loss | -17.9 +-0.2 | -15.2 | OUT |  |  |  |  |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | coherent share, arrival and tail | 0.116 +-0.012 | 0.786 +-0.034 | OUT | 1.124 +-0.007 | OUT |  |  | 0.414 +-0.003 | OUT |  |  |
| SBSBS+SBSBSB+BSBSBS | coherent share, at the peak | 0.144 +-0.025 | 0.914 +-0.099 | OUT | 1.105 +-0.014 | OUT |  |  | 0.479 +-0.056 | OUT |  |  |
| SBSBS+SBSBSB+BSBSBS | scintillation index | 0.74 +-0.04 | 0.23 +-0.17 | OUT | 0.01 +-0.01 | OUT |  |  | 0.84 +-0.14 | in | 1.25 +-0.50 | in |
| SBSBS+SBSBSB+BSBSBS | Doppler rms width (Hz) | 0.68 +-0.02 | 0.55 +-0.02 | OUT |  |  | 0.71 +-0.17 | in | 0.44 +-0.04 | OUT |  |  |
| SBSBS+SBSBSB+BSBSBS | Doppler 50 % half width (Hz) | 0.38 +-0.02 | 0.47 +-0.05 | in |  |  | 0.39 +-0.01 | in |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | Doppler shape w90/w50 | 2.99 +-0.06 | 2.21 +-0.16 | OUT |  |  | 4.12 +-0.28 | OUT |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | coherence time (ms) | 550 +-59 | 310 +-12 | OUT |  |  | 322 +-61 | OUT |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | tail decay time (ms) | 3.067 +-0.201 | 0.199 +-0.089 | OUT | - | - | - | - | - | - |  |  |
| SBSBS+SBSBSB+BSBSBS | tail mean excess delay (ms) | 0.149 +-0.005 | 0.170 +-0.061 | in | 0.160 +-0.002 | in | 0.049 +-0.010 | OUT |  |  |  |  |
| BSBSBSB | delay after direct (ms) | 2.880 +-0.017 | 3.053 +-0.027 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSB | level re whole channel (dB) | -18.0 +-0.1 | -15.7 +-0.3 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSB | the same with bubble loss | -18.0 +-0.1 | -15.5 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSB | coherent share, arrival and tail | 0.070 +-0.012 | 0.220 +-0.083 | in | 0.524 +-0.073 | OUT |  |  | 0.035 +-0.010 | in |  |  |
| BSBSBSB | coherent share, at the peak | 0.072 +-0.026 | 0.350 +-0.215 | in | 0.826 +-0.042 | OUT |  |  | 0.023 +-0.002 | in |  |  |
| BSBSBSB | scintillation index | 0.68 +-0.06 | 0.80 +-0.11 | in | 0.39 +-0.05 | OUT |  |  | 1.14 +-0.06 | OUT | 1.60 +-0.12 | OUT |
| BSBSBSB | Doppler rms width (Hz) | 0.72 +-0.02 | 0.69 +-0.07 | in |  |  | 0.78 +-0.05 | in | 0.61 +-0.06 | in |  |  |
| BSBSBSB | Doppler 50 % half width (Hz) | 0.41 +-0.01 | 0.53 +-0.04 | in |  |  | 0.53 +-0.03 | in |  |  |  |  |
| BSBSBSB | Doppler shape w90/w50 | 2.81 +-0.06 | 2.11 +-0.05 | OUT |  |  | 2.97 +-0.15 | in |  |  |  |  |
| BSBSBSB | coherence time (ms) | 462 +-7 | 325 +-15 | OUT |  |  | 323 +-44 | OUT |  |  |  |  |
| BSBSBSB | tail decay time (ms) | 1.282 +-0.143 | 0.191 +-0.164 | OUT | - | - | - | - | 0.078 +-0.022 | OUT |  |  |
| BSBSBSB | tail mean excess delay (ms) | 0.450 +-0.011 | 0.364 +-0.040 | in | 0.422 +-0.006 | in | 0.600 +-0.078 | in |  |  |  |  |
| SBSBSBS | delay after direct (ms) | 4.109 +-0.033 | 4.205 +-0.027 | in |  |  |  |  |  |  |  |  |
| SBSBSBS | level re whole channel (dB) | -22.6 +-0.2 | -16.9 +-0.1 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBS | the same with bubble loss | -22.6 +-0.2 | -16.9 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBS | coherent share, arrival and tail | 0.058 +-0.005 | 0.298 +-0.027 | OUT | 0.293 +-0.051 | OUT |  |  | 0.042 +-0.033 | in |  |  |
| SBSBSBS | coherent share, at the peak | 0.075 +-0.007 | 0.234 +-0.006 | OUT | 0.247 +-0.040 | OUT |  |  | 0.026 +-0.024 | in |  |  |
| SBSBSBS | scintillation index | 0.62 +-0.03 | 0.94 +-0.05 | OUT | 1.18 +-0.29 | in |  |  | 1.32 +-0.41 | in | 2.02 +-0.56 | OUT |
| SBSBSBS | Doppler rms width (Hz) | 1.14 +-0.01 | 0.66 +-0.05 | OUT |  |  | 0.92 +-0.09 | in | 0.62 +-0.04 | OUT |  |  |
| SBSBSBS | Doppler 50 % half width (Hz) | 0.70 +-0.03 | 0.49 +-0.03 | OUT |  |  | 0.57 +-0.09 | in |  |  |  |  |
| SBSBSBS | Doppler shape w90/w50 | 2.64 +-0.04 | 2.37 +-0.33 | in |  |  | 3.72 +-0.61 | in |  |  |  |  |
| SBSBSBS | coherence time (ms) | 206 +-6 | 316 +-5 | OUT |  |  | 269 +-2 | OUT |  |  |  |  |
| SBSBSBS | tail decay time (ms) | 1.239 +-0.358 | 0.054 | OUT | - | - | - | - | 0.192 +-0.050 | OUT |  |  |
| SBSBSBS | tail mean excess delay (ms) | 0.224 +-0.015 | 0.130 +-0.013 | OUT | 0.103 +-0.030 | OUT | 0.041 +-0.003 | OUT |  |  |  |  |
| SBSBSBSB+BSBSBSBS | delay after direct (ms) | 4.915 +-0.014 | 4.800 +-0.054 | in |  |  |  |  |  |  |  |  |
| SBSBSBSB+BSBSBSBS | level re whole channel (dB) | -22.8 +-0.2 | -17.3 +-0.2 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSB+BSBSBSBS | the same with bubble loss | -22.8 +-0.2 | -17.3 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSB+BSBSBSBS | coherent share, arrival and tail | 0.032 +-0.006 | 0.168 +-0.014 | OUT | 0.299 +-0.024 | OUT |  |  | 0.028 +-0.011 | in |  |  |
| SBSBSBSB+BSBSBSBS | coherent share, at the peak | 0.040 +-0.008 | 0.150 +-0.024 | OUT | 0.331 +-0.026 | OUT |  |  | 0.022 +-0.003 | in |  |  |
| SBSBSBSB+BSBSBSBS | scintillation index | 0.69 +-0.02 | 1.14 +-0.16 | OUT | 1.82 +-0.10 | OUT |  |  | 0.96 +-0.15 | in | 1.28 +-0.14 | OUT |
| SBSBSBSB+BSBSBSBS | Doppler rms width (Hz) | 1.09 +-0.02 | 0.74 +-0.01 | OUT |  |  | 1.03 +-0.26 | in | 0.63 +-0.01 | OUT |  |  |
| SBSBSBSB+BSBSBSBS | Doppler 50 % half width (Hz) | 0.64 +-0.02 | 0.52 +-0.07 | in |  |  | 0.73 +-0.10 | in |  |  |  |  |
| SBSBSBSB+BSBSBSBS | Doppler shape w90/w50 | 2.80 +-0.02 | 2.41 +-0.26 | in |  |  | 3.14 +-0.05 | OUT |  |  |  |  |
| SBSBSBSB+BSBSBSBS | coherence time (ms) | 229 +-5 | 284 +-15 | OUT |  |  | 257 +-21 | in |  |  |  |  |
| SBSBSBSB+BSBSBSBS | tail decay time (ms) | 2.370 +-1.649 | - | - | 0.265 | in | 0.214 +-0.075 | in | 0.181 +-0.065 | in |  |  |
| SBSBSBSB+BSBSBSBS | tail mean excess delay (ms) | 0.153 +-0.002 | 0.111 +-0.026 | in | 0.218 +-0.025 | in | 0.091 +-0.032 | in |  |  |  |  |
| BSBSBSBSB | delay after direct (ms) | 5.222 +-0.030 | 5.376 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSBSB | level re whole channel (dB) | -22.7 +-0.1 | -17.9 +-0.0 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSBSB | the same with bubble loss | -22.7 +-0.1 | -18.0 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSBSB | coherent share, arrival and tail | 0.034 +-0.004 | 0.020 +-0.003 | in | 0.036 +-0.023 | in |  |  | 0.022 +-0.007 | in |  |  |
| BSBSBSBSB | coherent share, at the peak | 0.027 +-0.010 | 0.031 +-0.006 | in | 0.093 +-0.049 | in |  |  | 0.024 +-0.002 | in |  |  |
| BSBSBSBSB | scintillation index | 0.71 +-0.02 | 1.16 +-0.08 | OUT | 1.41 +-0.32 | OUT |  |  | 1.01 +-0.04 | OUT | 1.59 +-0.17 | OUT |
| BSBSBSBSB | Doppler rms width (Hz) | 1.08 +-0.02 | 0.76 +-0.02 | OUT |  |  | 0.93 +-0.03 | OUT | 0.64 +-0.01 | OUT |  |  |
| BSBSBSBSB | Doppler 50 % half width (Hz) | 0.63 +-0.02 | 0.62 +-0.02 | in |  |  | 0.52 +-0.02 | OUT |  |  |  |  |
| BSBSBSBSB | Doppler shape w90/w50 | 2.81 +-0.12 | 2.07 +-0.17 | OUT |  |  | 4.51 +-0.36 | OUT |  |  |  |  |
| BSBSBSBSB | coherence time (ms) | 233 +-7 | 252 +-12 | in |  |  | 237 +-2 | in |  |  |  |  |
| BSBSBSBSB | tail decay time (ms) | 2.593 +-0.440 | 4.055 | OUT | - | - | - | - | 0.143 +-0.044 | OUT |  |  |
| BSBSBSBSB | tail mean excess delay (ms) | 0.487 +-0.008 | 0.483 +-0.017 | in | 0.509 +-0.000 | in | 0.552 +-0.047 | in |  |  |  |  |
| SBSBSBSBS | delay after direct (ms) | 6.605 +-0.044 | 6.912 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSBS | level re whole channel (dB) | -24.6 +-0.3 | -17.5 +-0.0 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSBS | the same with bubble loss | -24.6 +-0.3 | -17.6 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSBS | coherent share, arrival and tail | 0.028 +-0.003 | 0.032 +-0.015 | in | 0.024 +-0.004 | in |  |  | 0.020 +-0.001 | in |  |  |
| SBSBSBSBS | coherent share, at the peak | 0.033 +-0.006 | 0.023 +-0.009 | in | 0.009 +-0.000 | in |  |  | 0.015 +-0.010 | in |  |  |
| SBSBSBSBS | scintillation index | 0.71 +-0.05 | 0.93 +-0.06 | OUT | 1.29 +-0.17 | OUT |  |  | 1.33 +-0.00 | OUT | 1.74 +-0.32 | OUT |
| SBSBSBSBS | Doppler rms width (Hz) | 1.38 +-0.01 | 0.81 +-0.02 | OUT |  |  | 1.19 +-0.35 | in | 0.74 +-0.03 | OUT |  |  |
| SBSBSBSBS | Doppler 50 % half width (Hz) | 0.85 +-0.01 | 0.52 +-0.02 | OUT |  |  | 0.67 +-0.13 | in |  |  |  |  |
| SBSBSBSBS | Doppler shape w90/w50 | 2.63 +-0.06 | 2.60 +-0.14 | in |  |  | 4.32 +-0.84 | in |  |  |  |  |
| SBSBSBSBS | coherence time (ms) | 168 +-3 | 253 +-9 | OUT |  |  | 220 +-39 | in |  |  |  |  |
| SBSBSBSBS | tail decay time (ms) | 2.142 +-0.519 | 0.444 +-0.160 | OUT | - | - | 0.316 +-0.016 | OUT | 0.113 | OUT |  |  |
| SBSBSBSBS | tail mean excess delay (ms) | 0.146 +-0.012 | 0.072 +-0.002 | OUT | 0.029 +-0.003 | OUT | 0.106 +-0.019 | in |  |  |  |  |
| SBSBSBSBSB | delay after direct (ms) | 7.142 +-0.036 | 7.296 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSBSB | level re whole channel (dB) | -26.0 +-0.2 | -18.5 +-0.2 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSBSB | the same with bubble loss | -26.0 +-0.2 | -18.7 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSBSB | coherent share, arrival and tail | 0.024 +-0.006 | 0.033 +-0.013 | in | 0.051 +-0.030 | in |  |  | 0.020 +-0.003 | in |  |  |
| SBSBSBSBSB | coherent share, at the peak | 0.038 +-0.011 | 0.025 +-0.010 | in | 0.019 +-0.008 | in |  |  | 0.016 +-0.000 | in |  |  |
| SBSBSBSBSB | scintillation index | 0.82 +-0.06 | 1.23 +-0.24 | in | 1.52 +-0.26 | OUT |  |  | 1.13 +-0.05 | OUT | 1.87 +-0.47 | OUT |
| SBSBSBSBSB | Doppler rms width (Hz) | 1.48 +-0.03 | 0.77 +-0.01 | OUT |  |  | 1.08 +-0.21 | in | 0.77 +-0.01 | OUT |  |  |
| SBSBSBSBSB | Doppler 50 % half width (Hz) | 0.95 +-0.03 | 0.53 +-0.00 | OUT |  |  | 0.64 +-0.18 | in |  |  |  |  |
| SBSBSBSBSB | Doppler shape w90/w50 | 2.57 +-0.05 | 2.48 +-0.09 | in |  |  | 4.58 +-0.77 | OUT |  |  |  |  |
| SBSBSBSBSB | coherence time (ms) | 147 +-5 | 258 +-15 | OUT |  |  | 234 +-45 | in |  |  |  |  |
| SBSBSBSBSB | tail decay time (ms) | 1.540 +-0.201 | 0.813 +-0.508 | in | - | - | - | - | 0.222 +-0.010 | OUT |  |  |
| SBSBSBSBSB | tail mean excess delay (ms) | 0.146 +-0.002 | 0.075 +-0.046 | in | 0.000 +-0.000 | OUT | 0.118 +-0.024 | in |  |  |  |  |

## purple_11: 60 m, wind 9.14 m/s (fitted here)
rms height 0.447 m, mean-square slope 0.0498, rms vertical speed 0.591 m/s; estimator bandwidth 10.4 Hz; floor -32.9 dB per delay bin re the whole channel; 4 receivers, 53 s; D, E: rms height 0.145 m; E: tap_shape 0.37

| arrival | quantity | measured | A | | C | | B | | D | | E | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D+B | delay after direct (ms) | 0.000 +-0.000 | 0.000 +-0.000 | in |  |  |  |  |  |  |  |  |
| D+B | level re whole channel (dB) | -8.6 +-0.1 | -6.0 +-0.0 | OUT |  |  |  |  |  |  |  |  |
| D+B | the same with bubble loss | -8.6 +-0.1 | -5.6 | OUT |  |  |  |  |  |  |  |  |
| D+B | coherent share, at the peak | 0.141 +-0.047 | 1.002 +-0.000 | OUT | 1.003 +-0.000 | OUT |  |  | 1.002 +-0.000 | OUT |  |  |
| D+B | scintillation index | 1.21 +-0.04 | 0.00 +-0.00 | OUT | 0.00 +-0.00 | OUT |  |  | 0.00 +-0.00 | OUT | 0.00 +-0.00 | OUT |
| S | delay after direct (ms) | 3.149 +-0.104 | 2.957 +-0.000 | in |  |  |  |  |  |  |  |  |
| S | level re whole channel (dB) | -17.4 +-0.8 | -12.4 +-0.0 | OUT |  |  |  |  |  |  |  |  |
| S | the same with bubble loss | -17.4 +-0.8 | -12.9 | OUT |  |  |  |  |  |  |  |  |
| S | coherent share, arrival and tail | 0.004 +-0.001 | 0.002 +-0.000 | in | 0.002 +-0.001 | in |  |  | 0.002 +-0.000 | in |  |  |
| S | coherent share, at the peak | 0.003 +-0.003 | 0.000 +-0.000 | in | 0.000 +-0.000 | in |  |  | 0.000 +-0.000 | in |  |  |
| S | scintillation index | 4.81 +-1.00 | 0.58 +-0.01 | OUT | 0.66 +-0.01 | OUT |  |  | 0.58 +-0.01 | OUT | 4.70 +-2.81 | fit |
| S | Doppler rms width (Hz) FIT | 3.53 +-0.37 | 3.33 +-0.08 | fit |  |  | 4.04 +-0.83 | fit | 3.33 +-0.08 | fit |  |  |
| S | Doppler 50 % half width (Hz) | 2.74 +-0.41 | 2.17 +-0.02 | in |  |  | 2.14 +-0.38 | in |  |  |  |  |
| S | Doppler shape w90/w50 | 2.05 +-0.05 | 2.55 +-0.10 | OUT |  |  | 5.02 +-1.83 | in |  |  |  |  |
| S | coherence time (ms) | 57 +-10 | 72 +-5 | in |  |  | 66 +-20 | in |  |  |  |  |
| S | tail decay time (ms) | - | 0.614 +-0.006 | - | 1.236 +-0.091 | - | 1.171 +-0.106 | - | 0.614 +-0.006 | - |  |  |
| S | tail mean excess delay (ms) | 0.423 +-0.033 | 0.354 +-0.000 | in | 0.449 +-0.004 | in | 0.443 +-0.005 | in |  |  |  |  |
| S | closed forms | | grazing 21.0 deg, Gamma^2 282.94, coherent 0.000 | | Doppler rms 3.53 Hz | | Lorentzian half width 3.53 Hz | |
| SB | delay after direct (ms) | 4.070 +-0.083 | 4.570 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| SB | level re whole channel (dB) | -14.4 +-0.7 | -13.2 +-0.2 | in |  |  |  |  |  |  |  |  |
| SB | the same with bubble loss | -14.4 +-0.7 | -13.7 | in |  |  |  |  |  |  |  |  |
| SB | coherent share, arrival and tail | 0.003 +-0.002 | 0.001 +-0.000 | in | 0.001 +-0.001 | in |  |  | 0.001 +-0.000 | in |  |  |
| SB | coherent share, at the peak | 0.004 +-0.004 | 0.001 +-0.001 | in | 0.001 +-0.001 | in |  |  | 0.001 +-0.001 | in |  |  |
| SB | scintillation index | 3.07 +-0.66 | 0.61 +-0.03 | OUT | 0.61 +-0.02 | OUT |  |  | 0.61 +-0.03 | OUT | 2.83 +-0.27 | in |
| SB | Doppler rms width (Hz) | 3.24 +-0.48 | 3.65 +-0.12 | in |  |  | 4.81 +-0.84 | in | 3.65 +-0.12 | in |  |  |
| SB | Doppler 50 % half width (Hz) | 1.91 +-0.64 | 2.35 +-0.09 | in |  |  | 2.73 +-0.63 | in |  |  |  |  |
| SB | Doppler shape w90/w50 | 2.89 +-0.17 | 2.55 +-0.23 | in |  |  | 3.37 +-0.64 | in |  |  |  |  |
| SB | coherence time (ms) | 75 +-6 | 63 +-1 | in |  |  | 54 +-6 | OUT |  |  |  |  |
| SB | tail decay time (ms) | 0.591 +-2.950 | 0.608 +-0.011 | in | 1.625 +-0.457 | in | 1.608 +-0.015 | in | 0.608 +-0.011 | in |  |  |
| SB | tail mean excess delay (ms) | 0.235 +-0.041 | 0.160 +-0.002 | in | 0.182 +-0.002 | in | 0.180 +-0.004 | in |  |  |  |  |
| SB | closed forms | | grazing 25.7 deg, Gamma^2 414.11, coherent 0.000 | | Doppler rms 4.27 Hz | | Lorentzian half width 4.27 Hz | |
| BS | delay after direct (ms) | 4.877 +-0.081 | 5.146 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| BS | level re whole channel (dB) | -18.1 +-0.5 | -12.6 +-0.1 | OUT |  |  |  |  |  |  |  |  |
| BS | the same with bubble loss | -18.1 +-0.5 | -13.0 | OUT |  |  |  |  |  |  |  |  |
| BS | coherent share, arrival and tail | 0.003 +-0.002 | 0.003 +-0.004 | in | 0.005 +-0.005 | in |  |  | 0.003 +-0.004 | in |  |  |
| BS | coherent share, at the peak | 0.003 +-0.002 | 0.003 +-0.005 | in | 0.004 +-0.002 | in |  |  | 0.003 +-0.005 | in |  |  |
| BS | scintillation index | 2.21 +-0.05 | 0.57 +-0.00 | OUT | 0.60 +-0.04 | OUT |  |  | 0.57 +-0.00 | OUT | 2.73 +-1.01 | in |
| BS | Doppler rms width (Hz) | 4.05 +-0.60 | 3.76 +-0.04 | in |  |  | 4.58 +-0.41 | in | 3.76 +-0.04 | in |  |  |
| BS | Doppler 50 % half width (Hz) | 2.41 +-0.48 | 2.58 +-0.01 | in |  |  | 2.47 +-0.21 | in |  |  |  |  |
| BS | Doppler shape w90/w50 | 2.73 +-0.13 | 2.36 +-0.09 | in |  |  | 3.96 +-0.46 | OUT |  |  |  |  |
| BS | coherence time (ms) | 56 +-5 | 59 +-2 | in |  |  | 59 +-4 | in |  |  |  |  |
| BS | tail decay time (ms) | 7.659 +-0.637 | 0.663 +-0.016 | OUT | 1.157 +-0.009 | OUT | 1.134 +-0.018 | OUT | 0.663 +-0.016 | OUT |  |  |
| BS | tail mean excess delay (ms) | 0.517 +-0.022 | 0.374 +-0.002 | OUT | 0.442 +-0.000 | in | 0.444 +-0.002 | in |  |  |  |  |
| BS | closed forms | | grazing 27.4 deg, Gamma^2 464.45, coherent 0.000 | | Doppler rms 4.53 Hz | | Lorentzian half width 4.53 Hz | |
| BSB | delay after direct (ms) | 6.605 +-0.010 | 7.123 +-0.027 | OUT |  |  |  |  |  |  |  |  |
| BSB | level re whole channel (dB) | -21.8 +-0.5 | -21.1 +-0.3 | in |  |  |  |  |  |  |  |  |
| BSB | the same with bubble loss | -21.8 +-0.5 | -21.6 | in |  |  |  |  |  |  |  |  |
| BSB | coherent share, arrival and tail | 0.004 +-0.003 | 0.003 +-0.002 | in | 0.002 +-0.000 | in |  |  | 0.003 +-0.002 | in |  |  |
| BSB | coherent share, at the peak | 0.002 +-0.003 | 0.004 +-0.002 | in | 0.000 +-0.000 | in |  |  | 0.004 +-0.002 | in |  |  |
| BSB | scintillation index | 2.30 +-0.27 | 0.58 +-0.02 | OUT | 0.55 +-0.01 | OUT |  |  | 0.58 +-0.02 | OUT | 3.31 +-0.68 | in |
| BSB | Doppler rms width (Hz) | 3.91 +-0.30 | 4.13 +-0.13 | in |  |  | 5.25 +-0.31 | OUT | 4.13 +-0.13 | in |  |  |
| BSB | Doppler 50 % half width (Hz) | 2.39 +-0.42 | 2.84 +-0.16 | in |  |  | 3.07 +-0.04 | in |  |  |  |  |
| BSB | Doppler shape w90/w50 | 2.76 +-0.08 | 2.48 +-0.02 | in |  |  | 3.34 +-0.20 | in |  |  |  |  |
| BSB | coherence time (ms) | 58 +-4 | 48 +-3 | in |  |  | 47 +-3 | in |  |  |  |  |
| BSB | tail decay time (ms) | 0.529 +-0.859 | 0.523 +-0.026 | in | 3.013 +-0.676 | OUT | 3.507 +-1.113 | OUT | 0.523 +-0.026 | in |  |  |
| BSB | tail mean excess delay (ms) | 0.333 +-0.029 | 0.330 +-0.003 | in | 0.496 +-0.005 | OUT | 0.500 +-0.009 | OUT |  |  |  |  |
| BSB | closed forms | | grazing 31.6 deg, Gamma^2 603.22, coherent 0.000 | | Doppler rms 5.16 Hz | | Lorentzian half width 5.16 Hz | |

## purple_13: 200 m, wind 9.14 m/s (from the 60 m file of the same recording)
rms height 0.447 m, mean-square slope 0.0498, rms vertical speed 0.591 m/s; estimator bandwidth 10.4 Hz; floor -31.9 dB per delay bin re the whole channel; 4 receivers, 53 s; D, E: rms height 0.145 m; E: tap_shape 0.37

| arrival | quantity | measured | A | | C | | B | | D | | E | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D+B | delay after direct (ms) | 0.000 +-0.000 | 0.000 +-0.000 | in |  |  |  |  |  |  |  |  |
| D+B | level re whole channel (dB) | -5.5 +-0.1 | -8.8 +-0.0 | OUT |  |  |  |  |  |  |  |  |
| D+B | the same with bubble loss | -5.5 +-0.1 | -7.0 | OUT |  |  |  |  |  |  |  |  |
| D+B | coherent share, at the peak | 1.005 +-0.000 | 1.007 +-0.000 | in | 1.025 +-0.000 | in |  |  | 1.007 +-0.000 | in |  |  |
| D+B | scintillation index | 0.00 +-0.00 | 0.00 +-0.00 | in | 0.00 +-0.00 | in |  |  | 0.00 +-0.00 | in | 0.00 +-0.00 | in |
| S | delay after direct (ms) | 1.152 +-0.010 | 1.075 +-0.000 | in |  |  |  |  |  |  |  |  |
| S | level re whole channel (dB) | -11.9 +-0.5 | -14.5 +-0.2 | OUT |  |  |  |  |  |  |  |  |
| S | the same with bubble loss | -11.9 +-0.5 | -15.6 | OUT |  |  |  |  |  |  |  |  |
| S | coherent share, arrival and tail | 0.045 +-0.013 | 0.002 +-0.002 | in | 0.002 +-0.000 | in |  |  | 0.036 +-0.001 | fit |  |  |
| S | coherent share, at the peak | 0.044 +-0.012 | 0.001 +-0.000 | in | 0.001 +-0.000 | in |  |  | 0.010 +-0.002 | fit |  |  |
| S | scintillation index | 0.89 +-0.12 | 0.91 +-0.02 | in | 1.09 +-0.02 | in |  |  | 0.89 +-0.02 | in | 5.73 +-0.05 | OUT |
| S | Doppler rms width (Hz) | 1.02 +-0.12 | 1.09 +-0.13 | in |  |  | 1.85 +-0.68 | in | 1.11 +-0.15 | in |  |  |
| S | Doppler 50 % half width (Hz) | 0.71 +-0.12 | 0.70 +-0.09 | in |  |  | 1.08 +-0.45 | in |  |  |  |  |
| S | Doppler shape w90/w50 | 2.24 +-0.13 | 2.58 +-0.59 | in |  |  | 4.98 +-1.77 | in |  |  |  |  |
| S | coherence time (ms) | 242 +-43 | 205 +-8 | in |  |  | 168 +-39 | in |  |  |  |  |
| S | tail decay time (ms) | 1.083 +-0.976 | - | - | - | - | - | - | - | - |  |  |
| S | tail mean excess delay (ms) | 0.154 +-0.008 | 0.197 +-0.003 | in | 0.037 +-0.001 | OUT | 0.037 +-0.001 | OUT |  |  |  |  |
| S | closed forms | | grazing 6.7 deg, Gamma^2 29.78, coherent 0.000 | | Doppler rms 1.15 Hz | | Lorentzian half width 1.15 Hz | |
| SB+BS | delay after direct (ms) | 1.574 +-0.076 | 1.651 +-0.000 | in |  |  |  |  |  |  |  |  |
| SB+BS | level re whole channel (dB) | -11.6 +-0.0 | -12.6 +-0.1 | OUT |  |  |  |  |  |  |  |  |
| SB+BS | the same with bubble loss | -11.6 +-0.0 | -13.2 | OUT |  |  |  |  |  |  |  |  |
| SB+BS | coherent share, arrival and tail | 0.041 +-0.013 | 0.014 +-0.011 | in | 0.009 +-0.006 | in |  |  | 0.017 +-0.009 | in |  |  |
| SB+BS | coherent share, at the peak | 0.038 +-0.011 | 0.009 +-0.009 | in | 0.002 +-0.001 | in |  |  | 0.017 +-0.012 | in |  |  |
| SB+BS | scintillation index | 0.83 +-0.02 | 0.68 +-0.01 | OUT | 0.99 +-0.00 | OUT |  |  | 0.67 +-0.02 | OUT | 3.42 +-0.14 | OUT |
| SB+BS | Doppler rms width (Hz) | 1.38 +-0.13 | 1.37 +-0.06 | in |  |  | 2.18 +-0.05 | OUT | 1.38 +-0.04 | in |  |  |
| SB+BS | Doppler 50 % half width (Hz) | 0.87 +-0.10 | 0.91 +-0.01 | in |  |  | 1.21 +-0.18 | in |  |  |  |  |
| SB+BS | Doppler shape w90/w50 | 2.68 +-0.11 | 2.48 +-0.16 | in |  |  | 4.22 +-1.00 | in |  |  |  |  |
| SB+BS | coherence time (ms) | 172 +-19 | 170 +-7 | in |  |  | 142 +-9 | in |  |  |  |  |
| SB+BS | tail decay time (ms) | 1.510 +-0.378 | 2.100 +-0.034 | in | 1.818 +-0.259 | in | 1.411 +-0.007 | in | 1.448 +-0.079 | in |  |  |
| SB+BS | tail mean excess delay (ms) | 0.148 +-0.013 | 0.208 +-0.001 | in | 0.299 +-0.003 | OUT | 0.294 +-0.003 | OUT |  |  |  |  |
| SB+BS | closed forms | | grazing 8.1 deg, Gamma^2 43.85, coherent 0.000 | | Doppler rms 1.39 Hz | | Lorentzian half width 1.39 Hz | |
| BSB | delay after direct (ms) | 1.843 +-0.028 | 2.419 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| BSB | level re whole channel (dB) | -11.2 +-0.2 | -12.0 +-0.2 | in |  |  |  |  |  |  |  |  |
| BSB | the same with bubble loss | -11.2 +-0.2 | -12.4 | OUT |  |  |  |  |  |  |  |  |
| BSB | coherent share, arrival and tail | 0.013 +-0.004 | 0.004 +-0.001 | in | 0.006 +-0.001 | in |  |  | 0.004 +-0.001 | in |  |  |
| BSB | coherent share, at the peak | 0.022 +-0.008 | 0.003 +-0.002 | in | 0.006 +-0.000 | in |  |  | 0.003 +-0.002 | in |  |  |
| BSB | scintillation index | 0.81 +-0.07 | 0.78 +-0.08 | in | 0.83 +-0.01 | in |  |  | 0.82 +-0.09 | in | 2.21 +-0.63 | OUT |
| BSB | Doppler rms width (Hz) | 1.57 +-0.17 | 1.54 +-0.16 | in |  |  | 2.31 +-0.24 | OUT | 1.58 +-0.14 | in |  |  |
| BSB | Doppler 50 % half width (Hz) | 1.01 +-0.14 | 1.01 +-0.09 | in |  |  | 1.24 +-0.14 | in |  |  |  |  |
| BSB | Doppler shape w90/w50 | 2.57 +-0.16 | 2.40 +-0.03 | in |  |  | 4.35 +-0.56 | OUT |  |  |  |  |
| BSB | coherence time (ms) | 149 +-11 | 145 +-9 | in |  |  | 119 +-14 | in |  |  |  |  |
| BSB | tail decay time (ms) | 0.881 +-0.104 | 1.790 +-0.049 | OUT | 15.824 | OUT | 48.862 | OUT | 2.010 +-0.057 | OUT |  |  |
| BSB | tail mean excess delay (ms) | 0.419 +-0.013 | 0.452 +-0.007 | in | 0.478 +-0.002 | in | 0.489 +-0.008 | OUT |  |  |  |  |
| BSB | closed forms | | grazing 10.4 deg, Gamma^2 70.97, coherent 0.000 | | Doppler rms 1.77 Hz | | Lorentzian half width 1.77 Hz | |

## purple_15: 1000 m, wind 9.14 m/s (from the 60 m file of the same recording)
rms height 0.447 m, mean-square slope 0.0498, rms vertical speed 0.591 m/s; estimator bandwidth 10.4 Hz; floor -32.6 dB per delay bin re the whole channel; 4 receivers, 53 s; D, E: rms height 0.145 m; E: tap_shape 0.37

| arrival | quantity | measured | A | | C | | B | | D | | E | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D+B+S+SB+BS+BSB | delay after direct (ms) | 0.000 +-0.000 | 0.000 +-0.000 | in |  |  |  |  |  |  |  |  |
| D+B+S+SB+BS+BSB | level re whole channel (dB) | -4.8 +-0.1 | -4.2 +-0.0 | in |  |  |  |  |  |  |  |  |
| D+B+S+SB+BS+BSB | the same with bubble loss | -4.8 +-0.1 | -0.9 | OUT |  |  |  |  |  |  |  |  |
| D+B+S+SB+BS+BSB | coherent share, arrival and tail | 0.664 +-0.028 | 1.443 +-0.004 | OUT | 1.083 +-0.014 | OUT |  |  | 1.095 +-0.000 | OUT |  |  |
| D+B+S+SB+BS+BSB | coherent share, at the peak | 0.987 +-0.002 | 1.084 +-0.001 | OUT | 1.028 +-0.000 | OUT |  |  | 1.042 +-0.000 | OUT |  |  |
| D+B+S+SB+BS+BSB | scintillation index | 0.01 +-0.00 | 0.00 +-0.00 | in | 0.00 +-0.00 | in |  |  | 0.00 +-0.00 | in | 0.00 +-0.00 | in |
| D+B+S+SB+BS+BSB | Doppler rms width (Hz) | 0.38 +-0.07 | 0.00 +-0.00 | OUT |  |  | 0.00 +-0.00 | OUT | 0.00 +-0.00 | OUT |  |  |
| D+B+S+SB+BS+BSB | Doppler 50 % half width (Hz) | 0.29 +-0.02 | 0.00 +-0.00 | OUT |  |  | 0.00 +-0.00 | OUT |  |  |  |  |
| D+B+S+SB+BS+BSB | Doppler shape w90/w50 | 2.14 +-0.12 | 0.00 +-0.00 | OUT |  |  | 0.00 +-0.00 | OUT |  |  |  |  |
| D+B+S+SB+BS+BSB | coherence time (ms) | 1767 +-132 | 10 +-0 | OUT |  |  | 10 +-0 | OUT |  |  |  |  |
| D+B+S+SB+BS+BSB | tail decay time (ms) | 1.584 +-0.408 | - | - | - | - | - | - | - | - |  |  |
| D+B+S+SB+BS+BSB | tail mean excess delay (ms) | 0.494 +-0.009 | 0.000 +-0.000 | OUT | 0.000 +-0.000 | OUT | 0.000 +-0.000 | OUT |  |  |  |  |
| D+B+S+SB+BS+BSB | closed forms | | grazing 1.3 deg, Gamma^2 1.22, coherent 0.296 | | Doppler rms 0.28 Hz | | Lorentzian half width 0.23 Hz | |
| SBS+SBSB+BSBS+BSBSB | delay after direct (ms) | 1.613 +-0.021 | 1.421 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| SBS+SBSB+BSBS+BSBSB | level re whole channel (dB) | -14.5 +-0.5 | -8.9 +-0.8 | OUT |  |  |  |  |  |  |  |  |
| SBS+SBSB+BSBS+BSBSB | the same with bubble loss | -14.5 +-0.5 | -14.8 | in |  |  |  |  |  |  |  |  |
| SBS+SBSB+BSBS+BSBSB | coherent share, arrival and tail | 0.033 +-0.004 | 0.030 +-0.000 | in | - | - |  |  | 1.208 +-0.009 | OUT |  |  |
| SBS+SBSB+BSBS+BSBSB | coherent share, at the peak | 0.038 +-0.011 | 0.019 +-0.007 | in | - | - |  |  | 1.262 +-0.003 | OUT |  |  |
| SBS+SBSB+BSBS+BSBSB | scintillation index | 0.75 +-0.06 | 1.16 +-0.01 | OUT | - | - |  |  | 0.00 +-0.00 | OUT | 0.00 +-0.00 | OUT |
| SBS+SBSB+BSBS+BSBSB | Doppler rms width (Hz) | 0.91 +-0.05 | 0.37 +-0.04 | OUT |  |  | - | - | 0.00 +-0.00 | OUT |  |  |
| SBS+SBSB+BSBS+BSBSB | Doppler 50 % half width (Hz) | 0.51 +-0.02 | 0.35 +-0.05 | in |  |  | - | - |  |  |  |  |
| SBS+SBSB+BSBS+BSBSB | Doppler shape w90/w50 | 2.96 +-0.28 | 1.66 +-0.08 | OUT |  |  | - | - |  |  |  |  |
| SBS+SBSB+BSBS+BSBSB | coherence time (ms) | 324 +-21 | 737 +-107 | OUT |  |  | - | - |  |  |  |  |
| SBS+SBSB+BSBS+BSBSB | tail decay time (ms) | 2.030 +-0.256 | 0.267 +-0.016 | OUT | - | - | - | - | - | - |  |  |
| SBS+SBSB+BSBS+BSBSB | tail mean excess delay (ms) | 0.229 +-0.009 | 0.092 +-0.011 | OUT | - | - | - | - |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | delay after direct (ms) | 2.304 +-0.021 | 2.573 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | level re whole channel (dB) | -17.1 +-0.4 | -9.8 +-0.6 | OUT |  |  |  |  |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | the same with bubble loss | -17.1 +-0.4 | -16.9 | in |  |  |  |  |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | coherent share, arrival and tail | 0.023 +-0.008 | 0.033 +-0.017 | in | 0.017 +-0.003 | in |  |  | 0.122 +-0.046 | in |  |  |
| SBSBS+SBSBSB+BSBSBS | coherent share, at the peak | 0.026 +-0.012 | 0.016 +-0.008 | in | 0.018 +-0.009 | in |  |  | 0.087 +-0.051 | in |  |  |
| SBSBS+SBSBSB+BSBSBS | scintillation index | 0.76 +-0.04 | 1.01 +-0.03 | OUT | 1.05 +-0.00 | OUT |  |  | 1.35 +-0.01 | OUT | 7.43 +-0.73 | OUT |
| SBSBS+SBSBSB+BSBSBS | Doppler rms width (Hz) | 1.44 +-0.15 | 0.56 +-0.03 | OUT |  |  | 0.75 +-0.08 | OUT | 0.77 +-0.05 | OUT |  |  |
| SBSBS+SBSBSB+BSBSBS | Doppler 50 % half width (Hz) | 0.85 +-0.12 | 0.38 +-0.01 | OUT |  |  | 0.42 +-0.06 | OUT |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | Doppler shape w90/w50 | 2.77 +-0.11 | 2.31 +-0.07 | OUT |  |  | 5.22 +-2.73 | in |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | coherence time (ms) | 181 +-17 | 644 +-104 | OUT |  |  | 490 +-41 | OUT |  |  |  |  |
| SBSBS+SBSBSB+BSBSBS | tail decay time (ms) | 0.822 +-0.283 | 0.179 +-0.044 | OUT | 0.476 +-0.018 | in | 0.520 +-0.040 | in | - | - |  |  |
| SBSBS+SBSBSB+BSBSBS | tail mean excess delay (ms) | 0.229 +-0.024 | 0.074 +-0.001 | OUT | 0.103 +-0.005 | OUT | 0.098 +-0.002 | OUT |  |  |  |  |
| BSBSBSB | delay after direct (ms) | 3.110 +-0.056 | 2.842 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSB | level re whole channel (dB) | -19.9 +-0.2 | -16.5 +-0.7 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSB | the same with bubble loss | -19.9 +-0.2 | -21.5 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSB | coherent share, arrival and tail | 0.012 +-0.005 | 0.190 +-0.008 | OUT | 0.095 | OUT |  |  | 0.036 +-0.018 | in |  |  |
| BSBSBSB | coherent share, at the peak | 0.017 +-0.005 | 0.124 +-0.149 | in | 0.016 +-0.016 | in |  |  | 0.021 +-0.004 | in |  |  |
| BSBSBSB | scintillation index | 0.71 +-0.07 | 1.99 +-0.61 | OUT | 1.77 +-0.17 | OUT |  |  | 20.37 +-17.73 | in | 29.57 +-23.00 | in |
| BSBSBSB | Doppler rms width (Hz) | 1.59 +-0.12 | 0.54 +-0.03 | OUT |  |  | 0.54 +-0.00 | OUT | 0.66 +-0.03 | OUT |  |  |
| BSBSBSB | Doppler 50 % half width (Hz) | 0.89 +-0.08 | 0.38 +-0.00 | OUT |  |  | 0.46 +-0.07 | OUT |  |  |  |  |
| BSBSBSB | Doppler shape w90/w50 | 2.99 +-0.05 | 2.38 +-0.11 | OUT |  |  | 6.27 +-4.30 | in |  |  |  |  |
| BSBSBSB | coherence time (ms) | 145 +-7 | 414 +-12 | OUT |  |  | 366 +-25 | OUT |  |  |  |  |
| BSBSBSB | tail decay time (ms) | - | - | - | - | - | - | - | - | - |  |  |
| BSBSBSB | tail mean excess delay (ms) | 0.412 +-0.043 | 0.742 +-0.022 | OUT | 0.727 +-0.035 | OUT | 0.000 +-0.000 | OUT |  |  |  |  |
| SBSBSBS | delay after direct (ms) | 4.109 +-0.029 | 4.147 +-0.000 | in |  |  |  |  |  |  |  |  |
| SBSBSBS | level re whole channel (dB) | -18.8 +-0.3 | -15.5 +-0.1 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBS | the same with bubble loss | -18.8 +-0.3 | -21.1 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBS | coherent share, arrival and tail | 0.006 +-0.003 | 0.176 +-0.151 | in | 0.013 +-0.005 | in |  |  | 0.016 +-0.006 | in |  |  |
| SBSBSBS | coherent share, at the peak | 0.007 +-0.001 | 0.033 +-0.001 | OUT | 0.010 +-0.007 | in |  |  | 0.010 +-0.007 | in |  |  |
| SBSBSBS | scintillation index | 0.85 +-0.03 | 1.70 +-0.32 | OUT | 0.60 +-0.06 | OUT |  |  | 4.07 +-0.29 | OUT | 14.20 +-4.88 | OUT |
| SBSBSBS | Doppler rms width (Hz) | 1.68 +-0.07 | 0.80 +-0.05 | OUT |  |  | 1.22 +-0.07 | OUT | 0.96 +-0.23 | OUT |  |  |
| SBSBSBS | Doppler 50 % half width (Hz) | 1.00 +-0.08 | 0.50 +-0.01 | OUT |  |  | 0.68 +-0.12 | OUT |  |  |  |  |
| SBSBSBS | Doppler shape w90/w50 | 2.83 +-0.13 | 2.74 +-0.03 | in |  |  | 4.37 +-0.83 | in |  |  |  |  |
| SBSBSBS | coherence time (ms) | 140 +-4 | 253 +-18 | OUT |  |  | 217 +-17 | OUT |  |  |  |  |
| SBSBSBS | tail decay time (ms) | 2.292 +-0.141 | - | - | 0.149 +-0.009 | OUT | 0.073 +-0.005 | OUT | 0.333 +-0.062 | OUT |  |  |
| SBSBSBS | tail mean excess delay (ms) | 0.148 +-0.001 | 0.184 +-0.014 | in | 0.121 +-0.001 | in | 0.161 +-0.013 | in |  |  |  |  |
| SBSBSBSB+BSBSBSBS | delay after direct (ms) | 4.531 +-0.016 | 4.954 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSB+BSBSBSBS | level re whole channel (dB) | -19.5 +-0.3 | -13.1 +-0.8 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSB+BSBSBSBS | the same with bubble loss | -19.5 +-0.3 | -20.4 | in |  |  |  |  |  |  |  |  |
| SBSBSBSB+BSBSBSBS | coherent share, arrival and tail | 0.012 +-0.001 | 0.054 +-0.044 | in | 0.024 +-0.010 | in |  |  | 0.044 +-0.024 | in |  |  |
| SBSBSBSB+BSBSBSBS | coherent share, at the peak | 0.009 +-0.003 | 0.043 +-0.037 | in | 0.011 +-0.007 | in |  |  | 0.012 +-0.004 | in |  |  |
| SBSBSBSB+BSBSBSBS | scintillation index | 0.66 +-0.03 | 1.40 +-0.36 | in | 0.69 +-0.09 | in |  |  | 2.57 +-0.56 | OUT | 9.39 +-1.33 | OUT |
| SBSBSBSB+BSBSBSBS | Doppler rms width (Hz) | 1.85 +-0.09 | 0.59 +-0.05 | OUT |  |  | 1.05 +-0.11 | OUT | 1.02 +-0.11 | OUT |  |  |
| SBSBSBSB+BSBSBSBS | Doppler 50 % half width (Hz) | 1.16 +-0.07 | 0.38 +-0.00 | OUT |  |  | 0.59 +-0.06 | OUT |  |  |  |  |
| SBSBSBSB+BSBSBSBS | Doppler shape w90/w50 | 2.62 +-0.11 | 2.59 +-0.24 | in |  |  | 4.16 +-0.15 | OUT |  |  |  |  |
| SBSBSBSB+BSBSBSBS | coherence time (ms) | 118 +-1 | 512 +-129 | OUT |  |  | 349 +-20 | OUT |  |  |  |  |
| SBSBSBSB+BSBSBSBS | tail decay time (ms) | 0.891 +-0.177 | 0.445 | OUT | 0.236 +-0.112 | OUT | 0.085 +-0.034 | OUT | - | - |  |  |
| SBSBSBSB+BSBSBSBS | tail mean excess delay (ms) | 0.141 +-0.006 | 0.069 +-0.009 | OUT | 0.107 +-0.004 | in | 0.071 +-0.006 | OUT |  |  |  |  |
| BSBSBSBSB | delay after direct (ms) | 5.069 +-0.019 | 5.280 +-0.027 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSBSB | level re whole channel (dB) | -21.4 +-0.2 | -21.8 +-1.6 | in |  |  |  |  |  |  |  |  |
| BSBSBSBSB | the same with bubble loss | -21.4 +-0.2 | -23.8 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSBSB | coherent share, arrival and tail | 0.010 +-0.003 | 0.053 +-0.017 | in | - | - |  |  | 0.041 +-0.028 | in |  |  |
| BSBSBSBSB | coherent share, at the peak | 0.007 +-0.004 | 0.061 +-0.078 | in | 0.071 +-0.086 | in |  |  | 0.021 +-0.008 | in |  |  |
| BSBSBSBSB | scintillation index | 0.76 +-0.02 | 12.03 +-3.48 | OUT | 1.22 +-0.45 | in |  |  | 3.48 +-0.35 | OUT | 11.76 +-2.89 | OUT |
| BSBSBSBSB | Doppler rms width (Hz) | 2.10 +-0.08 | 0.75 +-0.02 | OUT |  |  | 1.19 +-0.31 | OUT | 0.98 +-0.07 | OUT |  |  |
| BSBSBSBSB | Doppler 50 % half width (Hz) | 1.32 +-0.07 | 0.45 +-0.09 | OUT |  |  | 0.56 +-0.12 | OUT |  |  |  |  |
| BSBSBSBSB | Doppler shape w90/w50 | 2.65 +-0.09 | 2.87 +-0.42 | in |  |  | 5.67 +-0.46 | OUT |  |  |  |  |
| BSBSBSBSB | coherence time (ms) | 101 +-2 | 406 +-75 | OUT |  |  | 290 +-23 | OUT |  |  |  |  |
| BSBSBSBSB | tail decay time (ms) | 3.147 +-0.675 | - | - | - | - | - | - | - | - |  |  |
| BSBSBSBSB | tail mean excess delay (ms) | 0.497 +-0.003 | 0.704 +-0.058 | OUT | 0.000 +-0.000 | OUT | 0.030 +-0.001 | OUT |  |  |  |  |
| SBSBSBSBS | delay after direct (ms) | 6.874 +-0.077 | 6.778 +-0.136 | in |  |  |  |  |  |  |  |  |
| SBSBSBSBS | level re whole channel (dB) | -22.4 +-0.3 | -18.0 +-1.1 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSBS | the same with bubble loss | -22.4 +-0.3 | -25.2 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSBS | coherent share, arrival and tail | 0.003 +-0.001 | 0.088 +-0.063 | in | 0.032 +-0.031 | in |  |  | 0.013 +-0.010 | in |  |  |
| SBSBSBSBS | coherent share, at the peak | 0.003 +-0.003 | 0.010 +-0.001 | in | 0.030 +-0.036 | in |  |  | 0.006 +-0.005 | in |  |  |
| SBSBSBSBS | scintillation index | 0.85 +-0.04 | 5.39 +-1.09 | OUT | 0.82 +-0.01 | in |  |  | 1.15 +-0.07 | OUT | 3.62 +-1.94 | in |
| SBSBSBSBS | Doppler rms width (Hz) | 2.28 +-0.07 | 0.96 +-0.03 | OUT |  |  | 1.54 +-0.09 | OUT | 1.16 +-0.06 | OUT |  |  |
| SBSBSBSBS | Doppler 50 % half width (Hz) | 1.48 +-0.05 | 0.70 +-0.09 | OUT |  |  | 0.80 +-0.05 | OUT |  |  |  |  |
| SBSBSBSBS | Doppler shape w90/w50 | 2.56 +-0.16 | 2.61 +-0.10 | in |  |  | 4.77 +-0.22 | OUT |  |  |  |  |
| SBSBSBSBS | coherence time (ms) | 94 +-1 | 183 +-5 | OUT |  |  | 314 +-28 | OUT |  |  |  |  |
| SBSBSBSBS | tail decay time (ms) | 1.309 +-0.978 | - | - | 0.272 +-0.036 | in | 4.255 +-5.847 | in | - | - |  |  |
| SBSBSBSBS | tail mean excess delay (ms) | 0.147 +-0.007 | 0.140 +-0.030 | in | 0.080 +-0.008 | OUT | 0.116 +-0.016 | in |  |  |  |  |
| SBSBSBSBSB | delay after direct (ms) | 7.219 +-0.016 | 7.219 +-0.000 | in |  |  |  |  |  |  |  |  |
| SBSBSBSBSB | level re whole channel (dB) | -22.7 +-0.3 | -14.0 +-1.1 | OUT |  |  |  |  |  |  |  |  |
| SBSBSBSBSB | the same with bubble loss | -22.7 +-0.3 | -21.8 | in |  |  |  |  |  |  |  |  |
| SBSBSBSBSB | coherent share, arrival and tail | 0.008 +-0.003 | 0.074 +-0.001 | OUT | 0.028 +-0.011 | in |  |  | 0.022 +-0.005 | in |  |  |
| SBSBSBSBSB | coherent share, at the peak | 0.008 +-0.001 | 0.058 +-0.005 | OUT | 0.036 +-0.012 | in |  |  | 0.009 +-0.007 | in |  |  |
| SBSBSBSBSB | scintillation index | 0.82 +-0.06 | 1.42 +-0.16 | OUT | 0.82 +-0.03 | in |  |  | 1.54 +-0.12 | OUT | 9.86 +-2.98 | OUT |
| SBSBSBSBSB | Doppler rms width (Hz) | 2.37 +-0.07 | 0.92 +-0.08 | OUT |  |  | 1.62 +-0.13 | OUT | 1.05 +-0.01 | OUT |  |  |
| SBSBSBSBSB | Doppler 50 % half width (Hz) | 1.55 +-0.11 | 0.55 +-0.09 | OUT |  |  | 0.86 +-0.13 | OUT |  |  |  |  |
| SBSBSBSBSB | Doppler shape w90/w50 | 2.55 +-0.07 | 2.79 +-0.48 | in |  |  | 4.16 +-0.93 | in |  |  |  |  |
| SBSBSBSBSB | coherence time (ms) | 87 +-2 | 331 +-74 | OUT |  |  | 206 +-55 | in |  |  |  |  |
| SBSBSBSBSB | tail decay time (ms) | 2.402 +-1.124 | 0.115 | in | 0.270 +-0.186 | in | 0.080 +-0.004 | OUT | - | - |  |  |
| SBSBSBSBSB | tail mean excess delay (ms) | 0.147 +-0.002 | 0.042 +-0.004 | OUT | 0.099 +-0.013 | in | 0.074 +-0.003 | OUT |  |  |  |  |
| BSBSBSBSBS | delay after direct (ms) | 7.450 +-0.044 | 7.680 +-0.000 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSBSBS | level re whole channel (dB) | -23.1 +-0.2 | -16.4 +-0.6 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSBSBS | the same with bubble loss | -23.1 +-0.2 | -22.2 | in |  |  |  |  |  |  |  |  |
| BSBSBSBSBS | coherent share, arrival and tail | 0.006 +-0.002 | 0.020 +-0.017 | in | 0.021 +-0.018 | in |  |  | 0.009 +-0.005 | in |  |  |
| BSBSBSBSBS | coherent share, at the peak | 0.008 +-0.003 | 0.008 +-0.005 | in | 0.020 +-0.018 | in |  |  | 0.005 +-0.000 | in |  |  |
| BSBSBSBSBS | scintillation index | 0.92 +-0.07 | 2.64 +-0.85 | in | 0.94 +-0.09 | in |  |  | 1.46 +-0.20 | OUT | 6.46 +-1.01 | OUT |
| BSBSBSBSBS | Doppler rms width (Hz) | 2.40 +-0.06 | 0.74 +-0.04 | OUT |  |  | 1.86 +-0.30 | in | 1.06 +-0.05 | OUT |  |  |
| BSBSBSBSBS | Doppler 50 % half width (Hz) | 1.65 +-0.07 | 0.57 +-0.07 | OUT |  |  | 0.99 +-0.01 | OUT |  |  |  |  |
| BSBSBSBSBS | Doppler shape w90/w50 | 2.43 +-0.06 | 2.49 +-0.55 | in |  |  | 3.95 +-0.65 | OUT |  |  |  |  |
| BSBSBSBSBS | coherence time (ms) | 84 +-1 | 254 +-14 | OUT |  |  | 126 +-18 | in |  |  |  |  |
| BSBSBSBSBS | tail decay time (ms) | 2.890 +-0.421 | 0.483 | OUT | 2.091 +-0.959 | in | 0.120 +-0.027 | OUT | 0.569 | OUT |  |  |
| BSBSBSBSBS | tail mean excess delay (ms) | 0.148 +-0.001 | 0.082 +-0.009 | OUT | 0.090 +-0.001 | OUT | 0.076 +-0.005 | OUT |  |  |  |  |
| BSBSBSBSBSB | delay after direct (ms) | 7.949 +-0.021 | 8.198 +-0.136 | in |  |  |  |  |  |  |  |  |
| BSBSBSBSBSB | level re whole channel (dB) | -23.9 +-0.3 | -16.1 +-1.0 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSBSBSB | the same with bubble loss | -23.9 +-0.3 | -22.5 | OUT |  |  |  |  |  |  |  |  |
| BSBSBSBSBSB | coherent share, arrival and tail | 0.005 +-0.001 | 0.043 +-0.001 | OUT | 0.080 +-0.054 | in |  |  | 0.039 +-0.041 | in |  |  |
| BSBSBSBSBSB | coherent share, at the peak | 0.003 +-0.002 | 0.022 +-0.020 | in | 0.051 +-0.063 | in |  |  | 0.034 +-0.037 | in |  |  |
| BSBSBSBSBSB | scintillation index | 1.02 +-0.05 | 1.59 +-0.48 | in | 2.24 +-0.47 | OUT |  |  | 5.64 +-0.88 | OUT | 39.64 +-30.59 | in |
| BSBSBSBSBSB | Doppler rms width (Hz) | 2.47 +-0.06 | 0.88 +-0.08 | OUT |  |  | 2.05 +-0.08 | OUT | 0.84 +-0.31 | OUT |  |  |
| BSBSBSBSBSB | Doppler 50 % half width (Hz) | 1.72 +-0.05 | 0.51 +-0.10 | OUT |  |  | 1.17 +-0.05 | OUT |  |  |  |  |
| BSBSBSBSBSB | Doppler shape w90/w50 | 2.48 +-0.08 | 2.96 +-0.19 | in |  |  | 3.83 +-0.44 | OUT |  |  |  |  |
| BSBSBSBSBSB | coherence time (ms) | 79 +-1 | 299 +-96 | OUT |  |  | 119 +-17 | in |  |  |  |  |
| BSBSBSBSBSB | tail decay time (ms) | 1.582 +-0.154 | 0.844 | OUT | - | - | - | - | - | - |  |  |
| BSBSBSBSBSB | tail mean excess delay (ms) | 0.470 +-0.005 | 0.559 +-0.062 | in | 0.592 +-0.002 | OUT | 0.523 +-0.014 | in |  |  |  |  |

