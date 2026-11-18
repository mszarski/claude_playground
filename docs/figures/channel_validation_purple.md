# Time spreading against measured channels (UACR Purple, SPACE'08)

A: gaussian Doppler spectrum, facet decay time, bounces convolved. C: gaussian, rule_of_thumb, convolve (the suite's defaults).
B: lorentzian, rule_of_thumb, sum (SST-style). One free parameter per recording (wind speed), fitted to the row marked FIT;
every other row is a prediction. +- on the measured column: standard error from the four quarters of the record; on the model
columns: spread of one record from seed to seed. `in`/`OUT`: within twice the larger of the two, plus the resolution.

## purple_1: 60 m, wind 3.96 m/s (fitted here)
rms height 0.084 m, mean-square slope 0.0233, rms vertical speed 0.256 m/s; estimator bandwidth 10.4 Hz; floor -35.3 dB per delay bin re the whole channel; 4 receivers, 53 s

| arrival | quantity | measured | A | | C | | B | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D+B | delay after direct (ms) | 0.000 +-0.000 | 0.000 +-0.000 | in |  |  |  |  |
| D+B | level re whole channel (dB) | -5.3 +-0.0 | -6.3 +-0.0 | OUT |  |  |  |  |
| D+B | the same with bubble loss | -5.3 +-0.0 | -6.2 | OUT |  |  |  |  |
| D+B | coherent share, at the peak | 1.002 +-0.000 | 1.001 +-0.000 | in | 1.001 +-0.000 | in |  |  |
| D+B | scintillation index | 0.00 +-0.00 | 0.00 +-0.00 | in | 0.00 +-0.00 | in |  |  |
| S | delay after direct (ms) | 2.995 +-0.021 | 2.918 +-0.000 | in |  |  |  |  |
| S | level re whole channel (dB) | -9.5 +-0.1 | -10.3 +-0.2 | OUT |  |  |  |  |
| S | the same with bubble loss | -9.5 +-0.1 | -10.4 | OUT |  |  |  |  |
| S | coherent share, arrival and tail | 0.012 +-0.005 | 0.002 +-0.000 | in | 0.005 +-0.003 | in |  |  |
| S | coherent share, at the peak | 0.005 +-0.007 | 0.001 +-0.001 | in | 0.002 +-0.003 | in |  |  |
| S | scintillation index | 0.87 +-0.15 | 0.67 +-0.01 | in | 0.62 +-0.16 | in |  |  |
| S | Doppler rms width (Hz) FIT | 1.53 +-0.17 | 1.55 +-0.07 | fit |  |  | 2.57 +-0.39 | OUT |
| S | Doppler 50 % half width (Hz) | 1.00 +-0.08 | 0.97 +-0.07 | in |  |  | 1.29 +-0.14 | in |
| S | Doppler shape w90/w50 | 2.53 +-0.15 | 2.64 +-0.16 | in |  |  | 4.26 +-0.26 | OUT |
| S | coherence time (ms) | 143 +-13 | 151 +-2 | in |  |  | 107 +-14 | in |
| S | tail decay time (ms) | 0.267 +-0.003 | 0.261 +-0.006 | in | 0.571 +-0.017 | OUT | 0.554 +-0.015 | OUT |
| S | tail mean excess delay (ms) | 0.200 +-0.004 | 0.244 +-0.005 | in | 0.357 +-0.006 | OUT | 0.355 +-0.002 | OUT |
| S | closed forms | | grazing 21.0 deg, Gamma^2 9.94, coherent 0.000 | | Doppler rms 1.53 Hz | | Lorentzian half width 1.53 Hz | |
| SB | delay after direct (ms) | 4.608 +-0.000 | 4.531 +-0.000 | in |  |  |  |  |
| SB | level re whole channel (dB) | -14.7 +-0.1 | -11.5 +-0.1 | OUT |  |  |  |  |
| SB | the same with bubble loss | -14.7 +-0.1 | -11.7 | OUT |  |  |  |  |
| SB | coherent share, arrival and tail | 0.002 +-0.002 | 0.003 +-0.003 | in | 0.002 +-0.001 | in |  |  |
| SB | coherent share, at the peak | 0.001 +-0.004 | 0.003 +-0.004 | in | 0.003 +-0.003 | in |  |  |
| SB | scintillation index | 1.14 +-0.16 | 0.64 +-0.06 | OUT | 0.61 +-0.04 | OUT |  |  |
| SB | Doppler rms width (Hz) | 1.98 +-0.10 | 1.79 +-0.15 | in |  |  | 2.99 +-0.18 | OUT |
| SB | Doppler 50 % half width (Hz) | 1.28 +-0.08 | 1.20 +-0.10 | in |  |  | 1.66 +-0.13 | OUT |
| SB | Doppler shape w90/w50 | 2.49 +-0.16 | 2.41 +-0.08 | in |  |  | 3.85 +-0.32 | OUT |
| SB | coherence time (ms) | 116 +-18 | 124 +-2 | in |  |  | 92 +-8 | in |
| SB | tail decay time (ms) | 0.481 +-0.031 | 0.357 +-0.006 | OUT | 0.644 +-0.050 | OUT | 0.575 +-0.112 | in |
| SB | tail mean excess delay (ms) | 0.163 +-0.006 | 0.134 +-0.001 | in | 0.165 +-0.002 | in | 0.162 +-0.003 | in |
| SB | closed forms | | grazing 25.7 deg, Gamma^2 14.55, coherent 0.000 | | Doppler rms 1.85 Hz | | Lorentzian half width 1.85 Hz | |
| BS | delay after direct (ms) | 4.992 +-0.052 | 5.107 +-0.000 | in |  |  |  |  |
| BS | level re whole channel (dB) | -15.9 +-0.2 | -11.3 +-0.2 | OUT |  |  |  |  |
| BS | the same with bubble loss | -15.9 +-0.2 | -11.1 | OUT |  |  |  |  |
| BS | coherent share, arrival and tail | 0.006 +-0.002 | 0.005 +-0.003 | in | 0.007 +-0.007 | in |  |  |
| BS | coherent share, at the peak | 0.002 +-0.001 | 0.004 +-0.003 | in | 0.004 +-0.004 | in |  |  |
| BS | scintillation index | 1.14 +-0.18 | 0.61 +-0.00 | OUT | 0.61 +-0.01 | OUT |  |  |
| BS | Doppler rms width (Hz) | 2.17 +-0.10 | 1.86 +-0.07 | OUT |  |  | 2.75 +-0.36 | in |
| BS | Doppler 50 % half width (Hz) | 1.40 +-0.09 | 1.26 +-0.01 | in |  |  | 1.50 +-0.17 | in |
| BS | Doppler shape w90/w50 | 2.50 +-0.09 | 2.43 +-0.17 | in |  |  | 3.89 +-0.17 | OUT |
| BS | coherence time (ms) | 108 +-7 | 119 +-6 | in |  |  | 99 +-18 | in |
| BS | tail decay time (ms) | 0.426 +-0.019 | 0.271 +-0.003 | OUT | 0.500 +-0.021 | in | 0.497 +-0.011 | in |
| BS | tail mean excess delay (ms) | 0.311 +-0.015 | 0.243 +-0.004 | in | 0.345 +-0.005 | in | 0.344 +-0.002 | in |
| BS | closed forms | | grazing 27.4 deg, Gamma^2 16.32, coherent 0.000 | | Doppler rms 1.96 Hz | | Lorentzian half width 1.96 Hz | |
| BSB | delay after direct (ms) | 7.219 +-0.000 | 7.104 +-0.000 | OUT |  |  |  |  |
| BSB | level re whole channel (dB) | -22.8 +-0.4 | -20.1 +-0.2 | OUT |  |  |  |  |
| BSB | the same with bubble loss | -22.8 +-0.4 | -20.3 | OUT |  |  |  |  |
| BSB | coherent share, arrival and tail | 0.011 +-0.001 | 0.007 +-0.006 | in | 0.004 +-0.003 | in |  |  |
| BSB | coherent share, at the peak | 0.004 +-0.001 | 0.005 +-0.007 | in | 0.002 +-0.002 | in |  |  |
| BSB | scintillation index | 1.30 +-0.07 | 0.67 +-0.00 | OUT | 0.62 +-0.06 | OUT |  |  |
| BSB | Doppler rms width (Hz) | 2.07 +-0.12 | 2.20 +-0.16 | in |  |  | 3.48 +-0.33 | OUT |
| BSB | Doppler 50 % half width (Hz) | 1.39 +-0.13 | 1.43 +-0.11 | in |  |  | 1.88 +-0.17 | OUT |
| BSB | Doppler shape w90/w50 | 2.47 +-0.08 | 2.57 +-0.01 | in |  |  | 4.11 +-0.93 | in |
| BSB | coherence time (ms) | 105 +-6 | 104 +-8 | in |  |  | 77 +-10 | OUT |
| BSB | tail decay time (ms) | 0.289 +-0.009 | 0.396 +-0.013 | OUT | 0.615 +-0.036 | OUT | 0.608 +-0.007 | OUT |
| BSB | tail mean excess delay (ms) | 0.248 +-0.010 | 0.285 +-0.005 | in | 0.371 +-0.011 | OUT | 0.373 +-0.005 | OUT |
| BSB | closed forms | | grazing 31.6 deg, Gamma^2 21.19, coherent 0.000 | | Doppler rms 2.23 Hz | | Lorentzian half width 2.23 Hz | |

## purple_3: 200 m, wind 3.96 m/s (from the 60 m file of the same recording)
rms height 0.084 m, mean-square slope 0.0233, rms vertical speed 0.256 m/s; estimator bandwidth 10.4 Hz; floor -32.3 dB per delay bin re the whole channel; 4 receivers, 53 s

| arrival | quantity | measured | A | | C | | B | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D+B | delay after direct (ms) | 0.000 +-0.000 | 0.000 +-0.000 | in |  |  |  |  |
| D+B | level re whole channel (dB) | -7.6 +-0.0 | -9.1 +-0.0 | OUT |  |  |  |  |
| D+B | the same with bubble loss | -7.6 +-0.0 | -8.9 | OUT |  |  |  |  |
| D+B | coherent share, at the peak | 1.008 +-0.092 | 1.002 +-0.000 | in | 1.004 +-0.000 | in |  |  |
| D+B | scintillation index | 0.00 +-0.00 | 0.00 +-0.00 | in | 0.00 +-0.00 | in |  |  |
| S | delay after direct (ms) | 1.037 +-0.000 | 0.806 +-0.000 | OUT |  |  |  |  |
| S | level re whole channel (dB) | -9.4 +-0.2 | -11.5 +-0.2 | OUT |  |  |  |  |
| S | the same with bubble loss | -9.4 +-0.2 | -11.8 | OUT |  |  |  |  |
| S | coherent share, arrival and tail | 0.136 +-0.044 | 0.599 +-0.015 | OUT | 0.781 +-0.018 | OUT |  |  |
| S | coherent share, at the peak | 0.156 +-0.042 | 0.848 +-0.012 | OUT | 0.992 +-0.000 | OUT |  |  |
| S | scintillation index | 0.67 +-0.07 | 0.12 +-0.01 | OUT | 0.00 +-0.00 | OUT |  |  |
| S | Doppler rms width (Hz) | 0.54 +-0.02 | 0.65 +-0.08 | in |  |  | 1.14 +-0.28 | in |
| S | Doppler 50 % half width (Hz) | 0.38 +-0.00 | 0.43 +-0.05 | in |  |  | 0.98 +-0.05 | OUT |
| S | Doppler shape w90/w50 | 2.12 +-0.08 | 2.51 +-0.34 | in |  |  | 11.40 +-1.12 | OUT |
| S | coherence time (ms) | 481 +-39 | 339 +-27 | OUT |  |  | 12 +-0 | OUT |
| S | tail decay time (ms) | 0.255 +-0.018 | 0.789 +-0.157 | OUT | - | - | - | - |
| S | tail mean excess delay (ms) | 0.121 +-0.004 | 0.206 +-0.003 | OUT | 0.284 +-0.019 | OUT | 0.266 +-0.042 | OUT |
| S | closed forms | | grazing 6.7 deg, Gamma^2 1.05, coherent 0.351 | | Doppler rms 0.62 Hz | | Lorentzian half width 0.50 Hz | |
| SB+BS | delay after direct (ms) | 1.459 +-0.017 | 1.382 +-0.000 | in |  |  |  |  |
| SB+BS | level re whole channel (dB) | -9.4 +-0.2 | -11.0 +-0.1 | OUT |  |  |  |  |
| SB+BS | the same with bubble loss | -9.4 +-0.2 | -11.0 | OUT |  |  |  |  |
| SB+BS | coherent share, arrival and tail | 0.042 +-0.025 | 0.200 +-0.026 | OUT | 0.319 +-0.035 | OUT |  |  |
| SB+BS | coherent share, at the peak | 0.045 +-0.035 | 0.422 +-0.033 | OUT | 0.656 +-0.047 | OUT |  |  |
| SB+BS | scintillation index | 0.73 +-0.08 | 0.41 +-0.02 | OUT | 0.36 +-0.04 | OUT |  |  |
| SB+BS | Doppler rms width (Hz) | 0.66 +-0.03 | 0.73 +-0.05 | in |  |  | 1.09 +-0.14 | OUT |
| SB+BS | Doppler 50 % half width (Hz) | 0.38 +-0.01 | 0.50 +-0.03 | in |  |  | 0.56 +-0.07 | in |
| SB+BS | Doppler shape w90/w50 | 2.72 +-0.18 | 2.32 +-0.03 | in |  |  | 5.57 +-1.86 | in |
| SB+BS | coherence time (ms) | 397 +-16 | 318 +-9 | OUT |  |  | 279 +-86 | in |
| SB+BS | tail decay time (ms) | 0.838 +-0.437 | 0.732 +-0.023 | in | 3.001 | OUT | 8.841 | OUT |
| SB+BS | tail mean excess delay (ms) | 0.132 +-0.005 | 0.314 +-0.009 | OUT | 0.338 +-0.055 | OUT | 0.361 +-0.015 | OUT |
| SB+BS | closed forms | | grazing 8.1 deg, Gamma^2 1.54, coherent 0.214 | | Doppler rms 0.68 Hz | | Lorentzian half width 0.60 Hz | |
| BSB | delay after direct (ms) | 1.843 +-0.017 | 2.266 +-0.000 | OUT |  |  |  |  |
| BSB | level re whole channel (dB) | -12.7 +-0.3 | -11.8 +-0.3 | in |  |  |  |  |
| BSB | the same with bubble loss | -12.7 +-0.3 | -11.7 | in |  |  |  |  |
| BSB | coherent share, arrival and tail | 0.023 +-0.009 | 0.030 +-0.004 | in | 0.061 +-0.011 | in |  |  |
| BSB | coherent share, at the peak | 0.014 +-0.013 | 0.069 +-0.011 | OUT | 0.239 +-0.024 | OUT |  |  |
| BSB | scintillation index | 0.66 +-0.02 | 0.55 +-0.02 | OUT | 0.63 +-0.04 | in |  |  |
| BSB | Doppler rms width (Hz) | 0.81 +-0.04 | 0.79 +-0.02 | in |  |  | 1.16 +-0.15 | in |
| BSB | Doppler 50 % half width (Hz) | 0.48 +-0.04 | 0.51 +-0.00 | in |  |  | 0.64 +-0.07 | in |
| BSB | Doppler shape w90/w50 | 2.67 +-0.16 | 2.42 +-0.15 | in |  |  | 4.10 +-0.37 | OUT |
| BSB | coherence time (ms) | 297 +-9 | 292 +-12 | in |  |  | 248 +-5 | OUT |
| BSB | tail decay time (ms) | 0.651 +-0.045 | 0.855 +-0.045 | OUT | 3.921 +-1.707 | in | 2.313 +-0.236 | OUT |
| BSB | tail mean excess delay (ms) | 0.369 +-0.005 | 0.402 +-0.009 | in | 0.500 +-0.009 | OUT | 0.494 +-0.004 | OUT |
| BSB | closed forms | | grazing 10.4 deg, Gamma^2 2.49, coherent 0.083 | | Doppler rms 0.80 Hz | | Lorentzian half width 0.77 Hz | |
| SBS | delay after direct (ms) | 5.069 +-0.000 | 4.838 +-0.133 | in |  |  |  |  |
| SBS | level re whole channel (dB) | -18.8 +-0.4 | -16.6 +-0.6 | OUT |  |  |  |  |
| SBS | the same with bubble loss | -18.8 +-0.4 | -16.4 | OUT |  |  |  |  |
| SBS | coherent share, arrival and tail | 0.006 +-0.003 | 0.003 +-0.002 | in | 0.006 +-0.002 | in |  |  |
| SBS | coherent share, at the peak | 0.011 +-0.008 | 0.003 +-0.003 | in | 0.008 +-0.004 | in |  |  |
| SBS | scintillation index | 1.75 +-0.19 | 0.55 +-0.06 | OUT | 0.73 +-0.02 | OUT |  |  |
| SBS | Doppler rms width (Hz) | 1.27 +-0.02 | 1.34 +-0.07 | in |  |  | 1.88 +-0.18 | OUT |
| SBS | Doppler 50 % half width (Hz) | 0.86 +-0.05 | 0.85 +-0.11 | in |  |  | 0.99 +-0.08 | in |
| SBS | Doppler shape w90/w50 | 2.40 +-0.12 | 2.71 +-0.48 | in |  |  | 4.75 +-1.04 | OUT |
| SBS | coherence time (ms) | 178 +-5 | 160 +-3 | in |  |  | 150 +-16 | in |
| SBS | tail decay time (ms) | 1.082 +-0.194 | 1.481 +-0.317 | in | 10.592 +-11.694 | in | 1.440 +-0.077 | in |
| SBS | tail mean excess delay (ms) | 0.461 +-0.006 | 0.462 +-0.015 | in | 0.520 +-0.004 | OUT | 0.450 +-0.003 | in |

## purple_5: 1000 m, wind 3.96 m/s (from the 60 m file of the same recording)
rms height 0.084 m, mean-square slope 0.0233, rms vertical speed 0.256 m/s; estimator bandwidth 10.4 Hz; floor -33.2 dB per delay bin re the whole channel; 4 receivers, 53 s

| arrival | quantity | measured | A | | C | | B | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D+B+S+SB+BS+BSB | delay after direct (ms) | 0.000 +-0.000 | 0.000 +-0.000 | in |  |  |  |  |
| D+B+S+SB+BS+BSB | level re whole channel (dB) | -9.7 +-0.1 | -4.2 +-0.0 | OUT |  |  |  |  |
| D+B+S+SB+BS+BSB | the same with bubble loss | -9.7 +-0.1 | -4.3 | OUT |  |  |  |  |
| D+B+S+SB+BS+BSB | coherent share, arrival and tail | 0.743 +-0.016 | 0.998 +-0.000 | OUT | 1.001 +-0.000 | OUT |  |  |
| D+B+S+SB+BS+BSB | coherent share, at the peak | 0.874 +-0.020 | 1.001 +-0.000 | OUT | 1.001 +-0.000 | OUT |  |  |
| D+B+S+SB+BS+BSB | scintillation index | 0.00 +-0.01 | 0.00 +-0.00 | in | 0.00 +-0.00 | in |  |  |
| D+B+S+SB+BS+BSB | Doppler rms width (Hz) | 0.40 +-0.01 | 0.00 +-0.00 | OUT |  |  | 0.00 +-0.00 | OUT |
| D+B+S+SB+BS+BSB | Doppler 50 % half width (Hz) | 0.31 +-0.00 | 11.95 +-1.12 | OUT |  |  | 12.92 +-2.16 | OUT |
| D+B+S+SB+BS+BSB | Doppler shape w90/w50 | 1.83 +-0.03 | 2.47 +-0.12 | OUT |  |  | 2.19 +-0.33 | in |
| D+B+S+SB+BS+BSB | coherence time (ms) | 843 +-60 | 10 +-0 | OUT |  |  | 10 +-0 | OUT |
| D+B+S+SB+BS+BSB | tail decay time (ms) | 0.755 +-0.093 | - | - | - | - | - | - |
| D+B+S+SB+BS+BSB | tail mean excess delay (ms) | 0.247 +-0.013 | 0.471 +-0.005 | OUT | 0.147 +-0.000 | OUT | 0.148 +-0.001 | OUT |
| D+B+S+SB+BS+BSB | closed forms | | grazing 1.3 deg, Gamma^2 0.04, coherent 0.958 | | Doppler rms 0.49 Hz | | Lorentzian half width 0.10 Hz | |
| SBS | delay after direct (ms) | 0.614 +-0.014 | 0.768 +-0.000 | OUT |  |  |  |  |
| SBS | level re whole channel (dB) | -9.5 +-0.1 | -9.3 +-0.1 | in |  |  |  |  |
| SBS | the same with bubble loss | -9.5 +-0.1 | -9.9 | in |  |  |  |  |
| SBS | coherent share, arrival and tail | 0.425 +-0.015 | 0.765 +-0.001 | OUT | 0.901 +-0.005 | OUT |  |  |
| SBS | coherent share, at the peak | 0.679 +-0.021 | 0.957 +-0.003 | OUT | 1.001 +-0.000 | OUT |  |  |
| SBS | scintillation index | 0.27 +-0.04 | 0.05 +-0.00 | OUT | 0.00 +-0.00 | OUT |  |  |
| SBS | Doppler rms width (Hz) | 0.47 +-0.00 | 0.55 +-0.04 | in |  |  | 0.00 +-0.00 | OUT |
| SBS | Doppler 50 % half width (Hz) | 0.33 +-0.00 | 0.38 +-0.00 | in |  |  | 11.61 +-2.33 | OUT |
| SBS | Doppler shape w90/w50 | 1.99 +-0.03 | 2.37 +-0.28 | in |  |  | 2.49 +-0.36 | in |
| SBS | coherence time (ms) | 713 +-16 | 399 +-31 | OUT |  |  | 10 +-0 | OUT |
| SBS | tail decay time (ms) | 0.668 +-0.046 | 9.160 | OUT | - | - | - | - |
| SBS | tail mean excess delay (ms) | 0.595 +-0.006 | 0.542 +-0.022 | in | 0.644 +-0.017 | in | 0.425 +-0.005 | OUT |

## purple_6: 60 m, wind 3.70 m/s (fitted here)
rms height 0.073 m, mean-square slope 0.0220, rms vertical speed 0.239 m/s; estimator bandwidth 10.4 Hz; floor -34.6 dB per delay bin re the whole channel; 4 receivers, 53 s

| arrival | quantity | measured | A | | C | | B | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D+B | delay after direct (ms) | 0.000 +-0.000 | 0.000 +-0.000 | in |  |  |  |  |
| D+B | level re whole channel (dB) | -7.1 +-0.2 | -6.3 +-0.0 | in |  |  |  |  |
| D+B | the same with bubble loss | -7.1 +-0.2 | -6.2 | in |  |  |  |  |
| D+B | coherent share, at the peak | 0.573 +-0.085 | 1.001 +-0.000 | OUT | 1.001 +-0.000 | OUT |  |  |
| D+B | scintillation index | 0.54 +-0.01 | 0.00 +-0.00 | OUT | 0.00 +-0.00 | OUT |  |  |
| S | delay after direct (ms) | 3.072 +-0.014 | 2.918 +-0.000 | OUT |  |  |  |  |
| S | level re whole channel (dB) | -11.1 +-0.1 | -10.2 +-0.2 | OUT |  |  |  |  |
| S | the same with bubble loss | -11.1 +-0.1 | -10.1 | OUT |  |  |  |  |
| S | coherent share, arrival and tail | 0.014 +-0.012 | 0.003 +-0.001 | in | 0.004 +-0.002 | in |  |  |
| S | coherent share, at the peak | 0.029 +-0.021 | 0.002 +-0.002 | in | 0.003 +-0.002 | in |  |  |
| S | scintillation index | 1.97 +-0.10 | 0.68 +-0.03 | OUT | 0.67 +-0.17 | OUT |  |  |
| S | Doppler rms width (Hz) FIT | 1.43 +-0.12 | 1.44 +-0.04 | fit |  |  | 2.46 +-0.34 | OUT |
| S | Doppler 50 % half width (Hz) | 1.01 +-0.10 | 0.89 +-0.03 | in |  |  | 1.26 +-0.14 | in |
| S | Doppler shape w90/w50 | 2.30 +-0.13 | 2.67 +-0.06 | in |  |  | 4.17 +-0.18 | OUT |
| S | coherence time (ms) | 172 +-17 | 160 +-1 | in |  |  | 115 +-9 | OUT |
| S | tail decay time (ms) | 0.415 +-0.006 | 0.220 +-0.004 | OUT | 0.546 +-0.011 | OUT | 0.525 +-0.017 | OUT |
| S | tail mean excess delay (ms) | 0.335 +-0.006 | 0.231 +-0.005 | OUT | 0.350 +-0.005 | in | 0.345 +-0.005 | in |
| S | closed forms | | grazing 21.0 deg, Gamma^2 7.61, coherent 0.000 | | Doppler rms 1.43 Hz | | Lorentzian half width 1.43 Hz | |
| SB | delay after direct (ms) | 4.646 +-0.016 | 4.531 +-0.000 | OUT |  |  |  |  |
| SB | level re whole channel (dB) | -15.8 +-0.4 | -11.4 +-0.2 | OUT |  |  |  |  |
| SB | the same with bubble loss | -15.8 +-0.4 | -11.6 | OUT |  |  |  |  |
| SB | coherent share, arrival and tail | 0.006 +-0.004 | 0.004 +-0.004 | in | 0.003 +-0.002 | in |  |  |
| SB | coherent share, at the peak | 0.004 +-0.004 | 0.003 +-0.004 | in | 0.004 +-0.004 | in |  |  |
| SB | scintillation index | 1.85 +-0.18 | 0.64 +-0.01 | OUT | 0.60 +-0.06 | OUT |  |  |
| SB | Doppler rms width (Hz) | 1.54 +-0.07 | 1.69 +-0.11 | in |  |  | 2.74 +-0.28 | OUT |
| SB | Doppler 50 % half width (Hz) | 0.89 +-0.03 | 1.13 +-0.04 | OUT |  |  | 1.52 +-0.22 | OUT |
| SB | Doppler shape w90/w50 | 3.08 +-0.08 | 2.40 +-0.06 | OUT |  |  | 4.00 +-0.47 | in |
| SB | coherence time (ms) | 152 +-7 | 133 +-2 | in |  |  | 101 +-11 | OUT |
| SB | tail decay time (ms) | 0.830 +-0.067 | 0.342 +-0.004 | OUT | 0.576 +-0.073 | OUT | 0.527 +-0.059 | OUT |
| SB | tail mean excess delay (ms) | 0.164 +-0.006 | 0.132 +-0.002 | in | 0.163 +-0.001 | in | 0.161 +-0.002 | in |
| SB | closed forms | | grazing 25.7 deg, Gamma^2 11.14, coherent 0.000 | | Doppler rms 1.73 Hz | | Lorentzian half width 1.73 Hz | |
| BS | delay after direct (ms) | 4.992 +-0.042 | 5.107 +-0.000 | in |  |  |  |  |
| BS | level re whole channel (dB) | -15.6 +-0.3 | -11.2 +-0.2 | OUT |  |  |  |  |
| BS | the same with bubble loss | -15.6 +-0.3 | -11.1 | OUT |  |  |  |  |
| BS | coherent share, arrival and tail | 0.005 +-0.005 | 0.005 +-0.004 | in | 0.006 +-0.005 | in |  |  |
| BS | coherent share, at the peak | 0.006 +-0.006 | 0.004 +-0.003 | in | 0.003 +-0.002 | in |  |  |
| BS | scintillation index | 1.31 +-0.15 | 0.60 +-0.02 | OUT | 0.63 +-0.02 | OUT |  |  |
| BS | Doppler rms width (Hz) | 2.01 +-0.06 | 1.74 +-0.11 | in |  |  | 2.62 +-0.35 | in |
| BS | Doppler 50 % half width (Hz) | 1.34 +-0.05 | 1.18 +-0.03 | in |  |  | 1.39 +-0.17 | in |
| BS | Doppler shape w90/w50 | 2.65 +-0.13 | 2.41 +-0.19 | in |  |  | 4.02 +-0.21 | OUT |
| BS | coherence time (ms) | 121 +-5 | 125 +-7 | in |  |  | 105 +-20 | in |
| BS | tail decay time (ms) | 0.741 +-0.114 | 0.218 +-0.003 | OUT | 0.464 +-0.008 | OUT | 0.458 +-0.008 | OUT |
| BS | tail mean excess delay (ms) | 0.397 +-0.013 | 0.229 +-0.003 | OUT | 0.334 +-0.004 | in | 0.330 +-0.003 | OUT |
| BS | closed forms | | grazing 27.4 deg, Gamma^2 12.50, coherent 0.000 | | Doppler rms 1.83 Hz | | Lorentzian half width 1.83 Hz | |
| BSB | delay after direct (ms) | 7.142 +-0.028 | 7.104 +-0.000 | in |  |  |  |  |
| BSB | level re whole channel (dB) | -25.7 +-0.2 | -20.0 +-0.2 | OUT |  |  |  |  |
| BSB | the same with bubble loss | -25.7 +-0.2 | -20.2 | OUT |  |  |  |  |
| BSB | coherent share, arrival and tail | 0.009 +-0.003 | 0.007 +-0.006 | in | 0.004 +-0.004 | in |  |  |
| BSB | coherent share, at the peak | 0.004 +-0.002 | 0.004 +-0.005 | in | 0.004 +-0.005 | in |  |  |
| BSB | scintillation index | 1.68 +-0.23 | 0.66 +-0.01 | OUT | 0.57 +-0.02 | OUT |  |  |
| BSB | Doppler rms width (Hz) | 1.99 +-0.10 | 2.06 +-0.14 | in |  |  | 3.17 +-0.26 | OUT |
| BSB | Doppler 50 % half width (Hz) | 1.20 +-0.08 | 1.34 +-0.06 | in |  |  | 1.73 +-0.15 | OUT |
| BSB | Doppler shape w90/w50 | 2.84 +-0.17 | 2.56 +-0.05 | in |  |  | 4.16 +-0.60 | in |
| BSB | coherence time (ms) | 108 +-10 | 109 +-9 | in |  |  | 78 +-8 | OUT |
| BSB | tail decay time (ms) | 0.451 +-0.010 | 0.353 +-0.009 | OUT | 0.570 +-0.021 | OUT | 0.562 +-0.022 | OUT |
| BSB | tail mean excess delay (ms) | 0.381 +-0.006 | 0.273 +-0.004 | OUT | 0.360 +-0.008 | in | 0.363 +-0.005 | in |
| BSB | closed forms | | grazing 31.6 deg, Gamma^2 16.23, coherent 0.000 | | Doppler rms 2.09 Hz | | Lorentzian half width 2.09 Hz | |

## purple_8: 200 m, wind 3.70 m/s (from the 60 m file of the same recording)
rms height 0.073 m, mean-square slope 0.0220, rms vertical speed 0.239 m/s; estimator bandwidth 10.4 Hz; floor -32.2 dB per delay bin re the whole channel; 4 receivers, 53 s

| arrival | quantity | measured | A | | C | | B | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D+B | delay after direct (ms) | 0.000 +-0.000 | 0.000 +-0.000 | in |  |  |  |  |
| D+B | level re whole channel (dB) | -5.8 +-0.1 | -9.1 +-0.0 | OUT |  |  |  |  |
| D+B | the same with bubble loss | -5.8 +-0.1 | -8.9 | OUT |  |  |  |  |
| D+B | coherent share, at the peak | 1.003 +-0.001 | 1.001 +-0.000 | in | 1.004 +-0.000 | in |  |  |
| D+B | scintillation index | 0.00 +-0.00 | 0.00 +-0.00 | in | 0.00 +-0.00 | in |  |  |
| S | delay after direct (ms) | 1.075 +-0.010 | 0.806 +-0.000 | OUT |  |  |  |  |
| S | level re whole channel (dB) | -9.8 +-0.3 | -10.6 +-0.1 | in |  |  |  |  |
| S | the same with bubble loss | -9.8 +-0.3 | -10.8 | in |  |  |  |  |
| S | coherent share, arrival and tail | 0.211 +-0.026 | 0.673 +-0.010 | OUT | 0.849 +-0.014 | OUT |  |  |
| S | coherent share, at the peak | 0.229 +-0.029 | 0.883 +-0.006 | OUT | 0.992 +-0.000 | OUT |  |  |
| S | scintillation index | 0.76 +-0.07 | 0.09 +-0.02 | OUT | 0.00 +-0.00 | OUT |  |  |
| S | Doppler rms width (Hz) | 0.55 +-0.03 | 0.67 +-0.07 | in |  |  | 2.31 +-0.25 | OUT |
| S | Doppler 50 % half width (Hz) | 0.38 +-0.00 | 0.46 +-0.04 | in |  |  | 1.43 +-0.11 | OUT |
| S | Doppler shape w90/w50 | 2.20 +-0.10 | 2.49 +-0.27 | in |  |  | 11.55 +-1.38 | OUT |
| S | coherence time (ms) | 518 +-35 | 324 +-21 | OUT |  |  | 11 +-0 | OUT |
| S | tail decay time (ms) | 1.681 +-0.128 | 0.659 +-0.092 | OUT | - | - | - | - |
| S | tail mean excess delay (ms) | 0.125 +-0.001 | 0.188 +-0.020 | in | 0.257 +-0.023 | OUT | 0.240 +-0.018 | OUT |
| S | closed forms | | grazing 6.7 deg, Gamma^2 0.80, coherent 0.449 | | Doppler rms 0.63 Hz | | Lorentzian half width 0.46 Hz | |
| SB+BS | delay after direct (ms) | 1.498 +-0.008 | 1.318 +-0.022 | OUT |  |  |  |  |
| SB+BS | level re whole channel (dB) | -10.4 +-0.3 | -10.7 +-0.4 | in |  |  |  |  |
| SB+BS | the same with bubble loss | -10.4 +-0.3 | -10.9 | in |  |  |  |  |
| SB+BS | coherent share, arrival and tail | 0.118 +-0.021 | 0.342 +-0.010 | OUT | 0.472 +-0.034 | OUT |  |  |
| SB+BS | coherent share, at the peak | 0.158 +-0.031 | 0.625 +-0.023 | OUT | 0.742 +-0.040 | OUT |  |  |
| SB+BS | scintillation index | 0.53 +-0.04 | 0.34 +-0.05 | OUT | 0.27 +-0.03 | OUT |  |  |
| SB+BS | Doppler rms width (Hz) | 0.65 +-0.02 | 0.74 +-0.06 | in |  |  | 1.10 +-0.13 | OUT |
| SB+BS | Doppler 50 % half width (Hz) | 0.38 +-0.02 | 0.50 +-0.05 | in |  |  | 0.61 +-0.08 | in |
| SB+BS | Doppler shape w90/w50 | 2.71 +-0.10 | 2.47 +-0.18 | in |  |  | 5.02 +-0.95 | OUT |
| SB+BS | coherence time (ms) | 466 +-40 | 321 +-8 | OUT |  |  | 256 +-43 | OUT |
| SB+BS | tail decay time (ms) | 0.834 +-0.251 | 1.838 +-0.915 | in | 1.947 +-0.690 | in | 11.273 +-6.133 | in |
| SB+BS | tail mean excess delay (ms) | 0.143 +-0.001 | 0.322 +-0.005 | OUT | 0.315 +-0.047 | OUT | 0.354 +-0.024 | OUT |
| SB+BS | closed forms | | grazing 8.1 deg, Gamma^2 1.18, coherent 0.307 | | Doppler rms 0.68 Hz | | Lorentzian half width 0.56 Hz | |
| BSB | delay after direct (ms) | 1.843 +-0.000 | 2.202 +-0.022 | OUT |  |  |  |  |
| BSB | level re whole channel (dB) | -10.6 +-0.3 | -11.9 +-0.2 | OUT |  |  |  |  |
| BSB | the same with bubble loss | -10.6 +-0.3 | -12.1 | OUT |  |  |  |  |
| BSB | coherent share, arrival and tail | 0.055 +-0.025 | 0.081 +-0.032 | in | 0.106 +-0.016 | in |  |  |
| BSB | coherent share, at the peak | 0.049 +-0.025 | 0.229 +-0.106 | in | 0.375 +-0.024 | OUT |  |  |
| BSB | scintillation index | 0.59 +-0.05 | 0.48 +-0.02 | in | 0.49 +-0.02 | in |  |  |
| BSB | Doppler rms width (Hz) | 0.73 +-0.04 | 0.76 +-0.02 | in |  |  | 1.16 +-0.13 | OUT |
| BSB | Doppler 50 % half width (Hz) | 0.44 +-0.03 | 0.50 +-0.01 | in |  |  | 0.65 +-0.01 | OUT |
| BSB | Doppler shape w90/w50 | 2.77 +-0.07 | 2.42 +-0.04 | OUT |  |  | 4.20 +-0.32 | OUT |
| BSB | coherence time (ms) | 360 +-18 | 298 +-14 | OUT |  |  | 268 +-18 | OUT |
| BSB | tail decay time (ms) | 0.649 +-0.010 | 0.802 +-0.017 | OUT | 2.331 +-0.256 | OUT | 2.435 +-0.540 | OUT |
| BSB | tail mean excess delay (ms) | 0.365 +-0.006 | 0.407 +-0.010 | in | 0.503 +-0.004 | OUT | 0.504 +-0.007 | OUT |
| BSB | closed forms | | grazing 10.4 deg, Gamma^2 1.91, coherent 0.148 | | Doppler rms 0.78 Hz | | Lorentzian half width 0.72 Hz | |
| SBS | delay after direct (ms) | 5.069 +-0.000 | 4.826 +-0.089 | in |  |  |  |  |
| SBS | level re whole channel (dB) | -20.9 +-1.1 | -16.3 +-0.3 | OUT |  |  |  |  |
| SBS | the same with bubble loss | -20.9 +-1.1 | -16.1 | OUT |  |  |  |  |
| SBS | coherent share, arrival and tail | 0.006 +-0.001 | 0.003 +-0.002 | in | 0.006 +-0.002 | in |  |  |
| SBS | coherent share, at the peak | 0.004 +-0.014 | 0.003 +-0.002 | in | 0.006 +-0.002 | in |  |  |
| SBS | scintillation index | 3.07 +-0.28 | 0.55 +-0.02 | OUT | 0.74 +-0.04 | OUT |  |  |
| SBS | Doppler rms width (Hz) | 1.08 +-0.05 | 1.34 +-0.01 | OUT |  |  | 1.74 +-0.12 | OUT |
| SBS | Doppler 50 % half width (Hz) | 0.64 +-0.04 | 0.84 +-0.12 | in |  |  | 0.90 +-0.02 | OUT |
| SBS | Doppler shape w90/w50 | 2.67 +-0.18 | 2.71 +-0.45 | in |  |  | 5.35 +-1.28 | in |
| SBS | coherence time (ms) | 194 +-10 | 162 +-2 | OUT |  |  | 154 +-15 | OUT |
| SBS | tail decay time (ms) | 0.977 +-0.148 | 1.334 +-0.174 | in | 9.174 +-4.136 | in | 1.882 +-0.249 | OUT |
| SBS | tail mean excess delay (ms) | 0.470 +-0.011 | 0.457 +-0.009 | in | 0.519 +-0.002 | in | 0.460 +-0.007 | in |

## purple_10: 1000 m, wind 3.70 m/s (from the 60 m file of the same recording)
rms height 0.073 m, mean-square slope 0.0220, rms vertical speed 0.239 m/s; estimator bandwidth 10.4 Hz; floor -34.5 dB per delay bin re the whole channel; 4 receivers, 53 s

| arrival | quantity | measured | A | | C | | B | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D+B+S+SB+BS+BSB | delay after direct (ms) | 0.000 +-0.000 | 0.000 +-0.000 | in |  |  |  |  |
| D+B+S+SB+BS+BSB | level re whole channel (dB) | -5.6 +-0.2 | -4.1 +-0.0 | OUT |  |  |  |  |
| D+B+S+SB+BS+BSB | the same with bubble loss | -5.6 +-0.2 | -4.3 | OUT |  |  |  |  |
| D+B+S+SB+BS+BSB | coherent share, arrival and tail | 0.842 +-0.011 | 0.998 +-0.000 | OUT | 1.001 +-0.000 | OUT |  |  |
| D+B+S+SB+BS+BSB | coherent share, at the peak | 0.931 +-0.007 | 1.000 +-0.000 | OUT | 1.001 +-0.000 | OUT |  |  |
| D+B+S+SB+BS+BSB | scintillation index | 0.11 +-0.01 | 0.00 +-0.00 | OUT | 0.00 +-0.00 | OUT |  |  |
| D+B+S+SB+BS+BSB | Doppler rms width (Hz) | 0.36 +-0.01 | 0.00 +-0.00 | OUT |  |  | 0.00 +-0.00 | OUT |
| D+B+S+SB+BS+BSB | Doppler 50 % half width (Hz) | 0.29 +-0.00 | 11.34 +-0.78 | OUT |  |  | 11.55 +-1.58 | OUT |
| D+B+S+SB+BS+BSB | Doppler shape w90/w50 | 1.76 +-0.07 | 2.71 +-0.40 | in |  |  | 2.48 +-0.42 | in |
| D+B+S+SB+BS+BSB | coherence time (ms) | 1350 +-90 | 10 +-0 | OUT |  |  | 10 +-0 | OUT |
| D+B+S+SB+BS+BSB | tail decay time (ms) | 0.434 +-0.096 | - | - | - | - | - | - |
| D+B+S+SB+BS+BSB | tail mean excess delay (ms) | 0.277 +-0.013 | 0.482 +-0.003 | OUT | 0.038 +-0.067 | OUT | 0.000 +-0.000 | OUT |
| D+B+S+SB+BS+BSB | closed forms | | grazing 1.3 deg, Gamma^2 0.03, coherent 0.968 | | Doppler rms 0.52 Hz | | Lorentzian half width 0.09 Hz | |
| SBS | delay after direct (ms) | 0.845 +-0.021 | 0.768 +-0.000 | in |  |  |  |  |
| SBS | level re whole channel (dB) | -10.0 +-0.3 | -8.9 +-0.1 | in |  |  |  |  |
| SBS | the same with bubble loss | -10.0 +-0.3 | -9.4 | in |  |  |  |  |
| SBS | coherent share, arrival and tail | 0.478 +-0.039 | 0.824 +-0.004 | OUT | 0.925 +-0.004 | OUT |  |  |
| SBS | coherent share, at the peak | 0.630 +-0.044 | 0.972 +-0.001 | OUT | 1.000 +-0.000 | OUT |  |  |
| SBS | scintillation index | 0.41 +-0.02 | 0.04 +-0.00 | OUT | 0.00 +-0.00 | OUT |  |  |
| SBS | Doppler rms width (Hz) | 0.47 +-0.02 | 0.56 +-0.04 | in |  |  | 0.00 +-0.00 | OUT |
| SBS | Doppler 50 % half width (Hz) | 0.34 +-0.01 | 0.38 +-0.00 | in |  |  | 8.47 +-0.53 | OUT |
| SBS | Doppler shape w90/w50 | 1.96 +-0.02 | 2.55 +-0.39 | in |  |  | 3.17 +-0.18 | OUT |
| SBS | coherence time (ms) | 908 +-84 | 378 +-17 | OUT |  |  | 10 +-0 | OUT |
| SBS | tail decay time (ms) | 0.582 +-0.013 | 382.259 +-645.869 | in | - | - | - | - |
| SBS | tail mean excess delay (ms) | 0.389 +-0.007 | 0.533 +-0.026 | OUT | 0.590 +-0.012 | OUT | 0.369 +-0.008 | in |

## purple_11: 60 m, wind 9.14 m/s (fitted here)
rms height 0.447 m, mean-square slope 0.0498, rms vertical speed 0.591 m/s; estimator bandwidth 10.4 Hz; floor -32.9 dB per delay bin re the whole channel; 4 receivers, 53 s

| arrival | quantity | measured | A | | C | | B | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D+B | delay after direct (ms) | 0.000 +-0.000 | 0.000 +-0.000 | in |  |  |  |  |
| D+B | level re whole channel (dB) | -8.6 +-0.1 | -6.0 +-0.0 | OUT |  |  |  |  |
| D+B | the same with bubble loss | -8.6 +-0.1 | -5.2 | OUT |  |  |  |  |
| D+B | coherent share, at the peak | 0.141 +-0.047 | 1.002 +-0.000 | OUT | 1.003 +-0.000 | OUT |  |  |
| D+B | scintillation index | 1.21 +-0.04 | 0.00 +-0.00 | OUT | 0.00 +-0.00 | OUT |  |  |
| S | delay after direct (ms) | 3.149 +-0.104 | 2.957 +-0.000 | in |  |  |  |  |
| S | level re whole channel (dB) | -17.4 +-0.8 | -12.3 +-0.2 | OUT |  |  |  |  |
| S | the same with bubble loss | -17.4 +-0.8 | -13.6 | OUT |  |  |  |  |
| S | coherent share, arrival and tail | 0.004 +-0.001 | 0.001 +-0.000 | in | 0.002 +-0.001 | in |  |  |
| S | coherent share, at the peak | 0.003 +-0.003 | 0.001 +-0.001 | in | 0.001 +-0.001 | in |  |  |
| S | scintillation index | 4.81 +-1.00 | 0.61 +-0.06 | OUT | 0.68 +-0.03 | OUT |  |  |
| S | Doppler rms width (Hz) FIT | 3.53 +-0.37 | 3.28 +-0.10 | fit |  |  | 4.18 +-0.63 | in |
| S | Doppler 50 % half width (Hz) | 2.74 +-0.41 | 2.15 +-0.04 | in |  |  | 2.32 +-0.41 | in |
| S | Doppler shape w90/w50 | 2.05 +-0.05 | 2.56 +-0.07 | OUT |  |  | 4.33 +-1.77 | in |
| S | coherence time (ms) | 57 +-10 | 73 +-4 | in |  |  | 64 +-15 | in |
| S | tail decay time (ms) | - | 0.610 +-0.008 | - | 1.216 +-0.073 | - | 1.213 +-0.104 | - |
| S | tail mean excess delay (ms) | 0.423 +-0.033 | 0.352 +-0.003 | in | 0.447 +-0.005 | in | 0.445 +-0.004 | in |
| S | closed forms | | grazing 21.0 deg, Gamma^2 282.94, coherent 0.000 | | Doppler rms 3.53 Hz | | Lorentzian half width 3.53 Hz | |
| SB | delay after direct (ms) | 4.070 +-0.083 | 4.570 +-0.000 | OUT |  |  |  |  |
| SB | level re whole channel (dB) | -14.4 +-0.7 | -13.2 +-0.1 | in |  |  |  |  |
| SB | the same with bubble loss | -14.4 +-0.7 | -14.2 | in |  |  |  |  |
| SB | coherent share, arrival and tail | 0.003 +-0.002 | 0.001 +-0.001 | in | 0.001 +-0.001 | in |  |  |
| SB | coherent share, at the peak | 0.004 +-0.004 | 0.001 +-0.001 | in | 0.001 +-0.001 | in |  |  |
| SB | scintillation index | 3.07 +-0.66 | 0.61 +-0.02 | OUT | 0.61 +-0.01 | OUT |  |  |
| SB | Doppler rms width (Hz) | 3.24 +-0.48 | 3.69 +-0.10 | in |  |  | 4.99 +-0.67 | OUT |
| SB | Doppler 50 % half width (Hz) | 1.91 +-0.64 | 2.37 +-0.07 | in |  |  | 2.84 +-0.48 | in |
| SB | Doppler shape w90/w50 | 2.89 +-0.17 | 2.55 +-0.17 | in |  |  | 3.29 +-0.47 | in |
| SB | coherence time (ms) | 75 +-6 | 62 +-1 | in |  |  | 52 +-6 | OUT |
| SB | tail decay time (ms) | 0.591 +-2.950 | 0.599 +-0.017 | in | 1.528 +-0.364 | in | 1.552 +-0.097 | in |
| SB | tail mean excess delay (ms) | 0.235 +-0.041 | 0.164 +-0.008 | in | 0.171 +-0.019 | in | 0.181 +-0.003 | in |
| SB | closed forms | | grazing 25.7 deg, Gamma^2 414.11, coherent 0.000 | | Doppler rms 4.27 Hz | | Lorentzian half width 4.27 Hz | |
| BS | delay after direct (ms) | 4.877 +-0.081 | 5.158 +-0.022 | OUT |  |  |  |  |
| BS | level re whole channel (dB) | -18.1 +-0.5 | -12.6 +-0.1 | OUT |  |  |  |  |
| BS | the same with bubble loss | -18.1 +-0.5 | -13.5 | OUT |  |  |  |  |
| BS | coherent share, arrival and tail | 0.003 +-0.002 | 0.002 +-0.003 | in | 0.003 +-0.004 | in |  |  |
| BS | coherent share, at the peak | 0.003 +-0.002 | 0.002 +-0.004 | in | 0.002 +-0.002 | in |  |  |
| BS | scintillation index | 2.21 +-0.05 | 0.57 +-0.00 | OUT | 0.59 +-0.03 | OUT |  |  |
| BS | Doppler rms width (Hz) | 4.05 +-0.60 | 3.84 +-0.14 | in |  |  | 4.66 +-0.31 | in |
| BS | Doppler 50 % half width (Hz) | 2.41 +-0.48 | 2.55 +-0.05 | in |  |  | 2.52 +-0.17 | in |
| BS | Doppler shape w90/w50 | 2.73 +-0.13 | 2.44 +-0.14 | in |  |  | 4.03 +-0.35 | OUT |
| BS | coherence time (ms) | 56 +-5 | 58 +-1 | in |  |  | 58 +-3 | in |
| BS | tail decay time (ms) | 7.659 +-0.637 | 0.660 +-0.012 | OUT | 1.159 +-0.008 | OUT | 1.174 +-0.071 | OUT |
| BS | tail mean excess delay (ms) | 0.517 +-0.022 | 0.374 +-0.002 | OUT | 0.443 +-0.000 | in | 0.446 +-0.004 | in |
| BS | closed forms | | grazing 27.4 deg, Gamma^2 464.45, coherent 0.000 | | Doppler rms 4.53 Hz | | Lorentzian half width 4.53 Hz | |
| BSB | delay after direct (ms) | 6.605 +-0.010 | 7.117 +-0.022 | OUT |  |  |  |  |
| BSB | level re whole channel (dB) | -21.8 +-0.5 | -21.2 +-0.2 | in |  |  |  |  |
| BSB | the same with bubble loss | -21.8 +-0.5 | -21.9 | in |  |  |  |  |
| BSB | coherent share, arrival and tail | 0.004 +-0.003 | 0.005 +-0.004 | in | 0.003 +-0.002 | in |  |  |
| BSB | coherent share, at the peak | 0.002 +-0.003 | 0.004 +-0.002 | in | 0.002 +-0.003 | in |  |  |
| BSB | scintillation index | 2.30 +-0.27 | 0.61 +-0.04 | OUT | 0.56 +-0.01 | OUT |  |  |
| BSB | Doppler rms width (Hz) | 3.91 +-0.30 | 3.97 +-0.29 | in |  |  | 5.22 +-0.23 | OUT |
| BSB | Doppler 50 % half width (Hz) | 2.39 +-0.42 | 2.77 +-0.16 | in |  |  | 3.06 +-0.03 | in |
| BSB | Doppler shape w90/w50 | 2.76 +-0.08 | 2.43 +-0.10 | in |  |  | 3.27 +-0.19 | in |
| BSB | coherence time (ms) | 58 +-4 | 50 +-5 | in |  |  | 47 +-3 | in |
| BSB | tail decay time (ms) | 0.529 +-0.859 | 0.530 +-0.022 | in | 2.916 +-0.507 | OUT | 2.978 +-1.207 | in |
| BSB | tail mean excess delay (ms) | 0.333 +-0.029 | 0.332 +-0.004 | in | 0.495 +-0.004 | OUT | 0.486 +-0.026 | OUT |
| BSB | closed forms | | grazing 31.6 deg, Gamma^2 603.22, coherent 0.000 | | Doppler rms 5.16 Hz | | Lorentzian half width 5.16 Hz | |

## purple_13: 200 m, wind 9.14 m/s (from the 60 m file of the same recording)
rms height 0.447 m, mean-square slope 0.0498, rms vertical speed 0.591 m/s; estimator bandwidth 10.4 Hz; floor -31.9 dB per delay bin re the whole channel; 4 receivers, 53 s

| arrival | quantity | measured | A | | C | | B | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D+B | delay after direct (ms) | 0.000 +-0.000 | 0.000 +-0.000 | in |  |  |  |  |
| D+B | level re whole channel (dB) | -5.5 +-0.1 | -8.8 +-0.0 | OUT |  |  |  |  |
| D+B | the same with bubble loss | -5.5 +-0.1 | -5.4 | in |  |  |  |  |
| D+B | coherent share, at the peak | 1.005 +-0.000 | 1.007 +-0.000 | in | 1.025 +-0.000 | in |  |  |
| D+B | scintillation index | 0.00 +-0.00 | 0.00 +-0.00 | in | 0.00 +-0.00 | in |  |  |
| S | delay after direct (ms) | 1.152 +-0.010 | 1.075 +-0.000 | in |  |  |  |  |
| S | level re whole channel (dB) | -11.9 +-0.5 | -14.4 +-0.1 | OUT |  |  |  |  |
| S | the same with bubble loss | -11.9 +-0.5 | -17.2 | OUT |  |  |  |  |
| S | coherent share, arrival and tail | 0.045 +-0.013 | 0.002 +-0.002 | in | 0.002 +-0.001 | in |  |  |
| S | coherent share, at the peak | 0.044 +-0.012 | 0.001 +-0.000 | in | 0.001 +-0.001 | in |  |  |
| S | scintillation index | 0.89 +-0.12 | 0.94 +-0.05 | in | 1.10 +-0.02 | in |  |  |
| S | Doppler rms width (Hz) | 1.02 +-0.12 | 1.12 +-0.11 | in |  |  | 2.09 +-0.63 | in |
| S | Doppler 50 % half width (Hz) | 0.71 +-0.12 | 0.72 +-0.08 | in |  |  | 1.19 +-0.37 | in |
| S | Doppler shape w90/w50 | 2.24 +-0.13 | 2.59 +-0.42 | in |  |  | 4.30 +-1.71 | in |
| S | coherence time (ms) | 242 +-43 | 196 +-16 | in |  |  | 153 +-38 | in |
| S | tail decay time (ms) | 1.083 +-0.976 | - | - | - | - | - | - |
| S | tail mean excess delay (ms) | 0.154 +-0.008 | 0.195 +-0.004 | in | 0.037 +-0.001 | OUT | 0.037 +-0.001 | OUT |
| S | closed forms | | grazing 6.7 deg, Gamma^2 29.78, coherent 0.000 | | Doppler rms 1.15 Hz | | Lorentzian half width 1.15 Hz | |
| SB+BS | delay after direct (ms) | 1.574 +-0.076 | 1.651 +-0.000 | in |  |  |  |  |
| SB+BS | level re whole channel (dB) | -11.6 +-0.0 | -12.5 +-0.1 | OUT |  |  |  |  |
| SB+BS | the same with bubble loss | -11.6 +-0.0 | -14.3 | OUT |  |  |  |  |
| SB+BS | coherent share, arrival and tail | 0.041 +-0.013 | 0.010 +-0.010 | in | 0.007 +-0.006 | in |  |  |
| SB+BS | coherent share, at the peak | 0.038 +-0.011 | 0.006 +-0.008 | in | 0.002 +-0.001 | in |  |  |
| SB+BS | scintillation index | 0.83 +-0.02 | 0.70 +-0.03 | OUT | 0.98 +-0.03 | OUT |  |  |
| SB+BS | Doppler rms width (Hz) | 1.38 +-0.13 | 1.40 +-0.06 | in |  |  | 2.42 +-0.41 | OUT |
| SB+BS | Doppler 50 % half width (Hz) | 0.87 +-0.10 | 0.90 +-0.01 | in |  |  | 1.37 +-0.31 | in |
| SB+BS | Doppler shape w90/w50 | 2.68 +-0.11 | 2.56 +-0.17 | in |  |  | 3.89 +-0.90 | in |
| SB+BS | coherence time (ms) | 172 +-19 | 168 +-6 | in |  |  | 129 +-23 | in |
| SB+BS | tail decay time (ms) | 1.510 +-0.378 | 1.996 +-0.183 | in | 1.881 +-0.213 | in | 1.486 +-0.130 | in |
| SB+BS | tail mean excess delay (ms) | 0.148 +-0.013 | 0.205 +-0.005 | in | 0.298 +-0.002 | OUT | 0.296 +-0.005 | OUT |
| SB+BS | closed forms | | grazing 8.1 deg, Gamma^2 43.85, coherent 0.000 | | Doppler rms 1.39 Hz | | Lorentzian half width 1.39 Hz | |
| BSB | delay after direct (ms) | 1.843 +-0.028 | 2.355 +-0.111 | OUT |  |  |  |  |
| BSB | level re whole channel (dB) | -11.2 +-0.2 | -12.2 +-0.3 | in |  |  |  |  |
| BSB | the same with bubble loss | -11.2 +-0.2 | -13.3 | OUT |  |  |  |  |
| BSB | coherent share, arrival and tail | 0.013 +-0.004 | 0.004 +-0.000 | in | 0.006 +-0.001 | in |  |  |
| BSB | coherent share, at the peak | 0.022 +-0.008 | 0.004 +-0.002 | in | 0.007 +-0.001 | in |  |  |
| BSB | scintillation index | 0.81 +-0.07 | 0.79 +-0.06 | in | 0.81 +-0.04 | in |  |  |
| BSB | Doppler rms width (Hz) | 1.57 +-0.17 | 1.48 +-0.15 | in |  |  | 2.42 +-0.26 | OUT |
| BSB | Doppler 50 % half width (Hz) | 1.01 +-0.14 | 0.98 +-0.09 | in |  |  | 1.30 +-0.14 | in |
| BSB | Doppler shape w90/w50 | 2.57 +-0.16 | 2.45 +-0.08 | in |  |  | 4.19 +-0.48 | OUT |
| BSB | coherence time (ms) | 149 +-11 | 153 +-15 | in |  |  | 112 +-16 | in |
| BSB | tail decay time (ms) | 0.881 +-0.104 | 2.267 +-0.828 | in | 15.824 | OUT | 31.041 +-25.203 | in |
| BSB | tail mean excess delay (ms) | 0.419 +-0.013 | 0.458 +-0.011 | in | 0.481 +-0.005 | in | 0.487 +-0.006 | OUT |
| BSB | closed forms | | grazing 10.4 deg, Gamma^2 70.97, coherent 0.000 | | Doppler rms 1.77 Hz | | Lorentzian half width 1.77 Hz | |

