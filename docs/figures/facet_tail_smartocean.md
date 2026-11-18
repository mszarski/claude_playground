# The reach of the facet tail against the measured tail of R1's surface reflection

Data: SFI Smart Ocean dataset (van Walree, Otnes, Tomasi, Henriksen, Danre, Bergh, IEEE DataPort 2025, doi:10.21227/3aa6-4k33, CC BY 4.0).

Made by `tools/channel_smartocean.py tail`: our time spreading for the R1 geometry with the surface fitted to the tail (two numbers: height, vertical speed; the slope is Cox-Munk's for the wind whose height that is, 0.0105, not fitted), sampled like the 4 Hz soundings and passed through the same fits as the measurement, with the taps spanning three mean decay times (as shipped) or the stated share of the shape's energy. Mean excess delay T read by each fit, ms; two seeds of the taps.

| Model | rule | T, exponential fit, first 0.9 ms | T, facet-shape fit, 0.9 ms | T, exponential fit, 3 ms | mean excess delay put in | last tap at, ms |
| --- | --- | --- | --- | --- | --- | --- |
| measured | | 0.50 | 1.08 | 0.96 | | |
| three mean decay times (as shipped), 24 taps | facet | 0.67 / 0.63 | 2.21 / 1.96 | 0.44 / 0.44 | 0.45 | 1.36 |
| three mean decay times (as shipped), 24 taps | rule_of_thumb | 4.00 / 4.00 | 4.00 / 4.00 | 0.96 / 1.02 | 0.85 | 2.56 |
| tail_quantile 0.9, 24 taps | facet | 0.39 / 0.39 | 0.53 / 0.50 | 0.29 / 0.27 | 0.45 | 1.20 |
| tail_quantile 0.99, 24 taps | facet | 0.37 / 0.37 | 0.56 / 0.53 | 0.56 / 0.60 | 0.45 | 2.82 |
| tail_quantile 0.99, 24 taps | rule_of_thumb | 1.54 / 1.45 | 4.00 / 4.00 | 0.96 / 0.96 | 0.85 | 3.93 |
| tail_quantile 0.99, 48 taps | facet | 0.39 / 0.37 | 0.53 / 0.47 | 0.56 / 0.60 | 0.45 | 2.82 |
| tail_quantile 0.999, 48 taps | facet | 0.39 / 0.37 | 0.53 / 0.50 | 0.60 / 0.63 | 0.45 | 4.53 |
