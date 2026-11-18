// What a detector's statistic is under noise alone, and the threshold that gives a stated
// false-alarm probability for it.
//
// A beamformed, pulse-compressed sample of complex Gaussian noise has a power that is exponentially
// distributed. A detector rarely tests one such sample: it tests the mean of n neighbouring samples
// (which are correlated when the map is sampled finer than the range resolution), and with stacked
// rows the sum of several independent maps. All of these are weighted sums of independent
// exponentials,
//
//     X = (1 / rows) * sum over rows r, components i of  w_i E_ri,      E_ri independent, mean 1,  sum w_i = 1,
//
// where the w_i are the eigenvalues of the samples' correlation matrix over n. One sample: w = {1}
// and X is Erlang with `rows` looks. The cell-averaging factor n_train (pfa^(-1/n_train) - 1) of
// cfar.hpp is the special case of one sample, one row, and a background estimated from n_train cells.
//
// The tail of X is computed exactly for the Erlang case and by the saddlepoint (Lugannani-Rice)
// formula otherwise; tests/cpp/test_statistic.cpp and tests/python/test_detector.py hold it to exact
// forms and to Monte Carlo.
#pragma once

#include <cstdint>
#include <vector>

namespace sonar::detect {

/// The weights w_i (they sum to 1) of the mean power of `n` consecutive samples whose complex
/// amplitudes correlate by `correlation[d]` at a lag of d samples (correlation[0] = 1; lags beyond
/// the vector are taken as uncorrelated). For more than `max_exact` samples the eigenvalues are not
/// computed: the mean is taken as `looks` equal components, looks = n / sum over lags of
/// |correlation|^2 (the variance-equivalent number), which is returned as that many equal weights
/// with the fractional remainder in the last one.
std::vector<double> mean_power_weights(const std::vector<double>& correlation, std::int64_t n, std::int64_t max_exact = 256);

/// 1 / sum w_i^2: the number of independent exponentials whose mean has the variance of X (one row).
double equivalent_looks(const std::vector<double>& weights);

/// P(X > x) under noise alone, X as above (mean 1).
double weighted_exponential_tail(const std::vector<double>& weights, std::int64_t rows, double x);

/// The x with P(X > x) = pfa: the threshold over the true mean noise power.
double weighted_exponential_threshold(const std::vector<double>& weights, std::int64_t rows, double pfa);

/// What a threshold over the TRUE mean must be multiplied by when the mean is estimated from
/// `n_train` independent exponential looks: n (pfa^(-1/n) - 1) / (-ln pfa). Exact for one sample of
/// one row (it is then the cell-averaging factor); for the other statistics it is the same
/// first-order correction (the false-alarm probability's slope against the threshold is the same at
/// the same pfa) and slightly conservative.
double training_correction(double pfa, double n_train);

}  // namespace sonar::detect
