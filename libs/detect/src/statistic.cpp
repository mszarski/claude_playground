#include "sonar/detect/statistic.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Dense>

namespace sonar::detect {
namespace {

// P(Erlang with `m` looks and mean 1 > x) = exp(-m x) sum_{j < m} (m x)^j / j!
double erlang_tail(std::int64_t m, double x) {
  const double y = static_cast<double>(m) * x;
  double term = 1.0, sum = 1.0;
  for (std::int64_t j = 1; j < m; ++j) {
    term *= y / static_cast<double>(j);
    sum += term;
  }
  return std::exp(-y) * sum;
}

}  // namespace

std::vector<double> mean_power_weights(const std::vector<double>& correlation, std::int64_t n, std::int64_t max_exact) {
  if (n < 1 || correlation.empty()) throw std::invalid_argument("mean_power_weights: n >= 1 and a correlation");
  const auto at = [&](std::int64_t d) { return d < static_cast<std::int64_t>(correlation.size()) ? correlation[static_cast<std::size_t>(d)] : 0.0; };
  if (n == 1) return {1.0};
  if (n > max_exact) {
    // Variance of the mean of n powers whose correlation at lag d is |rho_d|^2: (1 / n) sum_d (1 - |d| / n) |rho_d|^2.
    double s = 1.0;
    for (std::int64_t d = 1; d < n; ++d) s += 2.0 * (1.0 - static_cast<double>(d) / static_cast<double>(n)) * at(d) * at(d);
    const double looks = std::max(static_cast<double>(n) / s, 1.0);
    const auto whole = static_cast<std::int64_t>(std::floor(looks));
    std::vector<double> w(static_cast<std::size_t>(whole), 1.0 / looks);
    if (looks - static_cast<double>(whole) > 1e-9) w.push_back((looks - static_cast<double>(whole)) / looks);
    return w;
  }
  Eigen::MatrixXd c(n, n);
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t j = 0; j < n; ++j) c(i, j) = at(std::abs(i - j));
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(c, Eigen::EigenvaluesOnly);
  std::vector<double> w;
  double total = 0.0;
  for (std::int64_t i = 0; i < n; ++i) {
    const double v = std::max(es.eigenvalues()(i), 0.0);
    total += v;
    w.push_back(v);
  }
  for (double& v : w) v /= total;
  std::sort(w.begin(), w.end(), [](double a, double b) { return a > b; });
  while (w.size() > 1 && w.back() < 1e-12) w.pop_back();
  return w;
}

double equivalent_looks(const std::vector<double>& weights) {
  double s = 0.0, t = 0.0;
  for (double w : weights) {
    s += w * w;
    t += w;
  }
  return t * t / s;
}

double weighted_exponential_tail(const std::vector<double>& weights, std::int64_t rows, double x) {
  if (weights.empty() || rows < 1) throw std::invalid_argument("weighted_exponential_tail: weights and rows >= 1");
  if (!(x > 0.0)) return 1.0;
  if (weights.size() == 1) return erlang_tail(rows, x);
  const auto r = static_cast<double>(rows);
  double total = 0.0, top = 0.0;
  for (double w : weights) {
    total += w;
    top = std::max(top, w);
  }
  // Cumulant generating function K(t) = -rows sum ln(1 - a_i t), a_i = w_i / (rows total); K'(t) = x at the saddlepoint.
  const auto derivative = [&](double t) {
    double s = 0.0;
    for (double w : weights) {
      const double a = w / (r * total);
      s += r * a / (1.0 - a * t);
    }
    return s;
  };
  if (x <= 1.0 + 1e-9) return 0.5;                  // at or below the mean: not a threshold anyone asks for
  double lo = 0.0, hi = r * total / top * (1.0 - 1e-14);
  for (int it = 0; it < 200; ++it) {
    const double mid = 0.5 * (lo + hi);
    (derivative(mid) < x ? lo : hi) = mid;
  }
  const double t = 0.5 * (lo + hi);
  double k = 0.0, k2 = 0.0;
  for (double w : weights) {
    const double a = w / (r * total);
    k -= r * std::log1p(-a * t);
    k2 += r * a * a / ((1.0 - a * t) * (1.0 - a * t));
  }
  const double w0 = std::sqrt(std::max(2.0 * (t * x - k), 0.0)), u = t * std::sqrt(k2);
  if (!(w0 > 1e-6) || !(u > 0.0)) return 0.5;
  const double pdf = std::exp(-0.5 * w0 * w0) / std::sqrt(2.0 * 3.14159265358979323846);
  return std::clamp(0.5 * std::erfc(w0 / std::sqrt(2.0)) + pdf * (1.0 / u - 1.0 / w0), 0.0, 1.0);
}

double weighted_exponential_threshold(const std::vector<double>& weights, std::int64_t rows, double pfa) {
  if (!(pfa > 0.0 && pfa < 0.2)) throw std::invalid_argument("weighted_exponential_threshold: pfa in (0, 0.2)");
  double lo = 1.0 + 0.5 / std::sqrt(equivalent_looks(weights) * static_cast<double>(rows)), hi = 2.0 * lo;
  while (weighted_exponential_tail(weights, rows, hi) > pfa) hi *= 1.5;
  for (int it = 0; it < 100; ++it) {
    const double mid = 0.5 * (lo + hi);
    (weighted_exponential_tail(weights, rows, mid) > pfa ? lo : hi) = mid;
  }
  return 0.5 * (lo + hi);
}

double training_correction(double pfa, double n_train) {
  if (!(n_train > 0.0) || !std::isfinite(n_train)) return 1.0;
  return n_train * std::expm1(-std::log(pfa) / n_train) / (-std::log(pfa));
}

}  // namespace sonar::detect
