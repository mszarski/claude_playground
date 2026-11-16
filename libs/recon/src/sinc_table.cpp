#include "sonar/recon/sinc_table.hpp"

#include <cmath>

#include "sonar/recon/backproject.hpp"

namespace sonar::recon {

namespace {

double bessel_i0(double x) {
  // Power series; converges quickly for the arguments used here.
  double sum = 1.0, term = 1.0;
  for (int k = 1; k < 50; ++k) {
    term *= (x / (2.0 * k)) * (x / (2.0 * k));
    sum += term;
    if (term < 1e-16 * sum) break;
  }
  return sum;
}

}  // namespace

std::vector<float> make_sinc_table() {
  const double pi = 3.14159265358979323846;
  const int half = kSincTaps / 2;
  std::vector<float> table(static_cast<std::size_t>((kSincPhases + 1) * kSincTaps));
  const double i0_beta = bessel_i0(kSincKaiserBeta);
  for (int p = 0; p <= kSincPhases; ++p) {
    const double frac = static_cast<double>(p) / kSincPhases;
    double row[kSincTaps];
    double sum = 0.0;
    for (int k = 0; k < kSincTaps; ++k) {
      const double x = static_cast<double>(k - (half - 1)) - frac;  // distance in samples
      const double s = std::abs(x) < 1e-12 ? 1.0 : std::sin(pi * x) / (pi * x);
      const double u = x / half;
      const double win = std::abs(u) < 1.0
                             ? bessel_i0(kSincKaiserBeta * std::sqrt(1.0 - u * u)) / i0_beta
                             : 0.0;
      row[k] = s * win;
      sum += row[k];
    }
    for (int k = 0; k < kSincTaps; ++k) {
      table[static_cast<std::size_t>(p * kSincTaps + k)] = static_cast<float>(row[k] / sum);
    }
  }
  return table;
}

}  // namespace sonar::recon
