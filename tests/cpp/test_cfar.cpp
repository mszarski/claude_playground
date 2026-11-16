#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "sonar/detect/cfar.hpp"

using sonar::Backend;
using sonar::detect::cfar;
using sonar::detect::CfarOptions;

TEST(Cfar, ThresholdFactorMatchesTheClosedForm) {
  // For n training cells of exponential noise, P(x > a * mean) = (1 + a / n)^-n.
  for (int n : {8, 32, 64}) {
    const double a = sonar::detect::cfar_threshold_factor(1e-4, n);
    EXPECT_NEAR(std::pow(1.0 + a / n, -n), 1e-4, 1e-12);
  }
}

TEST(Cfar, FlatMapHasNoDetectionsAndASpikeIsFound) {
  const std::int64_t rows = 3, cols = 200;
  std::vector<float> power(rows * cols, 2.0F);
  std::vector<unsigned char> det(power.size(), 7);
  std::vector<float> ratio(power.size(), -1.0F);
  cfar(Backend::Host, power.data(), rows, cols, CfarOptions{}, det.data(), ratio.data());
  for (std::size_t i = 0; i < det.size(); ++i) {
    EXPECT_EQ(det[i], 0);
    EXPECT_FLOAT_EQ(ratio[i], 1.0F);
  }
  power[1 * cols + 100] = 200.0F;
  cfar(Backend::Host, power.data(), rows, cols, CfarOptions{}, det.data(), ratio.data());
  int total = 0;
  for (unsigned char d : det) total += d;
  EXPECT_EQ(total, 1);
  EXPECT_EQ(det[1 * cols + 100], 1);
  EXPECT_FLOAT_EQ(ratio[1 * cols + 100], 100.0F);
  // Cells whose training window holds the spike see a raised background.
  EXPECT_LT(ratio[1 * cols + 110], 1.0F);
  // Cells inside the guard do not.
  EXPECT_FLOAT_EQ(ratio[1 * cols + 103], 1.0F);
}

TEST(Cfar, EdgesUseTheCellsThatExist) {
  const std::int64_t cols = 50;
  std::vector<float> power(cols, 1.0F);
  power[0] = 1000.0F;
  power[cols - 1] = 1000.0F;
  std::vector<unsigned char> det(cols);
  CfarOptions o;
  o.guard = 2;
  o.train = 8;
  cfar(Backend::Host, power.data(), 1, cols, o, det.data());
  EXPECT_EQ(det[0], 1);
  EXPECT_EQ(det[cols - 1], 1);
}

TEST(Cfar, RejectsBadOptions) {
  std::vector<float> power(10, 1.0F);
  std::vector<unsigned char> det(10);
  CfarOptions o;
  o.train = 0;
  EXPECT_THROW(cfar(Backend::Host, power.data(), 1, 10, o, det.data()), std::invalid_argument);
  o = CfarOptions{};
  o.pfa = 0.0;
  EXPECT_THROW(cfar(Backend::Host, power.data(), 1, 10, o, det.data()), std::invalid_argument);
}
