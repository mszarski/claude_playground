// ASASIN Eq. 2 and its sign conventions (PAPER_NOTES.md, Sec. II-B).
#include <gtest/gtest.h>

#include <cmath>

#include "sonar/core/record.hpp"
#include "sonar/core/vec.hpp"

using namespace sonar;

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;
}  // namespace

TEST(RotationNed, MatchesEquation2EntryByEntry) {
  const double phi = 0.31, theta = -0.22, psi = 1.17;  // roll, pitch, yaw
  const Mat3d r = rotation_ned(phi, theta, psi);
  const double cph = std::cos(phi), sph = std::sin(phi);
  const double cth = std::cos(theta), sth = std::sin(theta);
  const double cps = std::cos(psi), sps = std::sin(psi);
  // Rows exactly as printed in the paper.
  EXPECT_NEAR(r.m[0][0], cth * cps, 1e-15);
  EXPECT_NEAR(r.m[0][1], cps * sth * sph - cph * sps, 1e-15);
  EXPECT_NEAR(r.m[0][2], cph * cps * sth + sph * sps, 1e-15);
  EXPECT_NEAR(r.m[1][0], cth * sps, 1e-15);
  EXPECT_NEAR(r.m[1][1], cph * cps + sth * sph * sps, 1e-15);
  EXPECT_NEAR(r.m[1][2], cph * sth * sps - cps * sph, 1e-15);
  EXPECT_NEAR(r.m[2][0], -sth, 1e-15);
  EXPECT_NEAR(r.m[2][1], cth * sph, 1e-15);
  EXPECT_NEAR(r.m[2][2], cth * cph, 1e-15);
}

TEST(RotationNed, IsOrthonormalWithUnitDeterminant) {
  const Mat3d r = rotation_ned(0.4, -0.3, 2.0);
  const Mat3d i = r * r.transposed();
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) EXPECT_NEAR(i.m[a][b], a == b ? 1.0 : 0.0, 1e-14);
  const Vec3d c0{r.m[0][0], r.m[1][0], r.m[2][0]};
  const Vec3d c1{r.m[0][1], r.m[1][1], r.m[2][1]};
  const Vec3d c2{r.m[0][2], r.m[1][2], r.m[2][2]};
  EXPECT_NEAR(dot(cross(c0, c1), c2), 1.0, 1e-14);
}

TEST(RotationNed, PositiveRollLowersStarboard) {
  // A point on the starboard side (body +y) moves down (NED +z).
  const Vec3d p = rotation_ned(10.0 * kDeg, 0.0, 0.0) * Vec3d{0.0, 1.0, 0.0};
  EXPECT_GT(p.z, 0.0);
}

TEST(RotationNed, PositivePitchRaisesBow) {
  // A point at the bow (body +x) moves up (NED -z).
  const Vec3d p = rotation_ned(0.0, 10.0 * kDeg, 0.0) * Vec3d{1.0, 0.0, 0.0};
  EXPECT_LT(p.z, 0.0);
}

TEST(RotationNed, PositiveYawRotatesToStarboard) {
  // The bow (body +x, initially north) swings towards east (NED +y).
  const Vec3d p = rotation_ned(0.0, 0.0, 10.0 * kDeg) * Vec3d{1.0, 0.0, 0.0};
  EXPECT_GT(p.y, 0.0);
}

TEST(WorldPosition, AppliesLeverArmAndFirstOrderMotion) {
  Pose pose;
  pose.position = {10.0, 20.0, -5.0};
  pose.rpy = {0.0, 0.0, kPi / 2.0};  // heading east
  pose.velocity = {0.0, 2.0, 0.0};
  pose.rates = {0.0, 0.0, 0.0};
  const Vec3d lever{1.0, 0.5, 0.0};  // 1 m forward, 0.5 m to starboard
  const Vec3d p0 = world_position(pose, lever);
  // Forward is east; starboard is south.
  EXPECT_NEAR(p0.x, 10.0 - 0.5, 1e-12);
  EXPECT_NEAR(p0.y, 20.0 + 1.0, 1e-12);
  EXPECT_NEAR(p0.z, -5.0, 1e-12);
  const Vec3d p1 = world_position(pose, lever, 0.5);
  EXPECT_NEAR(p1.y - p0.y, 1.0, 1e-12);
  EXPECT_NEAR(p1.x - p0.x, 0.0, 1e-12);
}

TEST(WorldPosition, AttitudeRateMovesLeverArm) {
  Pose pose;
  pose.rates = {0.0, 0.0, 0.1};  // yawing to starboard
  const Vec3d lever{2.0, 0.0, 0.0};
  const Vec3d v = world_velocity(pose, lever);
  // Bow sweeps east at omega * r.
  EXPECT_NEAR(v.y, 0.2, 1e-6);
  EXPECT_NEAR(v.x, 0.0, 1e-6);
}
