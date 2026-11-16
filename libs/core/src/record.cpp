#include "sonar/core/record.hpp"

namespace sonar {

Vec3d world_position(const Pose& pose, const Vec3d& lever_arm, double t) {
  const Vec3d rpy = pose.rpy + pose.rates * t;
  const Mat3d r = rotation_ned(rpy.x, rpy.y, rpy.z);
  return pose.position + pose.velocity * t + r * lever_arm;
}

Vec3d world_direction(const Pose& pose, const Vec3d& body_dir, double t) {
  const Vec3d rpy = pose.rpy + pose.rates * t;
  return rotation_ned(rpy.x, rpy.y, rpy.z) * body_dir;
}

Vec3d world_velocity(const Pose& pose, const Vec3d& lever_arm) {
  // Central difference of the first-order model; exact enough for the small
  // attitude rates involved and free of Euler-rate algebra.
  const double h = 1e-4;
  const Vec3d a = world_position(pose, lever_arm, -h);
  const Vec3d b = world_position(pose, lever_arm, +h);
  return (b - a) / (2.0 * h);
}

void set_default_channel_tables(PingRecord& record) {
  const auto n = static_cast<std::size_t>(record.samples.n_channels);
  auto& cal = record.calibration;
  if (cal.gain.size() != n) cal.gain.assign(n, 1.0);
  if (cal.phase.size() != n) cal.phase.assign(n, 0.0);
  if (cal.delay.size() != n) cal.delay.assign(n, 0.0);
  if (cal.position_offset.size() != n) cal.position_offset.assign(n, Vec3d{});
  auto& meta = record.metadata;
  if (meta.channel_element.size() != n) {
    meta.channel_element.resize(n);
    for (std::size_t i = 0; i < n; ++i) meta.channel_element[i] = static_cast<std::int32_t>(i);
  }
  if (meta.channel_board.size() != n) meta.channel_board.assign(n, 0);
  if (meta.boards.empty()) meta.boards.emplace_back();
}

}  // namespace sonar
