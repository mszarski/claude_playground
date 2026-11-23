#include "sonar/motion/dpc_solver.hpp"

#include <ceres/ceres.h>

#include <cmath>
#include <stdexcept>

namespace sonar::motion {

namespace {

/// Two-way time of flight of one ping to the footprint, with the receiver moved on to
/// its position at the reference time (transmission is instantaneous; Eq. 4).
template <typename T>
T time_of_flight(const T& y0, const T& z0, const T& y1, const T& z1, double x, double surge,
                 double dt, const Mat3d& rot, const Vec3d& tx_lever, const Vec3d& rx_lever,
                 const Vec3d& footprint, double t_ref, double c) {
  const Vec3d tx_off = rot * tx_lever;
  const Vec3d rx_off = rot * rx_lever;
  // Transmitter at the transmit instant.
  const T tx_x = T(x + tx_off.x), tx_y = y0 + T(tx_off.y), tx_z = z0 + T(tx_off.z);
  // Receiver after t_ref of first-order motion: p = p0 + v t, v = (p1 - p0) / dt.
  const T vy = (y1 - y0) / T(dt), vz = (z1 - z0) / T(dt);
  const T rx_x = T(x + surge * t_ref + rx_off.x);
  const T rx_y = y0 + vy * T(t_ref) + T(rx_off.y);
  const T rx_z = z0 + vz * T(t_ref) + T(rx_off.z);
  using std::sqrt;
  const T out = sqrt((tx_x - T(footprint.x)) * (tx_x - T(footprint.x)) +
                     (tx_y - T(footprint.y)) * (tx_y - T(footprint.y)) +
                     (tx_z - T(footprint.z)) * (tx_z - T(footprint.z)));
  const T back = sqrt((rx_x - T(footprint.x)) * (rx_x - T(footprint.x)) +
                      (rx_y - T(footprint.y)) * (rx_y - T(footprint.y)) +
                      (rx_z - T(footprint.z)) * (rx_z - T(footprint.z)));
  return (out + back) / T(c);
}

/// Eq. 4: g = (path_1 - path_2) / c + delta, scaled.
struct DelayResidual {
  const PingNav& a;
  const PingNav& b;
  DelayMeasurement m;
  double dt_a, dt_b, c, scale;
  Mat3d rot_a, rot_b;

  template <typename T>
  bool operator()(const T* const p0, const T* const p1, const T* const p2, T* residual) const {
    // p0, p1, p2: (y, z) at pings i, i + 1, i + 2.
    const T tof1 = time_of_flight(p0[0], p0[1], p1[0], p1[1], a.x, a.surge, dt_a, rot_a,
                                  m.tx_lever_1, m.rx_lever_1, m.footprint, m.t_ref, c);
    const T tof2 = time_of_flight(p1[0], p1[1], p2[0], p2[1], b.x, b.surge, dt_b, rot_b,
                                  m.tx_lever_2, m.rx_lever_2, m.footprint, m.t_ref, c);
    residual[0] = (tof1 - tof2 + T(m.delay)) * T(m.weight / scale);
    return true;
  }
};

/// Eq. 6: second difference of the velocity sequence, for sway and heave.
struct SmoothResidual {
  double dt0, dt1, dt2, scale;
  template <typename T>
  bool operator()(const T* const p0, const T* const p1, const T* const p2, const T* const p3,
                  T* residual) const {
    for (int k = 0; k < 2; ++k) {
      const T v0 = (p1[k] - p0[k]) / T(dt0);
      const T v1 = (p2[k] - p1[k]) / T(dt1);
      const T v2 = (p3[k] - p2[k]) / T(dt2);
      residual[k] = (v0 - T(2.0) * v1 + v2) / T(scale);
    }
    return true;
  }
};

/// Eq. 7: altitude of ping i against the DVL (altitude = -z; DECISIONS.md D-15).
struct DvlResidual {
  double altitude, scale;
  template <typename T>
  bool operator()(const T* const p, T* residual) const {
    residual[0] = (-p[1] - T(altitude)) / T(scale);
    return true;
  }
};

double interval(const std::vector<PingNav>& pings, std::size_t i) {
  const std::size_t n = pings.size();
  if (i + 1 < n) return pings[i + 1].time - pings[i].time;
  return pings[n - 1].time - pings[n - 2].time;  // the node after the last ping
}

}  // namespace

double predict_delay(const std::vector<PingNav>& pings, const DelayMeasurement& m,
                     const std::vector<double>& y, const std::vector<double>& z, double c) {
  const auto i = static_cast<std::size_t>(m.ping);
  const PingNav& a = pings.at(i);
  const PingNav& b = pings.at(i + 1);
  const Mat3d ra = rotation_ned(a.rpy.x, a.rpy.y, a.rpy.z);
  const Mat3d rb = rotation_ned(b.rpy.x, b.rpy.y, b.rpy.z);
  const double t1 = time_of_flight<double>(y.at(i), z.at(i), y.at(i + 1), z.at(i + 1), a.x, a.surge,
                                           interval(pings, i), ra, m.tx_lever_1, m.rx_lever_1,
                                           m.footprint, m.t_ref, c);
  const double t2 = time_of_flight<double>(y.at(i + 1), z.at(i + 1), y.at(i + 2), z.at(i + 2), b.x,
                                           b.surge, interval(pings, i + 1), rb, m.tx_lever_2,
                                           m.rx_lever_2, m.footprint, m.t_ref, c);
  return t2 - t1;
}

Solution solve(const std::vector<PingNav>& pings, const std::vector<DelayMeasurement>& measurements,
               const SolveOptions& opt) {
  const std::size_t n = pings.size();
  if (n < 3) throw std::invalid_argument("motion::solve needs at least three pings");
  const auto stated = [](double v) { return v > 0.0 && std::isfinite(v); };
  if (!stated(opt.sound_speed)) throw std::invalid_argument("motion::solve: sound_speed must be stated; there is no default");
  if (!stated(opt.residual_scale) || !stated(opt.smooth_scale) || !stated(opt.dvl_scale)) {
    throw std::invalid_argument(
        "motion::solve: residual_scale (s), smooth_scale (m/s) and dvl_scale (m) must be stated; they depend on "
        "the carrier and the platform and have no default (python/gpusonar/motion.py derives them)");
  }
  for (std::size_t i = 1; i < n; ++i) {
    if (!(pings[i].time > pings[i - 1].time)) {
      throw std::invalid_argument("motion::solve: ping times must increase");
    }
  }
  // Position nodes: one per ping plus one after the last, so the last ping has a velocity.
  std::vector<double> nodes(2 * (n + 1));
  for (std::size_t i = 0; i < n; ++i) {
    nodes[2 * i] = pings[i].y;
    nodes[2 * i + 1] = pings[i].z;
  }
  nodes[2 * n] = 2.0 * pings[n - 1].y - pings[n - 2].y;
  nodes[2 * n + 1] = 2.0 * pings[n - 1].z - pings[n - 2].z;

  ceres::Problem problem;
  for (std::size_t i = 0; i <= n; ++i) problem.AddParameterBlock(&nodes[2 * i], 2);
  // The first ping anchors the frame: its sway is the navigated one (DPC sees only relative
  // sway). Its height starts at minus its DVL altitude (D-15). Held there, one DVL sample's
  // noise offsets every height and, through the line of sight, the sway (sonar-jh8); so unless
  // fix_first_height, the height is free and its own Eq. 7 term below ties it to that sample.
  const bool first_height_free = !opt.fix_first_height && opt.lambda_dvl > 0.0 &&
                                 std::isfinite(pings[0].dvl_altitude);
  if (first_height_free) {
    problem.SetManifold(&nodes[0], new ceres::SubsetManifold(2, {0}));
  } else {
    problem.SetParameterBlockConstant(&nodes[0]);
  }

  for (const DelayMeasurement& m : measurements) {
    if (m.ping < 0 || static_cast<std::size_t>(m.ping) + 1 >= n) {
      throw std::invalid_argument("motion::solve: measurement refers to a ping pair that does not exist");
    }
    const auto i = static_cast<std::size_t>(m.ping);
    DelayMeasurement mm = m;
    if (!opt.use_weights) mm.weight = 1.0;
    auto* f = new DelayResidual{pings[i], pings[i + 1], mm, interval(pings, i), interval(pings, i + 1),
                                opt.sound_speed, opt.residual_scale,
                                rotation_ned(pings[i].rpy.x, pings[i].rpy.y, pings[i].rpy.z),
                                rotation_ned(pings[i + 1].rpy.x, pings[i + 1].rpy.y, pings[i + 1].rpy.z)};
    ceres::LossFunction* loss = opt.huber ? new ceres::HuberLoss(1.0) : nullptr;
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<DelayResidual, 1, 2, 2, 2>(f), loss,
                             &nodes[2 * i], &nodes[2 * (i + 1)], &nodes[2 * (i + 2)]);
  }
  if (opt.lambda_smooth > 0.0) {
    for (std::size_t i = 0; i + 3 <= n; ++i) {
      auto* f = new SmoothResidual{interval(pings, i), interval(pings, i + 1), interval(pings, i + 2),
                                   opt.smooth_scale};
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<SmoothResidual, 2, 2, 2, 2, 2>(f),
          new ceres::ScaledLoss(nullptr, opt.lambda_smooth, ceres::TAKE_OWNERSHIP), &nodes[2 * i],
          &nodes[2 * (i + 1)], &nodes[2 * (i + 2)], &nodes[2 * (i + 3)]);
    }
  }
  if (opt.lambda_dvl > 0.0) {
    for (std::size_t i = first_height_free ? 0 : 1; i < n; ++i) {
      if (std::isnan(pings[i].dvl_altitude)) continue;
      auto* f = new DvlResidual{pings[i].dvl_altitude, opt.dvl_scale};
      problem.AddResidualBlock(new ceres::AutoDiffCostFunction<DvlResidual, 1, 2>(f),
                               new ceres::ScaledLoss(nullptr, opt.lambda_dvl, ceres::TAKE_OWNERSHIP),
                               &nodes[2 * i]);
    }
  }

  ceres::Solver::Options so;
  so.minimizer_type = ceres::TRUST_REGION;
  so.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  so.linear_solver_type = ceres::DENSE_QR;  // small problems; no sparse backend needed
  so.max_num_iterations = opt.max_iterations;
  so.function_tolerance = 1e-12;
  so.parameter_tolerance = 1e-12;
  so.gradient_tolerance = 1e-14;
  so.logging_type = ceres::SILENT;
  so.num_threads = 4;
  ceres::Solver::Summary summary;
  ceres::Solve(so, &problem, &summary);

  Solution out;
  out.y.resize(n + 1);
  out.z.resize(n + 1);
  for (std::size_t i = 0; i <= n; ++i) {
    out.y[i] = nodes[2 * i];
    out.z[i] = nodes[2 * i + 1];
  }
  out.vy.resize(n);
  out.vz.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    out.vy[i] = (out.y[i + 1] - out.y[i]) / interval(pings, i);
    out.vz[i] = (out.z[i + 1] - out.z[i]) / interval(pings, i);
  }
  out.initial_cost = summary.initial_cost;
  out.final_cost = summary.final_cost;
  out.iterations = static_cast<int>(summary.iterations.size());
  out.converged = summary.termination_type == ceres::CONVERGENCE;
  out.report = summary.BriefReport();
  return out;
}

}  // namespace sonar::motion
