// Platform motion estimation: the Ceres solve of ASASIN Eq. 8.
//
//   L = L_DPC + lambda1 * L_smooth + lambda2 * L_DVL          (Eq. 8)
//
// with L_DPC built from the time-of-flight residual of Eq. 4, the smoothness term of
// Eq. 6 and the DVL altitude term of Eq. 7. Levenberg-Marquardt with automatic
// differentiation, Huber (Eq. 3) or squared loss.
//
// Parameterisation (DECISIONS.md D-30): the unknowns handed to Ceres are the sway and
// heave *positions* at each transmit, with the per-ping velocity defined as
// v_i = (p_{i+1} - p_i) / dt_i. This is the paper's first-order kinematic model and a
// linear, one-to-one change of variables from its velocity unknowns, so the loss and its
// minimiser are the same; it makes every residual depend on three neighbouring pings
// instead of on all earlier ones, which keeps the Jacobian sparse.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sonar/core/record.hpp"
#include "sonar/core/vec.hpp"

namespace sonar::motion {

/// What navigation knows about one ping before the solve.
struct PingNav {
  double time = 0.0;        ///< transmit time, s
  double x = 0.0;           ///< along-track position (from surge), m
  double surge = 0.0;       ///< along-track speed during the ping, m/s
  double y = 0.0;           ///< initial sway position, m
  double z = 0.0;           ///< initial heave position (NED, negative above the seafloor), m
  Vec3d rpy;                ///< attitude from the INS, rad
  double dvl_altitude = kNaN;  ///< m above the seafloor; NaN = no measurement
};

/// One time-delay measurement between ping `ping` and ping `ping + 1` (Eq. 4).
struct DelayMeasurement {
  std::int32_t ping = 0;
  Vec3d tx_lever_1, rx_lever_1;  ///< body-frame lever arms used on the first ping
  Vec3d tx_lever_2, rx_lever_2;  ///< and on the second
  double t_ref = 0.0;            ///< reference time t_k after transmit, s
  Vec3d footprint;               ///< p_s: the common seafloor point, world frame
  double delay = 0.0;            ///< delta = tau_2 - tau_1, s (DECISIONS.md D-13)
  double weight = 1.0;           ///< e.g. the correlation coefficient
};

/// The sound speed and the three scales have no default (docs/HARDCODED_AUDIT.md K-2, R-2):
/// `solve` throws when one is not stated. The scales set where the Huber loss turns from
/// quadratic to linear, so they belong to the sonar: the delay scale is a fraction of a
/// carrier period (python/gpusonar/motion.py uses a sixteenth; the 1e-6 s this struct used
/// to hold is that for a carrier of some hundreds of kHz and 1/200 of a period at 5 kHz).
struct SolveOptions {
  double sound_speed = kNaN;     ///< m/s; required
  double lambda_smooth = 1.0;    ///< lambda1
  double lambda_dvl = 1.0;       ///< lambda2
  bool huber = true;             ///< Eq. 3; false = squared loss
  double residual_scale = kNaN;  ///< s; residuals are divided by this before the loss; required
  double smooth_scale = kNaN;    ///< m/s; second differences of velocity are divided by this; required
  double dvl_scale = kNaN;       ///< m; altitude residuals are divided by this; required
  bool use_weights = true;
  int max_iterations = 100;
};

struct Solution {
  std::vector<double> y, z;    ///< positions at each transmit, plus one node after the last ping
  std::vector<double> vy, vz;  ///< velocity during each ping
  double initial_cost = 0.0;
  double final_cost = 0.0;
  int iterations = 0;
  bool converged = false;
  std::string report;
};

/// Minimise Eq. 8. `pings` are in time order; measurements refer to consecutive pairs.
Solution solve(const std::vector<PingNav>& pings, const std::vector<DelayMeasurement>& measurements,
               const SolveOptions& options);

/// The model's delay tau_2 - tau_1 for a measurement, given a trajectory (for phase
/// unwrapping and for tests). `y` and `z` have pings.size() + 1 entries.
double predict_delay(const std::vector<PingNav>& pings, const DelayMeasurement& m,
                     const std::vector<double>& y, const std::vector<double>& z, double sound_speed);

}  // namespace sonar::motion
