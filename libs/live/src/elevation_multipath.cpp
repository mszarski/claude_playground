// What unresolved boundary echoes do to a depth from stacked rows: the derivation is with the declaration
// (elevation.hpp, multipath_depth_limit). One function, called by both chain engines (the Python engine through the
// binding), so their reports carry the same numbers. The rays are the chain's own (raytrace.hpp).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "sonar/live/elevation.hpp"
#include "sonar/live/raytrace.hpp"

namespace sonar::live {
namespace {

constexpr double kPi = 3.14159265358979323846;
// How finely the calculation is laid out (none of these is a property of the water or of the sonar):
constexpr std::int64_t kMinDepths = 32, kMaxDepths = 256;   // target depths tried: half the angle's own depth sigma apart, within these
constexpr std::int64_t kMinRays = 256, kMaxRays = 2048;     // rays of the fan: a 64th of the depths tried apart at the far range, within these
constexpr std::int64_t kRanges = 8;                         // horizontal ranges the fan is read at (a deep target at a slant range is nearer)

struct Echo {
  double range, amplitude, elevation;   ///< half the two-way path (m), pressure re the strongest unit, rad up positive
  int count;                            ///< 1: out and back by one path; 2: a path pair and its mirror, added
};


}  // namespace

MultipathLimit multipath_depth_limit(const std::vector<double>& ssp_depth, const std::vector<double>& ssp_speed, double water_depth,
                                     double own_depth, double slant_range, double elevation, double sigma_angle, double reported_depth,
                                     double range_resolution, double aperture_resolution, bool first_arrival, double first_arrival_db,
                                     double explain_sigma, int max_bounces, double max_depth) {
  MultipathLimit out;
  if (ssp_depth.empty() || ssp_depth.size() != ssp_speed.size()) return out;
  if (!(slant_range > 0.0 && sigma_angle > 0.0 && range_resolution > 0.0 && std::isfinite(elevation) && std::isfinite(reported_depth))) return out;
  const RayProfile prof(ssp_depth, ssp_speed, water_depth);
  out.bottom = prof.has_bottom;
  // The depths a target at this slant range can be at.
  const double reach = 0.98 * slant_range;
  const double z_lo = std::max(own_depth - reach, 0.0);
  const double z_hi = std::min(prof.has_bottom ? prof.bottom : std::max(max_depth, own_depth), own_depth + reach);
  out.deepest = z_hi;
  if (!(z_hi > z_lo)) return out;
  const double own_sigma = slant_range * std::cos(elevation) * sigma_angle;   // the angle's own depth sigma, m
  const auto n_z = std::clamp(static_cast<std::int64_t>(std::ceil((z_hi - z_lo) / (0.5 * own_sigma))), kMinDepths, kMaxDepths);
  const double step = (z_hi - z_lo) / static_cast<double>(n_z);
  // A Gaussian narrower than the step between depths would fall between them: it is widened to the step (the same
  // widening is in the reference it is compared with below, so it adds nothing by itself).
  const double sigma_eff = std::hypot(sigma_angle, step / (slant_range * std::max(std::cos(elevation), 1e-6)));
  const double far = std::max(z_hi - own_depth, own_depth - z_lo);
  const double r_far = slant_range, r_near = std::sqrt(std::max(slant_range * slant_range - far * far, 1.0));
  std::vector<double> ranges;
  if (r_far - r_near < 1e-3 * r_far) {
    ranges = {0.5 * (r_near + r_far)};
  } else {
    for (std::int64_t i = 0; i < kRanges; ++i) ranges.push_back(r_near + (r_far - r_near) * static_cast<double>(i) / static_cast<double>(kRanges - 1));
  }
  const int bounces = std::max(max_bounces, 1);
  // The steepest ray any of these paths needs, by straight lines, with a margin for the bending (as multipath_rays.cpp).
  const double rise = prof.has_bottom ? own_depth + z_hi + 2.0 * prof.bottom * static_cast<double>(bounces)
                                      : own_depth + z_hi + 2.0 * z_hi * static_cast<double>(bounces - 1);
  const double max_angle = std::min(1.15 * std::atan2(rise, r_near) + 3.0 * kPi / 180.0, 80.0 * kPi / 180.0);
  const auto n_rays = std::clamp(static_cast<std::int64_t>(std::ceil(2.0 * max_angle * r_far / (step * static_cast<double>(n_z) / 64.0))), kMinRays, kMaxRays);
  const RayFan fan(prof, own_depth, ranges, max_angle, n_rays, bounces);
  const double c0 = prof.c(own_depth);
  const double floor_ratio = std::pow(10.0, -first_arrival_db / 10.0);

  // Over the depths: how well each can give the reading (1: exactly), and its squared distance from the reported depth
  // weighted by that; the same with the first path alone and resolved.
  double sum_with = 0.0, sum_with_d2 = 0.0, sum_alone = 0.0, sum_alone_d2 = 0.0, best = 0.0;
  bool direct = false;
  const double floor_can = std::exp(-0.5 * explain_sigma * explain_sigma);
  std::vector<RayArrival> paths;
  std::vector<Echo> echoes;
  for (std::int64_t j = 0; j < n_z; ++j) {
    const double z = z_lo + (static_cast<double>(j) + 0.5) * step;
    const double r = std::sqrt(std::max(slant_range * slant_range - (z - own_depth) * (z - own_depth), 1.0));
    std::size_t at = 0;
    for (std::size_t q = 1; q < ranges.size(); ++q) {
      if (std::abs(ranges[q] - r) < std::abs(ranges[at] - r)) at = q;
    }
    paths.clear();
    fan.arrivals(at, z, paths);
    if (paths.empty()) continue;                             // no sound reaches this depth at all (a shadow)
    // Every two-way echo: out by path i and back by path k. (i, k) and (k, i) are one echo of twice the pressure at
    // the mean of the two elevations.
    echoes.clear();
    std::size_t first = 0;
    for (std::size_t i = 0; i < paths.size(); ++i) {
      if (paths[i].time < paths[first].time) first = i;
      for (std::size_t k = i; k < paths.size(); ++k) {
        echoes.push_back({0.5 * c0 * (paths[i].time + paths[k].time), (i == k ? 1.0 : 2.0) * paths[i].amplitude * paths[k].amplitude,
                          -0.5 * (paths[i].launch + paths[k].launch), i == k ? 1 : 2});
      }
    }
    std::sort(echoes.begin(), echoes.end(), [](const Echo& a, const Echo& b) { return a.range < b.range; });
    // Clumps of echoes the range resolution cannot separate (they make one maximum along range), and the one the
    // stage reads: the earliest within first_arrival_db of the strongest (expected power).
    std::vector<std::size_t> start{0};
    for (std::size_t g = 1; g < echoes.size(); ++g) {
      if (echoes[g].range - echoes[g - 1].range >= range_resolution) start.push_back(g);
    }
    start.push_back(echoes.size());
    std::vector<double> power(start.size() - 1, 0.0);
    std::size_t strongest = 0;
    for (std::size_t c = 0; c + 1 < start.size(); ++c) {
      for (std::size_t g = start[c]; g < start[c + 1]; ++g) power[c] += echoes[g].amplitude * echoes[g].amplitude;
      if (power[c] > power[strongest]) strongest = c;
    }
    if (!(power[strongest] > 0.0)) continue;
    std::size_t pick = strongest;
    for (std::size_t c = 0; first_arrival && c < strongest; ++c) {
      if (power[c] >= power[strongest] * floor_ratio) {
        pick = c;
        break;
      }
    }
    // How well this depth can give the reading: by the echo of the clump that does it best. Which echo of a clump
    // the rows read is not known from here (it turns on their relative phases), so each counts fully.
    double can = 0.0;
    std::int64_t n_echoes = 0;
    for (std::size_t g = start[pick]; g < start[pick + 1]; ++g) {
      if (!(echoes[g].amplitude > 0.0)) continue;
      n_echoes += echoes[g].count;
      // What the rows read at this echo: its own elevation, pulled by every other echo as far as the compressed
      // pulse lets that one through at this range and as far as it can turn the phase across the rows' height.
      double pull = 0.0;
      for (std::size_t h = 0; h < echoes.size(); ++h) {
        if (h == g) continue;
        const double x = kPi * std::abs(echoes[h].range - echoes[g].range) / range_resolution;
        const double through = 2.0 * x * x > 1.0 ? 1.0 / (2.0 * x * x) : 1.0;       // a flat band's sinc^2, its lobes averaged
        const double d = std::min(std::abs(echoes[h].elevation - echoes[g].elevation), aperture_resolution / (2.0 * kPi));
        const double q = echoes[h].amplitude / echoes[g].amplitude;
        pull += 0.5 * q * q * through * d * d;                                      // the mean over the two echoes' relative phase
      }
      const double d = elevation - echoes[g].elevation;
      can = std::max(can, std::exp(-0.5 * d * d / (sigma_eff * sigma_eff + pull)));
    }
    const double dz = z - reported_depth;
    sum_with += can;
    sum_with_d2 += can * dz * dz;
    if (can >= floor_can) {
      if (!out.explained) out.depth_lo = z - 0.5 * step;
      out.depth_hi = z + 0.5 * step;
      out.explained = true;
    }
    if (can > best) {
      best = can;
      out.echoes = n_echoes;
    }
    // The reference: the first path alone, resolved, read with the angle's own sigma.
    const double d1 = elevation + paths[first].launch, alone = std::exp(-0.5 * d1 * d1 / (sigma_eff * sigma_eff));
    direct = direct || alone >= floor_can;
    sum_alone += alone;
    sum_alone_d2 += alone * dz * dz;
  }
  if (!out.explained) return out;
  const double flat = slant_range * std::cos(elevation) * sigma_eff;
  const double with = sum_with_d2 / sum_with, without = direct ? sum_alone_d2 / sum_alone : flat * flat;
  out.extra = std::sqrt(std::max(with - without, 0.0));
  out.depth_sigma = std::hypot(own_sigma, out.extra);
  out.limited = out.extra >= own_sigma;
  return out;
}

std::string multipath_limit_members(const MultipathLimit& limit) {
  std::ostringstream s;
  s << std::setprecision(10);
  if (!limit.explained) {
    s << "\"depth_unresolved_sigma_m\": null";
  } else {
    s << "\"depth_unresolved_sigma_m\": " << limit.extra << ", \"depth_multipath_limited\": " << (limit.limited ? "true" : "false")
      << ", \"depth_unresolved_echoes\": " << limit.echoes << ", \"depth_unresolved_span_m\": [" << limit.depth_lo << ", " << limit.depth_hi << "]";
  }
  if (!limit.bottom) {
    s << ", \"depth_unresolved_assumed\": \"water depth unknown: no bottom echoes, surface echoes only, target anywhere down to " << limit.deepest
      << " m\"";
  }
  return s.str();
}

}  // namespace sonar::live
