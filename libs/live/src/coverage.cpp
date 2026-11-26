// The coverage picture (coverage.hpp).
#include "sonar/live/coverage.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

#include "sonar/live/option_value.hpp"
#include "sonar/live/raytrace.hpp"

namespace sonar::live {
namespace {

using cd = std::complex<double>;
constexpr double kPi = 3.14159265358979323846;
constexpr double kInf = std::numeric_limits<double>::infinity();

double sinc(double x) {
  const double a = kPi * x;
  return std::abs(a) < 1e-12 ? 1.0 : std::sin(a) / a;
}

struct Leg {
  double t;
  cd coh;          // the coherent pressure re 1 m
  double power;    // everything that arrives, as power
  bool pair;       // direct or once surface-reflected: adds as a pressure
  bool direct;
};

// The arrivals of one leg at a point, strongest first, at most `most`.
void leg(const RayFan& fan, std::size_t i_range, double z, const CoverageOptions& o, double fc, double c_top, double c_bot, std::vector<RayArrival>& raw,
         std::vector<Leg>& out) {
  raw.clear();
  out.clear();
  fan.arrivals(i_range, z, raw);
  MultipathOptions surface;                      // the table's interpolation is that module's
  surface.surface_loss_db = o.surface_loss_db;
  surface.surface_loss_grazing = o.surface_loss_grazing;
  surface.surface_loss_table_db = o.surface_loss_table_db;
  for (const RayArrival& q : raw) {
    if (q.n_bottom > 0 && !o.bottom) continue;
    cd coh = q.amplitude;
    double power = q.amplitude * q.amplitude;
    if (q.n_surface > 0) {
      const double gamma = 2.0 * (2.0 * kPi * fc / c_top) * o.surface_rms_height * std::sin(q.grazing_surface);
      const double flat = o.surface_coefficient * std::pow(10.0, -multipath_surface_loss_db(surface, q.grazing_surface) / 20.0);
      coh *= std::pow(flat * std::exp(-0.5 * gamma * gamma), q.n_surface);
      power *= std::pow(flat * flat, q.n_surface);
    }
    if (q.n_bottom > 0) {
      const cd r = rayleigh_reflection(q.grazing_bottom, c_bot, o.bottom_speed, o.bottom_density, o.bottom_attenuation, o.bottom_shear_speed,
                                       o.bottom_shear_attenuation);
      coh *= std::pow(r, q.n_bottom);
      power *= std::pow(std::norm(r), q.n_bottom);
    }
    out.push_back({q.time, coh, power, q.n_bottom == 0 && q.n_surface <= 1, q.n_bottom == 0 && q.n_surface == 0});
  }
  std::stable_sort(out.begin(), out.end(), [](const Leg& a, const Leg& b) { return a.power > b.power; });
  if (static_cast<std::int64_t>(out.size()) > o.max_arrivals) out.resize(static_cast<std::size_t>(std::max<std::int64_t>(o.max_arrivals, 1)));
}

// A fan that is dense where the far ranges need it and reaches the steep rays the near ones need.
std::vector<double> launches(double rise, double r_first, double r_last, std::int64_t n) {
  const double deg = kPi / 180.0;
  const double steep = std::min(1.1 * std::atan2(rise, r_first) + 3.0 * deg, 75.0 * deg);
  const double flat = std::min(steep, 1.3 * std::atan2(rise, r_last / 3.0) + 3.0 * deg);
  const std::int64_t inner = std::max<std::int64_t>(n / 4, 8), outer = steep > flat ? std::max<std::int64_t>(n / 4, 8) : 0;
  std::vector<double> out;
  for (std::int64_t i = 0; i < outer; ++i) out.push_back(-steep + (steep - flat) * static_cast<double>(i) / static_cast<double>(outer));
  for (std::int64_t i = 0; i < 2 * inner; ++i) out.push_back(-flat + flat * (static_cast<double>(i) + 0.5) / static_cast<double>(inner));
  for (std::int64_t i = 0; i < outer; ++i) out.push_back(flat + (steep - flat) * static_cast<double>(i + 1) / static_cast<double>(outer));
  return out;
}

// A scattering-strength table of the configuration (linear, [g_in][g_out] on the grazing grid), bilinear, clamped.
double strength_at(const std::vector<double>& grid, const std::vector<double>& table, double g_in, double g_out) {
  const std::size_t n = grid.size();
  auto locate = [&](double g, std::size_t& i, double& u) {
    if (!(g > grid.front())) {
      i = 0;
      u = 0.0;
    } else if (g >= grid.back()) {
      i = n - 2;
      u = 1.0;
    } else {
      i = static_cast<std::size_t>(std::upper_bound(grid.begin(), grid.end(), g) - grid.begin()) - 1;
      u = (g - grid[i]) / (grid[i + 1] - grid[i]);
    }
  };
  std::size_t i = 0, j = 0;
  double u = 0.0, v = 0.0;
  locate(g_in, i, u);
  locate(g_out, j, v);
  const double a = table[i * n + j], b = table[i * n + j + 1], c = table[(i + 1) * n + j], d = table[(i + 1) * n + j + 1];
  return (1.0 - u) * ((1.0 - v) * a + v * b) + u * ((1.0 - v) * c + v * d);
}

// Int_0^y sinc^2(v) dv, sinc(v) = sin(pi v) / (pi v): (Si(2 pi y) - sin^2(pi y) / (pi y)) / pi; odd in y.
double sinc2_integral(double y) {
  if (y < 0.0) return -sinc2_integral(-y);
  if (y < 1e-6) return y;
  const double a = kPi * y, s = std::sin(a);
  return (sine_integral(2.0 * a) - s * s / a) / kPi;
}

struct ClassResult {
  bool present = false;
  double max_range = 0.0, direct_to = 0.0;
  double max_range_noise = 0.0;                  // the same without reverberation (reverb == 1 only)
  std::vector<double> worst;                     // [range]
  std::vector<std::vector<double>> blind;        // [r_lo, r_hi, z_lo, z_hi]
  std::vector<double> nulls;
};

std::vector<double> class_depths(const CoverageClass& c, double water_depth, std::int64_t n) {
  double lo = c.depth_lo, hi = c.depth_hi;
  if (water_depth > 0.0) hi = std::min(hi, water_depth - 1.0);
  std::vector<double> out;
  if (!(lo <= hi)) return out;
  if (hi - lo < 1e-9 || n < 2) return {0.5 * (lo + hi)};
  for (std::int64_t i = 0; i < n; ++i) out.push_back(lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(n - 1));
  return out;
}

// Every class at one own depth: fields, probabilities, what follows from them.
std::vector<ClassResult> evaluate(const CoverageOptions& o, const CoverageSonar& sonar, const std::vector<double>& ranges, const MultipathProfile& water,
                                  double water_depth, double array_depth, double projector_depth, bool details) {
  std::vector<double> depths;
  std::vector<std::size_t> first;
  for (const CoverageClass& c : o.classes) {
    first.push_back(depths.size());
    for (double z : class_depths(c, water_depth, o.n_depths)) depths.push_back(z);
  }
  first.push_back(depths.size());
  std::vector<ClassResult> out(o.classes.size());
  if (depths.empty()) return out;
  const CoverageField f = coverage_field(water, water_depth, array_depth, projector_depth, ranges, depths, o, sonar);
  const std::size_t n_r = ranges.size();
  for (std::size_t k = 0; k < o.classes.size(); ++k) {
    const std::size_t d0 = first[k], d1 = first[k + 1];
    ClassResult& r = out[k];
    if (d1 == d0) continue;
    r.present = true;
    std::vector<double> pd((d1 - d0) * n_r);
    r.worst.assign(n_r, 1.0);
    std::vector<double> worst_noise(o.reverb == 1 ? n_r : 0, 1.0);
    std::vector<char> direct(n_r, 1);
    for (std::size_t d = d0; d < d1; ++d) {
      for (std::size_t i = 0; i < n_r; ++i) {
        const std::size_t at = d * n_r + i;
        const double p = coverage_pd(o, coverage_snr_db(o, sonar, f.slant[at], o.classes[k].ts_db, f.factor_db[at], f.reverb_db[at]));
        pd[(d - d0) * n_r + i] = p;
        r.worst[i] = std::min(r.worst[i], p);
        if (!worst_noise.empty()) worst_noise[i] = std::min(worst_noise[i], coverage_pd(o, coverage_snr_db(o, sonar, f.slant[at], o.classes[k].ts_db, f.factor_db[at])));
        direct[i] = direct[i] && f.direct[at];
      }
    }
    // The greatest range at which the worst depth holds the stated probability (interpolated to the next range).
    auto reach = [&](const std::vector<double>& worst, std::ptrdiff_t& last) {
      last = -1;
      for (std::size_t i = 0; i < n_r; ++i) {
        if (worst[i] >= o.pd) last = static_cast<std::ptrdiff_t>(i);
      }
      if (last < 0) return 0.0;
      const auto u = static_cast<std::size_t>(last);
      double far = ranges[u];
      if (u + 1 < n_r && worst[u] > worst[u + 1]) far += (ranges[u + 1] - ranges[u]) * (worst[u] - o.pd) / (worst[u] - worst[u + 1]);
      return far;
    };
    std::ptrdiff_t last = -1, last_noise = -1;
    r.max_range = reach(r.worst, last);
    if (!worst_noise.empty()) r.max_range_noise = reach(worst_noise, last_noise);
    (void)last_noise;
    std::size_t run = 0;
    while (run < n_r && direct[run]) ++run;
    r.direct_to = run == 0 ? 0.0 : ranges[run - 1];
    if (!details) continue;
    for (std::ptrdiff_t i = 0; i < last && r.blind.size() < 4; ++i) {
      if (r.worst[static_cast<std::size_t>(i)] >= o.pd) continue;
      std::ptrdiff_t j = i;
      double z_lo = kInf, z_hi = -kInf;
      for (; j < last && r.worst[static_cast<std::size_t>(j)] < o.pd; ++j) {
        for (std::size_t d = d0; d < d1; ++d) {
          if (pd[(d - d0) * n_r + static_cast<std::size_t>(j)] < o.pd) {
            z_lo = std::min(z_lo, depths[d]);
            z_hi = std::max(z_hi, depths[d]);
          }
        }
      }
      r.blind.push_back({ranges[static_cast<std::size_t>(i)], ranges[static_cast<std::size_t>(j - 1)], z_lo, z_hi});
      i = j;
    }
    const std::size_t mid = d0 + (d1 - d0) / 2;
    for (std::size_t i = 1; i + 1 < n_r && r.nulls.size() < 4; ++i) {
      const double v = f.factor_db[mid * n_r + i];
      if (f.direct[mid * n_r + i] && v < o.null_db && v <= f.factor_db[mid * n_r + i - 1] && v < f.factor_db[mid * n_r + i + 1]) r.nulls.push_back(ranges[i]);
    }
  }
  return out;
}

}  // namespace

double sine_integral(double x) {
  if (x < 0.0) return -sine_integral(-x);
  if (x < 4.0) {                       // the series: sum (-1)^n x^(2n+1) / ((2n+1) (2n+1)!)
    double term = x, sum = x;
    const double x2 = x * x;
    for (int n = 1; n < 40; ++n) {
      term *= -x2 / static_cast<double>((2 * n) * (2 * n + 1));
      const double add = term / static_cast<double>(2 * n + 1);
      sum += add;
      if (std::abs(add) < 1e-17 * std::abs(sum)) break;
    }
    return sum;
  }
  // Abramowitz and Stegun 5.2.38-39 (x >= 1): Si = pi/2 - f cos x - g sin x, rational f and g.
  const double x2 = x * x, x4 = x2 * x2, x6 = x4 * x2, x8 = x4 * x4;
  const double f = (x8 + 38.027264 * x6 + 265.187033 * x4 + 335.677320 * x2 + 38.102495) /
                   (x8 + 40.021433 * x6 + 322.624911 * x4 + 570.236280 * x2 + 157.105423) / x;
  const double g = (x8 + 42.242855 * x6 + 302.757865 * x4 + 352.018498 * x2 + 21.821899) /
                   (x8 + 48.196927 * x6 + 482.485984 * x4 + 1114.978885 * x2 + 449.690326) / x2;
  return 0.5 * kPi - f * std::cos(x) - g * std::sin(x);
}

double compressed_power_average(double x, double width, double bandwidth) {
  if (!(width * bandwidth > 1e-3)) {
    const double v = sinc(bandwidth * x);
    return v * v;
  }
  return (sinc2_integral(bandwidth * (x + 0.5 * width)) - sinc2_integral(bandwidth * (x - 0.5 * width))) / (bandwidth * width);
}

double coverage_beam_integral(const std::vector<double>& x, const std::vector<double>& y, double frequency, double sound_speed, double steer, std::int64_t n) {
  const std::size_t m = std::min(x.size(), y.size());
  if (m == 0 || !(frequency > 0.0) || !(sound_speed > 0.0)) return 2.0 * kPi;
  const double k = 2.0 * kPi * frequency / sound_speed;
  double extent = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = i + 1; j < m; ++j) extent = std::max(extent, std::hypot(x[i] - x[j], y[i] - y[j]));
  }
  const auto count = static_cast<std::size_t>(std::clamp<double>(std::max<double>(static_cast<double>(n), std::ceil(32.0 * k * extent)), 64.0, 1048576.0));
  const double u0 = std::cos(steer), v0 = std::sin(steer);
  double total = 0.0;
  for (std::size_t a = 0; a < count; ++a) {
    const double phi = 2.0 * kPi * (static_cast<double>(a) + 0.5) / static_cast<double>(count);
    const double du = std::cos(phi) - u0, dv = std::sin(phi) - v0;
    cd sum = 0.0;
    for (std::size_t i = 0; i < m; ++i) sum += std::polar(1.0, k * (x[i] * du + y[i] * dv));
    total += std::norm(sum);
  }
  return total * (2.0 * kPi / static_cast<double>(count)) / (static_cast<double>(m) * static_cast<double>(m));
}

double coverage_beam_integral(const PingRecord& record) {
  std::vector<double> x, y;
  for (const Element& e : record.geometry.rx) {
    x.push_back(e.lever_arm.x);
    y.push_back(e.lever_arm.y);
  }
  const Waveform& w = record.transmit.waveform;
  double steer = 0.5 * kPi;                        // body y: broadside of a line along x
  if (!record.geometry.rx.empty()) {
    const Vec3d& b = record.geometry.rx.front().boresight;
    if (std::isfinite(b.x) && std::isfinite(b.y) && std::hypot(b.x, b.y) > 1e-9) steer = std::atan2(b.y, b.x);
  }
  return coverage_beam_integral(x, y, 0.5 * (w.f_start + w.f_stop), record.environment.sound_speed, steer);
}

std::vector<double> coverage_reverberation_db(const MultipathProfile& water, double water_depth, double array_depth, double projector_depth,
                                              const std::vector<double>& delays, const CoverageOptions& o, const CoverageSonar& sonar, double psi) {
  std::vector<double> out(delays.size(), -kInf);
  const std::size_t ng = o.reverb_grazing.size();
  const bool bottom = ng >= 2 && o.reverb_bottom_strength.size() == ng * ng && water_depth > 0.0;
  const bool surface = ng >= 2 && o.reverb_surface_strength.size() == ng * ng;
  double t_max = 0.0;
  for (double t : delays) {
    if (std::isfinite(t)) t_max = std::max(t_max, t);
  }
  if (!(bottom || surface) || !(psi > 0.0) || !(sonar.bandwidth > 0.0) || !(t_max > 0.0)) return out;
  const RayProfile prof(water.depth, water.speed, water_depth);
  const double c_bot = prof.speed.back();
  const double c_min = *std::min_element(prof.speed.begin(), prof.speed.end()), c_max = *std::max_element(prof.speed.begin(), prof.speed.end());
  const double tail = 8.0 / sonar.bandwidth;       // the compressed pulse's power beyond this is 2.5 % of it (0.1 dB)
  const std::int64_t n_rings = std::max<std::int64_t>(o.reverb_rings, 16);
  const double rho_max = 0.5 * c_max * (t_max + tail), d_rho = rho_max / static_cast<double>(n_rings);
  std::vector<double> rings(static_cast<std::size_t>(n_rings));
  for (std::size_t k = 0; k < rings.size(); ++k) rings[k] = (static_cast<double>(k) + 0.5) * d_rho;
  const double bin = 2.0 * d_rho / c_min;
  std::vector<double> energy(static_cast<std::size_t>(std::ceil((t_max + 2.0 * tail) / bin)) + 2, 0.0);
  const auto bounces = static_cast<int>(o.reverb_max_bounces > 0 ? static_cast<double>(o.reverb_max_bounces)
                                       : water_depth > 0.0 ? std::clamp(std::ceil(0.4 * rho_max / water_depth), 3.0, 40.0) : 3.0);
  const double own = std::max(array_depth, projector_depth);
  const double rise = own + (water_depth > 0.0 ? 2.0 * water_depth * bounces + water_depth : 2.0 * own);
  const std::vector<double> fan_angles = launches(rise, rings.front(), rings.back(), o.ray_fan);
  const RayFan rx(prof, array_depth, rings, fan_angles, bounces);
  const bool one = std::abs(projector_depth - array_depth) < 1e-6;
  const RayFan tx_own = one ? RayFan(prof, array_depth, {rings.front()}, std::vector<double>{-0.1, 0.1}, 0) : RayFan(prof, projector_depth, rings, fan_angles, bounces);
  const RayFan& tx = one ? rx : tx_own;
  const double alpha = std::isfinite(o.absorption_db_per_km) ? o.absorption_db_per_km / 1000.0 : 0.0;   // dB per metre of path
  const double c_path = 0.5 * (c_min + c_max);
  MultipathOptions loss;                         // the bubble law's table, as leg() reads it
  loss.surface_loss_db = o.surface_loss_db;
  loss.surface_loss_grazing = o.surface_loss_grazing;
  loss.surface_loss_table_db = o.surface_loss_table_db;
  struct Path {
    double t, power, grazing;
  };
  std::vector<RayArrival> raw;
  std::vector<Path> a, b;
  // The paths of one fan that reach the boundary at ring k going towards it, as powers (the coefficients of the
  // boundaries met before it, as in leg(): a rough surface's scattered part stays in the power).
  auto paths = [&](const RayFan& fan, std::size_t k, bool on_bottom, std::vector<Path>& got) {
    got.clear();
    raw.clear();
    const double z = on_bottom ? water_depth - 1e-3 * std::max(water_depth, 1.0) : 1e-3;
    fan.arrivals(k, z, raw);
    for (const RayArrival& q : raw) {
      if (on_bottom ? !(q.s_end > 0.0) : !(q.s_end < 0.0)) continue;
      if (q.n_bottom > 0 && !o.bottom) continue;
      double power = q.amplitude * q.amplitude;
      if (q.n_surface > 0) {
        const double flat = o.surface_coefficient * std::pow(10.0, -multipath_surface_loss_db(loss, q.grazing_surface) / 20.0);
        power *= std::pow(flat * flat, q.n_surface);
      }
      if (q.n_bottom > 0) {
        const cd r = rayleigh_reflection(q.grazing_bottom, c_bot, o.bottom_speed, o.bottom_density, o.bottom_attenuation, o.bottom_shear_speed,
                                         o.bottom_shear_attenuation);
        power *= std::pow(std::norm(r), q.n_bottom);
      }
      got.push_back({q.time, power, on_bottom ? q.grazing_bottom : q.grazing_surface});
    }
  };
  for (int side = 0; side < 2; ++side) {
    const bool on_bottom = side == 0;
    if (on_bottom ? !bottom : !surface) continue;
    const std::vector<double>& table = on_bottom ? o.reverb_bottom_strength : o.reverb_surface_strength;
    for (std::size_t k = 0; k < rings.size(); ++k) {
      paths(rx, k, on_bottom, b);
      if (b.empty()) continue;
      if (one) a = b; else paths(tx, k, on_bottom, a);
      const double area = rings[k] * d_rho;      // per radian of azimuth
      for (const Path& p : a) {
        for (const Path& q : b) {
          const double t = p.t + q.t;
          const auto at = static_cast<std::size_t>(t / bin);
          if (at >= energy.size()) continue;
          energy[at] += p.power * q.power * strength_at(o.reverb_grazing, table, p.grazing, q.grazing) * area * std::pow(10.0, -alpha * c_path * t / 10.0);
        }
      }
    }
  }
  for (std::size_t i = 0; i < delays.size(); ++i) {
    const double t = delays[i];
    if (!std::isfinite(t)) continue;
    const auto lo = static_cast<std::size_t>(std::max(0.0, std::floor((t - tail) / bin - 1.0)));
    const auto hi = std::min(energy.size(), static_cast<std::size_t>(std::ceil((t + tail) / bin + 1.0)));
    double sum = 0.0;
    for (std::size_t k = lo; k < hi; ++k) {
      if (energy[k] > 0.0) sum += energy[k] * compressed_power_average(t - (static_cast<double>(k) + 0.5) * bin, bin, sonar.bandwidth);
    }
    out[i] = sum > 0.0 ? 10.0 * std::log10(psi * sum) : -kInf;
  }
  return out;
}

MultipathProfile water_profile(const ProfileOptions& po, const LadderProfile* ladder, const std::vector<double>& ssp_depth,
                               const std::vector<double>& ssp_speed, bool use_gradient, double gradient, double gradient_sigma, double own,
                               double sound, double water) {
  MultipathProfile w;
  w.gradient_sigma = gradient_sigma;
  LadderProfile base;
  if (ladder != nullptr) {
    base = *ladder;
  } else if (!ssp_depth.empty()) {
    base.rung = "supplied";
    base.depth = ssp_depth;
    base.speed = ssp_speed;
  } else {
    base.rung = use_gradient ? "gradient" : "none";
  }
  const double g = use_gradient ? gradient : 0.0;
  w.rung = base.rung;
  w.depth = base.depth;
  w.speed = base.speed;
  w.sigma = base.sigma;
  if (w.depth.empty()) {
    const double bottom = std::max({std::isfinite(water) ? water : 0.0, own, 0.0}) + 10000.0;
    w.depth = {0.0, bottom};
    w.speed = {sound - g * own, sound + g * (bottom - own)};
  }
  if (po.ensemble) {
    std::vector<double> knots;
    std::vector<std::vector<double>> speeds;
    profile_ensemble(po, base, own, sound, g, gradient_sigma, water, knots, speeds);
    if (w.sigma.size() == w.depth.size()) {
      std::vector<double> sigma(knots.size());
      for (std::size_t i = 0; i < knots.size(); ++i) {
        const double z = knots[i];
        if (z <= w.depth.front()) sigma[i] = w.sigma.front();
        else if (z >= w.depth.back()) sigma[i] = w.sigma.back();
        else {
          const auto k = static_cast<std::size_t>(std::upper_bound(w.depth.begin(), w.depth.end(), z) - w.depth.begin());
          sigma[i] = w.sigma[k - 1] + (z - w.depth[k - 1]) / (w.depth[k] - w.depth[k - 1]) * (w.sigma[k] - w.sigma[k - 1]);
        }
      }
      w.sigma = sigma;
    }
    w.depth = knots;
    w.speed = speeds.front();
    w.ensemble.assign(speeds.begin() + 1, speeds.end());
  }
  return w;
}

std::string coverage_site_missing(const CoverageOptions& o) {
  const bool depth = std::isfinite(o.water_depth) && o.water_depth > 0.0;
  if (depth && o.bottom) return "";
  return std::string("coverage needs the site's water depth and bottom, and ") + (depth ? "" : "coverage_water_depth (metres)") + (!depth && !o.bottom ? " and " : "") +
         (o.bottom ? "" : "coverage_bottom=1 with coverage_bottom_speed, coverage_bottom_density, coverage_bottom_attenuation") + (!depth && !o.bottom ? " are" : " is") +
         " not given. Neither has a default: without them the picture would be the deep-water one, in which nothing returns from the bottom, and it would say "
         "that targets the chain detects cannot be seen";
}

bool set_coverage_option(CoverageOptions& o, const std::string& key, const std::string& value) {
  const std::string prefix = "coverage";
  if (key.compare(0, prefix.size(), prefix) != 0) return false;
  const auto v = [&] { return option_number(key, value); };
  const auto i = [&] { return option_integer(key, value); };
  const auto on = [&] { return option_flag(key, value); };
  if (key == prefix) {
    o.enabled = on();
    return true;
  }
  if (key.size() < prefix.size() + 2 || key[prefix.size()] != '_') return false;
  const std::string k = key.substr(prefix.size() + 1);
  if (k == "classes") {
    o.classes.clear();
    std::istringstream all(value);
    for (std::string item; std::getline(all, item, ';');) {
      std::istringstream one(item);
      std::string name, ts, lo, hi;
      if (!(std::getline(one, name, ':') && std::getline(one, ts, ':') && std::getline(one, lo, ':') && std::getline(one, hi, ':'))) return false;
      o.classes.push_back({name, option_number(key, ts), option_number(key, lo), option_number(key, hi)});
    }
  }
  else if (k == "source_level_db") o.source_level_db = v();
  else if (k == "noise_spectrum_db") o.noise_spectrum_db = v();
  else if (k == "array_gain_db") o.array_gain_db = v();
  else if (k == "absorption_db_per_km") o.absorption_db_per_km = v();
  else if (k == "pd") o.pd = v();
  else if (k == "pd_snr_db") o.pd_snr_db = option_numbers(key, value);
  else if (k == "pd_value") o.pd_value = option_numbers(key, value);
  else if (k == "range_first") o.range_first = v();
  else if (k == "range_last") o.range_last = v();
  else if (k == "n_ranges") o.n_ranges = i();
  else if (k == "n_depths") o.n_depths = i();
  else if (k == "n_report") o.n_report = i();
  else if (k == "recommend") o.recommend = on();
  else if (k == "own_depth_min") o.own_depth_min = v();
  else if (k == "own_depth_max") o.own_depth_max = v();
  else if (k == "bottom_clearance") o.bottom_clearance = v();
  else if (k == "n_own") o.n_own = i();
  else if (k == "refresh_depth") o.refresh_depth = v();
  else if (k == "refresh_water_depth") o.refresh_water_depth = v();
  else if (k == "refresh_speed") o.refresh_speed = v();
  else if (k == "surface_coefficient") o.surface_coefficient = v();
  else if (k == "surface_rms_height") o.surface_rms_height = v();
  else if (k == "surface_loss_db") o.surface_loss_db = v();
  else if (k == "surface_loss_grazing") o.surface_loss_grazing = option_numbers(key, value);
  else if (k == "surface_loss_table_db") o.surface_loss_table_db = option_numbers(key, value);
  else if (k == "bottom") o.bottom = on();
  else if (k == "bottom_speed") o.bottom_speed = v();
  else if (k == "bottom_density") o.bottom_density = v();
  else if (k == "bottom_attenuation") o.bottom_attenuation = v();
  else if (k == "bottom_shear_speed") o.bottom_shear_speed = option_nonnegative(key, value);
  else if (k == "bottom_shear_attenuation") o.bottom_shear_attenuation = option_nonnegative(key, value);
  else if (k == "water_depth") o.water_depth = v();
  else if (k == "ray_fan") o.ray_fan = i();
  else if (k == "max_bounces") o.max_bounces = i();
  else if (k == "max_arrivals") o.max_arrivals = i();
  else if (k == "null_db") o.null_db = v();
  else if (k == "surface_z") o.surface_z = v();
  else if (k == "reverb") {
    o.reverb = i();
    if (o.reverb < 0 || o.reverb > 2) throw std::invalid_argument(key + ": 0 (off), 1 (included) or 2 (not included: no scattering given)");
  }
  else if (k == "reverb_grazing") o.reverb_grazing = option_numbers(key, value);
  else if (k == "reverb_bottom_strength") o.reverb_bottom_strength = option_numbers(key, value);
  else if (k == "reverb_surface_strength") o.reverb_surface_strength = option_numbers(key, value);
  else if (k == "reverb_beam_integral") o.reverb_beam_integral = option_nonnegative(key, value);
  else if (k == "reverb_rings") o.reverb_rings = i();
  else if (k == "reverb_max_bounces") o.reverb_max_bounces = i();
  else return false;
  return true;
}

CoverageField coverage_field(const MultipathProfile& water, double water_depth, double array_depth, double projector_depth,
                             const std::vector<double>& ranges, const std::vector<double>& depths, const CoverageOptions& o, const CoverageSonar& sonar) {
  const std::size_t n_r = ranges.size(), n_d = depths.size();
  CoverageField f;
  f.factor_db.assign(n_d * n_r, -kInf);
  f.coherent_db.assign(n_d * n_r, -kInf);
  f.incoherent_db.assign(n_d * n_r, -kInf);
  f.direct.assign(n_d * n_r, 0);
  f.slant.assign(n_d * n_r, 0.0);
  f.delay.assign(n_d * n_r, kNaN);
  f.reverb_db.assign(n_d * n_r, -kInf);
  if (n_r == 0 || n_d == 0) return f;
  const RayProfile prof(water.depth, water.speed, water_depth);
  const double c_top = prof.speed.front(), c_bot = prof.speed.back();
  const double deepest = *std::max_element(depths.begin(), depths.end());
  const double own = std::max(array_depth, projector_depth);
  const auto bounces = static_cast<int>(std::max<std::int64_t>(o.max_bounces, 1));
  const double rise = own + deepest + (o.bottom && water_depth > 0.0 ? 2.0 * water_depth * bounces : 2.0 * std::max(own, deepest));
  const std::vector<double> fan_angles = launches(rise, ranges.front(), ranges.back(), o.ray_fan);
  const RayFan rx(prof, array_depth, ranges, fan_angles, bounces);
  const bool one = std::abs(projector_depth - array_depth) < 1e-6;
  const RayFan tx_own = one ? RayFan(prof, array_depth, {ranges.front()}, std::vector<double>{-0.1, 0.1}, 0) : RayFan(prof, projector_depth, ranges, fan_angles, bounces);
  const RayFan& tx = one ? rx : tx_own;
  std::vector<RayArrival> raw;
  std::vector<Leg> a, b;
  std::vector<double> tau, when, power;
  std::vector<cd> coherent;
  std::vector<char> is_pair;
  for (std::size_t d = 0; d < n_d; ++d) {
    for (std::size_t i = 0; i < n_r; ++i) {
      const std::size_t at = d * n_r + i;
      const double r_rx = std::hypot(ranges[i], depths[d] - array_depth), r_tx = std::hypot(ranges[i], depths[d] - projector_depth);
      f.slant[at] = std::sqrt(r_rx * r_tx);
      leg(rx, i, depths[d], o, sonar.f_centre, c_top, c_bot, raw, b);
      if (one) a = b; else leg(tx, i, depths[d], o, sonar.f_centre, c_top, c_bot, raw, a);
      if (a.empty() || b.empty()) continue;
      bool da = false, db = false;
      cd sum = 0.0;
      double all = 0.0;
      for (const Leg& q : a) da = da || q.direct;
      for (const Leg& q : b) {
        db = db || q.direct;
        sum += q.coh * std::polar(1.0, -2.0 * kPi * sonar.f_centre * q.t);
        all += q.power;
      }
      f.direct[at] = da && db;
      f.coherent_db[at] = 20.0 * std::log10(std::max(std::abs(sum) * r_rx, 1e-300));
      f.incoherent_db[at] = 10.0 * std::log10(std::max(all * r_rx * r_rx, 1e-300));
      // The two-way arrivals through the compressed pulse.
      tau.clear();
      coherent.clear();
      power.clear();
      is_pair.clear();
      for (const Leg& p : a) {
        for (const Leg& q : b) {
          const double t = p.t + q.t;
          const cd c = p.coh * q.coh * std::polar(1.0, -2.0 * kPi * sonar.f_centre * t);
          const bool pair = p.pair && q.pair;
          tau.push_back(t);
          is_pair.push_back(pair);
          coherent.push_back(pair ? c : cd(0.0, 0.0));
          power.push_back(pair ? std::max(p.power * q.power - std::norm(c), 0.0) : p.power * q.power);
        }
      }
      when = tau;
      std::sort(when.begin(), when.end());
      const std::size_t n_t = when.size();
      for (std::size_t k = 0; k + 1 < n_t; ++k) {
        if (when[k + 1] - when[k] < 2.0 / sonar.bandwidth) {
          for (double u : {0.25, 0.5, 0.75}) when.push_back(when[k] + u * (when[k + 1] - when[k]));
        }
      }
      double best = 0.0, t_best = kNaN;
      for (double t : when) {
        cd amp = 0.0;
        double p = 0.0;
        for (std::size_t k = 0; k < tau.size(); ++k) {
          const double s = sinc(sonar.bandwidth * (t - tau[k]));
          amp += coherent[k] * s;
          p += power[k] * s * s;
        }
        const double v = std::norm(amp) + p;
        if (v > best || std::isnan(t_best)) t_best = t;
        best = std::max(best, v);
      }
      f.delay[at] = t_best;
      f.factor_db[at] = 10.0 * std::log10(std::max(best * (r_rx * r_tx) * (r_rx * r_tx), 1e-300));
    }
  }
  if (o.reverb == 1) {
    const double psi = o.reverb_beam_integral > 0.0 ? o.reverb_beam_integral : sonar.beam_integral;
    f.reverb_db = coverage_reverberation_db(water, water_depth, array_depth, projector_depth, f.delay, o, sonar, psi);
  }
  return f;
}

double coverage_snr_db(const CoverageOptions& o, const CoverageSonar& sonar, double slant, double ts_db, double factor_db) {
  const double tl = 20.0 * std::log10(std::max(slant, 1.0)) + o.absorption_db_per_km * slant / 1000.0;
  return o.source_level_db - 2.0 * tl + ts_db - (o.noise_spectrum_db + 10.0 * std::log10(sonar.bandwidth)) + 10.0 * std::log10(sonar.bandwidth * sonar.pulse) +
         o.array_gain_db + factor_db;
}

double coverage_snr_db(const CoverageOptions& o, const CoverageSonar& sonar, double slant, double ts_db, double factor_db, double reverb_db) {
  const double noise_only = coverage_snr_db(o, sonar, slant, ts_db, factor_db);
  if (o.reverb != 1 || !(reverb_db > -kInf)) return noise_only;          // -inf and NaN: no reverberation
  const double tl = 20.0 * std::log10(std::max(slant, 1.0)) + o.absorption_db_per_km * slant / 1000.0;
  const double echo = o.source_level_db - 2.0 * tl + ts_db + factor_db;
  const double noise = echo - noise_only;          // the noise after processing, in the echo's units
  return noise_only - 10.0 * std::log10(1.0 + std::pow(10.0, (o.source_level_db + reverb_db - noise) / 10.0));
}

double coverage_pd(const CoverageOptions& o, double snr_db) {
  const std::vector<double>&x = o.pd_snr_db, &y = o.pd_value;
  if (x.empty() || x.size() != y.size()) return kNaN;
  if (!(snr_db > x.front())) return y.front();
  if (snr_db >= x.back()) return y.back();
  const auto hi = static_cast<std::size_t>(std::upper_bound(x.begin(), x.end(), snr_db) - x.begin());
  return y[hi - 1] + (snr_db - x[hi - 1]) / (x[hi] - x[hi - 1]) * (y[hi] - y[hi - 1]);
}

const std::string& Coverage::update(const PingRecord& record, const Vec3d& origin, double water_depth, const MultipathProfile& water) {
  const Waveform& w = record.transmit.waveform;
  const Samples& s = record.samples;
  const double f_lo = std::min(w.f_start, w.f_stop), f_hi = std::max(w.f_start, w.f_stop);
  CoverageSonar sonar;
  sonar.f_centre = 0.5 * (f_lo + f_hi);
  sonar.bandwidth = f_hi > f_lo ? f_hi - f_lo : 1.0 / w.duration;
  sonar.pulse = w.duration;
  const double c = record.environment.sound_speed;
  const double r0 = 0.5 * c * s.start_time, r1 = 0.5 * c * (s.start_time + static_cast<double>(s.n_samples - 1) / s.sample_rate - w.duration);
  const double projector = world_position(record.geometry.tx_pose, record.geometry.tx.lever_arm).z - opt_.surface_z;
  if (opt_.reverb == 1 && !(opt_.reverb_beam_integral > 0.0)) {     // the array's beam, once (the geometry does not change)
    if (!(psi_ > 0.0)) psi_ = coverage_beam_integral(record);
    sonar.beam_integral = psi_;
  }
  MultipathProfile through = water;
  if (through.depth.empty()) {                      // nothing known: isovelocity at the vehicle's sound speed
    through.depth = {0.0, 1.0};
    through.speed = {c, c};
  }
  return update(sonar, r0, r1, origin.z - opt_.surface_z, projector, water_depth, through);
}

const std::string& Coverage::update(const CoverageSonar& sonar, double range_first, double range_last, double array_depth, double projector_depth,
                                    double water_depth, const MultipathProfile& water) {
  // The water depth of this ping (elevation.water_depth, navigation), else the site's (`water_depth`), else unknown.
  const bool from_site = !(std::isfinite(water_depth) && water_depth > 0.0) && std::isfinite(opt_.water_depth) && opt_.water_depth > 0.0;
  const double water_now = from_site ? opt_.water_depth : std::isfinite(water_depth) && water_depth > 0.0 ? water_depth : 0.0;
  bool same = have_ && water.rung == rung_ && std::abs(array_depth - own_) < opt_.refresh_depth && std::abs(water_now - water_) < opt_.refresh_water_depth &&
              (water_now > 0.0) == (water_ > 0.0);
  // The profile then and now, each at the other's knots (a built profile gains knots as the vehicle samples).
  auto at = [](const std::vector<double>& d, const std::vector<double>& c, double z) {
    if (z <= d.front()) return c.front();
    if (z >= d.back()) return c.back();
    const auto k = static_cast<std::size_t>(std::upper_bound(d.begin(), d.end(), z) - d.begin());
    return c[k - 1] + (z - d[k - 1]) / (d[k] - d[k - 1]) * (c[k] - c[k - 1]);
  };
  for (std::size_t i = 0; same && i < depth_.size(); ++i) same = std::abs(at(water.depth, water.speed, depth_[i]) - speed_[i]) < opt_.refresh_speed;
  for (std::size_t i = 0; same && i < water.depth.size(); ++i) same = std::abs(at(depth_, speed_, water.depth[i]) - water.speed[i]) < opt_.refresh_speed;
  if (same) {
    refreshed_ = false;
    const std::size_t pos = json_.find("\"age_pings\": ");
    if (pos != std::string::npos) {
      const std::size_t end = json_.find_first_of(",}", pos);
      const long age = std::atol(json_.substr(pos + 13, end - pos - 13).c_str()) + 1;
      json_.replace(pos + 13, end - pos - 13, std::to_string(age));
    }
    return json_;
  }
  const auto t_start = std::chrono::steady_clock::now();
  refreshed_ = true;
  have_ = true;
  own_ = array_depth;
  water_ = water_now;
  rung_ = water.rung;
  depth_ = water.depth;
  speed_ = water.speed;
  pd_.clear();
  ranges_.clear();
  std::ostringstream s;
  s << std::setprecision(6);
  auto fail = [&](const std::string& why) -> const std::string& {
    json_ = "\"coverage\": {\"error\": \"" + why + "\", \"age_pings\": 0}";
    return json_;
  };
  if (opt_.classes.empty()) return fail("no target classes configured");
  if (!(std::isfinite(opt_.source_level_db) && std::isfinite(opt_.noise_spectrum_db) && std::isfinite(opt_.array_gain_db) && std::isfinite(opt_.absorption_db_per_km))) {
    return fail("the link budget is not configured (source level, noise spectrum, array gain, absorption)");
  }
  if (opt_.pd_snr_db.size() < 2 || opt_.pd_snr_db.size() != opt_.pd_value.size()) return fail("no detection-probability table");
  if (!(array_depth > 0.0) || !(projector_depth > 0.0)) return fail("the sonar is not below the surface");
  if (opt_.reverb == 1) {
    const std::size_t ng = opt_.reverb_grazing.size();
    if (ng < 2 || (opt_.reverb_bottom_strength.size() != ng * ng && opt_.reverb_surface_strength.size() != ng * ng)) {
      return fail("reverberation is included but no scattering-strength table matches the grazing grid");
    }
    if (!(opt_.reverb_beam_integral > 0.0 || sonar.beam_integral > 0.0)) return fail("reverberation is included but the beam's integral over azimuth is not known");
  }
  const double r0 = opt_.range_first > 0.0 ? opt_.range_first : std::max(range_first, 50.0), r1 = opt_.range_last > 0.0 ? opt_.range_last : range_last;
  if (!(r1 > r0)) return fail("no range window");
  const std::int64_t n_r = std::max<std::int64_t>(opt_.n_ranges, 4);
  for (std::int64_t i = 0; i < n_r; ++i) ranges_.push_back(r0 + (r1 - r0) * static_cast<double>(i) / static_cast<double>(n_r - 1));
  const std::vector<ClassResult> now = evaluate(opt_, sonar, ranges_, water, water_now, array_depth, projector_depth, true);
  for (const ClassResult& r : now) pd_.push_back(r.worst);
  auto reach = [](const std::vector<ClassResult>& rs) {
    double least = kInf, total = 0.0;
    for (const ClassResult& r : rs) {
      if (!r.present) continue;
      least = std::min(least, r.max_range);
      total += r.max_range;
    }
    return std::make_pair(std::isfinite(least) ? least : 0.0, total);
  };
  // Other own depths the platform may take.
  double best_depth = array_depth;
  std::vector<ClassResult> best = now;
  const double offset = projector_depth - array_depth;
  double deepest = opt_.own_depth_max;
  if (water_now > 0.0) deepest = std::min(deepest, water_now - opt_.bottom_clearance - std::max(offset, 0.0));
  if (opt_.recommend && opt_.n_own > 1 && deepest > opt_.own_depth_min) {
    for (std::int64_t i = 0; i < opt_.n_own; ++i) {
      const double z = opt_.own_depth_min + (deepest - opt_.own_depth_min) * static_cast<double>(i) / static_cast<double>(opt_.n_own - 1);
      if (!(z + offset > 0.0)) continue;
      const std::vector<ClassResult> there = evaluate(opt_, sonar, ranges_, water, water_now, z, z + offset, false);
      const auto a = reach(there), b = reach(best);
      const double step = (r1 - r0) / static_cast<double>(n_r - 1);
      if (a.first > b.first + 0.5 * step || (std::abs(a.first - b.first) <= 0.5 * step && a.second > b.second + 0.5 * step)) {
        best = there;
        best_depth = z;
      }
    }
  }
  // The profiles the water may also be (elevation.profile.ensemble): the span of each class's greatest range.
  std::vector<double> span_lo(now.size(), kInf), span_hi(now.size(), -kInf);
  for (const std::vector<double>& speeds : water.ensemble) {
    MultipathProfile member;
    member.depth = water.depth;
    member.speed = speeds;
    const std::vector<ClassResult> m = evaluate(opt_, sonar, ranges_, member, water_now, array_depth, projector_depth, false);
    for (std::size_t k = 0; k < m.size(); ++k) {
      span_lo[k] = std::min(span_lo[k], m[k].max_range);
      span_hi[k] = std::max(span_hi[k], m[k].max_range);
    }
  }
  const std::int64_t n_rep = std::clamp<std::int64_t>(opt_.n_report, 2, n_r);
  std::vector<std::size_t> shown;
  for (std::int64_t i = 0; i < n_rep; ++i) shown.push_back(static_cast<std::size_t>(std::llround(static_cast<double>(i) * static_cast<double>(n_r - 1) / static_cast<double>(n_rep - 1))));
  auto metres = [](double v) { return std::llround(v); };
  s << "\"coverage\": {\"own_depth_m\": " << std::round(array_depth * 10.0) / 10.0 << ", \"water_depth_m\": ";
  if (water_now > 0.0) s << std::round(water_now * 10.0) / 10.0; else s << "null";
  if (from_site) s << ", \"water_depth_from\": \"site\"";      // the configured depth of the site: this ping carried none
  s << ", \"profile\": \"" << (water.rung.empty() ? "none" : water.rung) << "\", \"pd\": " << opt_.pd << ", \"age_pings\": 0, \"ranges_m\": [";
  for (std::size_t i = 0; i < shown.size(); ++i) s << (i ? ", " : "") << metres(ranges_[shown[i]]);
  s << "], \"classes\": [";
  for (std::size_t k = 0; k < now.size(); ++k) {
    const ClassResult& r = now[k];
    s << (k ? ", " : "") << "{\"name\": \"" << opt_.classes[k].name << "\"";
    if (!r.present) {
      s << ", \"max_range_m\": null, \"note\": \"its depths are not in this water\"}";
      continue;
    }
    s << ", \"max_range_m\": " << metres(r.max_range) << ", \"pd\": [";
    for (std::size_t i = 0; i < shown.size(); ++i) s << (i ? ", " : "") << std::round(r.worst[shown[i]] * 100.0) / 100.0;
    s << "], \"direct_to_m\": " << metres(r.direct_to);
    if (!r.blind.empty()) {
      s << ", \"blind_m\": [";
      for (std::size_t i = 0; i < r.blind.size(); ++i) {
        s << (i ? ", " : "") << "[" << metres(r.blind[i][0]) << ", " << metres(r.blind[i][1]) << ", " << metres(r.blind[i][2]) << ", " << metres(r.blind[i][3]) << "]";
      }
      s << "]";
    }
    if (!r.nulls.empty()) {
      s << ", \"nulls_m\": [";
      for (std::size_t i = 0; i < r.nulls.size(); ++i) s << (i ? ", " : "") << metres(r.nulls[i]);
      s << "]";
    }
    if (!water.ensemble.empty()) s << ", \"max_range_ensemble_m\": [" << metres(std::min(span_lo[k], r.max_range)) << ", " << metres(std::max(span_hi[k], r.max_range)) << "]";
    if (opt_.recommend) s << ", \"recommended_max_range_m\": " << metres(best[k].max_range);
    if (opt_.reverb == 1) {
      // What noise alone would allow, and which of the two sets the reach (reverberation when it shortens it by more
      // than half a step of the grid).
      const double step = (r1 - r0) / static_cast<double>(n_r - 1);
      s << ", \"max_range_noise_only_m\": " << metres(r.max_range_noise) << ", \"limited_by\": \""
        << (r.max_range_noise > r.max_range + 0.5 * step ? "reverberation" : "noise") << "\"";
    }
    s << "}";
  }
  s << "]";
  if (opt_.recommend) {
    s << ", \"recommend\": {\"own_depth_m\": " << std::round(best_depth * 10.0) / 10.0 << ", \"gain_m\": " << metres(reach(best).first - reach(now).first) << "}";
  }
  if (opt_.reverb == 1) {
    const double psi = opt_.reverb_beam_integral > 0.0 ? opt_.reverb_beam_integral : sonar.beam_integral;
    s << ", \"reverberation\": \"included\", \"beam_integral_deg\": " << std::round(psi * 180.0 / kPi * 100.0) / 100.0;
  } else if (opt_.reverb == 2) {
    s << ", \"reverberation\": \"not included: the configuration gives no boundary scattering (coverage.reverberation), so ranges are noise-limited "
         "and optimistic where reverberation limits\"";
  }
  s << "}";
  json_ = s.str();
  seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  return json_;
}

double Coverage::seen(std::size_t i, double range) const {
  if (i >= pd_.size() || ranges_.empty() || pd_[i].empty()) return kNaN;
  if (range <= ranges_.front()) return pd_[i].front();
  if (range >= ranges_.back()) return pd_[i].back();
  const auto hi = static_cast<std::size_t>(std::upper_bound(ranges_.begin(), ranges_.end(), range) - ranges_.begin());
  return pd_[i][hi - 1] + (range - ranges_[hi - 1]) / (ranges_[hi] - ranges_[hi - 1]) * (pd_[i][hi] - pd_[i][hi - 1]);
}

}  // namespace sonar::live
