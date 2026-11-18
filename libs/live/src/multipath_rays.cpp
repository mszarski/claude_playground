// The multipath-depth stage's predicted pattern from rays through a profile (multipath_depth.hpp,
// options.rays; raytrace.hpp). One fan per leg (projector -> target, target -> array by reciprocity),
// read at the two ends of the horizontal ranges the candidate depths can lie at; an arrival at a depth
// and range in between comes from the fans by cubic Hermite interpolation, in depth along the wavefront
// (dt/dz = s / c) and in range between the two fans (dt/dr = p, the ray's horizontal slowness).
#include <algorithm>
#include <cmath>
#include <limits>

#include "sonar/live/environment.hpp"
#include "sonar/live/multipath_depth.hpp"
#include "sonar/live/raytrace.hpp"

namespace sonar::live {
namespace {

using cd = std::complex<double>;
constexpr double kPi = 3.14159265358979323846;
constexpr double kTiltStep = 0.05;     // the share of the profile's tilt the sigma is measured with (as profile_depth_sigma)

}  // namespace

double multipath_surface_loss_db(const MultipathOptions& o, double grazing) {
  double loss = o.surface_loss_db;
  const std::vector<double>&g = o.surface_loss_grazing, &l = o.surface_loss_table_db;
  if (g.empty() || g.size() != l.size()) return loss;
  if (grazing <= g.front()) return loss + l.front();
  if (grazing >= g.back()) return loss + l.back();
  const auto hi = static_cast<std::size_t>(std::upper_bound(g.begin(), g.end(), grazing) - g.begin());
  return loss + l[hi - 1] + (grazing - g[hi - 1]) / (g[hi] - g[hi - 1]) * (l[hi] - l[hi - 1]);
}

class RayPattern {
 public:
  RayPattern(const MultipathProfile& profile, const MultipathGeometry& g, const MultipathOptions& o, std::int64_t n_fan)
      : profile_(profile), g_(g), o_(o), n_fan_(std::max<std::int64_t>(n_fan, 64)) {
    g_.rays.reset();
    z_mid_ = 0.5 * (g.array_depth + g.projector_depth);
    z_hi_ = o.max_depth;
    if (g.water_depth > 0.0) z_hi_ = std::min(z_hi_, g.water_depth);
    z_hi_ = std::min(z_hi_, z_mid_ + 0.95 * g.range);
    t_nom_ = 2.0 * g.range / g.sound_speed;
    bottom_ = o.bottom && g.water_depth > 0.0;
    order_ = static_cast<int>(bottom_ ? std::max<std::int64_t>(o.bottom_order, 1) : std::max<std::int64_t>(o.ray_surface_order, 1));
    const RayProfile prof(profile.depth, profile.speed, g.water_depth);
    c_top_ = prof.speed.front();
    c_bot_ = prof.speed.back();
    const double far = std::max(std::abs(z_hi_ - z_mid_), z_mid_);
    const double r_hi = g.range * 1.005, r_lo = std::sqrt(std::max(g.range * g.range - far * far, 1.0)) * 0.995;
    // The steepest ray any path of the pattern needs, by straight lines, with a margin for the bending.
    const double own = std::max(g.array_depth, g.projector_depth);
    const double rise = bottom_ ? own + z_hi_ + 2.0 * g.water_depth * static_cast<double>(order_) : own + z_hi_ + 2.0 * z_hi_ * static_cast<double>(order_ - 1);
    const double max_angle = std::min(1.15 * std::atan2(rise, r_lo) + 3.0 * kPi / 180.0, 80.0 * kPi / 180.0);
    rx_ = std::make_shared<RayFan>(prof, g.array_depth, std::vector<double>{r_lo, r_hi}, max_angle, n_fan_, order_);
    tx_ = std::abs(g.projector_depth - g.array_depth) < 1e-6 ? rx_
                                                             : std::make_shared<RayFan>(prof, g.projector_depth, std::vector<double>{r_lo, r_hi}, max_angle, n_fan_, order_);
  }

  double z_hi() const { return z_hi_; }
  const MultipathProfile& profile() const { return profile_; }

  int paths(double z) const {
    prepare(z);
    return (tx_leg_.direct && rx_leg_.direct ? 1 : 0) | (tx_leg_.surface && rx_leg_.surface ? 2 : 0);
  }

  bool pattern(double z, double t0, std::vector<double>& delay, std::vector<cd>& amp) const {
    delay.clear();
    amp.clear();
    if (paths(z) != 3) return false;
    const Leg &a = tx_leg_, &b = rx_leg_;
    const double dr = (t0 - t_nom_) / (a.p[0] + b.p[0]);
    const cd unit = a.a[0] * b.a[0];
    for (std::size_t j = 0; j < a.t.size(); ++j) {
      for (std::size_t k = 0; k < b.t.size(); ++k) {
        delay.push_back((a.t[j] - a.t[0]) + (b.t[k] - b.t[0]) + ((a.p[j] - a.p[0]) + (b.p[k] - b.p[0])) * dr);
        amp.push_back(a.a[j] * b.a[k] / unit);
      }
    }
    return true;
  }

  /// tau_tx + tau_rx: the echo reflected at the surface both ways behind the direct one, s; NaN where a path is missing.
  double surface_delay(double z) const {
    if (paths(z) != 3) return std::numeric_limits<double>::quiet_NaN();
    return (tx_leg_.t[1] - tx_leg_.t[0]) + (rx_leg_.t[1] - rx_leg_.t[0]);
  }

  /// The depth whose surface delay is tau, searched from z0 (NaN: none nearby).
  double depth_for_delay(double tau, double z0) const {
    double z = z0;
    for (int it = 0; it < 12; ++it) {
      const double h = std::max(0.002 * z, 0.05);
      const double f = surface_delay(z) - tau, slope = (surface_delay(z + h) - surface_delay(z - h)) / (2.0 * h);
      if (!std::isfinite(f) || !std::isfinite(slope) || slope == 0.0) return std::numeric_limits<double>::quiet_NaN();
      const double step = std::clamp(-f / slope, -0.25 * z_hi_, 0.25 * z_hi_);
      z = std::clamp(z + step, 0.1, 2.0 * z_hi_);
      if (std::abs(step) < 1e-3) return z;
    }
    return std::numeric_limits<double>::quiet_NaN();
  }

  double sigma(double z) const {
    if (!plus_) {
      std::vector<double> sig = profile_.sigma;
      if (sig.size() != profile_.depth.size()) sig.assign(profile_.depth.size(), 1e300);   // no claim: the prior alone
      std::vector<double> knots, fast, slow;
      tilted_profiles(z_mid_, profile_.depth, profile_.speed, sig, profile_.gradient_sigma, knots, fast, slow);
      MultipathProfile up, down;
      up.depth = down.depth = knots;
      for (std::size_t i = 0; i < knots.size(); ++i) {
        const double mid = 0.5 * (fast[i] + slow[i]), half = 0.5 * (fast[i] - slow[i]);
        up.speed.push_back(mid + kTiltStep * half);
        down.speed.push_back(mid - kTiltStep * half);
      }
      const auto n = static_cast<std::int64_t>(static_cast<double>(n_fan_) * o_.ray_sigma_fan_fraction);
      plus_ = std::make_unique<RayPattern>(up, g_, o_, n);
      minus_ = std::make_unique<RayPattern>(down, g_, o_, n);
    }
    const double tau = surface_delay(z);
    const double a = plus_->depth_for_delay(tau, z), b = minus_->depth_for_delay(tau, z);
    if (std::isfinite(a) && std::isfinite(b)) return 0.5 * std::abs(a - b) / kTiltStep;
    if (std::isfinite(a)) return std::abs(a - z) / kTiltStep;
    if (std::isfinite(b)) return std::abs(b - z) / kTiltStep;
    return std::numeric_limits<double>::infinity();   // tilted a twentieth of what is not known, a path is gone
  }

  std::vector<double> ensemble(double z) const {
    std::vector<double> out;
    if (profile_.ensemble.empty()) return out;
    if (members_.empty()) {
      const auto n = static_cast<std::int64_t>(static_cast<double>(n_fan_) * o_.ray_sigma_fan_fraction);
      for (const std::vector<double>& speeds : profile_.ensemble) {
        MultipathProfile m;
        m.depth = profile_.depth;
        m.speed = speeds;
        members_.push_back(std::make_unique<RayPattern>(m, g_, o_, n));
      }
    }
    const double tau = surface_delay(z);
    for (const auto& m : members_) out.push_back(m->depth_for_delay(tau, z));
    return out;
  }

 private:
  struct Leg {
    std::vector<double> t, p;
    std::vector<cd> a;
    bool direct = false, surface = false;
  };
  struct One {
    double t, p;
    cd a;
    int ns, nb;
  };

  bool allowed(int ns, int nb) const { return bottom_ ? ns + nb <= order_ : (nb == 0 && ns <= order_); }

  cd coefficient(const RayArrival& q) const {
    cd a = q.amplitude;
    if (q.n_surface > 0) {
      const double k = 2.0 * kPi * g_.f_centre / c_top_;
      const double gamma = 2.0 * k * o_.surface_rms_height * std::sin(q.grazing_surface);
      const double mu = o_.surface_coefficient * std::pow(10.0, -multipath_surface_loss_db(o_, q.grazing_surface) / 20.0) * std::exp(-0.5 * gamma * gamma);
      a *= std::pow(mu, q.n_surface);
    }
    if (q.n_bottom > 0) a *= std::pow(rayleigh_reflection(q.grazing_bottom, c_bot_, o_.bottom_speed, o_.bottom_density, o_.bottom_attenuation), q.n_bottom);
    return a;
  }

  // The arrivals of one leg at depth z and horizontal range r: the direct one first, the once-surface-reflected one
  // second, then the rest.
  void leg(const RayFan& fan, double z, double r, Leg& out) const {
    std::vector<RayArrival> raw[2];
    std::vector<One> at[2];
    for (std::size_t q = 0; q < 2; ++q) {
      fan.arrivals(q, z, raw[q]);
      for (const RayArrival& x : raw[q]) {
        if (allowed(x.n_surface, x.n_bottom)) at[q].push_back({x.time, x.p, coefficient(x), x.n_surface, x.n_bottom});
      }
      std::stable_sort(at[q].begin(), at[q].end(), [](const One& a, const One& b) {
        return a.ns != b.ns ? a.ns < b.ns : a.nb != b.nb ? a.nb < b.nb : a.t < b.t;
      });
    }
    const double r_lo = fan.ranges()[0], r_hi = fan.ranges()[1], h = r_hi - r_lo, u = (r - r_lo) / h;
    const std::size_t near = u < 0.5 ? 0 : 1;
    const double r_near = near == 0 ? r_lo : r_hi;
    std::vector<One> all;
    auto count = [&](std::size_t q, int ns, int nb) {
      return std::count_if(at[q].begin(), at[q].end(), [&](const One& x) { return x.ns == ns && x.nb == nb; });
    };
    std::size_t i = 0, j = 0;
    while (i < at[0].size() || j < at[1].size()) {
      // the next class, in the sort's order
      const bool take0 = j >= at[1].size() || (i < at[0].size() && (at[0][i].ns != at[1][j].ns ? at[0][i].ns < at[1][j].ns : at[0][i].nb <= at[1][j].nb));
      const int ns = take0 ? at[0][i].ns : at[1][j].ns, nb = take0 ? at[0][i].nb : at[1][j].nb;
      const auto n0 = static_cast<std::size_t>(count(0, ns, nb)), n1 = static_cast<std::size_t>(count(1, ns, nb));
      if (n0 == n1) {
        for (std::size_t k = 0; k < n0; ++k) {
          const One &a = at[0][i + k], &b = at[1][j + k];
          const double u2 = u * u, u3 = u2 * u;
          One x;
          x.t = (2.0 * u3 - 3.0 * u2 + 1.0) * a.t + (u3 - 2.0 * u2 + u) * h * a.p + (-2.0 * u3 + 3.0 * u2) * b.t + (u3 - u2) * h * b.p;
          x.p = ((6.0 * u2 - 6.0 * u) * a.t + (3.0 * u2 - 4.0 * u + 1.0) * h * a.p + (-6.0 * u2 + 6.0 * u) * b.t + (3.0 * u2 - 2.0 * u) * h * b.p) / h;
          x.a = a.a + u * (b.a - a.a);
          x.ns = ns;
          x.nb = nb;
          all.push_back(x);
        }
      } else {
        // a path that is there at one end of the ranges and not at the other: from the nearer end, to first order
        const std::size_t first = near == 0 ? i : j, n = near == 0 ? n0 : n1;
        for (std::size_t k = 0; k < n; ++k) {
          One x = at[near][first + k];
          x.t += x.p * (r - r_near);
          all.push_back(x);
        }
      }
      i += n0;
      j += n1;
    }
    out = Leg{};
    std::ptrdiff_t direct = -1, surface = -1;
    for (std::size_t k = 0; k < all.size(); ++k) {
      if (direct < 0 && all[k].ns == 0 && all[k].nb == 0) direct = static_cast<std::ptrdiff_t>(k);
      if (surface < 0 && all[k].ns == 1 && all[k].nb == 0) surface = static_cast<std::ptrdiff_t>(k);
    }
    out.direct = direct >= 0;
    out.surface = surface >= 0;
    auto push = [&](const One& x) {
      out.t.push_back(x.t);
      out.p.push_back(x.p);
      out.a.push_back(x.a);
    };
    if (direct >= 0) push(all[static_cast<std::size_t>(direct)]);
    if (surface >= 0) push(all[static_cast<std::size_t>(surface)]);
    if (!(out.direct && out.surface)) return;
    const double floor = o_.ray_min_amplitude * std::abs(out.a[0]);
    for (std::size_t k = 0; k < all.size(); ++k) {
      if (static_cast<std::ptrdiff_t>(k) == direct || static_cast<std::ptrdiff_t>(k) == surface) continue;
      if (std::abs(all[k].a) >= floor) push(all[k]);
    }
  }

  // The two legs at depth z, at the horizontal range where the direct echo's two-way time is the detection's.
  void prepare(double z) const {
    if (z == cache_z_) return;
    cache_z_ = z;
    double r = std::sqrt(std::max(g_.range * g_.range - (z - z_mid_) * (z - z_mid_), 1.0));
    for (int it = 0; it < 4; ++it) {
      leg(*rx_, z, r, rx_leg_);
      if (tx_ == rx_) tx_leg_ = rx_leg_; else leg(*tx_, z, r, tx_leg_);
      if (it == 3 || !(tx_leg_.direct && rx_leg_.direct)) break;
      const double step = (t_nom_ - (tx_leg_.t[0] + rx_leg_.t[0])) / (tx_leg_.p[0] + rx_leg_.p[0]);
      if (std::abs(step) < 1e-4) break;
      r += step;
    }
  }

  MultipathProfile profile_;
  MultipathGeometry g_;
  MultipathOptions o_;
  std::int64_t n_fan_;
  double z_mid_ = 0.0, z_hi_ = 0.0, t_nom_ = 0.0, c_top_ = 1500.0, c_bot_ = 1500.0;
  bool bottom_ = false;
  int order_ = 1;
  std::shared_ptr<RayFan> tx_, rx_;
  mutable double cache_z_ = std::numeric_limits<double>::quiet_NaN();
  mutable Leg tx_leg_, rx_leg_;
  mutable std::unique_ptr<RayPattern> plus_, minus_;
  mutable std::vector<std::unique_ptr<RayPattern>> members_;
};

std::shared_ptr<const RayPattern> multipath_rays(const MultipathProfile& profile, const MultipathGeometry& g, const MultipathOptions& o) {
  return std::make_shared<const RayPattern>(profile, g, o, o.ray_fan);
}

int multipath_ray_paths(const RayPattern& rays, double z) { return rays.paths(z); }
double multipath_ray_sigma(const RayPattern& rays, double z) { return rays.sigma(z); }
std::vector<double> multipath_ray_ensemble(const RayPattern& rays, double z) { return rays.ensemble(z); }
bool multipath_ray_pattern(const RayPattern& rays, double z, double t0, std::vector<double>& delay, std::vector<cd>& amp) {
  return rays.pattern(z, t0, delay, amp);
}
double multipath_ray_surface_delay(const RayPattern& rays, double z) { return rays.surface_delay(z); }
double multipath_ray_z_hi(const RayPattern& rays) { return rays.z_hi(); }
const MultipathProfile& multipath_ray_profile(const RayPattern& rays) { return rays.profile(); }

}  // namespace sonar::live
