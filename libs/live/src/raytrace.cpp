// Rays through a layered profile (raytrace.hpp). Mirrors python/gpusonar/raytrace.py statement by
// statement where the two do the same thing (RayProfile, RayStepper, ray_depth_arcs, eigenrays);
// tests/python/test_raytrace_cpp.py holds them together. RayFan is this file's own.
#include "sonar/live/raytrace.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

namespace sonar::live {
namespace {

constexpr double kPi = 3.14159265358979323846;
using cd = std::complex<double>;

double sign(double v) { return v > 0.0 ? 1.0 : v < 0.0 ? -1.0 : 0.0; }

}  // namespace

RayProfile::RayProfile(const std::vector<double>& depths, const std::vector<double>& speeds, double water_depth) {
  std::vector<double> d = depths, c = speeds;
  if (d.empty() || d.size() != c.size()) throw std::invalid_argument("profile: depths must increase and be as many as the speeds");
  for (std::size_t i = 1; i < d.size(); ++i) {
    if (!(d[i] > d[i - 1])) throw std::invalid_argument("profile: depths must increase and be as many as the speeds");
  }
  if (d.size() == 1) {
    d = {d[0], d[0] + 1.0};
    c = {c[0], c[0]};
  }
  if (d.front() > 0.0) {                               // continued to the surface with the first speed
    d.insert(d.begin(), 0.0);
    c.insert(c.begin(), c.front());
  }
  has_bottom = std::isfinite(water_depth) && water_depth > 0.0;
  bottom = has_bottom ? water_depth : std::max(d.back(), 0.0) + 2.0e4;
  if (d.back() < bottom) {                             // continued to the bottom with the last gradient
    const std::size_t n = d.size();
    const double g = (c[n - 1] - c[n - 2]) / (d[n - 1] - d[n - 2]);
    c.push_back(c[n - 1] + g * (bottom - d[n - 1]));
    d.push_back(bottom);
  } else if (d.back() > bottom) {
    std::size_t keep = 0;
    while (keep < d.size() && d[keep] < bottom) ++keep;
    // numpy.interp(bottom, d, c)
    const std::size_t hi = std::min(keep, d.size() - 1);
    const double cb = hi == 0 ? c[0] : c[hi - 1] + (bottom - d[hi - 1]) / (d[hi] - d[hi - 1]) * (c[hi] - c[hi - 1]);
    d.resize(keep);
    c.resize(keep);
    d.push_back(bottom);
    c.push_back(cb);
  }
  depth = d;
  speed = c;
  gradient.resize(d.size() - 1);
  for (std::size_t i = 0; i + 1 < d.size(); ++i) gradient[i] = (c[i + 1] - c[i]) / (d[i + 1] - d[i]);
}

double RayProfile::c(double z) const {
  if (z <= depth.front()) return speed.front();
  if (z >= depth.back()) return speed.back();
  const auto hi = static_cast<std::size_t>(std::upper_bound(depth.begin(), depth.end(), z) - depth.begin());
  return speed[hi - 1] + (z - depth[hi - 1]) / (depth[hi] - depth[hi - 1]) * (speed[hi] - speed[hi - 1]);
}

std::size_t RayProfile::layer(double z, bool down) const {
  // numpy.searchsorted(depth, z, side = "right" if down else "left") - 1, clipped
  const auto at = down ? std::upper_bound(depth.begin(), depth.end(), z) : std::lower_bound(depth.begin(), depth.end(), z);
  const auto i = static_cast<std::ptrdiff_t>(at - depth.begin()) - 1;
  return static_cast<std::size_t>(std::clamp<std::ptrdiff_t>(i, 0, static_cast<std::ptrdiff_t>(gradient.size()) - 1));
}

RayStepper::RayStepper(const RayProfile& profile, double z0, double launch) : prof_(&profile) {
  const double c0 = profile.c(z0);
  p_ = std::cos(launch) / c0;
  s_ = std::sin(launch);
  z_ = z0;
  if (s_ == 0.0) {                                     // horizontal: it bends towards slower water
    const double gi = profile.gradient[profile.layer(z_, true)];
    if (gi == 0.0) dead_ = true;
    else s_ = -1e-12 * sign(gi);
  }
}

RayEnd RayStepper::to(double r_end, int max_bounces, std::int64_t max_steps) {
  RayEnd out;
  const std::vector<double>&d = prof_->depth, &c = prof_->speed, &g_all = prof_->gradient;
  const double p = p_;
  for (std::int64_t step = 0; step < max_steps && !dead_; ++step) {
    const bool down = s_ > 0.0;
    const std::size_t i = prof_->layer(z_, down);
    const double g = g_all[i], z_lo = d[i], z_hi = d[i + 1], c_lo = c[i], c_hi = c[i + 1];
    double s_b, zb, dr, dt;
    bool reach = true;
    if (g == 0.0) {
      zb = down ? z_hi : z_lo;
      dr = std::abs(zb - z_) * (p * c_lo) / std::abs(s_);
      if (r_ + dr >= r_end) {
        const double frac = r_end - r_;
        return {z_ + sign(s_) * frac * std::abs(s_) / (p * c_lo), t_ + frac / (p * c_lo * c_lo), s_, ns_, nb_, turns_, true};
      }
      dt = dr / (p * c_lo * c_lo);
      s_b = s_;
    } else {
      // where it leaves the layer: the boundary ahead if it gets there before turning, else the one behind
      if (down) {
        reach = p * c_hi < 1.0;
        s_b = reach ? std::sqrt(std::max(1.0 - (p * c_hi) * (p * c_hi), 0.0)) : -std::sqrt(std::max(1.0 - (p * c_lo) * (p * c_lo), 0.0));
        zb = reach ? z_hi : z_lo;
      } else {
        reach = p * c_lo < 1.0;
        s_b = reach ? -std::sqrt(std::max(1.0 - (p * c_lo) * (p * c_lo), 0.0)) : std::sqrt(std::max(1.0 - (p * c_hi) * (p * c_hi), 0.0));
        zb = reach ? z_lo : z_hi;
      }
      dr = (s_ - s_b) / (p * g);
      if (r_ + dr >= r_end) {
        const double s_e = s_ - p * g * (r_end - r_);
        const double z_end = z_lo + (std::sqrt(std::max(1.0 - s_e * s_e, 0.0)) / p - c_lo) / g;
        return {z_end, t_ + (std::atanh(s_) - std::atanh(s_e)) / g, s_e, ns_, nb_, turns_ + static_cast<int>(s_ * s_e < 0.0), true};
      }
      if (!reach) ++turns_;
      dt = (std::atanh(s_) - std::atanh(s_b)) / g;
    }
    r_ += dr;
    t_ += dt;
    z_ = zb;
    s_ = s_b;
    if (z_ <= 0.0 && s_ < 0.0) {
      s_ = -s_;
      ++ns_;
    } else if (z_ >= prof_->bottom && s_ > 0.0) {
      s_ = -s_;
      ++nb_;
    }
    if (s_ == 0.0) s_ = z_ <= 0.0 ? 1e-12 : -1e-12;   // grazing a boundary exactly: nudge back into the water
    if (ns_ + nb_ > max_bounces) dead_ = true;
  }
  dead_ = true;
  out.z = z_;
  out.time = std::numeric_limits<double>::quiet_NaN();
  out.s = s_;
  out.n_surface = ns_;
  out.n_bottom = nb_;
  out.turns = turns_;
  out.ok = false;
  return out;
}

RayEnd ray_trace(const RayProfile& profile, double z0, double launch, double r_end) { return RayStepper(profile, z0, launch).to(r_end); }

double ray_depth_arcs(double receiver_depth, double elevation, double slant_range, const std::vector<double>& ssp_depth,
                      const std::vector<double>& ssp_speed, double water_depth) {
  const RayProfile prof(ssp_depth, ssp_speed, water_depth);
  const double t_want = slant_range / prof.c(receiver_depth);
  double lo = 0.0, hi = slant_range * 1.2 + 1.0;
  for (int it = 0; it < 60; ++it) {
    const double mid = 0.5 * (lo + hi);
    const RayEnd ray = ray_trace(prof, receiver_depth, -elevation, mid);
    if (!ray.ok || ray.time > t_want) hi = mid; else lo = mid;
  }
  return ray_trace(prof, receiver_depth, -elevation, 0.5 * (lo + hi)).z;
}

cd rayleigh_reflection(double grazing, double c_water, double speed, double density, double attenuation) {
  const double delta = attenuation / (40.0 * kPi * std::log10(std::exp(1.0)));
  const cd c_b = speed / cd(1.0, delta);
  const cd n = c_water / c_b;
  cd root = std::sqrt(n * n - std::cos(grazing) * std::cos(grazing));
  if (root.imag() < 0.0) root = -root;
  const double ms = density * std::sin(grazing);
  return (ms - root) / (ms + root);
}

std::vector<Eigenray> eigenrays(const RayProfile& profile, double z_source, double z_receiver, double range, const EigenrayOptions& o) {
  const std::int64_t n_fan = o.n_fan | 1;
  std::vector<double> angles;
  for (std::int64_t i = 0; i < n_fan; ++i) {
    const double a = (-o.max_angle_deg + 2.0 * o.max_angle_deg * static_cast<double>(i) / static_cast<double>(n_fan - 1)) * kPi / 180.0;
    if (std::abs(a) > 1e-9) angles.push_back(a);
  }
  std::vector<RayEnd> rays(angles.size());
  for (std::size_t i = 0; i < angles.size(); ++i) rays[i] = ray_trace(profile, z_source, angles[i], range);
  const double c_s = profile.c(z_source), c_r = profile.c(z_receiver);
  std::vector<Eigenray> out;
  auto same = [](const RayEnd& a, const RayEnd& b) {
    return a.n_surface == b.n_surface && a.n_bottom == b.n_bottom && a.turns == b.turns && (a.s > 0.0) == (b.s > 0.0);
  };
  // Eigenrays with launch angles between two rays of the fan. Where the two have different histories the
  // interval is halved until its parts have one history each.
  std::function<void(double, const RayEnd&, double, const RayEnd&, int)> close = [&](double a_lo, const RayEnd& ray_lo, double a_hi,
                                                                                    const RayEnd& ray_hi, int level) {
    if (!(ray_lo.ok && ray_hi.ok)) return;
    if (!same(ray_lo, ray_hi)) {
      if (level < 30) {
        const double mid = 0.5 * (a_lo + a_hi);
        const RayEnd ray_mid = ray_trace(profile, z_source, mid, range);
        close(a_lo, ray_lo, mid, ray_mid, level + 1);
        close(mid, ray_mid, a_hi, ray_hi, level + 1);
      }
      return;
    }
    if ((ray_lo.z - z_receiver) * (ray_hi.z - z_receiver) > 0.0) return;
    if (ray_lo.n_surface + ray_lo.n_bottom > o.max_bounces || (ray_lo.n_bottom && !o.bottom.given)) return;
    double lo = a_lo, hi = a_hi;
    const double f_lo = ray_lo.z - z_receiver;
    for (int it = 0; it < 50; ++it) {
      const double mid = 0.5 * (lo + hi);
      const RayEnd q = ray_trace(profile, z_source, mid, range);
      if (!q.ok || !same(q, ray_lo)) break;
      if ((q.z - z_receiver) * f_lo > 0.0) lo = mid; else hi = mid;
    }
    const double launch = 0.5 * (lo + hi);
    const RayEnd q = ray_trace(profile, z_source, launch, range);
    if (!q.ok || !same(q, ray_lo) || std::abs(q.z - z_receiver) > 0.05) return;
    const double dz_dth = std::abs(ray_hi.z - ray_lo.z) / (a_hi - a_lo);
    const double cos_r = std::max(std::sqrt(std::max(1.0 - q.s * q.s, 0.0)), 1e-6);
    const double slant = std::hypot(range, z_receiver - z_source);
    const double intensity = std::cos(launch) / (range * std::max(dz_dth, 1e-12) * cos_r);
    const double cap = (o.caustic_clip / slant) * (o.caustic_clip / slant);
    const double amp = std::sqrt(std::min(intensity, cap) * c_r / c_s);
    cd coeff = std::pow(cd(o.surface_coefficient), q.n_surface);
    if (q.n_bottom) {
      const double c_b = profile.speed.back();
      coeff *= std::pow(rayleigh_reflection(std::acos(std::min(std::cos(launch) / c_s * c_b, 1.0)), c_b, o.bottom.speed, o.bottom.density, o.bottom.attenuation),
                        q.n_bottom);
    }
    out.push_back({launch, std::asin(std::clamp(q.s, -1.0, 1.0)), q.time, amp * coeff, q.n_surface, q.n_bottom, q.turns, intensity > cap});
  };
  for (std::size_t i = 0; i + 1 < angles.size(); ++i) close(angles[i], rays[i], angles[i + 1], rays[i + 1], 0);
  std::stable_sort(out.begin(), out.end(), [](const Eigenray& a, const Eigenray& b) { return a.time < b.time; });
  std::vector<Eigenray> kept;                          // the same ray found in two neighbouring brackets
  for (const Eigenray& e : out) {
    bool twice = false;
    for (const Eigenray& k : kept) twice = twice || (std::abs(e.time - k.time) < 1e-9 && e.n_surface == k.n_surface && e.n_bottom == k.n_bottom);
    if (!twice) kept.push_back(e);
  }
  return kept;
}

double transmission_loss(const std::vector<Eigenray>& rays, double frequency) {
  if (rays.empty()) return std::numeric_limits<double>::infinity();
  if (!(frequency > 0.0)) {
    double sum = 0.0;
    for (const Eigenray& e : rays) sum += std::norm(e.amplitude);
    return -10.0 * std::log10(sum);
  }
  cd sum = 0.0;
  for (const Eigenray& e : rays) sum += e.amplitude * std::polar(1.0, -2.0 * kPi * frequency * e.time);
  return -20.0 * std::log10(std::max(std::abs(sum), 1e-300));
}

RayFan::RayFan(const RayProfile& profile, double z_source, const std::vector<double>& ranges, double max_angle, std::int64_t n, int max_bounces,
               double caustic_clip)
    : prof_(profile), z_source_(z_source), c_source_(profile.c(z_source)), clip_(caustic_clip), n_(std::max<std::int64_t>(n, 2)), ranges_(ranges) {
  const double dtheta = 2.0 * max_angle / static_cast<double>(n_);
  launch_.resize(static_cast<std::size_t>(n_));
  for (std::int64_t i = 0; i < n_; ++i) launch_[static_cast<std::size_t>(i)] = -max_angle + (static_cast<double>(i) + 0.5) * dtheta;   // never exactly horizontal
  shoot(max_bounces);
}

RayFan::RayFan(const RayProfile& profile, double z_source, const std::vector<double>& ranges, const std::vector<double>& launches, int max_bounces,
               double caustic_clip)
    : prof_(profile), z_source_(z_source), c_source_(profile.c(z_source)), clip_(caustic_clip), n_(static_cast<std::int64_t>(launches.size())),
      ranges_(ranges), launch_(launches) {
  if (n_ < 2) throw std::invalid_argument("RayFan: needs two or more launch angles");
  shoot(max_bounces);
}

void RayFan::shoot(int max_bounces) {
  ends_.resize(ranges_.size() * static_cast<std::size_t>(n_));
  for (std::int64_t i = 0; i < n_; ++i) {
    const auto u = static_cast<std::size_t>(i);
    RayStepper ray(prof_, z_source_, launch_[u]);
    for (std::size_t k = 0; k < ranges_.size(); ++k) ends_[k * static_cast<std::size_t>(n_) + u] = ray.to(ranges_[k], max_bounces);
  }
}

void RayFan::arrivals(std::size_t i_range, double z, std::vector<RayArrival>& out) const {
  const RayEnd* e = ends_.data() + i_range * static_cast<std::size_t>(n_);
  const double r = ranges_[i_range];
  const double c_r = prof_.c(z), c_top = prof_.speed.front(), c_bot = prof_.speed.back();
  const double slant = std::hypot(r, z - z_source_);
  const double cap = (clip_ / slant) * (clip_ / slant);
  for (std::int64_t i = 0; i + 1 < n_; ++i) {
    const RayEnd &a = e[i], &b = e[i + 1];
    if (!(a.ok && b.ok) || a.n_surface != b.n_surface || a.n_bottom != b.n_bottom) continue;
    const double fa = a.z - z, fb = b.z - z;
    if (!(fa == 0.0 || fa * fb < 0.0)) continue;
    const double dz = b.z - a.z;
    const double u = dz != 0.0 ? -fa / dz : 0.0;
    // cubic Hermite in depth: dt/dz = s / c on the wavefront
    const double ma = a.s / prof_.c(a.z), mb = b.s / prof_.c(b.z);
    const double u2 = u * u, u3 = u2 * u;
    RayArrival q;
    q.time = (2.0 * u3 - 3.0 * u2 + 1.0) * a.time + (u3 - 2.0 * u2 + u) * dz * ma + (-2.0 * u3 + 3.0 * u2) * b.time + (u3 - u2) * dz * mb;
    const auto ui = static_cast<std::size_t>(i);
    const double dtheta = launch_[ui + 1] - launch_[ui];
    q.launch = launch_[ui] + u * dtheta;
    q.p = std::cos(q.launch) / c_source_;
    q.s_end = a.s + u * (b.s - a.s);
    const double cos_r = std::clamp(q.p * c_r, 1e-6, 1.0);                    // Snell: cos(theta) = p c
    const double intensity = std::cos(q.launch) / (r * std::max(std::abs(dz) / dtheta, 1e-12) * cos_r);
    q.caustic = intensity > cap;
    q.amplitude = std::sqrt(std::min(intensity, cap) * c_r / c_source_);
    q.grazing_surface = std::acos(std::min(q.p * c_top, 1.0));
    q.grazing_bottom = std::acos(std::min(q.p * c_bot, 1.0));
    q.n_surface = a.n_surface;
    q.n_bottom = a.n_bottom;
    out.push_back(q);
  }
}

}  // namespace sonar::live
