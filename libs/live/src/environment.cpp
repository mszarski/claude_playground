#include "sonar/live/environment.hpp"

#include "sonar/live/raytrace.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "sonar/live/elevation.hpp"

namespace sonar::live {
namespace {

constexpr double kNoClaim = 1e9;   // a sigma no rung claims: the prior decides
constexpr double kTiltFraction = 0.05;   // the part of the profile's error the sensitivity is measured with

// Where the value after "key": starts and ends in `extra`; false when the key is not there.
bool value_span(const std::string& extra, const std::string& key, std::size_t& key_at, std::size_t& first, std::size_t& last) {
  const std::string k = "\"" + key + "\"";
  key_at = extra.find(k);
  if (key_at == std::string::npos) return false;
  const auto colon = extra.find(':', key_at + k.size());
  if (colon == std::string::npos) return false;
  first = extra.find_first_not_of(" \t\n", colon + 1);
  if (first == std::string::npos) return false;
  int depth = 0;
  bool quoted = false;
  for (last = first; last < extra.size(); ++last) {
    const char ch = extra[last];
    if (quoted) {
      if (ch == '\\') ++last;
      else if (ch == '"') quoted = false;
    } else if (ch == '"') quoted = true;
    else if (ch == '[' || ch == '{') ++depth;
    else if (ch == ']' || ch == '}') {
      if (depth == 0) break;
      --depth;
    } else if (ch == ',' && depth == 0) break;
  }
  return true;
}

double interp(const std::vector<double>& x, const std::vector<double>& y, double at) {   // numpy.interp
  if (at <= x.front()) return y.front();
  if (at >= x.back()) return y.back();
  const auto hi = static_cast<std::size_t>(std::upper_bound(x.begin(), x.end(), at) - x.begin());
  const double w = (at - x[hi - 1]) / (x[hi] - x[hi - 1]);
  return y[hi - 1] + w * (y[hi] - y[hi - 1]);
}

}  // namespace

double extra_number(const std::string& extra, const std::string& key, double fallback) {
  std::size_t key_at, first, last;
  if (!value_span(extra, key, key_at, first, last)) return fallback;
  char* end = nullptr;
  const double v = std::strtod(extra.c_str() + first, &end);
  return end == extra.c_str() + first ? fallback : v;
}

std::vector<std::vector<double>> extra_rows(const std::string& extra, const std::string& key, std::size_t width) {
  std::vector<std::vector<double>> rows;
  std::size_t key_at, first, last;
  if (!value_span(extra, key, key_at, first, last) || extra[first] != '[') return rows;
  std::vector<double> row;
  for (std::size_t i = first + 1; i < last; ++i) {
    const char ch = extra[i];
    if (ch == '[') row.clear();
    else if (ch == ']') {
      if (row.size() == width) rows.push_back(row);
      row.clear();
    } else if (ch == '-' || ch == '+' || ch == '.' || (ch >= '0' && ch <= '9') || ch == 'N' || ch == 'n') {
      char* end = nullptr;
      const double v = std::strtod(extra.c_str() + i, &end);
      if (end == extra.c_str() + i) continue;        // "null": not a number
      row.push_back(v);
      i = static_cast<std::size_t>(end - extra.c_str()) - 1;
    }
  }
  return rows;
}

void extra_set(std::string& extra, const std::string& key, const std::string& value) {
  std::size_t key_at, first, last;
  if (value_span(extra, key, key_at, first, last)) {
    extra.replace(first, last - first, value);
    return;
  }
  const std::string item = "\"" + key + "\": " + value;
  const auto brace = extra.find('{');
  if (extra.empty() || brace == std::string::npos) {
    extra = "{" + item + "}";
    return;
  }
  const bool empty_object = extra.find_first_not_of(" \t\n", brace + 1) == extra.find('}');
  extra.insert(brace + 1, item + (empty_object ? "" : ", "));
}

WaterDepth WaterDepthTracker::update(std::int64_t time_ns, double x, double y, double depth, double altitude, bool lock, double altitude_sigma,
                                     double depth_sigma) {
  WaterDepth out;
  if (!opt_.from_altitude) return out;
  const bool valid = lock && std::isfinite(altitude) && altitude > 0.0 && std::isfinite(depth) &&
                     (opt_.max_altitude_m <= 0.0 || altitude <= opt_.max_altitude_m);
  if (valid) {
    const double sa = std::isfinite(altitude_sigma) ? altitude_sigma : opt_.altitude_sigma_m;
    const double sd = std::isfinite(depth_sigma) ? depth_sigma : opt_.depth_sigma_m;
    have_ = true;
    time_ns_ = time_ns;
    x_ = x;
    y_ = y;
    depth_ = depth + altitude + opt_.dvl_offset_m;
    sigma_ = std::hypot(sa, sd);
    out.depth = depth_;
    out.sigma = sigma_;
    return out;
  }
  if (!have_) return out;
  // No bottom lock: the last value, less certain the longer and the further it is carried.
  const double held = static_cast<double>(time_ns - time_ns_) * 1e-9;
  if (held < 0.0 || held > opt_.hold_s) return out;
  const double moved = std::hypot(x - x_, y - y_);
  out.depth = depth_;
  out.sigma = std::sqrt(sigma_ * sigma_ + std::pow(opt_.hold_sigma_m_per_s * held, 2) + std::pow(opt_.hold_sigma_m_per_m * moved, 2));
  out.held = true;
  out.held_s = held;
  return out;
}

void family_profile(const ProfileOptions& o, double own, double c_own, double bottom, std::vector<double>& depth, std::vector<double>& speed) {
  depth.clear();
  speed.clear();
  if (o.family == "isothermal") {
    if (!std::isfinite(o.family_gradient)) throw std::invalid_argument("elevation.profile.family: isothermal needs gradient_per_s");
    depth = {0.0, bottom};
    speed = {c_own - o.family_gradient * own, c_own + o.family_gradient * (bottom - own)};
  } else if (o.family == "mixed_layer") {
    if (!(std::isfinite(o.family_gradient) && std::isfinite(o.layer_depth) && std::isfinite(o.thermocline_gradient) &&
          std::isfinite(o.thermocline_thickness)) || o.layer_depth <= 0.0 || o.thermocline_thickness <= 0.0) {
      throw std::invalid_argument("elevation.profile.family: mixed_layer needs gradient_per_s, layer_depth_m, thermocline_gradient_per_s "
                                  "and thermocline_thickness_m");
    }
    const double d = o.layer_depth, t = o.thermocline_thickness, g = o.family_gradient, gt = o.thermocline_gradient;
    auto shape = [&](double z) {
      if (z <= d) return g * z;
      if (z <= d + t) return g * d + gt * (z - d);
      return g * d + gt * t + g * (z - d - t);
    };
    depth = {0.0, d, d + t, std::max(bottom, d + t + 1.0)};
    for (double z : depth) speed.push_back(c_own + shape(z) - shape(own));
  } else {
    throw std::invalid_argument("elevation.profile.family: '" + o.family + "' (isothermal or mixed_layer)");
  }
}

SoundSpeedLadder::SoundSpeedLadder(const ProfileOptions& options) : opt_(options) {
  if (!opt_.enabled) return;
  bool built = false;
  for (const std::string& r : opt_.ladder) {
    if (r != "supplied" && r != "built" && r != "family" && r != "local" && r != "prior") {
      throw std::invalid_argument("elevation.profile.ladder: '" + r + "' (supplied, built, family, local or prior)");
    }
    built = built || r == "built";
  }
  if (!(opt_.bin_m > 0.0)) throw std::invalid_argument("elevation.profile.built.bin_m must be positive");
  if (!built) return;
  if (!opt_.cast.empty()) load(opt_.cast, true);
  if (!opt_.file.empty() && std::ifstream(opt_.file).good()) load(opt_.file, false);
}

void SoundSpeedLadder::add(std::int64_t time_ns, double x, double y, double depth, double sound_speed) {
  if (!opt_.enabled || !std::isfinite(depth) || !std::isfinite(sound_speed) || sound_speed <= 0.0) return;
  const auto index = static_cast<std::int64_t>(std::floor(depth / opt_.bin_m));
  auto it = std::lower_bound(bins_.begin(), bins_.end(), index, [](const Bin& b, std::int64_t i) { return b.index < i; });
  if (it == bins_.end() || it->index != index) {
    bins_.insert(it, Bin{index, depth, sound_speed, 1, time_ns, x, y, false});
  } else {
    // A running mean that forgets: the water changes, and a bin visited again should follow it.
    if (it->fixed) it->count = 0;
    it->count = std::min<std::int64_t>(it->count + 1, 16);
    it->depth += (depth - it->depth) / static_cast<double>(it->count);
    it->speed += (sound_speed - it->speed) / static_cast<double>(it->count);
    it->time_ns = time_ns;
    it->x = x;
    it->y = y;
    it->fixed = false;
  }
  if (!opt_.file.empty() && ++unsaved_ >= 1) {
    save(opt_.file);
    unsaved_ = 0;
  }
}

bool SoundSpeedLadder::absorb(const PingRecord& record, double surface_z) {
  const std::string& extra = record.metadata.extra_json;
  const std::int64_t t = record.transmit.time_ns;
  for (const auto& row : extra_rows(extra, "sound_speed_samples", 5)) {
    add(t + static_cast<std::int64_t>(std::llround(row[0] * 1e9)), row[1], row[2], row[3] - surface_z, row[4]);
  }
  const bool measured = extra_number(extra, "sound_speed_measured", 0.0) != 0.0;
  const Vec3d& p = record.geometry.rx_pose.position;
  if (measured) add(t, p.x, p.y, p.z - surface_z, record.environment.sound_speed);
  return measured;
}

LadderProfile SoundSpeedLadder::evaluate(std::int64_t time_ns, double x, double y, double own, double sound, bool measured, double water,
                                         const std::vector<double>& supplied_depth, const std::vector<double>& supplied_speed,
                                         double gradient, double gradient_sigma) const {
  LadderProfile out;
  if (!opt_.enabled) return out;
  const bool have_water = std::isfinite(water) && water > 0.0;
  for (const std::string& rung : opt_.ladder) {
    if (rung == "supplied") {
      if (supplied_depth.empty()) continue;
      out.rung = rung;
      out.depth = supplied_depth;
      out.speed = supplied_speed;
      out.sigma.assign(out.depth.size(), opt_.supplied_sigma > 0.0 ? opt_.supplied_sigma : kNoClaim);
      return out;
    }
    if (rung == "built") {
      std::vector<double> d, c, s;
      for (const Bin& b : bins_) {
        double age = 0.0, far = 0.0;
        if (!b.fixed) {
          age = std::abs(static_cast<double>(time_ns - b.time_ns)) * 1e-9;
          far = std::hypot(x - b.x, y - b.y);
          if ((opt_.max_age_s > 0.0 && age > opt_.max_age_s) || (opt_.max_distance_m > 0.0 && far > opt_.max_distance_m)) continue;
        }
        if (!d.empty() && !(b.depth > d.back())) continue;      // bin means are ordered like the bins; guard the profile anyway
        d.push_back(b.depth);
        c.push_back(b.speed);
        s.push_back(std::sqrt(opt_.sample_sigma * opt_.sample_sigma + std::pow(opt_.age_sigma_per_hour * age / 3600.0, 2) +
                              std::pow(opt_.distance_sigma_per_km * far / 1000.0, 2)));
      }
      if (d.size() < 2 || d.back() - d.front() < opt_.min_span_m) continue;
      if (!(opt_.sample_sigma > 0.0)) {
        throw std::invalid_argument("elevation.profile.built.sample_sigma_m_s is not set: the sound-speed sensor's accuracy has no default");
      }
      out.rung = rung;
      out.span_lo = d.front();
      out.span_hi = d.back();
      out.bins = static_cast<std::int64_t>(d.size());
      // Outside the sampled depths: the assumed gradient (or none), less certain with every metre.
      const double widen = opt_.extrapolation_sigma_per_s > 0.0 ? opt_.extrapolation_sigma_per_s : gradient_sigma;
      if (d.front() > 0.0) {
        out.depth.push_back(0.0);
        out.speed.push_back(c.front() - gradient * d.front());
        out.sigma.push_back(s.front() + widen * d.front());
      }
      out.depth.insert(out.depth.end(), d.begin(), d.end());
      out.speed.insert(out.speed.end(), c.begin(), c.end());
      out.sigma.insert(out.sigma.end(), s.begin(), s.end());
      const double bottom = std::max({have_water ? water : 0.0, d.back(), own}) + 10000.0;
      out.depth.push_back(bottom);
      out.speed.push_back(c.back() + gradient * (bottom - d.back()));
      out.sigma.push_back(s.back() + widen * (bottom - d.back()));
      return out;
    }
    if (rung == "family") {
      if (opt_.family.empty()) continue;
      out.rung = rung + ":" + opt_.family;
      family_profile(opt_, own, sound, std::max({have_water ? water : 0.0, own, 0.0}) + 10000.0, out.depth, out.speed);
      out.sigma.assign(out.depth.size(), opt_.family_sigma > 0.0 ? opt_.family_sigma : kNoClaim);
      return out;
    }
    if (rung == "local") {
      if (!measured) continue;
      out.rung = rung;
      return out;
    }
    out.rung = "prior";
    return out;
  }
  return out;
}

void SoundSpeedLadder::save(const std::string& path) const {
  std::ofstream f(path + ".tmp");
  f.precision(12);
  f << "# sound-speed profile built by the vehicle: bin depth_m speed_m_s count time_ns x_m y_m fixed\n";
  for (const Bin& b : bins_) {
    f << b.index << " " << b.depth << " " << b.speed << " " << b.count << " " << b.time_ns << " " << b.x << " " << b.y << " " << (b.fixed ? 1 : 0) << "\n";
  }
  f.close();
  std::rename((path + ".tmp").c_str(), path.c_str());
}

void SoundSpeedLadder::load(const std::string& path, bool cast) {
  std::ifstream f(path);
  if (!f) throw std::invalid_argument("elevation.profile: cannot read " + path);
  std::string line;
  while (std::getline(f, line)) {
    for (char& ch : line) {
      if (ch == ',' || ch == ';' || ch == '\t') ch = ' ';
    }
    std::istringstream in(line.substr(0, line.find('#')));
    Bin b{};
    if (cast) {
      if (!(in >> b.depth >> b.speed)) continue;        // a header or an empty line
      b.index = static_cast<std::int64_t>(std::floor(b.depth / opt_.bin_m));
      b.count = 1;
      b.fixed = true;
    } else {
      int fixed = 0;
      if (!(in >> b.index >> b.depth >> b.speed >> b.count >> b.time_ns >> b.x >> b.y >> fixed)) continue;
      b.fixed = fixed != 0;
    }
    auto it = std::lower_bound(bins_.begin(), bins_.end(), b.index, [](const Bin& a, std::int64_t i) { return a.index < i; });
    if (it != bins_.end() && it->index == b.index) *it = b;
    else bins_.insert(it, b);
  }
}

void tilted_profiles(double own, const std::vector<double>& depth, const std::vector<double>& speed, const std::vector<double>& sigma,
                     double gradient_sigma, std::vector<double>& knots, std::vector<double>& fast, std::vector<double>& slow) {
  if (depth.empty() || depth.size() != speed.size() || depth.size() != sigma.size()) {
    throw std::invalid_argument("tilted_profiles: the profile needs depths and as many speeds and sigmas");
  }
  const double g = std::abs(gradient_sigma);
  auto error = [&](double z) { return std::min(interp(depth, sigma, z), g * std::abs(z - own)); };
  auto excess = [&](double z) { return interp(depth, sigma, z) - g * std::abs(z - own); };
  std::vector<double> base = depth;
  if (own > depth.front() && own < depth.back() && !std::binary_search(depth.begin(), depth.end(), own)) {
    base.insert(std::upper_bound(base.begin(), base.end(), own), own);
  }
  knots.clear();
  for (std::size_t i = 0; i < base.size(); ++i) {
    knots.push_back(base[i]);
    if (i + 1 == base.size()) break;
    // Both limits are straight between these knots: where they cross, the smaller one changes.
    const double a = excess(base[i]), b = excess(base[i + 1]);
    if ((a < 0.0) != (b < 0.0) && a != b) {
      const double z = base[i] + (base[i + 1] - base[i]) * a / (a - b);
      if (z > base[i] && z < base[i + 1]) knots.push_back(z);
    }
  }
  fast.clear();
  slow.clear();
  for (double z : knots) {
    const double c = interp(depth, speed, z), e = (z < own ? -1.0 : 1.0) * error(z);
    fast.push_back(c + e);
    slow.push_back(c - e);
  }
}

double profile_depth_sigma(double receiver_depth, double elevation, double slant_range, const std::vector<double>& depth,
                           const std::vector<double>& speed, const std::vector<double>& sigma, double gradient_sigma, double water_depth,
                           bool arcs) {
  std::vector<double> knots, fast, slow;
  tilted_profiles(receiver_depth, depth, speed, sigma, gradient_sigma, knots, fast, slow);
  // A twentieth of the tilt each way, scaled back up: the sensitivity, not the spread of two folded rays.
  for (std::size_t i = 0; i < knots.size(); ++i) {
    const double mid = 0.5 * (fast[i] + slow[i]), half = 0.5 * (fast[i] - slow[i]);
    fast[i] = mid + kTiltFraction * half;
    slow[i] = mid - kTiltFraction * half;
  }
  if (arcs) {   // the chain's one ray calculation (raytrace.hpp)
    return 0.5 * std::abs(ray_depth_arcs(receiver_depth, elevation, slant_range, knots, fast, water_depth) -
                          ray_depth_arcs(receiver_depth, elevation, slant_range, knots, slow, water_depth)) / kTiltFraction;
  }
  return 0.5 * std::abs(ray_depth(receiver_depth, elevation, slant_range, knots, fast, water_depth) -
                        ray_depth(receiver_depth, elevation, slant_range, knots, slow, water_depth)) / kTiltFraction;
}

void profile_ensemble(const ProfileOptions& o, const LadderProfile& ladder, double own, double sound, double gradient, double gradient_sigma,
                      double water, std::vector<double>& knots, std::vector<std::vector<double>>& speeds) {
  knots.clear();
  speeds.clear();
  const bool have_water = std::isfinite(water) && water > 0.0;
  // The rung's own profile, and the depths it has sampled (everything, for a supplied or an assumed one; nothing below).
  std::vector<double> d = ladder.depth, c = ladder.speed, s = ladder.sigma;
  double lo = 0.0, hi = 1e300;
  if (d.empty()) {
    const double bottom = std::max({have_water ? water : 0.0, own, 0.0}) + 10000.0;
    d = {0.0, bottom};
    c = {sound - gradient * own, sound + gradient * (bottom - own)};
    lo = hi = own;
  } else if (std::isfinite(ladder.span_lo) && std::isfinite(ladder.span_hi)) {
    lo = ladder.span_lo;
    hi = ladder.span_hi;
  } else if (ladder.rung.compare(0, 6, "family") == 0) {
    lo = hi = own;                                   // an assumed shape is not a sample: only the vehicle's own sound speed is
  }
  if (s.size() != d.size()) s.assign(d.size(), kNoClaim);
  auto interp = [&](const std::vector<double>& y, double z) {
    if (z <= d.front()) return y.front();
    if (z >= d.back()) return y.back();
    const auto k = static_cast<std::size_t>(std::upper_bound(d.begin(), d.end(), z) - d.begin());
    return y[k - 1] + (z - d[k - 1]) / (d[k] - d[k - 1]) * (y[k] - y[k - 1]);
  };
  std::vector<double> all = d;
  for (double z : {own, lo, hi}) {
    if (z > d.front() && z < d.back()) all.push_back(z);
  }
  for (double extent : o.ensemble_extents) {
    for (double z : {lo - extent, hi + extent}) {
      if (extent > 0.0 && z > d.front() && z < d.back()) all.push_back(z);
    }
  }
  std::sort(all.begin(), all.end());
  all.erase(std::unique(all.begin(), all.end(), [](double a, double b) { return std::abs(a - b) < 1e-6; }), all.end());
  knots = all;
  auto member = [&](double tilt, double g_above, double g_below, double extent) {
    std::vector<double> out(knots.size());
    for (std::size_t i = 0; i < knots.size(); ++i) {
      const double z = knots[i];
      // what the rung claims of its own profile, as one tilt about the vehicle, never more than the prior allows
      const double zc = std::clamp(z, lo, hi);
      const double e = (zc < own ? -1.0 : 1.0) * std::min(interp(s, zc), std::abs(gradient_sigma) * std::abs(zc - own));
      double v = interp(c, z) + tilt * e;
      if (z < lo) v -= (g_above - gradient) * std::min(lo - z, extent);       // dc/dz = g_above there
      if (z > hi) v += (g_below - gradient) * std::min(z - hi, extent);
      out[i] = v;
    }
    return out;
  };
  speeds.push_back(member(0.0, gradient, gradient, 0.0));
  const bool outside = lo > d.front() + 1e-6 || hi < d.back() - 1e-6;
  for (double extent : o.ensemble_extents) {
    if (!outside || !(extent > 0.0)) continue;
    for (double ga : o.ensemble_gradients) {
      for (double gb : o.ensemble_gradients) {
        if (ga == gradient && gb == gradient) continue;
        if (!(lo > d.front() + 1e-6) && ga != o.ensemble_gradients.front()) continue;   // nothing above: one pass
        if (!(hi < d.back() - 1e-6) && gb != o.ensemble_gradients.front()) continue;
        speeds.push_back(member(0.0, ga, gb, extent));
      }
    }
  }
  if (o.ensemble_sigma_members && lo < hi) {
    speeds.push_back(member(1.0, gradient, gradient, 0.0));
    speeds.push_back(member(-1.0, gradient, gradient, 0.0));
  }
}

}  // namespace sonar::live
