// Depth from multipath timing (multipath_depth.hpp). Mirrors python/gpusonar/multipath_depth.py
// statement by statement: a test holds the two to each other.
#include "sonar/live/multipath_depth.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>

#include "sonar/live/chain.hpp"
#include "sonar/live/elevation.hpp"
#include "sonar/live/raytrace.hpp"

namespace sonar::live {
namespace {

using cd = std::complex<double>;
constexpr double kPi = 3.14159265358979323846;
constexpr double kAnchor = 0.45;       // an arrival this strong re the pattern's strongest may be the echo's strongest sample
constexpr double kAnchorCells = 1.5;   // ... and is tried within this many 1 / B of it

double sinc(double x) {
  const double a = kPi * x;
  return std::abs(a) < 1e-12 ? 1.0 : std::sin(a) / a;
}

cd rayleigh(double grazing, double c_water, double speed, double density, double attenuation) {
  return rayleigh_reflection(grazing, c_water, speed, density, attenuation);   // raytrace.hpp: one implementation
}

struct Image {
  double z;
  int ns, nb;
};

// The images of a point at depth z in a layer 0..depth (depth <= 0: surface only), as lloyd._images.
std::vector<Image> images(double z, double depth, std::int64_t order) {
  std::vector<Image> out{{z, 0, 0}};
  for (int first = 0; first < 2; ++first) {
    if (first == 1 && !(depth > 0.0)) continue;
    double img = z;
    int ns = 0, nb = 0;
    bool surface_next = first == 0;
    for (std::int64_t k = 0; k < order; ++k) {
      if (surface_next) {
        img = -img;
        ++ns;
      } else {
        img = 2.0 * depth - img;
        ++nb;
      }
      surface_next = !surface_next;
      out.push_back({img, ns, nb});
      if (!(depth > 0.0)) break;
    }
  }
  return out;
}

// Path lengths and amplitudes re the direct path of the images joining depths z_own and z_target.
void one_way(double z_own, double z_target, double r_h, const MultipathGeometry& g, const MultipathOptions& o, std::vector<double>& length,
             std::vector<cd>& amp) {
  const double depth = (o.bottom && g.water_depth > 0.0) ? g.water_depth : 0.0;
  length.clear();
  amp.clear();
  const double r_d = std::sqrt(r_h * r_h + (z_own - z_target) * (z_own - z_target));
  const double k = 2.0 * kPi * g.f_centre / g.sound_speed;
  for (const Image& im : images(z_target, depth, depth > 0.0 ? o.bottom_order : 1)) {
    const double r_j = std::sqrt(r_h * r_h + (z_own - im.z) * (z_own - im.z));
    cd a = r_d / r_j;
    if (im.ns > 0 || im.nb > 0) {
      const double sin_g = std::clamp(std::abs(z_own - im.z) / r_j, 0.0, 1.0);
      if (im.ns > 0) {
        const double gamma = 2.0 * k * o.surface_rms_height * sin_g;
        const double loss_db = o.surface_loss_grazing.empty() ? o.surface_loss_db : multipath_surface_loss_db(o, std::asin(sin_g));
        const double mu = o.surface_coefficient * std::pow(10.0, -loss_db / 20.0) * std::exp(-0.5 * gamma * gamma);
        a *= std::pow(mu, im.ns);
      }
      if (im.nb > 0) a *= std::pow(rayleigh(std::asin(sin_g), g.sound_speed, o.bottom_speed, o.bottom_density, o.bottom_attenuation), im.nb);
    }
    length.push_back(r_j);
    amp.push_back(a);
  }
}

double pattern_energy(const std::vector<double>& delay, const std::vector<cd>& amp, double fc, double band, std::vector<cd>& w) {
  const std::size_t n = delay.size();
  w.resize(n);
  double out = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    w[j] = amp[j] * std::polar(1.0, -2.0 * kPi * fc * delay[j]);
    out += std::norm(w[j]);
    for (std::size_t i = 0; i < j; ++i) out += 2.0 * (w[j] * std::conj(w[i])).real() * sinc(band * (delay[j] - delay[i]));
  }
  return out;
}

double slack_s(const MultipathGeometry& g, const MultipathOptions& o) {
  return 2.0 * (g.slack > 0.0 ? g.slack : o.range_slack) / g.sound_speed + 2.0 / g.bandwidth;
}

// Depth change that moves the surface-surface echo by one resolution cell, m.
double envelope_lobe(double z, const MultipathGeometry& g) {
  const double d = std::max(0.01 * z, 0.05);
  const double slope = (multipath_surface_delay(z + d, g) - multipath_surface_delay(z - d, g)) / (2.0 * d);
  if (!std::isfinite(slope)) return std::max(0.02 * z, 0.1);   // rays: the edge of where the paths exist
  return 1.0 / (g.bandwidth * std::max(slope, 1e-12));
}

double model_sigma(double z, const MultipathGeometry& g, const MultipathOptions& o) {
  const double own = 0.5 * (g.array_depth + g.projector_depth);
  return z * std::hypot(o.own_depth_sigma / std::max(own, 1e-6), o.model_sigma_fraction);
}

// (index of the best, sigma, rivals) of a log-likelihood on a grid.
std::size_t posterior(const std::vector<double>& grid, const std::vector<double>& gain, double drop, double lobe, double& sigma,
                      std::vector<double>& rivals) {
  std::size_t b = 0;
  for (std::size_t k = 1; k < gain.size(); ++k) {
    if (gain[k] > gain[b]) b = k;
  }
  double sp = 0.0, sz = 0.0;
  for (std::size_t k = 0; k < gain.size(); ++k) {
    const double p = std::exp(gain[k] - gain[b]);
    sp += p;
    sz += p * (grid[k] - grid[b]) * (grid[k] - grid[b]);
  }
  sigma = std::sqrt(sz / sp);
  rivals.clear();
  for (std::size_t k = 1; k + 1 < gain.size(); ++k) {
    if (std::abs(grid[k] - grid[b]) > lobe && gain[k] >= gain[b] - drop && gain[k] >= gain[k - 1] && gain[k] > gain[k + 1]) rivals.push_back(grid[k]);
  }
  return b;
}

void num(std::ostringstream& s, double v) {
  if (std::isfinite(v)) s << v; else s << "null";
}

}  // namespace

bool set_multipath_option(MultipathOptions& o, const std::string& key, const std::string& value) {
  const std::string prefix = "multipath_depth";
  if (key.compare(0, prefix.size(), prefix) != 0) return false;
  const double v = std::atof(value.c_str());
  const auto i = static_cast<std::int64_t>(v);
  if (key == prefix) {
    o.enabled = v != 0.0;
    return true;
  }
  if (key.size() < prefix.size() + 2 || key[prefix.size()] != '_') return false;
  const std::string k = key.substr(prefix.size() + 1);
  if (k == "max_depth") o.max_depth = v;
  else if (k == "min_delay_bandwidth") o.min_delay_bandwidth = v;
  else if (k == "window_s") o.window_s = v;
  else if (k == "range_slack") o.range_slack = v;
  else if (k == "oversample") o.oversample = i;
  else if (k == "fringe_points") o.fringe_points = i;
  else if (k == "noise_cells") o.noise_cells = i;
  else if (k == "min_gain") o.min_gain = v;
  else if (k == "ambiguity_drop") o.ambiguity_drop = v;
  else if (k == "max_detections") o.max_detections = i;
  else if (k == "min_snr_db") o.min_snr_db = v;
  else if (k == "surface_coefficient") o.surface_coefficient = v;
  else if (k == "surface_rms_height") o.surface_rms_height = v;
  else if (k == "surface_loss_db") o.surface_loss_db = v;
  else if (k == "surface_min_amplitude") o.surface_min_amplitude = v;
  else if (k == "bottom") o.bottom = v != 0.0;
  else if (k == "bottom_speed") o.bottom_speed = v;
  else if (k == "bottom_density") o.bottom_density = v;
  else if (k == "bottom_attenuation") o.bottom_attenuation = v;
  else if (k == "bottom_order") o.bottom_order = i;
  else if (k == "bottom_grazing_limit_deg") o.bottom_grazing_limit_deg = v;
  else if (k == "water_depth") o.water_depth = v;
  else if (k == "surface_z") o.surface_z = v;
  else if (k == "own_depth_sigma") o.own_depth_sigma = v;
  else if (k == "model_sigma_fraction") o.model_sigma_fraction = v;
  else if (k == "gradient") o.gradient = v;
  else if (k == "gradient_sigma") o.gradient_sigma = v;
  else if (k == "track_gate") o.track_gate = v;
  else if (k == "track_step") o.track_step = v;
  else if (k == "track_min_gain") o.track_min_gain = v;
  else if (k == "track_memory_pings") o.track_memory_pings = v;
  else if (k == "consistency_sigma") o.consistency_sigma = v;
  else if (k == "rays") o.rays = v != 0.0;
  else if (k == "withhold_flagged") o.withhold_flagged = v != 0.0;
  else if (k == "ray_fan") o.ray_fan = i;
  else if (k == "ray_surface_order") o.ray_surface_order = i;
  else if (k == "ray_min_amplitude") o.ray_min_amplitude = v;
  else if (k == "ray_sigma_fan_fraction") o.ray_sigma_fan_fraction = v;
  else if (k == "surface_loss_grazing" || k == "surface_loss_table_db") {
    std::vector<double> list;                          // comma-separated (brackets and blanks allowed)
    std::string text = value;
    for (char& ch : text) {
      if (ch == '[' || ch == ']' || ch == ',') ch = ' ';
    }
    std::istringstream in(text);
    for (double x; in >> x;) list.push_back(x);
    (k == "surface_loss_grazing" ? o.surface_loss_grazing : o.surface_loss_table_db) = list;
  }
  else return false;
  return true;
}

void multipath_refraction(double flat_depth, const MultipathGeometry& g, const MultipathOptions& o, double& depth, double& sigma) {
  const double own = 0.5 * (g.array_depth + g.projector_depth);
  const double bend = g.range * g.range / (2.0 * g.sound_speed);          // r^2 / 2c: metres per unit gradient
  // z (1 + K / (z_own + z)) = z_flat, K = bend x gradient
  const double k = bend * o.gradient, b = own + k - flat_depth;
  depth = k == 0.0 ? flat_depth : 0.5 * (-b + std::sqrt(b * b + 4.0 * flat_depth * own));
  sigma = depth * bend * std::abs(o.gradient_sigma) / (own + depth);
}

void multipath_pattern(double z, double t0, const MultipathGeometry& g, const MultipathOptions& o, std::vector<double>& delay,
                       std::vector<cd>& amp) {
  if (g.rays) {
    // No direct or no surface path at this depth: one arrival, which no data fits better than L_1.
    if (!multipath_ray_pattern(*g.rays, z, t0, delay, amp)) {
      delay.assign(1, 0.0);
      amp.assign(1, cd(1.0, 0.0));
    }
    return;
  }
  const double z_mid = 0.5 * (g.array_depth + g.projector_depth);
  const double half = 0.5 * g.sound_speed * t0;
  const double r_h = std::sqrt(std::max(half * half - (z - z_mid) * (z - z_mid), 1.0));
  std::vector<double> l_tx, l_rx;
  std::vector<cd> a_tx, a_rx;
  one_way(g.projector_depth, z, r_h, g, o, l_tx, a_tx);
  one_way(g.array_depth, z, r_h, g, o, l_rx, a_rx);
  delay.clear();
  amp.clear();
  for (std::size_t j = 0; j < l_tx.size(); ++j) {
    for (std::size_t k = 0; k < l_rx.size(); ++k) {
      delay.push_back(((l_tx[j] - l_tx[0]) + (l_rx[k] - l_rx[0])) / g.sound_speed);
      amp.push_back(a_tx[j] * a_rx[k]);
    }
  }
}

double multipath_surface_delay(double z, const MultipathGeometry& g) {
  if (g.rays) return multipath_ray_surface_delay(*g.rays, z);
  const double z_mid = 0.5 * (g.array_depth + g.projector_depth);
  const double r2 = std::max(g.range * g.range - (z - z_mid) * (z - z_mid), 1.0);
  double sum = 0.0;
  for (double zo : {g.projector_depth, g.array_depth}) sum += std::sqrt(r2 + (z + zo) * (z + zo)) - std::sqrt(r2 + (z - zo) * (z - zo));
  return sum / g.sound_speed;
}

std::vector<double> multipath_depth_grid(const MultipathGeometry& g, const MultipathOptions& o) {
  double z_hi = o.max_depth;
  if (g.water_depth > 0.0) z_hi = std::min(z_hi, g.water_depth);
  z_hi = std::min(z_hi, 0.5 * (g.array_depth + g.projector_depth) + 0.95 * g.range);
  if (g.rays) {
    // Through the rays: only depths with a direct and a surface path on both legs have a delay; the grid runs from the
    // shallowest to the deepest of those that resolve, and the pattern says "one arrival" at the gaps in between.
    const int n = 1025;
    const double h = z_hi / n;
    std::vector<double> tau(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) tau[static_cast<std::size_t>(i)] = multipath_surface_delay(h * (i + 0.5), g);
    int first = -1, last = -1;
    double slope = 0.0;
    for (int i = 0; i < n; ++i) {
      const auto u = static_cast<std::size_t>(i);
      if (!(tau[u] * g.bandwidth >= o.min_delay_bandwidth && tau[u] <= o.window_s)) continue;
      if (first < 0) first = i;
      last = i;
      if (i > 0 && std::isfinite(tau[u - 1])) slope = std::max(slope, (tau[u] - tau[u - 1]) / h);
      if (i + 1 < n && std::isfinite(tau[u + 1])) slope = std::max(slope, (tau[u + 1] - tau[u]) / h);
    }
    if (first < 0 || !(slope > 0.0)) return {};
    const double z_lo = h * first, z_top = h * (last + 1);
    const std::int64_t per = std::max<std::int64_t>(o.fringe_points, 1);
    const double step = 1.0 / (g.f_centre * slope) / static_cast<double>(per);
    const std::int64_t count = std::max<std::int64_t>(static_cast<std::int64_t>(std::ceil((z_top - z_lo) / step / static_cast<double>(per))) * per, per);
    std::vector<double> out(static_cast<std::size_t>(count));
    for (std::int64_t i = 0; i < count; ++i) out[static_cast<std::size_t>(i)] = z_lo + step * (static_cast<double>(i) + 0.5);
    return out;
  }
  const int n_probe = 4001;
  std::vector<double> probe(n_probe), tau(n_probe);
  for (int i = 0; i < n_probe; ++i) {
    probe[static_cast<std::size_t>(i)] = z_hi * static_cast<double>(i) / (n_probe - 1);
    tau[static_cast<std::size_t>(i)] = multipath_surface_delay(probe[static_cast<std::size_t>(i)], g);
  }
  const double h = z_hi / (n_probe - 1);
  int first = -1, last = -1;
  double slope = 0.0;
  for (int i = 0; i < n_probe; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (!(tau[u] * g.bandwidth >= o.min_delay_bandwidth && tau[u] <= o.window_s)) continue;
    if (first < 0) first = i;
    last = i;
    const double d = i == 0 ? (tau[1] - tau[0]) / h : i == n_probe - 1 ? (tau[u] - tau[u - 1]) / h : (tau[u + 1] - tau[u - 1]) / (2.0 * h);
    slope = std::max(slope, d);
  }
  if (first < 0 || !(slope > 0.0)) return {};
  const double z_lo = probe[static_cast<std::size_t>(first)], z_top = probe[static_cast<std::size_t>(last)];
  const double fringe = 1.0 / (g.f_centre * slope);
  const std::int64_t n = std::max<std::int64_t>(o.fringe_points, 1);
  const double step = fringe / static_cast<double>(n);
  const std::int64_t count = std::max<std::int64_t>(static_cast<std::int64_t>(std::ceil((z_top - z_lo) / step / static_cast<double>(n))) * n, n);
  std::vector<double> out(static_cast<std::size_t>(count));
  for (std::int64_t i = 0; i < count; ++i) out[static_cast<std::size_t>(i)] = z_lo + step * (static_cast<double>(i) + 0.5);
  return out;
}

void multipath_window_times(const MultipathGeometry& g, const MultipathOptions& o, double spread, double& t_first, double& dt, std::int64_t& n,
                            std::int64_t& n_noise) {
  dt = 1.0 / (static_cast<double>(o.oversample) * g.bandwidth);
  const double guard = 4.0 / g.bandwidth;
  n_noise = o.noise_cells * o.oversample;
  const auto n_side = static_cast<std::int64_t>(std::ceil((spread + slack_s(g, o) + guard) / dt));
  t_first = -dt * static_cast<double>(n_noise + n_side);
  n = n_noise + 2 * n_side + 1;
}

std::vector<cd> multipath_window(const PingRecord& record, double bearing, double range, double t_first, double dt, std::int64_t n,
                                 const std::vector<std::vector<std::int64_t>>& rows, Backend backend, int device) {
  const double c = record.environment.sound_speed;
  std::vector<double> ranges(static_cast<std::size_t>(n));
  for (std::int64_t k = 0; k < n; ++k) ranges[static_cast<std::size_t>(k)] = range + 0.5 * c * (t_first + dt * static_cast<double>(k));
  std::vector<std::complex<float>> z;
  if (rows.size() < 2) {
    z = beam_samples(record, ranges, bearing, false, backend, device);
  } else {
    z.assign(ranges.size(), {0.0F, 0.0F});
    for (const auto& r : rows) {
      const std::vector<std::complex<float>> one = beam_samples(row_record(record, r), ranges, bearing, false, backend, device);
      for (std::size_t k = 0; k < z.size(); ++k) z[k] += one[k];
    }
  }
  const Waveform& w = record.transmit.waveform;
  const Samples& s = record.samples;
  const double fc = s.is_complex() && s.center_frequency > 0.0 ? s.center_frequency : 0.5 * (w.f_start + w.f_stop);
  std::vector<cd> y(ranges.size());
  for (std::int64_t k = 0; k < n; ++k) {
    const auto u = static_cast<std::size_t>(k);
    y[u] = cd(z[u].real(), z[u].imag()) * std::polar(1.0, -2.0 * kPi * fc * (t_first + dt * static_cast<double>(k)));
  }
  return y;
}

MultipathEstimate multipath_estimate(const cd* y, std::int64_t n, double t_first, double dt, std::int64_t n_noise, const MultipathGeometry& g,
                                     const MultipathOptions& o) {
  MultipathEstimate est;
  if (std::abs(o.surface_coefficient) * std::pow(10.0, -o.surface_loss_db / 20.0) < o.surface_min_amplitude) {
    est.reason = "no_surface_path";
    return est;
  }
  const std::vector<double> depths = multipath_depth_grid(g, o);
  if (g.rays) est.profile_source = multipath_ray_profile(*g.rays).rung;
  if (depths.empty()) {
    est.reason = "merged";
    if (g.rays) {
      // Which path the water does not hold, anywhere a target could be at this range.
      int seen = 0;
      const double top = multipath_ray_z_hi(*g.rays);
      for (int i = 0; i < 257; ++i) seen |= multipath_ray_paths(*g.rays, top * (i + 0.5) / 257.0);
      if (!(seen & 1)) est.reason = "no_direct_path";
      else if (!(seen & 2)) est.reason = "no_surface_path";
    }
    return est;
  }
  if (g.rays) {
    est.searched_lo = depths.front();
    est.searched_hi = depths.back();
  }
  bool any = false;
  for (std::int64_t k = 0; k < std::min(n, n_noise) && !any; ++k) any = std::abs(y[k]) > 0.0;
  if (n < n_noise + 8 || !any) {
    est.reason = "window";
    return est;
  }
  // Noise power: median |y|^2 / ln 2 over the leading samples.
  std::vector<double> p(static_cast<std::size_t>(n_noise));
  for (std::int64_t k = 0; k < n_noise; ++k) p[static_cast<std::size_t>(k)] = std::norm(y[k]);
  std::sort(p.begin(), p.end());
  const double median = n_noise % 2 ? p[static_cast<std::size_t>(n_noise / 2)] : 0.5 * (p[static_cast<std::size_t>(n_noise / 2 - 1)] + p[static_cast<std::size_t>(n_noise / 2)]);
  const double sigma2 = median / std::log(2.0);
  // The echo's strongest sample within the range slack of the detection: L_1, and the anchor of the search over t0.
  const double slack = slack_s(g, o);
  std::int64_t k_pk = -1;
  for (std::int64_t k = 0; k < n; ++k) {
    if (std::abs(t_first + dt * static_cast<double>(k)) > slack + 0.5 * dt) continue;
    if (k_pk < 0 || std::norm(y[k]) > std::norm(y[k_pk])) k_pk = k;
  }
  if (k_pk < 0) {
    est.reason = "window";
    return est;
  }
  const double t_pk = t_first + dt * static_cast<double>(k_pk);
  const double l1 = std::norm(y[k_pk]) / sigma2;
  est.snr_db = 10.0 * std::log10(std::max(l1, 1e-30));

  const int step = 2;
  const double t_det = 2.0 * g.range / g.sound_speed;
  const auto half = static_cast<std::int64_t>(std::ceil(kAnchorCells / g.bandwidth / (dt * step)));
  std::vector<double> big(depths.size());
  std::vector<double> delay;
  std::vector<cd> amp, w;
  std::vector<std::int64_t> ms;
  for (std::size_t i = 0; i < depths.size(); ++i) {
    // Only the t0 that put one of the pattern's strong arrivals on the strongest sample are tried.
    multipath_pattern(depths[i], t_det + t_pk, g, o, delay, amp);
    double top = 0.0;
    for (const cd& a : amp) top = std::max(top, std::abs(a));
    ms.clear();
    const std::size_t n_arr = amp.size();
    std::vector<char> strong(n_arr);
    std::vector<double> first(n_arr);
    for (std::size_t k = 0; k < n_arr; ++k) {
      strong[k] = std::abs(amp[k]) >= kAnchor * top;
      first[k] = delay[k];
    }
    for (std::size_t k = 0; k < n_arr; ++k) {
      if (!strong[k]) continue;
      double t0k = t_pk - first[k];
      for (int it = 0; it < 3; ++it) {       // the delay depends (weakly) on t0 itself
        multipath_pattern(depths[i], t_det + t0k, g, o, delay, amp);
        t0k = t_pk - delay[k];
      }
      const auto m0 = static_cast<std::int64_t>(std::nearbyint(t0k / (dt * step)));
      for (std::int64_t m = m0 - half; m <= m0 + half; ++m) ms.push_back(m);
    }
    std::sort(ms.begin(), ms.end());
    ms.erase(std::unique(ms.begin(), ms.end()), ms.end());
    double best = 0.0;
    for (std::int64_t m : ms) {
      const double t0 = dt * step * static_cast<double>(m);
      multipath_pattern(depths[i], t_det + t0, g, o, delay, amp);
      cd rake = 0.0;
      for (std::size_t k = 0; k < delay.size(); ++k) {
        const double u = (t0 + delay[k] - t_first) / dt;
        const auto iu = static_cast<std::int64_t>(std::floor(u));
        if (iu < 0 || iu >= n - 1) continue;
        const double f = u - static_cast<double>(iu);
        rake += std::conj(amp[k]) * std::polar(1.0, 2.0 * kPi * g.f_centre * delay[k]) * (y[iu] * (1.0 - f) + y[iu + 1] * f);
      }
      const double t = std::norm(rake) / std::max(pattern_energy(delay, amp, g.f_centre, g.bandwidth, w), 1e-12);
      best = std::max(best, t);
    }
    big[i] = best / sigma2 - l1;
  }
  // The fringe envelope: the largest value in each run of fringe_points depths.
  const auto per = static_cast<std::size_t>(std::max<std::int64_t>(o.fringe_points, 1));
  const std::size_t cells = depths.size() / per;
  std::vector<double> at(cells);
  est.grid.resize(cells);
  est.envelope.resize(cells);
  for (std::size_t q = 0; q < cells; ++q) {
    std::size_t k = q * per;
    double mean = 0.0;
    for (std::size_t j = q * per; j < (q + 1) * per; ++j) {
      mean += depths[j];
      if (big[j] > big[k]) k = j;
    }
    est.grid[q] = mean / static_cast<double>(per);
    est.envelope[q] = big[k];
    at[q] = depths[k];
  }
  std::size_t b0 = 0;
  for (std::size_t q = 1; q < cells; ++q) {
    if (est.envelope[q] > est.envelope[b0]) b0 = q;
  }
  const double lobe = envelope_lobe(est.grid[b0], g);
  double sigma = 0.0;
  std::vector<double> rivals;
  const std::size_t b = posterior(est.grid, est.envelope, o.ambiguity_drop, lobe, sigma, rivals);
  est.gain = est.envelope[b];
  multipath_pattern(at[b], t_det, g, o, delay, amp);
  double strongest = 0.0;
  for (std::size_t k = 0; k < amp.size(); ++k) {
    if (std::abs(amp[k]) >= 0.1) ++est.arrivals;
    if (k > 0) strongest = std::max(strongest, std::abs(amp[k]));
  }
  if (est.gain < o.min_gain) {
    est.reason = "single_arrival";
    return est;
  }
  if (amp.size() < 2 || std::abs(amp[1]) < o.surface_min_amplitude) {   // the one-leg surface echo (roughness at its grazing angle)
    est.reason = "no_surface_path";
    return est;
  }
  const double cell = cells > 1 ? est.grid[1] - est.grid[0] : 0.0;
  est.valid = true;
  est.method = multipath_surface_delay(at[b], g) * g.bandwidth >= 4.0 ? "delay" : "ripple";
  const double flat_depth = at[b];
  if (g.rays) {
    // The pattern was the profile's own: the depth stands, and what the rung does not know goes through the rays.
    est.depth = flat_depth;
    est.refraction_sigma = multipath_ray_sigma(*g.rays, flat_depth);
    est.ensemble = multipath_ray_ensemble(*g.rays, flat_depth);
  } else {
    multipath_refraction(flat_depth, g, o, est.depth, est.refraction_sigma);
  }
  const double msig = model_sigma(est.depth, g, o);
  est.sigma = std::sqrt(sigma * sigma + cell * cell / 12.0 + msig * msig + est.refraction_sigma * est.refraction_sigma);
  est.ambiguous = !rivals.empty();
  if (rivals.size() > 4) rivals.resize(4);
  est.alternatives = rivals;
  if (g.rays) {
    // Depths the direct path does not reach at this range: a target there is heard by other paths only, which the
    // pattern of a depth that is reached may fit. The depth is then one explanation among others, and the report says so.
    const double top = multipath_ray_z_hi(*g.rays);
    for (int i = 0; i < 257; ++i) {
      const double z = top * (i + 0.5) / 257.0;
      if (multipath_ray_paths(*g.rays, z) & 1) continue;
      if (!std::isfinite(est.shadow_lo)) est.shadow_lo = z;
      est.shadow_hi = z;
    }
    if (std::isfinite(est.shadow_lo)) {
      est.ambiguous = true;
      est.reason = "shadow";
    }
  }
  // Paths the pattern leaves out but the water may hold: the depth is then one explanation among others.
  if (g.water_depth > 0.0 && !o.bottom) {
    const double z_mid = 0.5 * (g.array_depth + g.projector_depth);
    const double grazing = std::atan2(2.0 * g.water_depth - z_mid - flat_depth, g.range) * 180.0 / kPi;
    if (grazing < o.bottom_grazing_limit_deg) {
      est.ambiguous = true;
      est.reason = "bottom_unmodelled";
    }
  } else if (g.water_depth > 0.0 && o.bottom && !g.rays) {
    MultipathOptions more = o;
    more.bottom_order = o.bottom_order + 1;
    std::vector<double> d2;
    std::vector<cd> a2;
    multipath_pattern(flat_depth, t_det, g, more, d2, a2);
    const auto legs = static_cast<std::size_t>(1 + 2 * more.bottom_order), had = static_cast<std::size_t>(1 + 2 * o.bottom_order);
    double left_out = 0.0;      // the strongest two-way arrival with a leg of order + 1 (legs come as d, s.., b..)
    for (std::size_t j = 0; j < legs; ++j) {
      for (std::size_t k = 0; k < legs; ++k) {
        const bool fresh = j == had / 2 + 1 || j == legs - 1 || k == had / 2 + 1 || k == legs - 1;
        if (fresh && d2[j * legs + k] <= o.window_s) left_out = std::max(left_out, std::abs(a2[j * legs + k]));
      }
    }
    if (left_out >= o.surface_min_amplitude) {
      est.ambiguous = true;
      est.reason = "bottom_order";
    }
  }
  if (o.withhold_flagged && !est.reason.empty()) {
    est.candidate = est.depth;
    est.valid = false;
    est.depth = est.sigma = kNaN;
    est.method = "none";
    est.grid.clear();
    est.envelope.clear();
  }
  return est;
}

MultipathGeometry multipath_geometry(const PingRecord& record, const Vec3d& origin, double range, const MultipathOptions& o, double first_range,
                                     double extent) {
  const Waveform& w = record.transmit.waveform;
  const double f_lo = std::min(w.f_start, w.f_stop), f_hi = std::max(w.f_start, w.f_stop);
  MultipathGeometry g;
  g.range = range;
  if (std::isfinite(first_range) && std::isfinite(extent) && extent >= 0.0) {
    g.range = first_range + 0.5 * extent;
    g.slack = o.range_slack + 0.5 * extent;
  }
  g.array_depth = origin.z - o.surface_z;
  g.projector_depth = world_position(record.geometry.tx_pose, record.geometry.tx.lever_arm).z - o.surface_z;
  g.sound_speed = record.environment.sound_speed;
  g.f_centre = 0.5 * (f_lo + f_hi);
  g.bandwidth = f_hi > f_lo ? f_hi - f_lo : 1.0 / w.duration;
  const double water = o.water_depth > 0.0 ? o.water_depth : record.environment.water_depth;
  g.water_depth = std::isfinite(water) && water > 0.0 ? water : 0.0;
  return g;
}

MultipathEstimate multipath_detection(const PingRecord& record, const Vec3d& origin, double bearing, double range, const MultipathOptions& o,
                                      double first_range, double extent, const std::vector<std::vector<std::int64_t>>& rows, Backend backend,
                                      int device, const MultipathProfile* profile, MultipathGeometry* used) {
  MultipathEstimate est;
  MultipathGeometry g = multipath_geometry(record, origin, range, o, first_range, extent);
  est.reason = "merged";
  if (!(g.array_depth > 0.0) || !(g.projector_depth > 0.0)) return est;
  if (o.rays) {
    MultipathProfile water;
    if (profile != nullptr && !profile->depth.empty()) {
      water = *profile;
    } else {                                           // nothing known: isovelocity at the vehicle's sound speed, the prior as its error
      water.depth = {0.0, 1.0};
      water.speed = {g.sound_speed, g.sound_speed};
      water.gradient_sigma = profile != nullptr && profile->gradient_sigma > 0.0 ? profile->gradient_sigma : o.gradient_sigma;
      water.rung = profile != nullptr && !profile->rung.empty() ? profile->rung : "prior";
    }
    g.rays = multipath_rays(water, g, o);
  }
  if (used != nullptr) *used = g;
  const std::vector<double> depths = multipath_depth_grid(g, o);
  if (depths.empty()) return g.rays ? multipath_estimate(nullptr, 0, 0.0, 0.0, 0, g, o) : est;   // rays: says which path is missing
  double t_first = 0.0, dt = 0.0;
  std::int64_t n = 0, n_noise = 0;
  multipath_window_times(g, o, multipath_surface_delay(depths.back(), g), t_first, dt, n, n_noise);
  const Samples& s = record.samples;
  const double t_det = 2.0 * g.range / g.sound_speed;
  est.reason = "window";
  if (t_det + t_first < s.start_time ||
      t_det + t_first + dt * static_cast<double>(n - 1) > s.start_time + static_cast<double>(s.n_samples - 1) / s.sample_rate - record.transmit.waveform.duration) {
    return est;
  }
  const std::vector<cd> y = multipath_window(record, bearing, g.range, t_first, dt, n, rows, backend, device);
  return multipath_estimate(y.data(), n, t_first, dt, n_noise, g, o);
}

MultipathFused multipath_fuse(double depth_mp, double sigma_mp, double depth_el, double sigma_el, const std::vector<double>& alternatives,
                              double consistency_sigma, double refraction_mp, double refraction_el) {
  MultipathFused out;
  double d = depth_el;
  for (std::size_t k = 0; k < alternatives.size(); ++k) {
    if (std::abs(alternatives[k] - depth_mp) < std::abs(d - depth_mp)) {
      d = alternatives[k];
      out.selected = static_cast<std::int64_t>(k) + 1;
    }
  }
  out.consistent = std::abs(d - depth_mp) <= consistency_sigma * std::hypot(sigma_mp, sigma_el);
  if (!out.consistent) {
    out.depth = depth_mp;
    out.sigma = sigma_mp;
    return out;
  }
  // Only what differs between the two is averaged down; the gradient's part is one error in both.
  const double floor = 1e-6;
  const double s1 = std::sqrt(std::max(sigma_mp * sigma_mp - refraction_mp * refraction_mp, floor * floor));
  const double s2 = std::sqrt(std::max(sigma_el * sigma_el - refraction_el * refraction_el, floor * floor));
  const double w1 = 1.0 / (s1 * s1), w2 = 1.0 / (s2 * s2);
  out.depth = (w1 * depth_mp + w2 * d) / (w1 + w2);
  const double common = (w1 * refraction_mp + w2 * refraction_el) / (w1 + w2);
  out.sigma = std::sqrt(1.0 / (w1 + w2) + common * common);
  return out;
}

void MultipathDepth::add(std::int64_t track, const MultipathEstimate& est, const MultipathGeometry& g) {
  if (est.grid.size() < 2) return;
  const double step = opt_.track_step;
  const auto n = static_cast<std::size_t>(std::ceil(opt_.max_depth / step));
  Sum& sum = sums_[track];
  if (sum.total.size() != n) sum.total.assign(n, 0.0);
  const double fade = opt_.track_memory_pings > 0.0 ? std::exp(-1.0 / opt_.track_memory_pings) : 1.0;
  const double cell = est.grid[1] - est.grid[0];
  std::vector<double> value(n, 0.0);
  if (cell > step) {
    // linear interpolation between the ping's cells; outside them: no better than one arrival
    std::size_t j = 0;
    for (std::size_t k = 0; k < n; ++k) {
      const double z = step * (static_cast<double>(k) + 0.5);
      if (z < est.grid.front() || z > est.grid.back()) continue;
      while (j + 2 < est.grid.size() && est.grid[j + 1] < z) ++j;
      const double f = (z - est.grid[j]) / (est.grid[j + 1] - est.grid[j]);
      value[k] = est.envelope[j] * (1.0 - f) + est.envelope[j + 1] * f;
    }
  } else {
    std::vector<double> low(n, -std::numeric_limits<double>::infinity());
    for (std::size_t q = 0; q < est.grid.size(); ++q) {
      const double k = std::floor(est.grid[q] / step);
      if (k < 0.0 || k >= static_cast<double>(n)) continue;
      low[static_cast<std::size_t>(k)] = std::max(low[static_cast<std::size_t>(k)], est.envelope[q]);
    }
    for (std::size_t k = 0; k < n; ++k) value[k] = std::isfinite(low[k]) ? low[k] : 0.0;
  }
  std::size_t b = 0;
  for (std::size_t k = 0; k < n; ++k) {
    sum.total[k] = fade * sum.total[k] + value[k];
    if (sum.total[k] > sum.total[b]) b = k;
  }
  ++sum.pings;
  sum.cell = std::max(sum.cell, cell);
  sum.last_ping = pings_;
  const double z = step * (static_cast<double>(b) + 0.5);
  sum.lobe = envelope_lobe(z, g);
  sum.geometry = g;
}

MultipathEstimate MultipathDepth::track(std::int64_t id) const {
  MultipathEstimate est;
  est.reason = "single_arrival";
  const auto it = sums_.find(id);
  if (it == sums_.end() || it->second.pings == 0) return est;
  const Sum& sum = it->second;
  est.arrivals = sum.pings;
  std::vector<double> grid(sum.total.size());
  for (std::size_t k = 0; k < grid.size(); ++k) grid[k] = opt_.track_step * (static_cast<double>(k) + 0.5);
  double sigma = 0.0;
  std::vector<double> rivals;
  const std::size_t b = posterior(grid, sum.total, opt_.ambiguity_drop, sum.lobe, sigma, rivals);
  est.gain = sum.total[b];
  if (est.gain < opt_.track_min_gain) return est;
  est.reason.clear();
  est.valid = true;
  if (sum.geometry.rays) {
    est.depth = grid[b];
    est.refraction_sigma = multipath_ray_sigma(*sum.geometry.rays, grid[b]);
    est.ensemble = multipath_ray_ensemble(*sum.geometry.rays, grid[b]);
    est.profile_source = multipath_ray_profile(*sum.geometry.rays).rung;
  } else {
    multipath_refraction(grid[b], sum.geometry, opt_, est.depth, est.refraction_sigma);
  }
  const double msig = model_sigma(est.depth, sum.geometry, opt_);
  // half a cell of the track's grid, and the ping's own fringe cell (the envelope has no finer structure); the gradient's
  // part does not average down over pings
  est.sigma = std::sqrt(sigma * sigma + opt_.track_step * opt_.track_step / 4.0 + sum.cell * sum.cell / 12.0 + msig * msig +
                        est.refraction_sigma * est.refraction_sigma);
  est.ambiguous = !rivals.empty();
  if (rivals.size() > 4) rivals.resize(4);
  est.alternatives = rivals;
  est.method = "track";
  return est;
}

MultipathReport MultipathDepth::process(const PingRecord& record, const Vec3d& origin, const std::vector<MultipathHit>& hits,
                                        const std::vector<MultipathTrack>& tracks, const std::vector<std::vector<std::int64_t>>& rows,
                                        Backend backend, int device) {
  const auto t_start = std::chrono::steady_clock::now();
  MultipathReport out;
  out.detections.assign(hits.size(), "");
  out.tracks.assign(tracks.size(), "");
  ++pings_;
  // The strongest detections, at most max_detections of them.
  std::vector<std::size_t> order;
  for (std::size_t i = 0; i < hits.size(); ++i) {
    if (hits[i].snr_db >= opt_.min_snr_db) order.push_back(i);
  }
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return hits[a].snr_db > hits[b].snr_db; });
  if (static_cast<std::int64_t>(order.size()) > opt_.max_detections) order.resize(static_cast<std::size_t>(std::max<std::int64_t>(opt_.max_detections, 0)));
  std::vector<MultipathEstimate> ests(hits.size());
  std::vector<char> done(hits.size(), 0);
  std::vector<MultipathGeometry> used(hits.size());
  auto fused = [&](std::ostringstream& s, const MultipathEstimate& e, bool has, double depth, double depth_sigma, double depth_refraction,
                   const std::vector<double>& alts, bool select) {
    if (!e.valid || !has || !(depth_sigma > 0.0)) return;
    const MultipathFused f = multipath_fuse(e.depth, e.sigma, depth, depth_sigma, alts, opt_.consistency_sigma, e.refraction_sigma, depth_refraction);
    s << ", \"depth_fused_m\": " << f.depth << ", \"depth_fused_sigma_m\": " << f.sigma << ", \"depth_consistent\": " << (f.consistent ? "true" : "false");
    if (select && !alts.empty()) s << ", \"elevation_selected\": " << f.selected;
  };
  auto members = [&](std::ostringstream& s, const MultipathEstimate& e, bool is_track) {
    s << "\"depth_multipath_m\": ";
    num(s, e.valid ? e.depth : kNaN);
    if (e.valid) {
      s << ", \"depth_multipath_sigma_m\": " << e.sigma << ", \"depth_multipath_refraction_sigma_m\": " << e.refraction_sigma << (is_track ? ", \"multipath_pings\": " : ", \"multipath_arrivals\": ") << e.arrivals
        << ", \"multipath_method\": \"" << e.method << "\"";
      if (e.ambiguous) {
        s << ", \"multipath_ambiguous\": true, \"multipath_alternatives_m\": [";
        for (std::size_t k = 0; k < e.alternatives.size(); ++k) s << (k ? ", " : "") << e.alternatives[k];
        s << "]";
      }
      if (!e.reason.empty()) s << ", \"multipath_note\": \"" << e.reason << "\"";
      if (!e.ensemble.empty()) {
        // The same delay through every member of the profile ensemble: where the depth may lie.
        double lo = e.depth, hi = e.depth;
        bool all = true;
        for (double z : e.ensemble) {
          all = all && std::isfinite(z);
          if (std::isfinite(z)) {
            lo = std::min(lo, z);
            hi = std::max(hi, z);
          }
        }
        s << ", \"depth_multipath_ensemble_m\": [" << lo << ", " << hi << "]";
        if (!all) s << ", \"depth_multipath_ensemble_complete\": false";   // a member has no such path
      }
    } else {
      s << ", \"multipath_reason\": \"" << e.reason << "\"";
      if (std::isfinite(e.candidate)) s << ", \"multipath_candidate_m\": " << e.candidate;
      if (is_track) s << ", \"multipath_pings\": " << e.arrivals;
    }
    if (!e.profile_source.empty()) s << ", \"multipath_profile_source\": \"" << e.profile_source << "\"";
    if (std::isfinite(e.searched_lo)) s << ", \"multipath_searched_m\": [" << e.searched_lo << ", " << e.searched_hi << "]";
    if (std::isfinite(e.shadow_lo)) s << ", \"multipath_shadow_m\": [" << e.shadow_lo << ", " << e.shadow_hi << "]";
    s << ", \"multipath_gain\": ";
    num(s, e.gain);
  };
  for (std::size_t i : order) {
    const MultipathHit& h = hits[i];
    ests[i] = multipath_detection(record, origin, h.bearing, h.range, opt_, h.first_range, h.extent, rows, backend, device, &profile_, &used[i]);
    done[i] = 1;
    std::ostringstream s;
    s << std::setprecision(10);
    members(s, ests[i], false);
    fused(s, ests[i], h.has_depth, h.depth, h.depth_sigma, h.depth_refraction_sigma, h.depth_alternatives, true);
    out.detections[i] = s.str();
  }
  // Each track takes the nearest examined detection inside its gate; its envelope goes into the track's sum.
  for (std::size_t q = 0; q < tracks.size(); ++q) {
    const MultipathTrack& t = tracks[q];
    double best = opt_.track_gate;
    std::int64_t pick = -1;
    for (std::size_t i = 0; i < hits.size(); ++i) {
      if (!done[i]) continue;
      const double d = std::hypot(hits[i].x - t.x, hits[i].y - t.y);
      if (d <= best) {
        best = d;
        pick = static_cast<std::int64_t>(i);
      }
    }
    if (pick >= 0) {
      const auto u = static_cast<std::size_t>(pick);
      add(t.id, ests[u], used[u]);
    }
    const auto it = sums_.find(t.id);
    if (it != sums_.end()) it->second.last_ping = pings_;
    const MultipathEstimate e = track(t.id);
    std::ostringstream s;
    s << std::setprecision(10);
    members(s, e, true);
    fused(s, e, t.has_depth, t.depth, t.depth_sigma, 0.0, {}, false);
    out.tracks[q] = s.str();
  }
  // Sums of tracks that have not been reported for a while are dropped.
  for (auto it = sums_.begin(); it != sums_.end();) {
    if (pings_ - it->second.last_ping > 20) it = sums_.erase(it); else ++it;
  }
  out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  return out;
}

}  // namespace sonar::live
