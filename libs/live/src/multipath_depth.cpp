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

#include "sonar/recon/backproject.hpp"

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

// Plane-wave pressure reflection coefficient of a fluid half-space (Rayleigh), as sim/propagation.py:
// R = (m sin g - sqrt(n^2 - cos^2 g)) / (m sin g + sqrt(n^2 - cos^2 g)), m = rho_b / rho_w, n = c_w / c_b,
// attenuation (dB per wavelength) as an imaginary part of c_b.
cd rayleigh(double grazing, double c_water, double speed, double density, double attenuation) {
  const double delta = attenuation / (40.0 * kPi * std::log10(std::exp(1.0)));
  const cd c_b = speed / cd(1.0, delta);
  const cd n = c_water / c_b;
  cd root = std::sqrt(n * n - std::cos(grazing) * std::cos(grazing));
  if (root.imag() < 0.0) root = -root;
  const double ms = density * std::sin(grazing);
  return (ms - root) / (ms + root);
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
        const double mu = o.surface_coefficient * std::pow(10.0, -o.surface_loss_db / 20.0) * std::exp(-0.5 * gamma * gamma);
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

double slack_s(const MultipathGeometry& g, const MultipathOptions& o) { return 2.0 * o.range_slack / g.sound_speed + 2.0 / g.bandwidth; }

// Depth change that moves the surface-surface echo by one resolution cell, m.
double envelope_lobe(double z, const MultipathGeometry& g) {
  const double d = std::max(0.01 * z, 0.05);
  const double slope = (multipath_surface_delay(z + d, g) - multipath_surface_delay(z - d, g)) / (2.0 * d);
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
  else if (k == "water_depth") o.water_depth = v;
  else if (k == "surface_z") o.surface_z = v;
  else if (k == "own_depth_sigma") o.own_depth_sigma = v;
  else if (k == "model_sigma_fraction") o.model_sigma_fraction = v;
  else if (k == "track_gate") o.track_gate = v;
  else if (k == "track_step") o.track_step = v;
  else if (k == "track_min_gain") o.track_min_gain = v;
  else if (k == "track_memory_pings") o.track_memory_pings = v;
  else if (k == "consistency_sigma") o.consistency_sigma = v;
  else return false;
  return true;
}

void multipath_pattern(double z, double t0, const MultipathGeometry& g, const MultipathOptions& o, std::vector<double>& delay,
                       std::vector<cd>& amp) {
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

std::vector<cd> multipath_window(const PingRecord& record, const Vec3d& origin, const Vec3d& direction, double range, double t_first, double dt,
                                 std::int64_t n, Backend backend, int device) {
  const Vec3d tx = world_position(record.geometry.tx_pose, record.geometry.tx.lever_arm) - origin;
  const double c = record.environment.sound_speed;
  std::vector<Vec3d> points(static_cast<std::size_t>(n));
  for (std::int64_t k = 0; k < n; ++k) {
    const double path = 2.0 * range + c * (t_first + dt * static_cast<double>(k));
    const double rho = (path * path - dot(tx, tx)) / (2.0 * (path - dot(direction, tx)));
    points[static_cast<std::size_t>(k)] = origin + direction * rho;
  }
  recon::BpOptions bp;
  bp.interp = recon::Interp::Sinc;
  bp.device = device;
  const std::vector<const PingRecord*> pings{&record};
  const recon::BpInputs inputs = recon::make_inputs(pings);
  std::vector<std::complex<float>> z(points.size(), {0.0F, 0.0F});
  recon::backproject(backend, inputs, points, bp, z.data());
  const Waveform& w = record.transmit.waveform;
  const Samples& s = record.samples;
  const double fc = s.is_complex() && s.center_frequency > 0.0 ? s.center_frequency : 0.5 * (w.f_start + w.f_stop);
  std::vector<cd> y(points.size());
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
  if (depths.empty()) {
    est.reason = "merged";
    return est;
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
  if (strongest < o.surface_min_amplitude) {
    est.reason = "no_surface_path";
    return est;
  }
  const double cell = cells > 1 ? est.grid[1] - est.grid[0] : 0.0;
  est.valid = true;
  est.depth = at[b];
  const double msig = model_sigma(est.depth, g, o);
  est.sigma = std::sqrt(sigma * sigma + cell * cell / 12.0 + msig * msig);
  est.method = multipath_surface_delay(est.depth, g) * g.bandwidth >= 2.0 ? "delay" : "ripple";
  est.ambiguous = !rivals.empty();
  if (rivals.size() > 4) rivals.resize(4);
  est.alternatives = rivals;
  return est;
}

MultipathGeometry multipath_geometry(const PingRecord& record, const Vec3d& origin, double range, const MultipathOptions& o) {
  const Waveform& w = record.transmit.waveform;
  const double f_lo = std::min(w.f_start, w.f_stop), f_hi = std::max(w.f_start, w.f_stop);
  MultipathGeometry g;
  g.range = range;
  g.array_depth = origin.z - o.surface_z;
  g.projector_depth = world_position(record.geometry.tx_pose, record.geometry.tx.lever_arm).z - o.surface_z;
  g.sound_speed = record.environment.sound_speed;
  g.f_centre = 0.5 * (f_lo + f_hi);
  g.bandwidth = f_hi > f_lo ? f_hi - f_lo : 1.0 / w.duration;
  const double water = o.water_depth > 0.0 ? o.water_depth : record.environment.water_depth;
  g.water_depth = std::isfinite(water) && water > 0.0 ? water : 0.0;
  return g;
}

MultipathEstimate multipath_detection(const PingRecord& record, const Vec3d& origin, const Vec3d& direction, double range,
                                      const MultipathOptions& o, Backend backend, int device) {
  MultipathEstimate est;
  const MultipathGeometry g = multipath_geometry(record, origin, range, o);
  est.reason = "merged";
  if (!(g.array_depth > 0.0) || !(g.projector_depth > 0.0)) return est;
  const std::vector<double> depths = multipath_depth_grid(g, o);
  if (depths.empty()) return est;
  double t_first = 0.0, dt = 0.0;
  std::int64_t n = 0, n_noise = 0;
  multipath_window_times(g, o, multipath_surface_delay(depths.back(), g), t_first, dt, n, n_noise);
  const Samples& s = record.samples;
  const double t_det = 2.0 * range / g.sound_speed;
  est.reason = "window";
  if (t_det + t_first < s.start_time ||
      t_det + t_first + dt * static_cast<double>(n - 1) > s.start_time + static_cast<double>(s.n_samples - 1) / s.sample_rate - record.transmit.waveform.duration) {
    return est;
  }
  const std::vector<cd> y = multipath_window(record, origin, direction, range, t_first, dt, n, backend, device);
  return multipath_estimate(y.data(), n, t_first, dt, n_noise, g, o);
}

MultipathFused multipath_fuse(double depth_mp, double sigma_mp, double depth_el, double sigma_el, const std::vector<double>& alternatives,
                              double consistency_sigma) {
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
  const double w1 = 1.0 / (sigma_mp * sigma_mp), w2 = 1.0 / (sigma_el * sigma_el);
  out.depth = (w1 * depth_mp + w2 * d) / (w1 + w2);
  out.sigma = 1.0 / std::sqrt(w1 + w2);
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
  sum.last_ping = pings_;
  const double z = step * (static_cast<double>(b) + 0.5);
  sum.lobe = envelope_lobe(z, g);
  sum.model = model_sigma(z, g, opt_);
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
  est.depth = grid[b];
  est.sigma = std::sqrt(sigma * sigma + opt_.track_step * opt_.track_step / 12.0 + sum.model * sum.model);
  est.ambiguous = !rivals.empty();
  if (rivals.size() > 4) rivals.resize(4);
  est.alternatives = rivals;
  est.method = "track";
  return est;
}

MultipathReport MultipathDepth::process(const PingRecord& record, const Vec3d& origin, const std::vector<MultipathHit>& hits,
                                        const std::vector<MultipathTrack>& tracks, Backend backend, int device) {
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
  auto fused = [&](std::ostringstream& s, const MultipathEstimate& e, bool has, double depth, double depth_sigma, const std::vector<double>& alts,
                   bool select) {
    if (!e.valid || !has || !(depth_sigma > 0.0)) return;
    const MultipathFused f = multipath_fuse(e.depth, e.sigma, depth, depth_sigma, alts, opt_.consistency_sigma);
    s << ", \"depth_fused_m\": " << f.depth << ", \"depth_fused_sigma_m\": " << f.sigma << ", \"depth_consistent\": " << (f.consistent ? "true" : "false");
    if (select && !alts.empty()) s << ", \"elevation_selected\": " << f.selected;
  };
  auto members = [&](std::ostringstream& s, const MultipathEstimate& e, bool is_track) {
    s << "\"depth_multipath_m\": ";
    num(s, e.valid ? e.depth : kNaN);
    if (e.valid) {
      s << ", \"depth_multipath_sigma_m\": " << e.sigma << (is_track ? ", \"multipath_pings\": " : ", \"multipath_arrivals\": ") << e.arrivals
        << ", \"multipath_method\": \"" << e.method << "\"";
      if (e.ambiguous) {
        s << ", \"multipath_ambiguous\": true, \"multipath_alternatives_m\": [";
        for (std::size_t k = 0; k < e.alternatives.size(); ++k) s << (k ? ", " : "") << e.alternatives[k];
        s << "]";
      }
    } else {
      s << ", \"multipath_reason\": \"" << e.reason << "\"";
      if (is_track) s << ", \"multipath_pings\": " << e.arrivals;
    }
    s << ", \"multipath_gain\": ";
    num(s, e.gain);
  };
  for (std::size_t i : order) {
    const MultipathHit& h = hits[i];
    ests[i] = multipath_detection(record, origin, h.direction, h.range, opt_, backend, device);
    done[i] = 1;
    std::ostringstream s;
    s << std::setprecision(10);
    members(s, ests[i], false);
    fused(s, ests[i], h.has_depth, h.depth, h.depth_sigma, h.depth_alternatives, true);
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
      add(t.id, ests[u], multipath_geometry(record, origin, hits[u].range, opt_));
    }
    const auto it = sums_.find(t.id);
    if (it != sums_.end()) it->second.last_ping = pings_;
    const MultipathEstimate e = track(t.id);
    std::ostringstream s;
    s << std::setprecision(10);
    members(s, e, true);
    fused(s, e, t.has_depth, t.depth, t.depth_sigma, {}, false);
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
