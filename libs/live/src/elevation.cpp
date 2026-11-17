#include "sonar/live/elevation.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>

#include "sonar/core/parallel.hpp"
#include "sonar/dsp/fft.hpp"
#include "sonar/recon/backproject.hpp"

namespace sonar::live {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 == 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Unit vector along the rows, body frame: the median step between consecutive channels.
Vec3d row_axis(const PingRecord& record) {
  const auto& rx = record.geometry.rx;
  std::vector<double> dx, dy, dz;
  for (std::size_t i = 1; i < rx.size(); ++i) {
    const Vec3d d = rx[i].lever_arm - rx[i - 1].lever_arm;
    const double n = norm(d);
    if (!(n > 0.0)) continue;
    dx.push_back(d.x / n);
    dy.push_back(d.y / n);
    dz.push_back(d.z / n);
  }
  if (dx.empty()) throw std::invalid_argument("row_axis: needs at least two elements at different positions");
  return normalized(Vec3d{median(dx), median(dy), median(dz)});
}

std::vector<double> elevation_grid(double first_deg, double last_deg, double step_deg) {
  // The same count as numpy.arange(first, last + step / 2, step).
  const auto n = static_cast<std::int64_t>(std::ceil((last_deg + 0.5 * step_deg - first_deg) / step_deg));
  std::vector<double> g(static_cast<std::size_t>(std::max<std::int64_t>(n, 1)));
  for (std::size_t i = 0; i < g.size(); ++i) g[i] = (first_deg + static_cast<double>(i) * step_deg) * kDeg;
  return g;
}

// Sub-grid position of a peak: a parabola through the three points about the maximum.
double parabola(double left, double mid, double right) {
  const double den = left - 2.0 * mid + right;
  return den < 0.0 ? 0.5 * (left - right) / den : 0.0;
}

}  // namespace

std::vector<std::vector<std::int64_t>> find_rows(const PingRecord& record, double tolerance, std::int64_t min_elements) {
  const auto& rx = record.geometry.rx;
  std::vector<std::vector<std::int64_t>> rows;
  if (rx.size() < 2) {
    rows.emplace_back();
    for (std::size_t i = 0; i < rx.size(); ++i) rows[0].push_back(static_cast<std::int64_t>(i));
    return rows;
  }
  const Vec3d a = row_axis(record);
  std::vector<Vec3d> anchors;
  for (std::size_t i = 0; i < rx.size(); ++i) {
    const Vec3d q = rx[i].lever_arm - a * dot(rx[i].lever_arm, a);
    bool placed = false;
    for (std::size_t k = 0; k < anchors.size() && !placed; ++k) {
      if (norm(q - anchors[k]) <= tolerance) {
        rows[k].push_back(static_cast<std::int64_t>(i));
        placed = true;
      }
    }
    if (!placed) {
      rows.push_back({static_cast<std::int64_t>(i)});
      anchors.push_back(q);
    }
  }
  std::vector<std::vector<std::int64_t>> kept;
  for (auto& r : rows) {
    if (static_cast<std::int64_t>(r.size()) >= min_elements) kept.push_back(std::move(r));
  }
  return kept;
}

PingRecord row_record(const PingRecord& record, const std::vector<std::int64_t>& rows) {
  PingRecord out;
  out.transmit = record.transmit;
  out.geometry = record.geometry;
  out.navigation = record.navigation;
  out.calibration = record.calibration;
  out.environment = record.environment;
  out.metadata = record.metadata;
  const Samples& old = record.samples;
  Samples& s = out.samples;
  s.kind = old.kind;
  s.n_samples = old.n_samples;
  s.sample_rate = old.sample_rate;
  s.center_frequency = old.center_frequency;
  s.start_time = old.start_time;
  s.storage = old.storage;
  s.storage_scale = old.storage_scale;
  s.n_channels = static_cast<std::int64_t>(rows.size());
  const std::size_t per = static_cast<std::size_t>(old.n_samples) * old.stride();
  s.data.reserve(per * rows.size());
  for (std::int64_t r : rows) s.data.insert(s.data.end(), old.channel(r), old.channel(r) + per);
  const auto n_old = static_cast<std::size_t>(old.n_channels);
  auto keep = [&](auto& v, const auto& from) {
    if (from.size() != n_old) return;
    v.clear();
    for (std::int64_t r : rows) v.push_back(from[static_cast<std::size_t>(r)]);
  };
  keep(out.geometry.rx, record.geometry.rx);
  keep(out.calibration.gain, record.calibration.gain);
  keep(out.calibration.phase, record.calibration.phase);
  keep(out.calibration.delay, record.calibration.delay);
  keep(out.calibration.position_offset, record.calibration.position_offset);
  keep(out.metadata.channel_element, record.metadata.channel_element);
  keep(out.metadata.channel_board, record.metadata.channel_board);
  for (Board& b : out.metadata.boards) b.reference_channel = -1;   // row numbers no longer apply
  return out;
}

ElevationFrame elevation_frame(const PingRecord& record, const std::vector<std::vector<std::int64_t>>& rows, bool starboard, const Vec3d& rpy) {
  ElevationFrame f;
  const Mat3d rot = rotation_ned(rpy.x, rpy.y, rpy.z);
  std::vector<Vec3d> pos;
  for (const auto& row : rows) {
    for (std::int64_t i : row) {
      Vec3d lever = record.geometry.rx[static_cast<std::size_t>(i)].lever_arm;
      if (record.calibration.position_offset.size() == record.geometry.rx.size()) lever += record.calibration.position_offset[static_cast<std::size_t>(i)];
      pos.push_back(record.geometry.rx_pose.position + rot * lever);
    }
  }
  f.origin = Vec3d{0.0, 0.0, 0.0};
  for (const Vec3d& q : pos) f.origin += q / static_cast<double>(pos.size());
  f.along = rot * row_axis(record);
  const Vec3d up = rot * Vec3d{0.0, 0.0, -1.0};
  f.up = normalized(up - f.along * dot(f.along, up));
  f.look = cross(f.along, f.up) * (starboard ? 1.0 : -1.0);
  for (const Vec3d& q : pos) f.heights.push_back(dot(q - f.origin, f.up));
  return f;
}

ElevationFrame elevation_frame(const PingRecord& record, const std::vector<std::vector<std::int64_t>>& rows, bool starboard) {
  return elevation_frame(record, rows, starboard, record.geometry.rx_pose.rpy);
}

bool attitude_at(const PingRecord& record, double t, Vec3d& rpy, double& gap) {
  const auto& ins = record.navigation.ins;
  if (ins.empty()) return false;
  std::size_t i = 0;
  while (i < ins.size() && ins[i].t < t) ++i;            // numpy.searchsorted(times, t)
  if (i == 0) {
    rpy = ins.front().rpy;
    gap = ins.front().t - t;
  } else if (i == ins.size()) {
    rpy = ins.back().rpy;
    gap = t - ins.back().t;
  } else {
    const double w = (t - ins[i - 1].t) / std::max(ins[i].t - ins[i - 1].t, 1e-12);
    auto turn = [&](double a, double b) { return a + w * std::atan2(std::sin(b - a), std::cos(b - a)); };   // the short way round
    rpy = Vec3d{turn(ins[i - 1].rpy.x, ins[i].rpy.x), turn(ins[i - 1].rpy.y, ins[i].rpy.y), turn(ins[i - 1].rpy.z, ins[i].rpy.z)};
    gap = 0.0;
  }
  return true;
}

double refraction_sigma(double slant_range, double sound_speed, double gradient_sigma) {
  return slant_range * slant_range * std::abs(gradient_sigma) / (2.0 * sound_speed);
}

Vec3d arrival_direction(const ElevationFrame& frame, double sin_cone, double sin_elevation) {
  const double out = std::sqrt(std::max(1.0 - sin_cone * sin_cone - sin_elevation * sin_elevation, 0.0));
  return frame.along * sin_cone + frame.up * sin_elevation + frame.look * out;
}

double rms_wavenumber(double f_low, double f_high, double sound_speed) {
  const double fc = 0.5 * (f_low + f_high), band = std::abs(f_high - f_low);
  return 2.0 * kPi / sound_speed * std::sqrt(fc * fc + band * band / 12.0);
}

double weight_power(std::int64_t n_elements, bool hann) {
  if (!hann || n_elements < 2) return static_cast<double>(n_elements);
  double sum = 0.0;
  for (std::int64_t i = 0; i < n_elements; ++i) {
    const double x = (static_cast<double>(i) - 0.5 * static_cast<double>(n_elements - 1)) / static_cast<double>(n_elements);
    const double c = std::cos(kPi * x);
    sum += c * c * c * c;
  }
  return sum;
}

std::vector<ElevationEstimate> estimate_elevation(const PingRecord& record, const std::vector<std::vector<std::int64_t>>& rows,
                                                  const std::vector<ElevationHit>& hits,
                                                  const std::vector<double>& backgrounds, const ElevationFrame& frame,
                                                  const ElevationOptions& o, double noise_weight, double range_cell,
                                                  Backend backend, int device) {
  std::vector<ElevationEstimate> out(hits.size());
  if (hits.empty()) return out;
  const double c = record.environment.sound_speed;
  const Waveform& w = record.transmit.waveform;
  const double f_low = std::min(w.f_start, w.f_stop), f_high = std::max(w.f_start, w.f_stop);
  const double band = f_high > f_low ? f_high - f_low : 1.0 / w.duration;
  const std::vector<double> grid = elevation_grid(o.first_deg, o.last_deg, o.step_deg);
  const auto n_e = static_cast<std::int64_t>(grid.size());
  const double half = o.range_window_m > 0.0 ? o.range_window_m : 1.5 * range_cell;
  const double dr = 0.25 * c / band;                      // half a range resolution cell
  const auto n_hi = static_cast<std::int64_t>(std::ceil(half / dr));
  const auto n_k = static_cast<std::int64_t>(rows.size());
  const auto n_h = static_cast<std::int64_t>(hits.size());
  double n_total = 0.0, sum_sq = 0.0;
  for (const auto& r : rows) {
    n_total += static_cast<double>(r.size());
    sum_sq += static_cast<double>(r.size()) * static_cast<double>(r.size());
  }
  const double coherent_gain = n_total * n_total / sum_sq;   // |sum of rows|^2 over sum of |row|^2, for one plane wave
  double mean_h = 0.0, spread = 0.0;
  for (double h : frame.heights) mean_h += h / static_cast<double>(frame.heights.size());
  for (double h : frame.heights) spread += (h - mean_h) * (h - mean_h);
  const double k_rms = rms_wavenumber(f_low, f_high, c);
  const Vec3d tx_world = world_position(record.geometry.tx_pose, record.geometry.tx.lever_arm);
  const bool starboard = dot(cross(frame.along, frame.up), frame.look) > 0.0;
  const Pose& pose = record.geometry.rx_pose;
  const bool turning = pose.rates.x != 0.0 || pose.rates.y != 0.0 || pose.rates.z != 0.0;
  const double water = o.water_depth > 0.0 ? o.water_depth : record.environment.water_depth;
  // The array as the record's pose has it when each echo arrives (the kernel turns it with the pose's
  // rates), and as the attitude history says it then was: the angles are found in the first frame
  // and turned into a world direction with the second.
  std::vector<ElevationFrame> kf(hits.size(), frame), wf(hits.size(), frame);
  std::vector<double> extra(hits.size(), 0.0);
  for (std::size_t i = 0; i < hits.size(); ++i) {
    const double t_arr = 2.0 * hits[i].range / c;
    if (turning) kf[i] = elevation_frame(record, rows, starboard, pose.rpy + pose.rates * t_arr);
    wf[i] = kf[i];
    if (o.attitude_arrival) {
      Vec3d rpy;
      double gap = 0.0;
      // The compressed echo is the whole pulse, received from t_arr to t_arr + its length: its middle.
      const double t_att = t_arr + 0.5 * w.duration;
      if (attitude_at(record, t_att, rpy, gap)) {
        wf[i] = elevation_frame(record, rows, starboard, rpy);
        out[i].attitude_gap = gap;
      } else {
        out[i].attitude_gap = t_att;                       // nothing but the pose at the transmit
      }
      extra[i] = o.attitude_rate_sigma * out[i].attitude_gap;
    }
  }

  // Points of constant two-way path (projector -> point -> array centre) along elevation: the
  // projector is not at the array's centre, and at constant range from the array the path
  // would change with elevation by far more than the rows' own differences.
  // The range searched: about the detection, and back to the leading edge of its cluster (the
  // direct arrival is the first), but no further than eight half-windows.
  std::vector<std::int64_t> n_lo(hits.size()), first_point(hits.size() + 1, 0);
  for (std::size_t i = 0; i < hits.size(); ++i) {
    double lead = half;
    if (std::isfinite(hits[i].first_range)) lead = std::min(std::max(hits[i].range - hits[i].first_range + range_cell, half), 8.0 * half);
    n_lo[i] = static_cast<std::int64_t>(std::ceil(lead / dr));
    first_point[i + 1] = first_point[i] + n_e * (n_lo[i] + n_hi + 1);
  }
  std::vector<Vec3d> points(static_cast<std::size_t>(first_point.back()));
  std::vector<double> rho(points.size());
  for (std::int64_t i = 0; i < n_h; ++i) {
    const double sb = std::sin(hits[static_cast<std::size_t>(i)].bearing);
    const std::int64_t n_half = n_lo[static_cast<std::size_t>(i)], n_r = n_half + n_hi + 1;
    for (std::int64_t e = 0; e < n_e; ++e) {
      const ElevationFrame& kfi = kf[static_cast<std::size_t>(i)];
      const Vec3d tx = tx_world - kfi.origin;
      const Vec3d u = arrival_direction(kfi, sb, std::sin(grid[static_cast<std::size_t>(e)]));
      for (std::int64_t j = 0; j < n_r; ++j) {
        const double path = 2.0 * (hits[static_cast<std::size_t>(i)].range + dr * static_cast<double>(j - n_half));
        const double r = (path * path - dot(tx, tx)) / (2.0 * (path - dot(u, tx)));
        const auto at = static_cast<std::size_t>(first_point[static_cast<std::size_t>(i)] + e * n_r + j);
        rho[at] = r;
        points[at] = kfi.origin + u * r;
      }
    }
  }
  // Each row's coherent sum at every point (one row's samples in memory at a time).
  std::vector<std::vector<std::complex<float>>> z(static_cast<std::size_t>(n_k));
  recon::BpOptions bp;
  bp.interp = recon::Interp::Sinc;   // linear interpolation ripples each row by a per cent or two with elevation: 0.1 degree of bias
  bp.device = device;
  for (std::int64_t k = 0; k < n_k; ++k) {
    const PingRecord sub = row_record(record, rows[static_cast<std::size_t>(k)]);
    const std::vector<const PingRecord*> pings{&sub};
    const recon::BpInputs inputs = recon::make_inputs(pings);
    z[static_cast<std::size_t>(k)].assign(points.size(), {0.0F, 0.0F});
    recon::backproject(backend, inputs, points, bp, z[static_cast<std::size_t>(k)].data());
  }
  std::vector<double> p;
  for (std::int64_t i = 0; i < n_h; ++i) {
    const auto base = static_cast<std::size_t>(first_point[static_cast<std::size_t>(i)]);
    const std::int64_t n_r = n_lo[static_cast<std::size_t>(i)] + n_hi + 1;
    p.assign(static_cast<std::size_t>(n_e * n_r), 0.0);
    for (std::size_t q = 0; q < p.size(); ++q) {
      std::complex<double> sum{0.0, 0.0};
      for (std::int64_t k = 0; k < n_k; ++k) sum += std::complex<double>(z[static_cast<std::size_t>(k)][base + q]);
      p[q] = std::norm(sum);
    }
    const ElevationFrame& wfi = wf[static_cast<std::size_t>(i)];
    const double sb = std::sin(hits[static_cast<std::size_t>(i)].bearing);
    if (o.sector_from_depth && std::isfinite(water) && water > 0.0) {
      // Straight rays with so many bounces cannot arrive from steeper than this: do not look there.
      const double own = wfi.origin.z - o.surface_z, rng = hits[static_cast<std::size_t>(i)].range;
      const double top = std::atan((own + o.sector_bounces * water) / rng) + o.sector_margin;
      const double bottom = -std::atan((water - own + o.sector_bounces * water) / rng) - o.sector_margin;
      std::vector<char> allowed(static_cast<std::size_t>(n_e), 0);
      std::int64_t n_allowed = 0;
      for (std::int64_t ee = 0; ee < n_e; ++ee) {
        const double level = std::asin(std::clamp(-arrival_direction(wfi, sb, std::sin(grid[static_cast<std::size_t>(ee)])).z, -1.0, 1.0));
        allowed[static_cast<std::size_t>(ee)] = level <= top && level >= bottom;
        n_allowed += allowed[static_cast<std::size_t>(ee)];
      }
      if (n_allowed >= 3) {
        for (std::int64_t ee = 0; ee < n_e; ++ee) {
          if (allowed[static_cast<std::size_t>(ee)]) continue;
          for (std::int64_t jj = 0; jj < n_r; ++jj) p[static_cast<std::size_t>(ee * n_r + jj)] = 0.0;
        }
      }
    }
    const double noise = backgrounds[static_cast<std::size_t>(i)] / noise_weight;   // at one hydrophone
    std::vector<double> over(static_cast<std::size_t>(n_r), 0.0);                    // best over elevation, at each range
    for (std::int64_t ee = 0; ee < n_e; ++ee) {
      for (std::int64_t jj = 0; jj < n_r; ++jj) over[static_cast<std::size_t>(jj)] = std::max(over[static_cast<std::size_t>(jj)], p[static_cast<std::size_t>(ee * n_r + jj)]);
    }
    auto j = static_cast<std::int64_t>(std::max_element(over.begin(), over.end()) - over.begin());
    if (o.first_arrival) {
      // Bounce paths are longer than the direct one: take the earliest arrival that stands clear
      // of the noise and is not far below the strongest.
      const double floor = std::max(over[static_cast<std::size_t>(j)] * std::pow(10.0, -o.first_arrival_db / 10.0),
                                    n_total * noise * std::pow(10.0, o.first_arrival_snr_db / 10.0));
      for (std::int64_t jj = 1; jj < j; ++jj) {
        const double v = over[static_cast<std::size_t>(jj)];
        if (v >= floor && v >= over[static_cast<std::size_t>(jj - 1)] && v > over[static_cast<std::size_t>(jj + 1)]) {
          j = jj;
          break;
        }
      }
    }
    auto col = [&](std::int64_t ee) { return p[static_cast<std::size_t>(ee * n_r + j)]; };
    std::int64_t e = 0;
    for (std::int64_t ee = 1; ee < n_e; ++ee) {
      if (col(ee) > col(e)) e = ee;
    }
    ElevationEstimate& el = out[static_cast<std::size_t>(i)];
    el.range = rho[base + static_cast<std::size_t>(e * n_r + j)];
    auto refined = [&](std::int64_t k) {             // sub-grid peak: a parabola through the three points about a maximum
      if (k == 0 || k == n_e - 1) return grid[static_cast<std::size_t>(k)];
      return grid[static_cast<std::size_t>(k)] + parabola(col(k - 1), col(k), col(k + 1)) * o.step_deg * kDeg;
    };
    const bool edge = e == 0 || e == n_e - 1 || col(std::max<std::int64_t>(e - 1, 0)) == 0.0 || col(std::min<std::int64_t>(e + 1, n_e - 1)) == 0.0;
    const double angle = refined(e);
    // The other lobes: maxima outside the main lobe (which runs down to its first minima).
    std::int64_t lo = e, hi = e;
    while (lo > 0 && col(lo - 1) < col(lo)) --lo;
    while (hi < n_e - 1 && col(hi + 1) < col(hi)) ++hi;
    double second = 0.0;
    for (std::int64_t ee = 0; ee < n_e; ++ee) {
      if (ee < lo || ee > hi) second = std::max(second, col(ee));
    }
    // Would noise swap the first and the second? Their difference in amplitude is (1 - g) of the peak,
    // g the second's amplitude over the first's, and its noise sqrt(1 - g) of the sum's.
    const double g = col(e) > 0.0 ? std::sqrt(second / col(e)) : 1.0;
    const double margin_sigma = noise > 0.0 ? std::sqrt(col(e) / (n_total * noise) * std::max(1.0 - g, 0.0))
                                            : std::numeric_limits<double>::infinity();
    el.ambiguous = edge || second > col(e) * std::pow(10.0, -o.ambiguity_db / 10.0) || margin_sigma < o.ambiguity_sigma;
    if (el.ambiguous) {
      const double floor_alt = col(e) * std::pow(10.0, -o.alternatives_db / 10.0);
      for (std::int64_t k = 1; k < n_e - 1; ++k) {
        if ((k < lo || k > hi) && col(k) >= floor_alt && col(k) >= col(k - 1) && col(k) > col(k + 1)) {
          el.alternatives.push_back(arrival_direction(wfi, sb, std::sin(refined(k))));
        }
      }
    }
    const std::int64_t j0 = std::max<std::int64_t>(j - 1, 0), j1 = std::min<std::int64_t>(j + 2, n_r);
    double coherent = 0.0, incoherent = 0.0;
    for (std::int64_t jj = j0; jj < j1; ++jj) {
      coherent += p[static_cast<std::size_t>(e * n_r + jj)];
      for (std::int64_t k = 0; k < n_k; ++k) {
        incoherent += std::norm(std::complex<double>(z[static_cast<std::size_t>(k)][base + static_cast<std::size_t>(e * n_r + jj)]));
      }
    }
    el.quality = incoherent > 0.0 ? coherent / (coherent_gain * incoherent) : 0.0;
    const double snr = noise > 0.0 ? col(e) / (n_total * n_total * noise) : std::numeric_limits<double>::infinity();
    el.snr_db = 10.0 * std::log10(std::max(snr, 1e-30));
    el.sin_elevation = std::sin(angle);
    const double sigma_sin = spread > 0.0 && snr > 0.0 ? 1.0 / (k_rms * std::sqrt(2.0 * snr * spread))
                                                        : std::numeric_limits<double>::infinity();
    const double from_noise = sigma_sin / std::max(std::cos(angle), 1e-6), from_time = extra[static_cast<std::size_t>(i)];
    el.sigma = std::sqrt(from_noise * from_noise + o.roll_sigma * o.roll_sigma + from_time * from_time);
    el.world = arrival_direction(wfi, sb, el.sin_elevation);
    el.valid = el.quality >= o.min_quality && std::isfinite(el.sigma);
  }
  return out;
}

std::pair<double, std::string> fold_depth(double z, double water_depth, double surface_z) {
  const double d = z - surface_z;
  if (!std::isfinite(d)) return {std::numeric_limits<double>::quiet_NaN(), "unknown"};
  if (!(std::isfinite(water_depth) && water_depth > 0.0)) return {surface_z + std::abs(d), d < 0.0 ? "surface" : "direct"};
  const double h = water_depth;
  double m = std::fmod(d, 2.0 * h);
  if (m < 0.0) m += 2.0 * h;
  const double folded = m <= h ? m : 2.0 * h - m;
  const char* path = (d >= 0.0 && d <= h) ? "direct" : (d >= -h && d < 0.0) ? "surface" : (d > h && d <= 2.0 * h) ? "bottom" : "multiple";
  return {surface_z + folded, path};
}

double ray_depth(double receiver_depth, double elevation, double slant_range, const std::vector<double>& zs,
                 const std::vector<double>& cs, double water_depth) {
  if (zs.empty() || zs.size() != cs.size()) throw std::invalid_argument("ray_depth: the profile needs depths and as many speeds");
  auto speed = [&](double z) {                      // numpy.interp: linear between knots, held outside them
    if (z <= zs.front()) return cs.front();
    if (z >= zs.back()) return cs.back();
    const auto hi = static_cast<std::size_t>(std::upper_bound(zs.begin(), zs.end(), z) - zs.begin());
    const double w = (z - zs[hi - 1]) / (zs[hi] - zs[hi - 1]);
    return cs[hi - 1] + w * (cs[hi] - cs[hi - 1]);
  };
  const double step = std::min(std::max(slant_range / 2000.0, 0.5), 10.0);
  const double c0 = speed(receiver_depth);
  const double snell = std::cos(elevation) / c0;
  double z = receiver_depth, remaining = slant_range / c0;
  bool down = elevation < 0.0;
  const bool bottom = std::isfinite(water_depth) && water_depth > 0.0;
  while (remaining > 0.0) {
    const double c = speed(z);
    const double sin_t = std::sqrt(std::max(1.0 - (snell * c) * (snell * c), 0.0));
    const double ds = std::min(step, remaining * c);
    double z_next = z + (down ? 1.0 : -1.0) * std::max(sin_t, 1e-4) * ds;
    if (z_next < 0.0) {
      z_next = -z_next;
      down = true;
    } else if (bottom && z_next > water_depth) {
      z_next = 2.0 * water_depth - z_next;
      down = false;
    } else if (snell * speed(z_next) >= 1.0 && snell * c < 1.0) {
      down = !down;                                 // a turning point: back towards slower water
      z_next = z;
    }
    z = z_next;
    remaining -= ds / c;
  }
  return z;
}

bool outside_water(double z, double sigma, double water_depth, double surface_z, double n_sigma, double water_depth_sigma) {
  if (!(std::isfinite(z) && std::isfinite(sigma))) return false;
  const double d = z - surface_z;
  if (d < -n_sigma * sigma) return true;
  return std::isfinite(water_depth) && water_depth > 0.0 && d > water_depth + n_sigma * std::hypot(sigma, water_depth_sigma);
}

PassiveElevation passive_elevation(const PingRecord& record, const std::vector<std::vector<std::int64_t>>& rows, double bearing,
                                   double f_low, double f_high, std::int64_t n_fft, std::int64_t hop, std::int64_t first,
                                   std::int64_t n_blocks, double first_deg, double last_deg, double step_deg, bool starboard) {
  using Cd = std::complex<double>;
  const Samples& s = record.samples;
  if (n_fft < 8 || hop < 1) throw std::invalid_argument("passive_elevation: bad block sizes");
  const std::int64_t available = (s.n_samples - first - n_fft) / hop + 1;
  if (n_blocks < 0 || n_blocks > available) n_blocks = available;
  if (n_blocks < 1) throw std::invalid_argument("passive_elevation: the record is shorter than one block");
  std::vector<std::int64_t> bins;
  std::vector<double> freqs;
  for (std::int64_t k = 0; k < n_fft; ++k) {
    double f;
    if (s.is_complex()) {
      f = s.center_frequency + static_cast<double>(k <= n_fft / 2 ? k : k - n_fft) * s.sample_rate / static_cast<double>(n_fft);
    } else {
      if (k > n_fft / 2) break;
      f = static_cast<double>(k) * s.sample_rate / static_cast<double>(n_fft);
    }
    if (f >= f_low && f <= f_high && f > 0.0) {
      bins.push_back(k);
      freqs.push_back(f);
    }
  }
  if (bins.empty()) throw std::invalid_argument("passive_elevation: no FFT bin inside the band asked for");
  const ElevationFrame frame = elevation_frame(record, rows, starboard);
  const double c = record.environment.sound_speed;
  const double sb = std::sin(bearing);
  const Vec3d u0 = arrival_direction(frame, sb, 0.0);
  const auto n_k = static_cast<std::int64_t>(rows.size());
  const auto n_f = static_cast<std::int64_t>(bins.size());
  // Every row steered at the bearing (level), per block and bin.
  std::vector<Cd> beams(static_cast<std::size_t>(n_k * n_blocks * n_f));
  std::vector<Vec3d> centres(static_cast<std::size_t>(n_k));
  std::vector<std::vector<Vec3d>> pos(static_cast<std::size_t>(n_k));
  for (std::int64_t k = 0; k < n_k; ++k) {
    Vec3d centre{0.0, 0.0, 0.0};
    for (std::int64_t i : rows[static_cast<std::size_t>(k)]) {
      Vec3d lever = record.geometry.rx[static_cast<std::size_t>(i)].lever_arm;
      if (static_cast<std::size_t>(i) < record.calibration.position_offset.size()) lever += record.calibration.position_offset[static_cast<std::size_t>(i)];
      pos[static_cast<std::size_t>(k)].push_back(world_position(record.geometry.rx_pose, lever) - frame.origin);
      centre += pos[static_cast<std::size_t>(k)].back() / static_cast<double>(rows[static_cast<std::size_t>(k)].size());
    }
    centres[static_cast<std::size_t>(k)] = centre;
  }
  parallel_for_host(n_k * n_blocks, [&](std::int64_t job) {
    const std::int64_t k = job / n_blocks, b = job % n_blocks;
    const auto& row = rows[static_cast<std::size_t>(k)];
    std::vector<std::complex<float>> work(static_cast<std::size_t>(n_fft));
    std::vector<Cd> sum(static_cast<std::size_t>(n_f), Cd{});
    for (std::size_t e = 0; e < row.size(); ++e) {
      const std::int64_t start = first + b * hop;
      for (std::int64_t q = 0; q < n_fft; ++q) {
        const auto wq = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * (static_cast<double>(q) + 0.5) / static_cast<double>(n_fft)));
        work[static_cast<std::size_t>(q)] = s.is_complex() ? s.channel_complex(row[e])[start + q] * wq
                                                           : std::complex<float>(s.channel(row[e])[start + q] * wq, 0.0F);
      }
      dsp::fft(work.data(), work.size(), false);
      const double lead = dot(pos[static_cast<std::size_t>(k)][e] - centres[static_cast<std::size_t>(k)], u0) / c;
      for (std::int64_t f = 0; f < n_f; ++f) {
        sum[static_cast<std::size_t>(f)] += Cd(work[static_cast<std::size_t>(bins[static_cast<std::size_t>(f)])]) *
                                            std::polar(1.0, -2.0 * kPi * freqs[static_cast<std::size_t>(f)] * lead);
      }
    }
    std::copy(sum.begin(), sum.end(), beams.begin() + (k * n_blocks + b) * n_f);
  });
  const std::vector<double> grid = elevation_grid(first_deg, last_deg, step_deg);
  std::vector<double> power(grid.size(), 0.0);
  parallel_for_host(static_cast<std::int64_t>(grid.size()), [&](std::int64_t e) {
    const Vec3d u = arrival_direction(frame, sb, std::sin(grid[static_cast<std::size_t>(e)]));
    std::vector<double> lead(static_cast<std::size_t>(n_k));
    for (std::int64_t k = 0; k < n_k; ++k) lead[static_cast<std::size_t>(k)] = dot(centres[static_cast<std::size_t>(k)], u) / c;
    double total = 0.0;
    for (std::int64_t f = 0; f < n_f; ++f) {
      std::vector<Cd> steer(static_cast<std::size_t>(n_k));
      for (std::int64_t k = 0; k < n_k; ++k) {
        steer[static_cast<std::size_t>(k)] = std::polar(1.0, -2.0 * kPi * freqs[static_cast<std::size_t>(f)] * lead[static_cast<std::size_t>(k)]);
      }
      for (std::int64_t b = 0; b < n_blocks; ++b) {
        Cd sum{};
        for (std::int64_t k = 0; k < n_k; ++k) sum += beams[static_cast<std::size_t>((k * n_blocks + b) * n_f + f)] * steer[static_cast<std::size_t>(k)];
        total += std::norm(sum);
      }
    }
    power[static_cast<std::size_t>(e)] = total;
  });
  const auto e = static_cast<std::int64_t>(std::max_element(power.begin(), power.end()) - power.begin());
  const auto n_e = static_cast<std::int64_t>(grid.size());
  const double frac = (e == 0 || e == n_e - 1) ? 0.0
                                               : parabola(power[static_cast<std::size_t>(e - 1)], power[static_cast<std::size_t>(e)],
                                                          power[static_cast<std::size_t>(e + 1)]);
  double n_total = 0.0, sum_sq = 0.0, incoherent = 0.0;
  for (const auto& r : rows) {
    n_total += static_cast<double>(r.size());
    sum_sq += static_cast<double>(r.size()) * static_cast<double>(r.size());
  }
  for (const Cd& v : beams) incoherent += std::norm(v);
  PassiveElevation out;
  out.elevation = grid[static_cast<std::size_t>(e)] + frac * step_deg * kDeg;
  out.quality = incoherent > 0.0 ? power[static_cast<std::size_t>(e)] / (n_total * n_total / sum_sq * incoherent) : 0.0;
  return out;
}

}  // namespace sonar::live
